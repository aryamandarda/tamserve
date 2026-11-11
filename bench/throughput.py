"""The engine against the best ways to run the stock predictor on many cameras.

All arms use synthetic frames already on the GPU (cost depends on shapes, not
pixels), one object per camera, every memory full, one GPU:
  stock-eager      stock, one camera at a time
  stock-compiled   the same with its modules compiled as SAM2's vos_optimized
                   mode compiles SAM2 (EdgeTAM's repo ships no compiled mode)
  procs            N stock processes at once on the GPU, one camera each, with
                   and without NVIDIA MPS (which lets processes share the GPU)
  procs-compiled   the same with every process compiled: the strongest setup
                   that needs no engine
  engine-eager     tamserve at N cameras
  engine-compiled  tamserve at N cameras, Engine(compile=True) unless --engine-mode
                   names one torch.compile mode
  engine-procs     compiled engines in processes of their own, each batching N
                   cameras: with one process, the engine measured exactly as
                   the stock processes are
Prints frames per second, GPU memory, host memory and time to ready per arm.
Stock leaves its masks on the GPU; --gpu-masks has the engine do the same.
Compilation changes rounding; accuracy is measured by bench/accuracy.py.
--stock sam2 runs the stock arms on Meta's current sam2 package instead of the
tracker in EdgeTAM's repository (SAM 2.1 models only; see bench/meta_sam2.py),
compiled the same way; --stock sam2-vos compiles it with its own vos_optimized
mode instead.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import psutil
import torch
from PIL import Image

import meta_sam2
from tamserve.models import build_video_predictor
from tamserve.engine import Engine, prepare
from tamserve.reference import clock, precision

H, W = 360, 640
FULL = 16  # frames until an object's memory is full
compiled = lambda mode: {"": False, "off": False, "best": True}.get(mode, mode)  # Engine's compile argument


def frame_ms(pred, frames_dir):
    """Stock's median time per frame once the memory is full, one object."""
    times = []
    with torch.inference_mode(), precision("cuda"):
        state = pred.init_state(frames_dir)
        seed = torch.zeros(H, W, dtype=torch.bool)
        seed[40:80, 60:120] = True
        pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed)
        t = clock("cuda")
        for f, _, _ in pred.propagate_in_video(state):
            now = clock("cuda")
            if f >= FULL:
                times.append((now - t) * 1e3)
            t = now
    return float(np.median(times))


def _clone(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    return x


def compile_stock(pred, mode, warm_dir):
    """SAM2's vos_optimized compilation applied to a stock predictor, plus EdgeTAM's
    spatial perceiver. Outputs are cloned because CUDA-graph modes reuse output
    buffers. One uncompiled pass first fills the cached position encodings; filled
    inside a CUDA graph, a later replay would overwrite them."""
    transformer = sys.modules[type(pred.sam_mask_decoder.transformer).__module__]

    frame_ms(pred, warm_dir)
    # As the engine does: where stock pins its attention kernel through a context
    # manager, that cuts its attention modules into many graphs; its own fallback
    # switch lets each compile whole.
    if hasattr(transformer, "ALLOW_ALL_KERNELS"):
        transformer.ALLOW_ALL_KERNELS = True
    torch._dynamo.config.cache_size_limit = 64

    def wrap(fn, dynamic):
        compiled = torch.compile(fn, mode=mode, dynamic=dynamic)
        return lambda *a, **k: _clone(compiled(*a, **k))

    pred.image_encoder.forward = wrap(pred.image_encoder.forward, False)
    pred.memory_encoder.forward = wrap(pred.memory_encoder.forward, False)
    if getattr(pred, "spatial_perceiver", None) is not None:  # EdgeTAM only
        pred.spatial_perceiver.forward = wrap(pred.spatial_perceiver.forward, False)
    pred.memory_attention.forward = wrap(pred.memory_attention.forward, True)  # memory length varies
    pred.sam_prompt_encoder.forward = wrap(pred.sam_prompt_encoder.forward, False)
    pred.sam_mask_decoder.forward = wrap(pred.sam_mask_decoder.forward, False)


def stock(model, which, mode, warm_dir):
    """The stock predictor, compiled when `mode` is set: the tracker in EdgeTAM's
    repository, or the one in Meta's current sam2 package, compiled the same way
    ("sam2") or by its own vos_optimized mode ("sam2-vos")."""
    if which == "sam2-vos":
        return meta_sam2.build(model, compiled=bool(mode))
    pred = meta_sam2.build(model) if which == "sam2" else build_video_predictor("cuda", model)
    if mode:
        compile_stock(pred, mode, warm_dir)
    return pred


def synthetic_video(d, frames):
    rng = np.random.default_rng(0)
    pool = [Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)) for _ in range(8)]
    for i in range(frames):
        pool[i % len(pool)].save(f"{d}/{i}.jpg")
    seed = torch.zeros(H, W, dtype=torch.bool)
    seed[40:80, 60:120] = True
    return [prepare(p) for p in pool], {1: seed}


def one_camera(pred, d):
    """Steady-state stock frames per second; the first run warms up (and compiles)."""
    t = time.perf_counter()
    frame_ms(pred, d)
    warm_s = time.perf_counter() - t
    ms = frame_ms(pred, d)
    return {"frame_ms": round(ms, 2), "frames_per_s": round(1000 / ms, 1), "warmup_s": round(warm_s, 1)}


def gpu_used_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return round(int(out.split()[0]) / 1024, 2)


def engine_fps(engine, frames, seed, n, warm=20, steps=30):
    """n cameras of one object through one engine; each camera has its own frame tensors."""
    own = [[(f.clone(), size) for f, size in frames] for _ in range(n)]
    engine.step({i: own[i][0] for i in range(n)}, {i: seed for i in range(n)})
    for k in range(1, warm):
        engine.step({i: own[i][(k + i) % len(frames)] for i in range(n)})
    torch.cuda.reset_peak_memory_stats()
    times = []
    for k in range(steps):
        batch = {i: own[i][(k + i) % len(frames)] for i in range(n)}
        t0 = clock("cuda")
        engine.step(batch)
        times.append((clock("cuda") - t0) * 1e3)
    ms = float(np.median(times))
    out = {"cameras": n, "step_ms": round(ms, 2), "frames_per_s": round(n / ms * 1e3, 1),
           "gpu_gb": gpu_used_gb(), "gpu_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "host_gb": round(psutil.Process().memory_info().rss / 2**30, 2)}
    for i in range(n):
        engine.drop(i)
    return out


def _wait_for_go(sync, i, skip, window):
    """Tell the parent this worker is ready; return the window every worker measures in."""
    Path(sync, f"ready.{i}").touch()
    while not Path(sync, "go").exists():
        time.sleep(0.05)
    lo = float(Path(sync, "go").read_text()) + skip
    return lo, lo + window


def worker(d, sync, i, skip, window, model, mode, which):
    """One stock process: track a synthetic video held on the GPU, in a loop. Reports
    the frames it tracked with a full memory inside the window all workers share,
    and the time those took. Compiled first when `mode` is set."""
    t = time.time()
    pred = stock(model, which, mode, d)
    if mode:
        frame_ms(pred, d)  # compiles
    seed = torch.zeros(H, W, dtype=torch.bool)
    seed[40:80, 60:120] = True
    with torch.inference_mode(), precision("cuda"):
        state = pred.init_state(d)
        pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed)
        ready_s = time.time() - t
        lo, hi = _wait_for_go(sync, i, skip, window)
        count, busy = 0, 0.0
        while time.time() < hi:
            prev = clock("cuda")
            for f, _, _ in pred.propagate_in_video(state):
                now = clock("cuda")
                if f >= FULL and lo <= time.time() < hi:
                    count, busy = count + 1, busy + now - prev
                prev = now
                if time.time() >= hi:
                    break
    print(json.dumps({"frames": count, "busy_s": busy, "ready_s": round(ready_s, 1),
                      "host_gb": round(psutil.Process().memory_info().rss / 2**30, 2)}), flush=True)


def engine_worker(sync, i, skip, window, model, mode, n, gpu_masks):
    """One engine process following one object on each of n cameras; reports as worker() does."""
    t = time.time()
    rng = np.random.default_rng(int(i))
    frames = [prepare(Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)), device="cuda") for _ in range(8)]
    seed = torch.zeros(H, W, dtype=torch.bool)
    seed[40:80, 60:120] = True
    engine = Engine(build_video_predictor("cuda", model), compile=compiled(mode), gpu_masks=gpu_masks)
    own = [[(f.clone(), size) for f, size in frames] for _ in range(n)]
    engine.step({j: own[j][0] for j in range(n)}, {j: {1: seed} for j in range(n)})
    for k in range(1, 20):
        engine.step({j: own[j][(k + j) % 8] for j in range(n)})
    ready_s = time.time() - t
    lo, hi = _wait_for_go(sync, i, skip, window)
    count, busy, k, prev = 0, 0.0, 0, clock("cuda")
    while time.time() < hi:
        engine.step({j: own[j][(k + j) % 8] for j in range(n)})
        now = clock("cuda")
        if lo <= time.time() < hi:
            count, busy = count + n, busy + now - prev
        prev, k = now, k + 1
    print(json.dumps({"frames": count, "busy_s": busy, "ready_s": round(ready_s, 1),
                      "host_gb": round(psutil.Process().memory_info().rss / 2**30, 2)}), flush=True)


def procs(n, d, mps, model, mode=None, window=20, skip=10, patience=3600, cameras=0, gpu_masks=False, which="edgetam"):
    if mps:
        ok = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True).returncode == 0
        if not ok:
            return {"procs": n, "mps": True, "error": "MPS daemon could not start on this host"}
    try:
        with tempfile.TemporaryDirectory() as sync:
            cmd = [sys.executable, __file__, "--worker", d, "--sync", sync, "--skip", str(skip), "--window", str(window),
                   "--model", model, "--mode", mode or "", "--engine-cameras", str(cameras), "--stock", which]
            cmd += ["--gpu-masks"] if gpu_masks else []
            ps = [subprocess.Popen(cmd + ["--index", str(i)], stdout=subprocess.PIPE,
                                   stderr=open(Path(sync, f"err.{i}"), "w"), text=True) for i in range(n)]
            t = time.time()
            while len(list(Path(sync).glob("ready.*"))) < n:  # every worker loaded (and compiled)
                dead = [i for i, p in enumerate(ps) if p.poll() is not None]
                if dead or time.time() - t > patience:
                    for p in ps:
                        p.kill()
                    why = Path(sync, f"err.{dead[0]}").read_text().strip().splitlines()[-1:] if dead else ["timed out"]
                    return {"procs": n, "mps": mps, "compiled": bool(mode), "cameras_each": cameras,
                            "error": f"{len(dead)} of {n} workers died before ready", "why": why[0][:300] if why else ""}
                time.sleep(1)
            ready_s = time.time() - t
            Path(sync, "go.tmp").write_text(str(time.time() + 5))
            Path(sync, "go.tmp").rename(Path(sync, "go"))
            time.sleep(5 + skip + window / 2)
            gpu_gb = gpu_used_gb()  # mid-window, every process loaded
            outs = [p.communicate()[0].strip().splitlines() for p in ps]
        if not all(outs):
            return {"procs": n, "mps": mps, "compiled": bool(mode),
                    "error": f"{sum(not o for o in outs)} of {n} workers died"}
        res = [json.loads(o[-1]) for o in outs]
    finally:
        if mps:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, capture_output=True)
    rates = [r["frames"] / r["busy_s"] for r in res]
    return {"procs": n, "mps": mps, "compiled": bool(mode), "cameras_each": cameras,
            "frames_per_s": round(sum(rates), 1), "per_process": round(float(np.mean(rates)), 1),
            "gpu_gb": gpu_gb, "host_gb": round(sum(r["host_gb"] for r in res), 1), "ready_s": round(ready_s, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="stock-eager,stock-compiled,engine-eager,engine-compiled,procs,procs-compiled")
    ap.add_argument("--sizes", default="4,8,12,16,32,64")
    ap.add_argument("--procs", default="4,8,16")
    ap.add_argument("--mode", default="max-autotune", help="torch.compile mode for stock")
    ap.add_argument("--stock", default="edgetam", choices=["edgetam", "sam2", "sam2-vos"],
                    help="whose tracker the stock arms run: EdgeTAM's repository's, or Meta's current sam2 package "
                         "compiled the same way (sam2) or by its own vos_optimized mode (sam2-vos)")
    ap.add_argument("--engine-mode", default="best",
                    help="for engine-compiled and engine-procs: 'best' for Engine(compile=True), a torch.compile "
                         "mode, or 'off'")
    ap.add_argument("--model", default="edgetam")
    ap.add_argument("--worker")
    ap.add_argument("--sync")
    ap.add_argument("--index")
    ap.add_argument("--engine-cameras", type=int, default=0, help="worker: an engine with this many cameras")
    ap.add_argument("--gpu-masks", action="store_true", help="engine arms: leave masks on the GPU, as stock does")
    ap.add_argument("--skip", type=float)
    ap.add_argument("--window", type=float)
    args = ap.parse_args()
    if args.worker and args.engine_cameras:
        return engine_worker(args.sync, args.index, args.skip, args.window, args.model, args.mode,
                             args.engine_cameras, args.gpu_masks)
    if args.worker:
        return worker(args.worker, args.sync, args.index, args.skip, args.window, args.model, args.mode, args.stock)

    arms = args.arms.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    with tempfile.TemporaryDirectory() as d48, tempfile.TemporaryDirectory() as dlong:
        frames, seed = synthetic_video(d48, 48)
        # Each stock worker holds its video on the GPU as float32 (12.6 MB a frame),
        # so keep it short; workers loop over it.
        synthetic_video(dlong, 64)
        if not set(arms) <= {"procs", "procs-compiled", "engine-procs"}:  # those arms' parent stays off the GPU
            frames = [(f.cuda(), size) for f, size in frames]
        if "stock-eager" in arms:
            print(json.dumps({"model": args.model, "arm": "stock-eager", "stock": args.stock,
                              **one_camera(stock(args.model, args.stock, None, d48), d48)}), flush=True)
        if "engine-eager" in arms:
            engine = Engine(build_video_predictor("cuda", args.model), gpu_masks=args.gpu_masks)
            for n in sizes:
                print(json.dumps({"model": args.model, "arm": "engine-eager", **engine_fps(engine, frames, seed, n)}), flush=True)
            del engine
        if "engine-compiled" in arms:
            t = time.time()
            engine = Engine(build_video_predictor("cuda", args.model), compile=compiled(args.engine_mode),
                            gpu_masks=args.gpu_masks)
            ready_s = round(time.time() - t, 1)
            for n in sizes:
                print(json.dumps({"model": args.model, "arm": "engine-compiled", "mode": args.engine_mode, "ready_s": ready_s,
                                  **engine_fps(engine, frames, seed, n)}), flush=True)
            del engine
        torch.cuda.empty_cache()
        for arm, mode in (("procs", None), ("procs-compiled", args.mode)):
            if arm in arms:
                for mps in ((False, True) if mode is None else (True,)):
                    for n in (int(p) for p in args.procs.split(",")):
                        print(json.dumps({"model": args.model, "arm": arm, "mode": mode, "stock": args.stock,
                                          **procs(n, dlong, mps, args.model, mode, which=args.stock)}), flush=True)
        if "engine-procs" in arms:
            for n in (int(p) for p in args.procs.split(",")):
                for k in sizes:
                    print(json.dumps({"model": args.model, "arm": "engine-procs", "mode": args.engine_mode,
                                      "gpu_masks": args.gpu_masks,
                                      **procs(n, dlong, n > 1, args.model, args.engine_mode, cameras=k,
                                              gpu_masks=args.gpu_masks)}), flush=True)
        if "stock-compiled" in arms:
            pred = stock(args.model, args.stock, args.mode, d48)
            print(json.dumps({"model": args.model, "arm": "stock-compiled", "mode": args.mode, "stock": args.stock,
                              **one_camera(pred, d48)}), flush=True)


if __name__ == "__main__":
    main()
