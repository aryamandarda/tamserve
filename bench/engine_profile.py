"""Where an engine step's time goes at each camera count: one object per camera,
synthetic frames (cost depends on shapes, not pixels), steady state.

  frames_per_s      unprofiled
  gpu_idle_pct      share of a step the GPU had nothing queued: above zero, the
                    CPU launching work is the limit, not the GPU
  kernels_per_step  GPU kernels launched
  per_camera_ms     each stage, timed with the GPU synchronised around it
Options change one thing at a time.
"""
import argparse
import json
import os
import time
from collections import defaultdict

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from PIL import Image
from torch.profiler import ProfilerActivity, profile

import tamserve.engine as E
from tamserve.models import build_video_predictor

H, W = 360, 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="edgetam")
    ap.add_argument("--mode", default="best", help="'best' for Engine(compile=True), a torch.compile mode, or 'off'")
    ap.add_argument("--sizes", default="1,2,4,8,16,32,64")
    ap.add_argument("--encode-batch", type=int, help="frames per image-encoder call, past 8 frames")
    ap.add_argument("--object-batch", type=int, help="objects per decoder and memory-encoder call")
    ap.add_argument("--gpu-masks", action="store_true", help="leave masks on the GPU")
    ap.add_argument("--stages", action="store_true", help="add the profiled columns")
    args = ap.parse_args()
    E._ENCODE_BATCH = args.encode_batch or E._ENCODE_BATCH
    E._OBJECT_BATCH = args.object_batch or E._OBJECT_BATCH

    rng = np.random.default_rng(0)
    base = [E.prepare(Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)), device="cuda")
            for _ in range(8)]
    seed = torch.zeros(H, W, dtype=torch.bool)
    seed[40:80, 60:120] = True
    t = time.time()
    engine = E.Engine(build_video_predictor("cuda", args.model), gpu_masks=args.gpu_masks,
                      compile={"off": False, "best": True}.get(args.mode, args.mode))
    ready = round(time.time() - t, 1)
    m = engine.model

    for n in (int(s) for s in args.sizes.split(",")):
        own = [[(f.clone(), size) for f, size in base] for _ in range(n)]
        step = lambda k: engine.step({i: own[i][(k + i) % 8] for i in range(n)})

        def run(first, steps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            for k in range(first, first + steps):
                step(k)
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / steps * 1e3

        engine.step({i: own[i][0] for i in range(n)}, {i: {1: seed} for i in range(n)})
        run(1, 24)
        wall = run(25, 40)
        row = {"model": args.model, "mode": args.mode, "encode_batch": E._ENCODE_BATCH,
               "object_batch": E._OBJECT_BATCH, "gpu_masks": args.gpu_masks, "ready_s": ready, "cameras": n,
               "step_ms": round(wall, 2), "ms_per_camera": round(wall / n, 2), "frames_per_s": round(n / wall * 1e3, 1)}
        if args.stages:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                run(65, 10)
            ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            busy = sum(e.device_time for e in ev) / 1e3 / 10
            acc = defaultdict(float)

            def timed(name, fn):
                def wrapper(*a, **k):
                    torch.cuda.synchronize()
                    s = time.perf_counter()
                    out = fn(*a, **k)
                    torch.cuda.synchronize()
                    acc[name] += (time.perf_counter() - s) * 1e3 / 20
                    return out
                return wrapper

            patched = [(engine, "_encode"), (engine, "_attend"), (engine, "_outputs"), (engine, "_rows"),
                       (m, "_forward_sam_heads"), (m, "_encode_new_memory")]
            for obj, name in patched:
                setattr(obj, name, timed(name, getattr(obj, name)))
            total = run(75, 20)
            for obj, name in patched:
                del obj.__dict__[name]
            acc["other"] = total - sum(acc.values())
            row.update({"gpu_idle_pct": round(100 * (1 - busy / wall), 1), "kernels_per_step": len(ev) // 10,
                        "per_camera_ms": {k: round(v / n, 2) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}})
        print(json.dumps(row), flush=True)
        for i in range(n):
            engine.drop(i)


if __name__ == "__main__":
    main()
