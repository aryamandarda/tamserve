"""The engine against stock on DAVIS 2017 val: every video is a camera, all
batched together, compared with stock tracking each video alone and with the
ground truth.

Cameras join at staggered steps, advance one frame per step and leave when their
video ends, so every step mixes objects that are starting, still filling their
memory, and in steady state.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

import davis
from tamserve.compare import compare
from tamserve.models import build_video_predictor
from tamserve.engine import Engine, prepare
from tamserve.reference import clock, track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis", default="data/davis/DAVIS")
    ap.add_argument("--split", default="val")
    ap.add_argument("--stagger", type=int, default=3, help="steps between camera starts")
    ap.add_argument("--fp32", action="store_true", help="the logic check: everything in fp32")
    ap.add_argument("--model", default="edgetam")
    ap.add_argument("--compile", default="off",
                    help="'best' for Engine(compile=True), a torch.compile mode, or 'off' (the stock reference "
                         "stays uncompiled)")
    ap.add_argument("--out", default="results/accuracy.jsonl")
    args = ap.parse_args()

    root = Path(args.davis)
    bf16 = not args.fp32
    pred = build_video_predictor("cuda" if torch.cuda.is_available() else "cpu", args.model)
    seqs = davis.names(root, args.split)
    seeds = {s: davis.seeds(root, s) for s in seqs}
    paths = {s: sorted(davis.frames_dir(root, s).glob("*.jpg")) for s in seqs}
    start = {s: i * args.stagger for i, s in enumerate(seqs)}

    serial_s, refs = 0.0, {}
    for s in seqs:
        st = {}
        refs[s] = track(pred, str(davis.frames_dir(root, s)), seeds[s], bf16=bf16, stats=st)
        serial_s += st["propagate_s"]

    engine = Engine(pred, bf16=bf16, compile={"off": False, "best": True}.get(args.compile, args.compile))
    got = {s: {} for s in seqs}
    engine_s, live_counts = 0.0, []
    steps = max(start[s] + len(paths[s]) for s in seqs)
    with ThreadPoolExecutor(16) as pool:
        for k in range(steps):
            live = [s for s in seqs if start[s] <= k < start[s] + len(paths[s])]
            frames = dict(zip(live, pool.map(lambda s: prepare(paths[s][k - start[s]], device=engine.device), live)))
            joining = {s: seeds[s] for s in live if k == start[s]}
            t = clock(engine.device)
            out = engine.step(frames, joining)
            engine_s += clock(engine.device) - t
            live_counts.append(len(live))
            for s, masks in out.items():
                got[s][k - start[s]] = masks
            for s in live:
                if k == start[s] + len(paths[s]) - 1:
                    engine.drop(s)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("w") as fh:
        for s in seqs:
            gt = davis.ground_truth(root, s, seeds[s])
            row = {"seq": s, "objects": len(seeds[s]), "frames": len(paths[s]),
                   **compare(refs[s], got[s]),
                   "acc_stock": compare(gt, refs[s])["mean_iou"],
                   "acc_engine": compare(gt, got[s])["mean_iou"]}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
    deltas = [r["acc_engine"] - r["acc_stock"] for r in rows]
    print(json.dumps({
        "precision": "fp32" if args.fp32 else "bf16",
        "model": args.model,
        "compile": args.compile,
        "cameras": len(rows),
        "steps": steps,
        "live_cameras_mean": round(float(np.mean(live_counts)), 1),
        "live_cameras_max": max(live_counts),
        "mean_iou": round(float(np.mean([r["mean_iou"] for r in rows])), 5),
        "worst_p1_iou": round(min(r["p1_iou"] for r in rows), 4),
        "worst_min_iou": round(min(r["min_iou"] for r in rows), 4),
        "missing": sum(r["missing"] for r in rows),
        "extra": sum(r["extra"] for r in rows),
        "object_frames": sum(r["object_frames"] for r in rows),
        "lost": sum(r["lost"] for r in rows),
        "phantom": sum(r["phantom"] for r in rows),
        "acc_stock": round(float(np.mean([r["acc_stock"] for r in rows])), 5),
        "acc_engine": round(float(np.mean([r["acc_engine"] for r in rows])), 5),
        "acc_delta_mean": round(float(np.mean(deltas)), 5),
        "acc_delta_range": [round(min(deltas), 4), round(max(deltas), 4)],
        "better_worse_same": [sum(d > 0 for d in deltas), sum(d < 0 for d in deltas),
                              sum(d == 0 for d in deltas)],
        "serial_s": round(serial_s, 2),
        "engine_s": round(engine_s, 2),
        "speedup": round(serial_s / engine_s, 2),
    }))


if __name__ == "__main__":
    main()
