"""How much stock EdgeTAM's own output varies. That variation is the bar batching must meet.

Per DAVIS sequence:
  repeat       the same run twice: is stock deterministic?
  composition  each object tracked alone vs alongside the others: the variation
               stock already shows just from the batch size changing
  precision    bf16 vs fp32
  accuracy     each variant against DAVIS ground truth: does a difference
               between runs make either one worse?
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

import davis
from tamserve.compare import compare, only
from tamserve.models import build_video_predictor
from tamserve.reference import track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis", default="data/davis/DAVIS")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/noise_floor.jsonl")
    args = ap.parse_args()

    root = Path(args.davis)
    seqs = davis.names(root, args.split)
    if args.limit:
        seqs = seqs[:args.limit]
    pred = build_video_predictor("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_test = defaultdict(list)
    with out.open("w") as fh:
        for seq in seqs:
            frames = str(davis.frames_dir(root, seq))
            seeds = davis.seeds(root, seq)
            gt = davis.ground_truth(root, seq, seeds)
            together = track(pred, frames, seeds)
            fp32 = track(pred, frames, seeds, bf16=False)
            rows = [
                {"test": "repeat", **compare(together, track(pred, frames, seeds))},
                {"test": "precision", **compare(fp32, together)},
            ]
            for o, m in seeds.items():
                acc = {"together": compare(only(gt, o), only(together, o))["mean_iou"],
                       "fp32": compare(only(gt, o), only(fp32, o))["mean_iou"]}
                if len(seeds) > 1:
                    solo = track(pred, frames, {o: m})
                    rows.append({"test": "composition", "obj": o,
                                 **compare(solo, only(together, o))})
                    acc["solo"] = compare(only(gt, o), solo)["mean_iou"]
                rows.append({"test": "accuracy", "obj": o, **acc})
            for r in rows:
                fh.write(json.dumps({"seq": seq, "objects": len(seeds), **r}) + "\n")
                by_test[r["test"]].append(r)
            print(seq, " ".join(f"{r['test']}={r['mean_iou']:.4f}"
                                for r in rows if "mean_iou" in r))

    acc = by_test.pop("accuracy", [])
    multi = [r for r in acc if "solo" in r]
    deltas = [r["solo"] - r["together"] for r in multi]
    print(json.dumps({
        "test": "accuracy_vs_ground_truth",
        "objects": len(acc),
        "bf16": round(float(np.mean([r["together"] for r in acc])), 5),
        "fp32": round(float(np.mean([r["fp32"] for r in acc])), 5),
        "multi_object_objects": len(multi),
        "together": round(float(np.mean([r["together"] for r in multi])), 5),
        "solo": round(float(np.mean([r["solo"] for r in multi])), 5),
        "solo_minus_together_mean": round(float(np.mean(deltas)), 5),
        "solo_minus_together_range": [round(min(deltas), 4), round(max(deltas), 4)],
    }))
    for test, rs in by_test.items():
        print(json.dumps({
            "test": test,
            "runs": len(rs),
            "mean_iou": round(float(np.mean([r["mean_iou"] for r in rs])), 5),
            "worst_p1_iou": round(min(r["p1_iou"] for r in rs), 4),
            "worst_min_iou": round(min(r["min_iou"] for r in rs), 4),
            "lost": sum(r["lost"] for r in rs),
            "phantom": sum(r["phantom"] for r in rs),
        }))


if __name__ == "__main__":
    main()
