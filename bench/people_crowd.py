"""PeopleTracker in a packed crowd: MOT20's training videos, 60 to 200 people a frame
(https://motchallenge.net/data/MOT20.zip, unpacked into --data).

MOT20 labels people with boxes, not outlines, so each mask is scored as the box around
it, with TrackEval's box HOTA (scripts/setup_trackeval.sh). Its boxes cover the whole
person, hidden parts included, and a mask covers only what can be seen, so anyone
partly hidden is scored too low: read HOTA here as a floor. The plainer number is
"covered": the share of labelled people that some tracked box overlaps by at least a
half, given for everyone and for those at least three quarters visible.

The four videos run together as four cameras, each one's first --frames frames.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from PIL import Image

from tamserve import Engine, PeopleFinder, PeopleTracker, build_video_predictor
from tamserve.reference import clock

ROOT = Path(__file__).resolve().parents[1]


def covered(labelled, tracked):
    """Of each frame's labelled boxes (x0, y0, x1, y1), which a tracked box overlaps by at least a half."""
    if not len(tracked):
        return np.zeros(len(labelled), bool)
    a, b = np.array(labelled)[:, None], np.array(tracked)[None]
    inter = (np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
             * np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None))
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return (inter / (area(a) + area(b) - inter)).max(1) >= 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/MOT20")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--look-every", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.4, help="the finder's confidence threshold")
    ap.add_argument("--start", type=float, default=0.7, help="the confidence that starts following someone")
    ap.add_argument("--sam3", help="path to sam3.pt; without it the weights come from Hugging Face")
    ap.add_argument("--compile", nargs="?", const=True, default=False,
                    help="compile the engine (optionally a mode) and the looks")
    ap.add_argument("--out", default="results/crowd")
    args = ap.parse_args()
    train, out = Path(args.data) / "train", Path(args.out)
    seqs = sorted(p.name for p in train.iterdir() if (p / "gt" / "gt.txt").exists())
    name = f"people-every{args.look_every}-start{args.start}-t{args.threshold}" + ("-compiled" if args.compile else "")

    # The labels cut to the same frames, laid out as TrackEval wants them. A label row is
    # frame, id, left, top, width, height, scored (1 or 0), class (1: a person on foot), visible share.
    gt_root, labelled = out / "gt" / "MOT20-train", {}
    for s in seqs:
        (gt_root / s / "gt").mkdir(parents=True, exist_ok=True)
        rows = [r for r in (train / s / "gt" / "gt.txt").read_text().splitlines() if int(r.split(",")[0]) <= args.frames]
        (gt_root / s / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
        ini = (train / s / "seqinfo.ini").read_text().splitlines()
        (gt_root / s / "seqinfo.ini").write_text("\n".join(f"seqLength={args.frames}" if l.startswith("seqLength") else l
                                                           for l in ini) + "\n")
        for f, _, x, y, w, h, use, kind, visible in (list(map(float, r.split(",")[:9])) for r in rows):
            if use and kind == 1:
                labelled.setdefault((s, int(f)), []).append((x, y, x + w, y + h, visible))
    seqmap = out / "gt" / "seqmaps" / "MOT20-train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")

    finder = PeopleFinder(checkpoint=args.sam3, threshold=args.threshold, compile="default" if args.compile else None)
    tracker = PeopleTracker(Engine(build_video_predictor("cuda"), compile=args.compile), finder=finder,
                            look_every=args.look_every, start=args.start)
    paths = {s: sorted((train / s / "img1").glob("*.jpg"))[:args.frames] for s in seqs}
    lines, seconds, frames, tracked = {s: [] for s in seqs}, 0.0, 0, 0
    hit = {"everyone": [0, 0], "three quarters visible or more": [0, 0]}
    torch.cuda.reset_peak_memory_stats()
    with ThreadPoolExecutor(8) as pool:
        for k in range(args.frames):
            live = [s for s in seqs if k < len(paths[s])]
            images = dict(zip(live, pool.map(lambda s: np.asarray(Image.open(paths[s][k]).convert("RGB")), live)))
            t = clock("cuda")
            result = tracker.step(images)
            seconds += clock("cuda") - t
            frames += len(live)
            for s, people in result.items():
                boxes = []
                for i, m in people.items():
                    ys, xs = m.any(1).nonzero().flatten(), m.any(0).nonzero().flatten()
                    boxes.append((int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1))
                    x0, y0, x1, y1 = boxes[-1]
                    lines[s].append(f"{k + 1},{i},{x0},{y0},{x1 - x0},{y1 - y0},1,-1,-1,-1")
                tracked += len(boxes)
                there = labelled.get((s, k + 1), [])
                if there:
                    got = covered([b[:4] for b in there], boxes)
                    for (*_, visible), g in zip(there, got):
                        for key in ["everyone"] + ["three quarters visible or more"] * (visible >= 0.75):
                            hit[key][0] += int(g)
                            hit[key][1] += 1
    data = out / "trackers" / "MOT20-train" / name / "data"
    data.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        (data / f"{s}.txt").write_text("\n".join(lines[s]) + "\n" if lines[s] else "")

    sys.path.insert(0, str(ROOT / "third_party" / "TrackEval"))
    import trackeval
    quiet = {"PRINT_RESULTS": False, "PRINT_CONFIG": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
             "PLOT_CURVES": False, "TIME_PROGRESS": False, "USE_PARALLEL": False}
    evaluator = trackeval.Evaluator({**trackeval.Evaluator.get_default_eval_config(), **quiet})
    dataset = trackeval.datasets.MotChallenge2DBox({
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(), "GT_FOLDER": str(out / "gt"),
        "TRACKERS_FOLDER": str(out / "trackers"), "BENCHMARK": "MOT20", "SPLIT_TO_EVAL": "train",
        "TRACKERS_TO_EVAL": [name], "SEQMAP_FILE": str(seqmap), "PRINT_CONFIG": False})
    res, _ = evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR()])
    r = res["MotChallenge2DBox"][name]["COMBINED_SEQ"]["pedestrian"]
    row = {"system": name, "frames": frames, "people_per_frame_labelled": round(hit["everyone"][1] / frames, 1),
           "people_per_frame_followed": round(tracked / frames, 1),
           **{f"covered, {k}": round(100 * a / max(b, 1), 1) for k, (a, b) in hit.items()},
           "HOTA": round(float(np.mean(r["HOTA"]["HOTA"])) * 100, 2), "IDSW": int(r["CLEAR"]["IDSW"]),
           "ms_per_frame": round(seconds / frames * 1e3, 1),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(json.dumps(row), flush=True)
    with (out / "results.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
