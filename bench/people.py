"""How well PeopleTracker follows people and what it costs, against SAM 3's own video tracker.

A dataset is laid out like MOTS20: train/<sequence>/img1/*.jpg and gt/gt.txt, people
with a mask and an id on every labelled frame. Each sequence is one camera and all of
them run together, so looks interleave as they would across live cameras. Systems:
  people  tamserve's PeopleTracker
  sam3    SAM 3's own video tracking as Meta ships it, one video at a time
Scored with TrackEval (scripts/setup_trackeval.sh): HOTA, which combines finding people
(DetA) with keeping the same id on each one (AssA), IDF1 and ID switches. Cost is wall
time per frame once the GPU has finished; reading frames from disk is not counted.

  --long N   memory over a long run instead of a score: the first sequence's frames,
             repeated to N frames, as one video
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pycocotools.mask as rle
import torch
from PIL import Image

from tamserve import Engine, PeopleFinder, PeopleTracker, build_video_predictor
from tamserve.reference import clock

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sequences(data):
    return sorted(p.name for p in (Path(data) / "train").iterdir() if (p / "gt" / "gt.txt").exists())


def first_frames(data, n, out):
    """A copy of the dataset cut to each sequence's first n frames (images linked, labels
    filtered): SAM 3's video tracker cannot finish a full MOTS20 sequence in 24 GB."""
    root = Path(out) / f"{Path(data).resolve().name}_first{n}"
    for seq in sequences(data):
        src, dst = Path(data) / "train" / seq, root / "train" / seq
        if (dst / "seqinfo.ini").exists():
            continue
        (dst / "img1").mkdir(parents=True, exist_ok=True)
        (dst / "gt").mkdir(exist_ok=True)
        for p in sorted((src / "img1").glob("*.jpg"))[:n]:
            (dst / "img1" / p.name).symlink_to(p.resolve())
        last = n  # the last labelled frame kept: labels are numbered by scored frame
        if (src / "scored.txt").exists():
            marks = [k for k in (src / "scored.txt").read_text().split() if int(k) <= n]
            (dst / "scored.txt").write_text("\n".join(marks) + "\n")
            last = len(marks)
        if (src / "prompt.txt").exists():
            (dst / "prompt.txt").write_text((src / "prompt.txt").read_text())
        rows = [r for r in (src / "gt" / "gt.txt").read_text().splitlines() if int(r.split(" ", 1)[0]) <= last]
        (dst / "gt" / "gt.txt").write_text("\n".join(rows) + "\n" if rows else "")
        ini = (src / "seqinfo.ini").read_text().splitlines()
        (dst / "seqinfo.ini").write_text("\n".join(f"seqLength={last}" if l.startswith("seqLength") else l
                                                   for l in ini) + "\n")
    return root


def long_video(data, n, out):
    """One sequence of n frames: the first sequence's frames, repeated. Nothing to score."""
    src = sorted((Path(data) / "train" / sequences(data)[0] / "img1").glob("*.jpg"))
    dst = Path(out) / f"long{n}" / "train" / "long" / "img1"
    if not dst.exists():
        dst.mkdir(parents=True)
        for k in range(n):
            (dst / f"{k + 1:06d}.jpg").symlink_to(src[k % len(src)].resolve())
    return dst.parents[2]


def scored(seq_dir):
    """Run frame -> scored frame number. Every frame is scored unless the sequence lists
    its labelled frames in scored.txt (the home videos are labelled once a second)."""
    f = seq_dir / "scored.txt"
    return {int(k): j for j, k in enumerate(f.read_text().split(), start=1)} if f.exists() else None


def mots_lines(frame, tracks, h, w):
    """MOTS rows for one frame; masks may not overlap, so older tracks keep shared pixels."""
    taken, rows = np.zeros((h, w), bool), []
    for tid in sorted(tracks):
        m = np.asarray(tracks[tid], dtype=bool) & ~taken
        taken |= m
        if m.any():
            counts = rle.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()
            rows.append(f"{frame} {2000 + tid} 2 {h} {w} {counts}")
    return rows


class Timed:
    """A finder, with the time its looks take."""

    def __init__(self, finder):
        self.finder, self.seconds, self.looks = finder, 0.0, 0

    def __call__(self, frames, *prompts):
        t = clock(DEVICE)
        out = self.finder(frames, *prompts)
        self.seconds += clock(DEVICE) - t
        self.looks += len(frames)
        return out

    def __getattr__(self, name):  # scores, names
        return getattr(self.finder, name)


def run_people(args, seqs, train):
    finder = Timed(PeopleFinder(checkpoint=args.sam3, threshold=args.threshold,
                                compile="default" if args.compile else None))
    tracker = PeopleTracker(Engine(build_video_predictor(DEVICE), compile=args.compile), finder=finder,
                            look_every=args.look_every, hold=args.hold)
    torch.cuda.reset_peak_memory_stats()
    paths = {s: sorted((train / s / "img1").glob("*.jpg")) for s in seqs}
    keep = {s: scored(train / s) for s in seqs}
    lines = {s: [] for s in seqs}
    seconds, frames = 0.0, 0
    with ThreadPoolExecutor(8) as pool:
        for k in range(max(len(p) for p in paths.values())):
            live = [s for s in seqs if k < len(paths[s])]
            images = dict(zip(live, pool.map(lambda s: np.asarray(Image.open(paths[s][k]).convert("RGB")), live)))
            t = clock(DEVICE)
            out = tracker.step(images)
            seconds += clock(DEVICE) - t
            frames += len(live)
            for s, people in out.items():
                f = k + 1 if keep[s] is None else keep[s].get(k + 1)
                if f is not None:
                    lines[s] += mots_lines(f, people, *images[s].shape[:2])
                if k == len(paths[s]) - 1:
                    tracker.drop(s)
            if args.long and k % 500 == 499:
                print(json.dumps({"frame": k + 1, "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2)}),
                      flush=True)
    return lines, {"frames": frames, "seconds": seconds, "look_seconds": finder.seconds, "looks": finder.looks,
                   "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def run_sam3(args, seqs, train, prompts=None):
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(checkpoint_path=args.sam3)
    lines, seconds, frames = {s: [] for s in seqs}, 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    for s in seqs:
        w, h = Image.open(next((train / s / "img1").glob("*.jpg"))).size
        keep = scored(train / s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Frames and tracking state in CPU memory, or a 24 GB GPU runs out sooner still.
            sid = predictor.handle_request(dict(type="start_session", offload_video_to_cpu=True,
                                                offload_state_to_cpu=True,
                                                resource_path=str(train / s / "img1")))["session_id"]
            t = clock(DEVICE)
            predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0,
                                          text=prompts[s] if prompts else "person"))
            for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                o = r["outputs"]
                tracks = {int(i): m for i, m in zip(o["out_obj_ids"], o["out_binary_masks"])}
                f = r["frame_index"] + 1 if keep is None else keep.get(r["frame_index"] + 1)
                if f is not None:
                    lines[s] += mots_lines(f, tracks, h, w)
                frames += 1
                if r["frame_index"] % 100 == 99:  # its memory grows with the video's length
                    print(json.dumps({"seq": s, "frame": r["frame_index"] + 1,
                                      "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 1)}), flush=True)
            seconds += clock(DEVICE) - t
            predictor.handle_request(dict(type="close_session", session_id=sid))
    return lines, {"frames": frames, "seconds": seconds,
                   "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def score(name, lines, seqs, data, out_dir):
    sys.path.insert(0, str(ROOT / "third_party" / "TrackEval"))
    import trackeval

    dataset = Path(data).resolve().name
    gt_root, trk_root = out_dir / "gt" / dataset, out_dir / "trackers" / dataset
    split = gt_root / "MOTS-train"
    if not split.exists():
        split.parent.mkdir(parents=True, exist_ok=True)
        split.symlink_to((Path(data) / "train").resolve())
    seqmap = gt_root / "seqmaps" / "MOTS-train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")
    out = trk_root / "MOTS-train" / name / "data"
    out.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        # Nobody tracked in a sequence: an empty file (TrackEval cannot parse a lone newline).
        (out / f"{s}.txt").write_text("\n".join(lines[s]) + "\n" if lines[s] else "")
    quiet = {"PRINT_RESULTS": False, "PRINT_CONFIG": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
             "PLOT_CURVES": False, "TIME_PROGRESS": False, "USE_PARALLEL": False}
    evaluator = trackeval.Evaluator({**trackeval.Evaluator.get_default_eval_config(), **quiet})
    mots = trackeval.datasets.MOTSChallenge({
        **trackeval.datasets.MOTSChallenge.get_default_dataset_config(),
        "GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trk_root), "SPLIT_TO_EVAL": "train",
        "TRACKERS_TO_EVAL": [name], "SEQMAP_FILE": str(seqmap), "PRINT_CONFIG": False})
    res, _ = evaluator.evaluate([mots], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
                                         trackeval.metrics.Identity()])

    def summary(r):
        r = r["pedestrian"]
        # DetRe and DetPr: the share of what is there that is found, and of what is found that is there
        return {**{k: round(float(np.mean(r["HOTA"][k])) * 100, 2) for k in ("HOTA", "DetA", "AssA", "DetRe", "DetPr")},
                "IDF1": round(float(r["Identity"]["IDF1"]) * 100, 2), "IDSW": int(r["CLEAR"]["IDSW"])}

    per = res["MOTSChallenge"][name]
    return {**summary(per["COMBINED_SEQ"]), "per_sequence": {s: summary(per[s])["HOTA"] for s in seqs}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mots/MOTS", help="a folder laid out like MOTS20")
    ap.add_argument("--system", choices=["people", "sam3"], default="people")
    ap.add_argument("--sam3", help="path to sam3.pt; without it the weights come from Hugging Face")
    ap.add_argument("--look-every", type=int, default=15)
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.4, help="the finder's confidence threshold")
    ap.add_argument("--compile", nargs="?", const=True, default=False,
                    help="compile the engine (optionally a mode) and the looks")
    ap.add_argument("--max-frames", type=int, help="each sequence's first N frames only")
    ap.add_argument("--every-nth", type=int, help="every N-th sequence only (a fixed subset)")
    ap.add_argument("--long", type=int, help="memory over one video of N frames; not scored")
    ap.add_argument("--out", default="results/people")
    args = ap.parse_args()
    out_dir = Path(args.out)
    if args.max_frames:
        args.data = str(first_frames(args.data, args.max_frames, out_dir))
    if args.long:
        args.data = str(long_video(args.data, args.long, out_dir))
    train = Path(args.data) / "train"
    seqs = ["long"] if args.long else sequences(args.data)[::args.every_nth or 1]
    name = "sam3" if args.system == "sam3" else (
        f"people-every{args.look_every}-hold{args.hold}" + (f"-t{args.threshold}" if args.threshold != 0.4 else "")
        + ("-compiled" if args.compile else ""))
    lines, cost = (run_sam3 if args.system == "sam3" else run_people)(args, seqs, train)
    result = {"system": name, "dataset": Path(args.data).resolve().name, "sequences": len(seqs), **cost,
              "ms_per_frame": round(cost["seconds"] / cost["frames"] * 1e3, 2)}
    if not args.long:
        result.update(score(name, lines, seqs, args.data, out_dir))
    print(json.dumps(result), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("a") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
