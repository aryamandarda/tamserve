"""How well Tracker follows what it is asked for and what it costs, against SAM 3's own video tracker.

A dataset is laid out like MOTS20, with one sequence per video and prompt:
train/<sequence>/img1/*.jpg, gt/gt.txt with a mask and an id for everything the prompt
names on every labelled frame, and prompt.txt (bench/named_saco.py builds one from
Meta's SA-Co/VEval). Systems:
  named  tamserve's Tracker. Each sequence is a camera and all of them run together,
         so looks interleave as they would across live cameras.
  sam3   SAM 3's own video tracking as Meta ships it, one video and one prompt at a time
Scored with TrackEval (scripts/setup_trackeval.sh) as bench/people.py scores people.
Cost is wall time per frame and prompt once the GPU has finished; reading frames from
disk is not counted.

  --share    sequences that are one video with different prompts run as one camera
             with several prompts, which SAM 3 cannot do: it takes a prompt a session
  --late N   also score the answers Tracker would give N frames late. SAM 3 holds each
             answer back 15 frames and never shows an object it drops in that time;
             Tracker answers at once, so a false find shows until it is removed
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pycocotools.mask as rle
import torch
from PIL import Image

from people import DEVICE, Timed, first_frames, mots_lines, run_sam3, score, scored, sequences
from tamserve import Engine, Finder, Tracker, build_video_predictor
from tamserve.reference import clock
from tamserve.tracker import _TRIAL


class Late:
    """One sequence's answers, each given `delay` frames late: an object removed during
    its trial (its first frames, when most looks missed it) is then never shown."""

    def __init__(self, delay):
        self.delay, self.buffer, self.first = delay, [], {}

    def push(self, frame, objects, followed):
        """objects: id -> mask, this frame's answer. followed: the ids still followed after
        it. Returns the (frame, objects) pairs that are now `delay` frames old."""
        for t in objects:
            self.first.setdefault(t, frame)
        for t in [t for t, born in self.first.items() if t not in followed]:
            if frame - self.first.pop(t) <= _TRIAL:
                for _, shown in self.buffer:
                    shown.pop(t, None)
        self.buffer.append((frame, {t: rle.encode(np.asfortranarray(np.asarray(m, dtype=np.uint8)))
                                    for t, m in objects.items()}))
        return self.flush(len(self.buffer) - self.delay)

    def flush(self, n=None):
        n = len(self.buffer) if n is None else max(n, 0)
        out = [(f, {t: rle.decode(m).astype(bool) for t, m in shown.items()}) for f, shown in self.buffer[:n]]
        del self.buffer[:n]
        return out


def run_named(args, seqs, train, prompts):
    paths = {s: sorted((train / s / "img1").glob("*.jpg")) for s in seqs}
    cams = {}  # camera -> the sequences it serves: one, or with --share every prompt of its video
    for s in seqs:
        cams.setdefault(os.path.dirname(os.path.realpath(paths[s][0])) if args.share else s, []).append(s)
    cams = {ss[0]: ss for ss in cams.values()}
    finder = Timed(Finder(checkpoint=args.sam3, compile="default" if args.compile else None))
    tracker = Tracker(Engine(build_video_predictor(DEVICE, args.model), compile=args.compile),
                      {c: [prompts[s] for s in ss] for c, ss in cams.items()}, finder=finder,
                      look_every=args.look_every, hold=args.hold, still=args.still)
    torch.cuda.reset_peak_memory_stats()
    keep = {s: scored(train / s) for s in seqs}
    lines, late_lines = {s: [] for s in seqs}, {s: [] for s in seqs}
    late = {s: Late(args.late) for s in seqs} if args.late else {}
    seconds, frames, shape = 0.0, 0, {}

    def write(to, s, answers):
        for f, objects in answers:
            f = f if keep[s] is None else keep[s].get(f)
            if f is not None:
                to[s] += mots_lines(f, objects, *shape[s])

    with ThreadPoolExecutor(8) as pool:
        for k in range(max(len(paths[c]) for c in cams)):
            live = [c for c in cams if k < len(paths[c])]
            images = dict(zip(live, pool.map(lambda c: np.asarray(Image.open(paths[c][k]).convert("RGB")), live)))
            t = clock(DEVICE)
            out = tracker.step(images)
            seconds += clock(DEVICE) - t
            frames += sum(len(cams[c]) for c in live)  # a camera with two prompts counts twice, as it would for SAM 3
            for c, objects in out.items():
                names = tracker.names(c)
                for s in cams[c]:
                    shape[s] = images[c].shape[:2]
                    mine = {t: m for t, m in objects.items() if names[t] == prompts[s]}
                    write(lines, s, [(k + 1, mine)])
                    if late:
                        write(late_lines, s, late[s].push(k + 1, mine, names))
                        if k == len(paths[c]) - 1:
                            write(late_lines, s, late[s].flush())
                if k == len(paths[c]) - 1:
                    tracker.drop(c)
    cost = {"frames": frames, "cameras": len(cams), "seconds": seconds, "look_seconds": finder.seconds,
            "looks": finder.looks, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    return lines, late_lines, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/saco", help="a folder laid out like MOTS20, a prompt.txt in each sequence")
    ap.add_argument("--system", choices=["named", "sam3"], default="named")
    ap.add_argument("--sam3", help="path to sam3.pt; without it the weights come from Hugging Face")
    ap.add_argument("--model", default="sam2.1-large", help="the model that follows between looks")
    ap.add_argument("--look-every", type=int, default=3)
    ap.add_argument("--hold", type=int, default=300)
    ap.add_argument("--still", action="store_true", help="the cameras do not move")
    ap.add_argument("--share", action="store_true", help="one camera per video, with all of its prompts")
    ap.add_argument("--late", type=int, default=0, help="also score the answers given this many frames late")
    ap.add_argument("--compile", nargs="?", const=True, default=False,
                    help="compile the engine (optionally a mode) and the looks")
    ap.add_argument("--max-frames", type=int, help="each sequence's first N frames only")
    ap.add_argument("--every-nth", type=int, help="every N-th sequence only (a fixed subset)")
    ap.add_argument("--out", default="results/named")
    args = ap.parse_args()
    if args.late and args.hold < _TRIAL:
        ap.error("--late tells a removed false find from a forgotten object by its age: it needs --hold of 15 or more")
    out_dir = Path(args.out)
    if args.max_frames:
        args.data = str(first_frames(args.data, args.max_frames, out_dir))
    train = Path(args.data) / "train"
    seqs = sequences(args.data)[::args.every_nth or 1]
    prompts = {s: (train / s / "prompt.txt").read_text().strip() for s in seqs}
    name = "sam3" if args.system == "sam3" else (
        f"named-{args.model}-every{args.look_every}" + (f"-hold{args.hold}" if args.hold != 300 else "")
        + ("-still" if args.still else "") + ("-share" if args.share else "") + ("-compiled" if args.compile else ""))
    late_lines = None
    if args.system == "sam3":
        lines, cost = run_sam3(args, seqs, train, prompts)
    else:
        lines, late_lines, cost = run_named(args, seqs, train, prompts)
    result = {"system": name, "dataset": Path(args.data).resolve().name, "sequences": len(seqs), **cost,
              "ms_per_frame": round(cost["seconds"] / cost["frames"] * 1e3, 2)}
    rows = [{**result, **score(name, lines, seqs, args.data, out_dir)}]
    if args.late and late_lines is not None:
        rows.append({**result, "system": f"{name}-late{args.late}",
                     **score(f"{name}-late{args.late}", late_lines, seqs, args.data, out_dir)})
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("a") as fh:
        for row in rows:
            print(json.dumps(row), flush=True)
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
