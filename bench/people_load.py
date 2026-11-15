"""How many cameras one GPU keeps up with when PeopleTracker follows the people on them,
or Tracker what each camera is asked for, and whether that holds over time.

Cameras replay a dataset's sequences (train/<sequence>/img1/*.jpg) in a loop, so any
number of them can run for any length of time. Camera i plays sequence i mod n from a
staggered start. Frames are read ahead on CPU threads, outside the timed step, as
frames from live cameras would already be decoded.

  capacity  --cameras 8,16,32,64: for each count, the time of a step (median and 95th
            percentile), and so the frame rate each camera gets
  soak      --cameras 16 --minutes 12: one line a minute with step times, GPU memory,
            process memory and people followed, to show that nothing creeps

  --prompts N   Tracker in place of PeopleTracker, on a dataset with a prompt.txt in each
                sequence (bench/named_saco.py): every camera is asked for N things, its
                sequence's own prompt and then others of the dataset, most of which name
                nothing in view, as most prompts on a real camera would. Cameras replay
                the sequences whose prompt names something, from staggered starts, and
                a camera whose video starts over is dropped first, as a new camera: the
                cut is not something a real camera shows, and Tracker would go on
                following what was there before it.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import psutil
import torch
from PIL import Image

from tamserve import Engine, Finder, PeopleFinder, PeopleTracker, Tracker, build_video_predictor
from tamserve.reference import clock


class Cameras:
    """n cameras replaying the dataset's sequences in a loop, frames read ahead."""

    def __init__(self, root, n, stagger=37, named=False):
        seqs = sorted(p for p in (Path(root) / "train").iterdir() if (p / "img1").is_dir())
        if named:  # those with something to find
            seqs = [s for s in seqs if (s / "gt" / "gt.txt").stat().st_size]
        self.paths = [sorted((s / "img1").glob("*.jpg")) for s in seqs]
        self.own = [(s / "prompt.txt").read_text().strip() if (s / "prompt.txt").exists() else None for s in seqs]
        self.cams = [(i % len(seqs), i * stagger % len(self.paths[i % len(seqs)]) if named else (i // len(seqs)) * stagger)
                     for i in range(n)]
        self.k = 0
        self.pool = ThreadPoolExecutor(16)
        self.next = self.pool.submit(self._read, 0)

    def prompts(self, n):
        """camera -> n prompts: its sequence's own, then the dataset's others in turn."""
        every = sorted(set(self.own))
        return {c: ([self.own[seq]] + [p for k in range(len(every)) for p in [every[(c * 7 + k) % len(every)]]
                                       if p != self.own[seq]])[:n] for c, (seq, _) in enumerate(self.cams)}

    def restarted(self):
        """The cameras whose video starts over on the frames about to be given."""
        return [c for c, (seq, start) in enumerate(self.cams) if self.k and (start + self.k) % len(self.paths[seq]) == 0]

    def _read(self, k):
        def one(c):
            seq, start = self.cams[c]
            p = self.paths[seq]
            return np.asarray(Image.open(p[(start + k) % len(p)]).convert("RGB"))
        return dict(zip(range(len(self.cams)), self.pool.map(one, range(len(self.cams)))))

    def frames(self):
        out = self.next.result()
        self.k += 1
        self.next = self.pool.submit(self._read, self.k)
        return out


def run(tracker, cams, steps):
    times, people = [], []
    for _ in range(steps):
        restarted = cams.restarted() if isinstance(tracker, Tracker) else ()
        frames = cams.frames()
        t = clock("cuda")
        for c in restarted:
            tracker.drop(c)
        out = tracker.step(frames)
        times.append((clock("cuda") - t) * 1e3)
        people.append(sum(len(v) for v in out.values()))
    return np.array(times), np.array(people)


def summary(times, people, n):
    med = float(np.median(times))
    return {"cameras": n, "step_ms_mean": round(float(times.mean()), 1), "step_ms_median": round(med, 1),
            "step_ms_p95": round(float(np.percentile(times, 95)), 1),
            "fps_per_camera": round(1e3 / med, 1), "camera_frames_per_s": round(n * 1e3 / med, 1),
            "followed_per_step": round(float(people.mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a folder laid out like MOTS20, e.g. data/homes")
    ap.add_argument("--cameras", default="8,16,32,64")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60, help="steps before timing")
    ap.add_argument("--minutes", type=float, help="soak: run this long at one camera count")
    ap.add_argument("--look-every", type=int, help="15 for people, 3 with --prompts")
    ap.add_argument("--prompts", type=int, help="Tracker, with this many prompts on every camera")
    ap.add_argument("--model", default="edgetam", help="the model that follows between looks")
    ap.add_argument("--sam3", help="path to sam3.pt; without it the weights come from Hugging Face")
    ap.add_argument("--compile", nargs="?", const=True, default=False,
                    help="compile the engine (optionally a mode) and the looks")
    args = ap.parse_args()

    args.look_every = args.look_every or (3 if args.prompts else 15)
    finder = (Finder if args.prompts else PeopleFinder)(checkpoint=args.sam3, compile="default" if args.compile else None)
    predictor = build_video_predictor("cuda", args.model)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "data": Path(args.data).name, "model": args.model,
                      "look_every": args.look_every, "prompts": args.prompts, "compile": args.compile}), flush=True)
    for n in map(int, args.cameras.split(",")):
        cams = Cameras(args.data, n, named=bool(args.prompts))
        engine = Engine(predictor, compile=args.compile)
        tracker = (Tracker(engine, cams.prompts(args.prompts), finder=finder, look_every=args.look_every)
                   if args.prompts else PeopleTracker(engine, finder=finder, look_every=args.look_every))
        run(tracker, cams, args.warmup)
        if not args.minutes:
            print(json.dumps(summary(*run(tracker, cams, args.steps), n)), flush=True)
            continue
        proc, end, minute = psutil.Process(), time.time() + args.minutes * 60, 0
        while time.time() < end:
            torch.cuda.reset_peak_memory_stats()
            times, people = [], []
            t_end = time.time() + 60
            while time.time() < t_end:
                tt, pp = run(tracker, cams, 10)
                times.append(tt)
                people.append(pp)
            minute += 1
            print(json.dumps({"minute": minute, **summary(np.concatenate(times), np.concatenate(people), n),
                              "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
                              "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                              "process_rss_gb": round(proc.memory_info().rss / 2**30, 2),
                              "people_followed": len(tracker.engine.tracks)}), flush=True)


if __name__ == "__main__":
    main()
