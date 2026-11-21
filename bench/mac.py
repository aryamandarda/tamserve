"""tamserve on an Apple-silicon Mac: what the Neural Engine buys, and what it costs in accuracy.

  speed      frames per second with everything on the GPU, with the image encoder on the Neural
             Engine, and with each round's next frames given a round early (step's next_frames).
             Made-up 720p frames with moving discs: cost depends on shapes, not pixels.
  live       cameras that send frames on a clock, as live ones do, through a Router: frames tracked
             a second and how long after a frame arrived its masks came, on the GPU alone and
             with the Neural Engine, which starts on each frame as it arrives.
  accuracy   DAVIS 2017 val scored as bench/accuracy.py scores it (every video a camera, joining
             three steps apart, against the hand-drawn masks), from frames as a decoder gives
             them, on the GPU alone and with the Neural Engine.
"""
import argparse
import functools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import davis
import numpy as np
import torch
from PIL import Image

from tamserve import Box, Engine, Router, build_video_predictor
from tamserve.compare import compare


def discs(n, objects, h=720, w=1280):
    """n frames with `objects` discs drifting to the right, and a box round each disc on the first."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 80 * xx / w, 70 + 60 * yy / h, 110 + 40 * (xx + yy) / (h + w)], -1).astype(np.uint8)
    at = [(h * (k + 1) // (objects + 1), 200 + 150 * k) for k in range(objects)]
    frames = []
    for i in range(n):
        frame = base.copy()
        for y, x in at:
            frame[(yy - y) ** 2 + (xx - x - 3 * i) ** 2 <= 60 * 60] = (220, 40, 40)
        frames.append(frame)
    return frames, {k: Box(x - 60, y - 60, x + 60, y + 60) for k, (y, x) in enumerate(at)}


def speed(args):
    frames, boxes = discs(args.frames + 1, args.objects)
    predictor = build_video_predictor("mps")
    for cameras in args.cameras:
        cams = [f"cam{i}" for i in range(cameras)]
        row = {"cameras": cameras, "objects": args.objects, "compile": args.compile}
        for name, neural, ahead in (("gpu", False, False), ("neural_engine", True, False), ("next_frames", True, True)):
            engine = Engine(predictor, neural_engine=neural, compile=args.compile or False)
            feed = lambda k, cams=cams: {c: frames[k].copy() for c in cams}  # each camera its own array
            now, times = feed(0), []
            for k in range(args.frames):
                coming = feed(k + 1)
                t = time.perf_counter()
                engine.step(now, {c: boxes for c in cams} if k == 0 else None, next_frames=coming if ahead else None)
                torch.mps.synchronize()
                times.append(time.perf_counter() - t)
                now = coming
            row[name + "_frames_per_s"] = round(cameras / float(np.median(times[args.frames // 2:])), 1)
        print(json.dumps(row), flush=True)


class Paced:
    """A camera a Router's worker reads: made-up frames on a clock, the discs drifting right and back for ever."""

    def __init__(self, objects, fps, seconds):
        self.frames, _ = discs(45, objects)
        self.fps, self.left, self.k, self.t0 = fps, int(fps * seconds), 0, None

    def read(self):
        if self.k == self.left:
            return None
        self.t0 = self.t0 or time.perf_counter()
        wait = self.t0 + self.k / self.fps - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        at = self.k % 88  # 0 to 44 and back
        self.k += 1
        return self.frames[min(at, 88 - at)].copy()


def live(args):
    _, boxes = discs(1, args.objects)
    for cameras in args.cameras:
        row = {"cameras": cameras, "objects": args.objects, "camera_frames_per_s": args.fps}
        for name, neural in (("gpu", False), ("neural_engine", True)):
            with Router(neural_engine=neural) as router:
                for c in range(cameras):
                    router.add_camera(c, functools.partial(Paced, args.objects, args.fps, args.seconds))
                    for obj, box in boxes.items():
                        router.start(c, obj, box)
                got, ended = [], 0
                for u in router.results(timeout=120):
                    if u.masks and u.frame >= args.fps * args.seconds / 3:  # the first third is warm-up
                        got.append((time.time(), time.time() - u.time))
                    ended += bool(u.ended)
                    if ended == cameras:
                        break
            row[name + "_frames_per_s"] = round((len(got) - 1) / (got[-1][0] - got[0][0]), 1)
            row[name + "_ms_after_arrival"] = round(float(np.median([late for _, late in got])) * 1e3)
        print(json.dumps(row), flush=True)


def accuracy(args):
    root = Path(args.davis)
    seqs = davis.names(root, "val")
    seeds = {s: davis.seeds(root, s) for s in seqs}
    paths = {s: sorted(davis.frames_dir(root, s).glob("*.jpg")) for s in seqs}
    start = {s: i * 3 for i, s in enumerate(seqs)}
    predictor = build_video_predictor("mps")
    for name, neural in (("gpu", False), ("neural_engine", True)):
        engine, got = Engine(predictor, neural_engine=neural), {s: {} for s in seqs}
        for k in range(max(start[s] + len(paths[s]) for s in seqs)):
            live = [s for s in seqs if start[s] <= k < start[s] + len(paths[s])]
            frames = {s: np.asarray(Image.open(paths[s][k - start[s]]).convert("RGB")) for s in live}
            for s, masks in engine.step(frames, {s: seeds[s] for s in live if k == start[s]}).items():
                got[s][k - start[s]] = masks
            for s in live:
                if k == start[s] + len(paths[s]) - 1:
                    engine.drop(s)
        each = [compare(davis.ground_truth(root, s, seeds[s]), got[s])["mean_iou"] for s in seqs]
        print(json.dumps({"engine": name, "videos": len(seqs), "accuracy": round(float(np.mean(each)), 4)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("speed")
    s.add_argument("--cameras", type=lambda v: [int(x) for x in v.split(",")], default=[1, 4, 8])
    s.add_argument("--objects", type=int, default=1, help="objects on each camera")
    s.add_argument("--frames", type=int, default=120)
    s.add_argument("--compile", default=None, help='a torch.compile mode: "default" works on a Mac')
    v = sub.add_parser("live")
    v.add_argument("--cameras", type=lambda v: [int(x) for x in v.split(",")], default=[1, 2, 4])
    v.add_argument("--objects", type=int, default=1, help="objects on each camera")
    v.add_argument("--fps", type=int, default=30, help="frames a second each camera sends")
    v.add_argument("--seconds", type=float, default=10)
    a = sub.add_parser("accuracy")
    a.add_argument("--davis", default="data/davis/DAVIS")
    args = ap.parse_args()
    {"speed": speed, "live": live, "accuracy": accuracy}[args.what](args)
