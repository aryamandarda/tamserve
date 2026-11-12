"""A live loop from video to masks. The other benchmarks start from frames that are
already resized and on the GPU; this one starts from a video file.

  per frame   decoding a frame, resizing it on the CPU with prepare(), and what the
              engine's own upload and resize cost when it is handed the frame as it is
  live loop   N cameras with one object each, frames per second from decoder to masks:
                prepare() in the loop    every frame decoded, then resized on the CPU
                raw frames               every frame decoded, then resized by the engine
                raw frames, read ahead   each camera decoded on a thread of its own while
                                         the round before is tracked, as in the README
              next to the engine alone, on frames prepared beforehand.
Needs OpenCV (pip install opencv-python-headless) and a video file to read.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("TQDM_DISABLE", "1")

import cv2
import torch

from tamserve import Box, Engine, build_video_predictor, prepare


class Camera:
    """A video file read in a loop, as a camera that never ends."""

    def __init__(self, path, start=0):
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self.cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ms(fn, reps=80):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def fps(step, n, warm=10, rounds=60):
    """Frames per second over `rounds` calls of step(), each one round of n cameras."""
    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(rounds):
        step()
    torch.cuda.synchronize()
    return round(rounds * n / (time.perf_counter() - t), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cameras", default="1,4,8,16")
    ap.add_argument("--model", default="edgetam")
    ap.add_argument("--compile", default="best", help="'best', a torch.compile mode, or 'off'")
    args = ap.parse_args()
    engine = Engine(build_video_predictor("cuda", args.model), compile={"off": False, "best": True}.get(args.compile, args.compile))
    cam = Camera(args.video)
    eight = [cam.read() for _ in range(8)]
    h, w = eight[0].shape[:2]
    video = f"{w}x{h}"
    print(json.dumps({"part": "per frame, ms", "video": video,
                      "decode": round(ms(cam.read), 2),
                      "prepare() on the CPU": round(ms(lambda: prepare(eight[0]), reps=30), 2),
                      "the engine's upload and resize, 8 frames together": round(ms(lambda: engine._prepared(eight, range(8))) / 8, 2)}),
          flush=True)

    box = Box(w * 0.3, h * 0.3, w * 0.6, h * 0.7)
    for n in (int(c) for c in args.cameras.split(",")):
        cams = [Camera(args.video, 7 * i) for i in range(n)]  # each camera somewhere else in the video
        row = {"part": "live loop, frames/s", "video": video, "cameras": n}
        engine.step({i: c.read() for i, c in enumerate(cams)}, {i: {1: box} for i in range(n)})
        ready = [engine._prepared({i: c.read() for i, c in enumerate(cams)}, range(n)) for _ in range(8)]
        k = iter(range(10 ** 9))
        row["engine alone"] = fps(lambda: engine.step(ready[next(k) % 8]), n)
        row["prepare() in the loop"] = fps(lambda: engine.step({i: prepare(c.read(), device="cuda") for i, c in enumerate(cams)}), n,
                                           warm=3, rounds=20)
        row["raw frames"] = fps(lambda: engine.step({i: c.read() for i, c in enumerate(cams)}), n)
        with ThreadPoolExecutor(n) as pool:
            ahead = [pool.submit(c.read) for c in cams]

            def step():
                frames = {i: f.result() for i, f in enumerate(ahead)}
                ahead[:] = [pool.submit(c.read) for c in cams]  # the next round decodes while this one is tracked
                engine.step(frames)

            row["raw frames, read ahead"] = fps(step, n)
            for f in ahead:
                f.result()
        print(json.dumps(row), flush=True)
        for i in range(n):
            engine.drop(i)


if __name__ == "__main__":
    main()
