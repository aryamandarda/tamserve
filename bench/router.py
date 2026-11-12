"""Cameras over GPUs through a Router: frames per second, and the delay from a frame
being decoded to its masks reaching the router.

Every camera plays the same video file in a loop at the file's frame rate, as a live
camera sends frames, each starting somewhere else in it, with one object each. Give
more cameras than the GPUs can keep up with and every camera skips frames, so the
total is what the GPUs can do. With --source the cameras are real streams instead,
such as rtsp://127.0.0.1:8554/cam{i}, opened by whichever worker gets them.
Needs OpenCV (pip install opencv-python-headless) and a video file.
"""
import argparse
import functools
import json
import os
import statistics
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import cv2

from tamserve import Box, Router


class Paced:
    """A video file played in a loop at its own frame rate, from frame `start` on."""

    def __init__(self, path, start=0):
        self.capture = cv2.VideoCapture(path, cv2.CAP_ANY, [cv2.CAP_PROP_N_THREADS, 1])  # one decoding thread, as tamserve opens streams
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        self.gap = 1 / (self.capture.get(cv2.CAP_PROP_FPS) or 30)
        self.due = time.perf_counter()

    def read(self):
        self.due = max(self.due + self.gap, time.perf_counter() - self.gap)  # a late frame does not bring a burst
        time.sleep(max(0.0, self.due - time.perf_counter()))
        ok, bgr = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self.capture.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gpus", default="0", help="the GPUs to use, such as 0 or 0,1")
    ap.add_argument("--workers", help="instead of --gpus: devices and worker addresses, such as cuda:0,10.0.0.12:7300")
    ap.add_argument("--token-file", help="the token of the workers on other machines")
    ap.add_argument("--source", help="a stream address for camera i, with {i} in it; the video file plays otherwise")
    ap.add_argument("--cameras", type=int, default=16, help="cameras for each worker")
    ap.add_argument("--model", default="edgetam")
    ap.add_argument("--compile", default="best", help="'best', a torch.compile mode, or 'off'")
    ap.add_argument("--warmup", type=float, default=15)
    ap.add_argument("--seconds", type=float, default=30)
    args = ap.parse_args()
    gpus = args.workers.split(",") if args.workers else [f"cuda:{g}" for g in args.gpus.split(",")]
    token = open(args.token_file).read().strip() if args.token_file else None
    n = args.cameras * len(gpus)
    probe = cv2.VideoCapture(args.video)
    w, h, fps = (probe.get(p) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    t = time.time()
    with Router(args.model, gpus, token=token, compile={"off": False, "best": True}.get(args.compile, args.compile)) as router:
        ready = round(time.time() - t, 1)
        for i in range(n):
            router.add_camera(i, args.source.format(i=i) if args.source else functools.partial(Paced, args.video, 37 * i))
            router.start(i, "thing", Box(w * 0.3, h * 0.3, w * 0.6, h * 0.7))
        begin = time.time() + args.warmup
        end = begin + args.seconds
        delays, per_camera, by_worker = [], dict.fromkeys(range(n), 0), {}
        for u in router.results(timeout=60):
            now = time.time()
            if u.error:
                print(json.dumps({"camera": u.camera, "error": u.error}), flush=True)
                continue
            if now >= end:
                break
            if now >= begin and u.masks:
                delays.append(now - u.time)
                by_worker.setdefault(router.home[u.camera], []).append(now - u.time)
                per_camera[u.camera] += 1
        on = [sorted(router.home.values()).count(i) for i in range(len(gpus))]
        order = [w.device for w in router._workers]
    counts = sorted(per_camera.values())
    print(json.dumps({
        "model": args.model, "video": f"{int(w)}x{int(h)}", "video_fps": round(fps, 1), "gpus": len(gpus),
        "cameras": n, "cameras_on_each_worker": on, "ready_s": ready,
        "frames_per_s": round(sum(counts) / args.seconds, 1),
        "per_camera_fps_min_median_max": [round(c / args.seconds, 1) for c in (counts[0], statistics.median(counts), counts[-1])],
        "delay_ms_median": round(1e3 * statistics.median(delays), 1),
        "delay_ms_p95": round(1e3 * statistics.quantiles(delays, n=20)[-1], 1),
        "delay_ms_median_by_worker": {order[i]: round(1e3 * statistics.median(d), 1) for i, d in sorted(by_worker.items())}}),
        flush=True)


if __name__ == "__main__":
    main()
