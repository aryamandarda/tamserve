"""A worker: one engine on one GPU, the cameras placed on it, and the loop that
tracks them. A Router starts one per GPU and talks to it in small messages, commands
in and compact masks out, so video never leaves the worker that decodes it.

`tamserve worker` runs one as a service on another machine, for a Router to reach
over the network (main(), below)."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from collections import deque
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

from tamserve import wire
from tamserve.engine import Engine
from tamserve.models import MODELS, build_video_predictor
from tamserve.wire import Mask, Update

_COMMANDS = ("add_camera", "remove_camera", "start", "stop")
_ASIDE = "tamserve: catching up"  # the camera id an object is followed under while it catches up


def encode(masks: dict) -> dict:
    """{object: H x W bool tensor} for one frame as {object: Mask}. The runs are found
    where the tensors are, so only their ends leave the GPU."""
    if not masks:
        return {}
    names = list(masks)
    x = torch.stack([masks[o] for o in names])
    n, h, w = x.shape
    flat = x.view(n, h * w)
    rows, cols = torch.nonzero(flat[:, 1:] != flat[:, :-1], as_tuple=True)  # row by row, so rows come sorted
    rows, cols, first, ys, xs = (t.cpu().numpy() for t in (rows, cols, flat[:, 0], x.any(2), x.any(1)))
    cut = np.searchsorted(rows, np.arange(n + 1))
    out = {}
    for i, o in enumerate(names):
        ends = (cols[cut[i]:cut[i + 1]] + 1).astype(np.int32)
        y, xx = np.flatnonzero(ys[i]), np.flatnonzero(xs[i])
        box = (int(xx[0]), int(y[0]), int(xx[-1]) + 1, int(y[-1]) + 1) if len(y) else None
        runs = np.diff(np.concatenate(([0], ends, [h * w])))
        out[o] = Mask((h, w), bool(first[i]), ends, box, int(runs[0 if first[i] else 1::2].sum()))
    return out


def _open(source):
    """A reader whose read() gives the next frame as an H x W x 3 uint8 RGB array, or
    None once the stream has ended."""
    if callable(source):
        return source()
    try:
        import cv2
    except ImportError:
        raise ImportError("reading a camera by its address needs OpenCV: pip install opencv-python-headless") from None
    return _Stream(cv2, source)


class _Stream:
    def __init__(self, cv2, address):
        # One decoding thread per camera: the parallelism comes from the cameras. Left to
        # itself, FFmpeg gives every camera a thread per core, and 32 of them at 1080p held
        # two GPUs to 55 frames/s on a 61-core machine.
        self.cv2, self.capture = cv2, cv2.VideoCapture(address, cv2.CAP_ANY, [cv2.CAP_PROP_N_THREADS, 1])
        if not self.capture.isOpened():
            raise OSError("could not open the stream")  # not the address: it can hold a password

    def read(self):
        ok, bgr = self.capture.read()
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB) if ok else None


class _Camera:
    """One source, read on a thread of its own, its last `keep` frames kept."""

    def __init__(self, source, keep, skip, fresh):
        self.frames = deque(maxlen=keep)  # (number, time, frame), oldest first
        self.taken = -1                   # the newest frame number the loop has taken
        self.count = 0                    # frames read so far
        self.ended = self.reported = self.stopped = False
        self.error = None
        self.skip, self.fresh = skip, fresh
        self.lock = threading.Condition()
        threading.Thread(target=self._read, args=(source,), daemon=True).start()

    def _read(self, source):
        try:
            reader = _open(source)
            while not self.stopped:
                frame = reader.read()
                if frame is None:
                    break
                t = time.time()
                with self.lock:
                    while not (self.skip or self.stopped or self.taken == self.count - 1):
                        self.lock.wait()  # every frame is wanted: wait until the loop has the last one
                    self.frames.append((self.count, t, np.asarray(frame)))
                    self.count += 1
                self.fresh.set()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        with self.lock:
            self.ended = True
        self.fresh.set()

    def newest(self):
        """The newest frame the loop has not taken yet, now taken; None if there is none."""
        with self.lock:
            if not self.frames or self.frames[-1][0] <= self.taken:
                return None
            item = self.frames[-1]
            self.taken = item[0]
            self.lock.notify_all()
            return item

    def kept(self, first, last):
        """The kept frames numbered `first` to `last`, oldest first."""
        with self.lock:
            return [f for f in self.frames if first <= f[0] <= last]

    def done(self):
        with self.lock:
            return self.ended and self.taken == self.count - 1

    def stop(self):
        with self.lock:
            self.stopped = True
            self.lock.notify_all()


def serve(commands, events, device, model, options, keep):
    """A worker process: builds its engine, then tracks until it is told to close."""
    options = {k: v for k, v in options.items() if k != "gpu_masks"}  # masks are encoded where they are
    try:
        engine = Engine(build_video_predictor(device, model), gpu_masks=device != "cpu", **options)
    except Exception as e:
        traceback.print_exc()  # the whole story goes to the worker's error output; the router gets one line
        events.send(("error", f"{type(e).__name__}: {e}"))
        return
    events.send(("ready", None))
    _Worker(engine, events, keep).run(commands)


class _Worker:
    def __init__(self, engine, events, keep):
        self.engine, self.events, self.keep = engine, events, keep
        self.cameras: dict = {}  # camera -> _Camera
        self.waiting: dict = {}  # camera -> {object: (prompt, frame)}: starts on frames not reached yet
        self.fresh = threading.Event()
        self.since, self.spent = time.perf_counter(), 0.0

    def run(self, commands):
        """Serves the router until it says close or goes away, then forgets everything."""
        try:
            self._serve(commands)
        except (EOFError, OSError, TimeoutError):
            pass  # the router has gone
        except ValueError as e:
            print(f"tamserve worker: {e}; ending this router's session", file=sys.stderr, flush=True)
        finally:
            self.forget()

    def forget(self):
        for cam in self.cameras.values():
            cam.stop()
        for camera in {c for c, _ in self.engine.tracks}:
            self.engine.drop(camera)
        self.cameras.clear()
        self.waiting.clear()

    def _serve(self, commands):
        while True:
            while commands.poll():
                name, *args = commands.recv()
                if name == "close":
                    return
                try:
                    if name not in _COMMANDS:
                        raise ValueError(f"no command {name!r}")
                    getattr(self, name)(*args)
                except Exception as e:
                    self._send([Update(args[0] if args else None, None, None, {}, error=f"{type(e).__name__}: {e}")])
            self.fresh.clear()
            if not self._round():
                self.fresh.wait(0.005)
            now = time.perf_counter()
            if now - self.since >= 1:  # how busy the GPU has been, for the router to place cameras by
                self.events.send(("load", self.spent / (now - self.since), len(self.cameras)))
                self.since, self.spent = now, 0.0

    def add_camera(self, camera, source, skip):
        self.cameras[camera] = _Camera(source, self.keep, skip, self.fresh)
        self.waiting[camera] = {}

    def remove_camera(self, camera):
        self.cameras.pop(camera).stop()
        del self.waiting[camera]
        self.engine.drop(camera)

    def start(self, camera, obj, prompt, frame):
        cam = self.cameras[camera]
        if isinstance(prompt, np.ndarray):
            prompt = torch.from_numpy(prompt)
        self.waiting[camera].pop(obj, None)
        if frame is None or frame > cam.taken or not self._catch_up(camera, obj, prompt, frame, cam.taken):
            self.waiting[camera][obj] = (prompt, frame)

    def stop(self, camera, obj):
        self.engine.drop(camera, obj)
        if obj is None:
            self.waiting[camera].clear()
        else:
            self.waiting[camera].pop(obj, None)

    def _round(self):
        """One step over every camera with a frame the loop has not taken. False if none had one."""
        frames, seeds, when, updates = {}, {}, {}, []
        for name, cam in self.cameras.items():
            item = cam.newest()
            if item is None:
                if not cam.reported and cam.done():
                    cam.reported = True
                    updates.append(Update(name, None, None, {}, error=cam.error, ended=True))
                continue
            n, t, frames[name] = item
            when[name] = (n, t)
            for obj, (prompt, f) in list(self.waiting[name].items()):
                if f is not None and f > n:
                    continue
                del self.waiting[name][obj]
                try:  # a frame the loop skipped: start the object there and follow it up to now
                    caught = f is not None and f < n and self._catch_up(name, obj, prompt, f, n - 1)
                except Exception as e:
                    updates.append(Update(name, f, None, {}, error=f"{type(e).__name__}: {e}"))
                    continue
                if not caught:
                    seeds.setdefault(name, {})[obj] = prompt
        if frames:
            t0 = time.perf_counter()
            try:
                out = self.engine.step(frames, seeds)
                updates += [Update(c, *when[c], encode(m)) for c, m in out.items() if m]
            except Exception as e:
                updates += [Update(c, *when[c], {}, error=f"{type(e).__name__}: {e}") for c in frames]
            self.spent += time.perf_counter() - t0
        self._send(updates)
        return bool(frames)

    def _catch_up(self, camera, obj, prompt, first, last):
        """Start `obj` on frame `first`, which the loop has passed, and follow it alone
        through the kept frames up to `last`, so it joins its camera's next step as if
        it had been there all along. If `first` is no longer kept it starts on the
        oldest frame that is. False if no frame between them is kept."""
        kept = self.cameras[camera].kept(first, last)
        if not kept:
            return False
        t0 = time.perf_counter()
        aside = (_ASIDE, camera)
        self.engine.drop(camera, obj)  # a new prompt for a tracked object replaces it
        updates = []
        for i, (n, t, frame) in enumerate(kept):
            out = self.engine.step({aside: frame}, {aside: {obj: prompt}} if i == 0 else None)
            updates.append(Update(camera, n, t, encode(out[aside])))
        self.engine.tracks[(camera, obj)] = self.engine.tracks.pop((aside, obj))
        self.spent += time.perf_counter() - t0
        self._send(updates)
        return True

    def _send(self, updates):
        if updates:
            self.events.send(("updates", updates))


class _Wire:
    """A connection to a router on another machine, carrying plain data (tamserve.wire)."""

    def __init__(self, conn):
        self.conn = conn

    def poll(self, timeout=0):
        return self.conn.poll(timeout)

    def recv(self):
        return wire.recv(self.conn)

    def send(self, message):
        wire.send(self.conn, message)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tamserve")
    sub = ap.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker", help="track cameras on one GPU for a Router on another machine")
    w.add_argument("--listen", default="127.0.0.1:7300",
                   help="address and port to listen on; 0.0.0.0:7300 listens on every network this machine is on")
    w.add_argument("--token-file", required=True, help="a file holding the secret a router must know to connect")
    w.add_argument("--gpu", type=int, help="which GPU to use, counted from 0; the first if left out")
    w.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu is for trying it out")
    w.add_argument("--model", default="edgetam", choices=sorted(MODELS))
    w.add_argument("--compile", nargs="?", const="best", default=None,
                   help="compile for speed: on its own for the fastest setup, or a torch.compile mode")
    w.add_argument("--keep", type=int, default=32, help="frames each camera keeps, for prompts on frames already passed")
    args = ap.parse_args(argv)
    if args.gpu is not None:  # before CUDA starts: this process then sees only that GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    token = Path(args.token_file).read_bytes().strip()
    if len(token) < 16:
        sys.exit('tamserve worker: the token must be at least 16 characters; make one with '
                 'python -c "import secrets; print(secrets.token_hex(32))"')
    host, _, port = args.listen.rpartition(":")
    compile = {None: False, "best": True}.get(args.compile, args.compile)
    engine = Engine(build_video_predictor(args.device, args.model), gpu_masks=args.device != "cpu", compile=compile)
    gpu = torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
    about = {"model": args.model, "compiled": bool(compile), "device": gpu}
    listener = Listener((host, int(port)), family="AF_INET")
    print(f"tamserve worker: {args.model} on {gpu}, listening on {host}:{port}", flush=True)
    while True:
        conn = listener.accept()
        peer = listener.last_accepted[0]
        try:
            wire.prove_worker(conn, token)
            kind, _ = wire.recv(conn, timeout=10)
            if kind != "hello":
                raise ValueError("the router did not say hello")
            wire.send(conn, ("ready", {**about, "time": time.time()}))
        except (PermissionError, TimeoutError, ValueError, EOFError, OSError) as e:
            print(f"tamserve worker: refused a connection from {peer}: {e}", file=sys.stderr, flush=True)
            conn.close()
            continue
        print(f"tamserve worker: router connected from {peer}", flush=True)
        _Worker(engine, _Wire(conn), args.keep).run(_Wire(conn))
        conn.close()
        print("tamserve worker: router gone; its cameras are forgotten", flush=True)


if __name__ == "__main__":
    main()
