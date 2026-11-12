"""A worker: one engine on one GPU, the cameras placed on it, and the loop that
tracks them. A Router starts one per GPU and talks to it in small messages, commands
in and compact masks out, so video never leaves the worker that decodes it."""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from tamserve.engine import Engine
from tamserve.models import build_video_predictor

_COMMANDS = ("add_camera", "remove_camera", "start", "stop")
_ASIDE = "tamserve: catching up"  # the camera id an object is followed under while it catches up


class Mask(NamedTuple):
    """One object's mask, stored as runs of equal pixels, row by row: `ends` holds
    where each run but the last ends, `first` is the first run's value. A few KB for
    a typical object, where the full mask of a 1080p frame is 2 MB."""
    shape: tuple        # (height, width)
    first: bool
    ends: np.ndarray
    box: tuple | None   # (left, top, right, bottom) around the object, right and bottom exclusive; None if empty
    area: int           # pixels on the object

    def array(self) -> np.ndarray:
        """The mask as an H x W bool array, True on the object."""
        h, w = self.shape
        bounds = np.concatenate(([0], self.ends, [h * w]))
        values = np.zeros(len(bounds) - 1, dtype=bool)
        values[0 if self.first else 1::2] = True
        return np.repeat(values, np.diff(bounds)).reshape(h, w)


class Update(NamedTuple):
    """One frame of one camera: the masks of the objects tracked on it, by name. An
    object started on an earlier frame also sends its masks for the frames between,
    so one frame can arrive in more than one Update."""
    camera: object
    frame: int | None     # the frame's number on its camera, counted from 0
    time: float | None    # when the frame was decoded, as time.time() gives it
    masks: dict
    error: str | None = None
    ended: bool = False   # the camera's stream has ended and all of it has been tracked


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
        while True:
            while commands.poll():
                name, *args = commands.recv()
                if name == "close":
                    for cam in self.cameras.values():
                        cam.stop()
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
