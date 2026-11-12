"""A worker: one engine on one GPU, the cameras placed on it, and the loop that
tracks them. A Router starts one per GPU and talks to it in small messages, commands
in and compact masks out, so video never leaves the worker that decodes it.

`tamserve worker` runs one as a service on another machine, for a Router to reach
over the network (main(), below)."""
from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from importlib.metadata import version
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

from tamserve import wire
from tamserve.engine import Box, Clicks, Engine
from tamserve.models import MODELS, build_video_predictor
from tamserve.wire import Mask, Update

log = logging.getLogger("tamserve")

_COMMANDS = ("add_camera", "remove_camera", "start", "stop")
_ASIDE = "tamserve: catching up"  # the camera id an object is followed under while it catches up
_SILENCE = 15.0  # seconds without a word from the router before a worker ends its session
_FAILURES = 3    # failed steps in a row before a worker decides its GPU is broken and stops
_NETWORK = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://", "srt://")


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
        # two GPUs to 55 frames/s on a 61-core machine. A stream that sends nothing for
        # 10 s counts as dropped.
        params = [cv2.CAP_PROP_N_THREADS, 1, cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000]
        self.cv2, self.capture = cv2, cv2.VideoCapture(address, cv2.CAP_ANY, params)
        self.live = address.lower().startswith(_NETWORK)
        if not self.capture.isOpened():
            raise ConnectionError("could not open the stream")  # not the address: it can hold a password

    def read(self):
        ok, bgr = self.capture.read()
        if ok:
            return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        if self.live:
            raise ConnectionError("the stream stopped sending frames")
        return None  # the end of a file

    def close(self):
        self.capture.release()


def _checked(frame):
    a = np.asarray(frame)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or 0 in a.shape:
        raise ValueError(f"a frame must be an H x W x 3 uint8 RGB array, not {a.dtype} {a.shape}")
    return a


def _problem(prompt, h, w):
    """Why `prompt` cannot start an object on an h x w frame, or None if it can."""
    if isinstance(prompt, Box):
        if all(map(math.isfinite, prompt)) and prompt.x1 > prompt.x0 and prompt.y1 > prompt.y0:
            return None
        return f"a box needs finite edges with right > left and bottom > top, not {tuple(prompt)}"
    if isinstance(prompt, Clicks):
        if prompt.xy and all(len(p) == 2 and all(map(math.isfinite, p)) for p in prompt.xy) \
                and (prompt.labels is None or len(prompt.labels) == len(prompt.xy)):
            return None
        return "clicks need at least one (x, y) point, and one label for each point if labels are given"
    if torch.is_tensor(prompt):
        return None if tuple(prompt.shape) == (h, w) else f"a mask must be {h} x {w}, the frame's size, not {tuple(prompt.shape)}"
    return f"a prompt is a Box, Clicks or a bool mask, not a {type(prompt).__name__}"


class _Camera:
    """One source, read on a thread of its own, its last `keep` frames kept.

    A network stream or a reader function that fails is opened again, waiting 1, 2,
    4 ... and at most 60 s between tries, for `retry_for` seconds (None: for ever). A
    file, or a reader that returns None, has ended. What happens is queued in
    `statuses` for the loop to report."""

    def __init__(self, name, source, keep, skip, retry_for, fresh):
        self.name = name
        self.frames = deque(maxlen=keep)  # (number, time, frame), oldest first
        self.taken = -1                   # the newest frame number the loop has taken
        self.count = 0                    # frames read so far
        self.ended = self.reported = self.stopped = False
        self.error = None
        self.statuses: deque = deque()    # (status, why) for the loop to send
        self.skip, self.retry_for, self.fresh = skip, retry_for, fresh
        self.retries = callable(source) or (isinstance(source, str) and source.lower().startswith(_NETWORK))
        self.lock = threading.Condition()
        threading.Thread(target=self._run, args=(source,), daemon=True).start()

    def _run(self, source):
        down, tries = None, 0  # when the stream went down, and the tries since
        while not self.stopped:
            reader = None
            try:
                reader = _open(source)
                while not self.stopped:
                    frame = reader.read()
                    if frame is None:
                        return self._end(None)
                    frame = _checked(frame)
                    if down is not None:
                        self._say("reconnected", None, f"after {time.time() - down:.0f} s")
                        down, tries = None, 0
                    self._add(frame)
            except Exception as e:
                why = f"{type(e).__name__}: {e}"
                if not self.retries:
                    return self._end(why)
                now = time.time()
                if down is None:
                    down = now
                    self._say("disconnected", why)
                elif self.retry_for is not None and now - down >= self.retry_for:
                    return self._end(f"gave up after {now - down:.0f} s: {why}")
                else:
                    self._say("reconnecting", f"try {tries} failed: {why}")
                wait = min(60.0, 2.0 ** tries)
                tries += 1
                with self.lock:
                    self.lock.wait_for(lambda: self.stopped, timeout=wait)
            finally:
                if hasattr(reader, "close"):
                    reader.close()

    def _add(self, frame):
        t = time.time()
        with self.lock:
            while not (self.skip or self.stopped or self.taken == self.count - 1):
                self.lock.wait()  # every frame is wanted: wait until the loop has the last one
            self.frames.append((self.count, t, frame))
            self.count += 1
        self.fresh.set()

    def _say(self, status, why, note=""):
        level = logging.INFO if status == "reconnecting" else logging.WARNING
        log.log(level, "camera %s %s%s%s", self.name, status, f" {note}" if note else "", f": {why}" if why else "")
        with self.lock:
            self.statuses.append((status, why))
        self.fresh.set()

    def _end(self, why):
        with self.lock:
            self.ended, self.error = True, why
        self.fresh.set()

    def news(self):
        """The statuses queued since the last call."""
        with self.lock:
            out = list(self.statuses)
            self.statuses.clear()
            return out

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
        self.over = threading.Event()  # the session is over
        self.sending = threading.Lock()  # the loop and the heartbeat both send
        self.heard = time.monotonic()
        self.since, self.spent, self.failures = time.perf_counter(), 0.0, 0

    def run(self, commands):
        """Serves the router until it says close, goes quiet or goes away, then forgets
        everything. False if the GPU kept failing and this worker should stop."""
        threading.Thread(target=self._heartbeat, daemon=True).start()
        try:
            self._serve(commands)
        except (EOFError, OSError, TimeoutError):
            log.info("the router has gone")
        except Exception as e:  # anything malformed ends this router's session, not the worker
            log.warning("ending the router's session: %s: %s", type(e).__name__, e)
        finally:
            self.over.set()
            self.forget()
        return self.failures < _FAILURES

    def _heartbeat(self):
        """Every second: tell the router how busy the GPU has been, and end the session
        if the router has said nothing for _SILENCE seconds."""
        while not self.over.wait(1.0):
            if time.monotonic() - self.heard > _SILENCE:
                log.warning("no word from the router for %.0f s; ending its session", _SILENCE)
                self.over.set()
                return
            now = time.perf_counter()
            busy, self.since, self.spent = self.spent / (now - self.since), now, 0.0
            try:
                self._emit(("load", busy, len(self.cameras)))
            except (OSError, ValueError):
                self.over.set()
                return

    def forget(self):
        for cam in self.cameras.values():
            cam.stop()
        for camera in {c for c, _ in self.engine.tracks}:
            self.engine.drop(camera)
        self.cameras.clear()
        self.waiting.clear()

    def _serve(self, commands):
        while not self.over.is_set():
            while commands.poll():
                message = commands.recv()
                self.heard = time.monotonic()
                name, *args = message  # anything but a tuple ends the session (run)
                if name == "close":
                    return
                if name == "ping":
                    continue
                try:
                    if name not in _COMMANDS:
                        raise ValueError(f"no command {name!r}")
                    getattr(self, name)(*args)
                except Exception as e:
                    self._send([Update(args[0] if args else None, None, None, {}, error=f"{type(e).__name__}: {e}")])
            self.fresh.clear()
            if not self._round():
                self.fresh.wait(0.005)
            if self.failures >= _FAILURES:
                log.error("%d steps in a row failed; this worker's GPU looks broken, stopping", self.failures)
                return

    def add_camera(self, camera, source, skip, retry_for=None):
        if camera in self.cameras:
            self.remove_camera(camera)
        self.cameras[camera] = _Camera(camera, source, self.keep, skip, retry_for, self.fresh)
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
            for status, why in cam.news():
                updates.append(Update(name, None, None, {}, error=why, status=status))
            item = cam.newest()
            if item is None:
                if not cam.reported and cam.done():
                    cam.reported = True
                    updates.append(Update(name, None, None, {}, error=cam.error, ended=True, status="ended"))
                continue
            n, t, frames[name] = item
            when[name] = (n, t)
            h, w = frames[name].shape[:2]
            for obj, (prompt, f) in list(self.waiting[name].items()):
                if f is not None and f > n:
                    continue
                del self.waiting[name][obj]
                why = _problem(prompt, h, w)
                if why is None and f is not None and f < n:
                    try:  # a frame the loop has passed: start the object there and follow it up to now
                        if self._catch_up(name, obj, prompt, f, n - 1):
                            continue
                    except Exception as e:
                        why = f"{type(e).__name__}: {e}"
                if why is not None:
                    updates.append(Update(name, f, None, {}, error=f"could not start {obj!r}: {why}"))
                else:
                    seeds.setdefault(name, {})[obj] = prompt
        if frames:
            t0 = time.perf_counter()
            try:
                out = self.engine.step(frames, seeds)
                updates += [Update(c, *when[c], encode(m)) for c, m in out.items() if m]
                self.failures = 0
            except Exception as e:
                self.failures += 1
                log.error("a step failed: %s: %s", type(e).__name__, e)
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
        why = _problem(prompt, *kept[0][2].shape[:2])
        if why is not None:
            raise ValueError(why)
        t0 = time.perf_counter()
        aside = (_ASIDE, camera)
        self.engine.drop(camera, obj)  # a new prompt for a tracked object replaces it
        updates = []
        try:
            for i, (n, t, frame) in enumerate(kept):
                out = self.engine.step({aside: frame}, {aside: {obj: prompt}} if i == 0 else None)
                updates.append(Update(camera, n, t, encode(out[aside])))
            self.engine.tracks[(camera, obj)] = self.engine.tracks.pop((aside, obj))
        finally:
            self.engine.drop(aside)  # nothing is left behind if a step failed
        self.spent += time.perf_counter() - t0
        self._send(updates)
        return True

    def _send(self, updates):
        if updates:
            self._emit(("updates", updates))

    def _emit(self, message):
        with self.sending:
            self.events.send(message)


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s tamserve worker: %(message)s")
    if args.gpu is not None:  # before CUDA starts: this process then sees only that GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        token = wire.check_token(Path(args.token_file).read_bytes())
    except ValueError as e:
        sys.exit(f"tamserve worker: {e}")
    host, _, port = args.listen.rpartition(":")
    compile = {None: False, "best": True}.get(args.compile, args.compile)
    engine = Engine(build_video_predictor(args.device, args.model), gpu_masks=args.device != "cpu", compile=compile)
    gpu = torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
    about = {"model": args.model, "compiled": bool(compile), "device": gpu, "version": version("tamserve")}
    listener = Listener((host, int(port)), family="AF_INET")
    log.info("%s on %s, listening on %s:%s", args.model, gpu, host, port)

    arrivals: queue.Queue = queue.Queue()
    taken = threading.Lock()   # held while a router is connected
    greeters = threading.BoundedSemaphore(8)  # handshakes under way at once

    def greet(conn, peer):
        """A newcomer: it must prove it knows the token and speak this protocol, and it is
        turned away at once if another router is connected."""
        try:
            wire.prove_worker(conn, token)
            kind, protocol, _ = wire.recv(conn, timeout=5)
            if kind != "hello" or protocol != wire.PROTOCOL:
                wire.send(conn, ("refused", f"this worker speaks protocol {wire.PROTOCOL} (tamserve {about['version']}) "
                                            f"and the router {protocol}: install the same tamserve on both"))
                raise ValueError(f"it speaks protocol {protocol}")
            if not taken.acquire(timeout=5):  # a router that just left may still be letting go
                wire.send(conn, ("refused", "another router is connected to this worker"))
                raise ValueError("another router is connected")
            try:
                wire.send(conn, ("ready", {**about, "time": time.time()}))
            except BaseException:
                taken.release()
                raise
            arrivals.put((conn, peer))
        except Exception as e:
            log.warning("turned away a connection from %s: %s", peer, e)
            conn.close()
        finally:
            greeters.release()

    def doorman():
        while True:
            conn = listener.accept()
            peer = listener.last_accepted[0]
            if not greeters.acquire(blocking=False):  # too many at once: someone is flooding the door
                conn.close()
                continue
            threading.Thread(target=greet, args=(conn, peer), daemon=True).start()

    threading.Thread(target=doorman, daemon=True).start()
    while True:
        conn, peer = arrivals.get()
        log.info("router connected from %s", peer)
        healthy = _Worker(engine, _Wire(conn), args.keep).run(_Wire(conn))
        conn.close()
        taken.release()
        log.info("router gone; its cameras are forgotten")
        if not healthy:
            sys.exit("tamserve worker: the GPU kept failing; exiting so this worker can be restarted")


if __name__ == "__main__":
    main()
