"""Cameras over several GPUs, used through one object.

A Router puts each camera on the worker with the most room, where it stays: its
objects' memory lives on that GPU. Workers open and decode their own cameras, so
video never passes between processes; the router sends them commands and receives
compact masks. A worker is either a process the router starts on a GPU of this
machine, or a `tamserve worker` on another machine, reached over the network.

When something goes wrong the router keeps going and says so: a worker that stops
answering is replaced by the others, which take over its cameras, and is brought
back when it can be; every change is an Update in results(), a log line under the
"tamserve" logger, and visible in status().
"""
from __future__ import annotations

import logging
import os
import queue
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import Client, wait

import torch

from tamserve import wire
from tamserve.wire import Mask, Update
from tamserve.worker import serve

log = logging.getLogger("tamserve")

_ENVIRON = threading.Lock()
_PING = 2.0      # seconds between the router's signs of life to each worker
_SILENCE = 15.0  # seconds without a word from a worker before it counts as lost
_WINDOW = 5.0    # seconds over which status() measures frame rate and delay
_STATES = {"disconnected": "reconnecting", "reconnecting": "reconnecting", "reconnected": "tracking",
           "ended": "ended", "moved": "starting", "waiting": "waiting"}


def _visible(device):
    """(CUDA_VISIBLE_DEVICES for the worker, the device it uses). A GPU worker sees
    only its own GPU, as its first, so every worker reuses the code the first one
    compiled: a GPU addressed as cuda:1 would be compiled for again."""
    if torch.device(device).type != "cuda":
        return None, device
    ids = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = ids.split(",") if ids else [str(i) for i in range(torch.cuda.device_count())]
    index = torch.device(device).index or 0
    if index >= len(ids):
        raise ValueError(f"{device} is not a GPU this process can see")
    return ids[index], "cuda"


class _Outbox:
    """Sends a worker's commands from a thread of its own, in order. The router's own
    thread is the only reader of every worker's messages, so it must never wait on a
    send: a worker can be waiting for it to read just then, and a large message (a
    mask, to move an object) fills a pipe. A send that fails sets `broken`."""

    def _open_outbox(self, deliver):
        self._queue, self._deliver, self.broken = queue.Queue(), deliver, None
        self._sender = threading.Thread(target=self._drain, daemon=True)
        self._sender.start()

    def _drain(self):
        while (command := self._queue.get()) is not None:
            try:
                self._deliver(command)
            except Exception as e:
                self.broken = f"{type(e).__name__}: {e}"
                return

    def send(self, *command):
        if not self.alive:
            raise RuntimeError(f"the worker at {self.device} has stopped")
        self._queue.put(command)

    def _close_outbox(self):
        """Delivers what is queued, the close command included, waiting at most 5 s."""
        self._queue.put(None)
        self._sender.join(timeout=5)


class _Local(_Outbox):
    """A worker process on this machine, which the router starts."""
    remote = False

    def __init__(self, device, model, options, keep):
        ctx = torch.multiprocessing.get_context("spawn")
        self.events, events = ctx.Pipe(duplex=False)
        commands, self.commands = ctx.Pipe(duplex=False)
        visible, inner = _visible(device)
        with _ENVIRON:  # a new process starts with this one's environment
            before = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible
            try:
                self.process = ctx.Process(target=serve, args=(commands, events, inner, model, options, keep), daemon=True)
                self.process.start()
            finally:
                if visible is not None and before is None:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                elif visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = before
        commands.close()
        events.close()  # this process's copies, so that the worker exiting reaches us as the end of the pipe
        self.device, self.cameras, self.busy, self.counted = device, set(), 0.0, 0
        self.alive, self.lock, self.heard = False, threading.Lock(), time.monotonic()
        self._open_outbox(self.commands.send)

    def ready(self):
        """Waits until the worker's engine is built."""
        try:
            kind, detail = self.events.recv()
        except EOFError:
            kind, detail = "error", "it stopped while starting"
        if kind != "ready":
            raise RuntimeError(f"the worker for {self.device} did not start: {detail}")
        self.alive, self.heard = True, time.monotonic()

    def fileno(self):
        return self.events.fileno()

    def recv(self):
        return self.events.recv()

    def close(self):
        if self.alive:
            self.send("close")
        self.alive = False
        self._close_outbox()
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()


class _Remote(_Outbox):
    """A `tamserve worker` on another machine, reached over the network."""
    remote = True

    def __init__(self, address, token, model):
        host, _, port = address.rpartition(":")
        self.device, self.conn = address, Client((host, int(port)), family="AF_INET")
        try:
            wire.prove_router(self.conn, token)
            sent = time.time()
            wire.send(self.conn, ("hello", wire.PROTOCOL, sent))
            kind, about = wire.recv(self.conn, timeout=30)
            back = time.time()
            if kind == "refused":
                raise RuntimeError(f"the worker at {address} turned this router away: {about}")
            if kind != "ready":
                raise RuntimeError(f"the worker at {address} did not get ready")
            if about["model"] != model:
                raise ValueError(f"the worker at {address} runs {about['model']}, not {model}")
        except BaseException:
            self.conn.close()
            raise
        # How far its clock is ahead of ours, to the precision of half the round trip:
        # Update.time is then on this machine's clock, wherever the frame was decoded.
        self.offset = about["time"] - (sent + back) / 2
        self.about, self.cameras, self.busy, self.counted = about, set(), 0.0, 0
        self.alive, self.lock, self.heard = True, threading.Lock(), time.monotonic()
        self._open_outbox(lambda command: wire.send(self.conn, command))

    def ready(self):
        pass  # a remote worker builds its engine when it starts, before anyone connects

    def fileno(self):
        return self.conn.fileno()

    def recv(self):
        message = wire.recv(self.conn)
        if message[0] == "updates":
            message = ("updates", [u._replace(time=u.time - self.offset) if u.time is not None else u
                                   for u in message[1]])
        return message

    def close(self):
        if self.alive:
            self.send("close")
        self.alive = False
        self._close_outbox()
        self.conn.close()


@dataclass
class _Cam:
    """What the router knows of a camera: enough to move it, and to report on it."""
    source: object
    skip: bool
    retry_for: float | None
    objects: dict = field(default_factory=dict)  # object -> its prompt, then its latest Mask
    worker: int | None = None
    state: str = "starting"
    since: float = field(default_factory=time.time)
    last_frame: float | None = None
    seen: deque = field(default_factory=deque)  # (arrival, delay) of recent masks


class Router:
    """Cameras spread over GPUs, used through one object.

    workers: "local" for a worker on every GPU this process can see, or a list in
        which each entry is a device of this machine, such as "cuda:0" or "cpu",
        or the address of a `tamserve worker` on another machine, such as
        "10.0.0.12:7300". The two can be mixed.
    token: the secret the workers on other machines were started with, at least
        32 characters.
    keep: frames each camera on this machine's workers keeps, for prompts made on
        a frame its worker has already passed (start(..., frame=n)). Workers on
        other machines take it as a command-line option.
    Every other argument goes to the Engine of each worker on this machine, such
    as compile=True; workers on other machines are set up when they start.

    Create it under `if __name__ == "__main__":`. Workers on this machine are new
    Python processes, and they import your main module as they start.
    """

    def __init__(self, model: str = "edgetam", workers="local", *, token: str | bytes | None = None,
                 keep: int = 32, **options):
        if workers == "local":
            workers = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if not workers:
                raise ValueError("no GPU is visible; pass workers=['cpu'] to run on the CPU")
        local = [w for w in workers if w.partition(":")[0] in ("cpu", "cuda", "mps")]
        remote = [w for w in workers if w not in local]
        if remote and not token:
            raise ValueError("workers on other machines need the token they were started with")
        self._token = wire.check_token(token) if remote else b""
        self._model, self._options, self._keep = model, options, keep
        self.home: dict = {}  # camera -> index of its worker
        self._cams: dict = {}
        self._updates: queue.Queue = queue.Queue()
        self._workers: list = []
        self._specs: list = []  # how to start each worker again
        self._lock = threading.RLock()
        self._closing = threading.Event()
        # Running before any worker is ready, so that a worker hears from the router as soon
        # as it is, even while the next one is still compiling (which can take minutes).
        self._watcher = threading.Thread(target=self._watch, daemon=True)
        self._watcher.start()
        try:
            for address in remote:
                w = _Remote(address, self._token, model)
                with self._lock:
                    self._workers.append(w)
                    self._specs.append(address)
            # Compiling, the first local worker goes alone and the others start from its compiled code.
            for group in ([local[:1], local[1:]] if options.get("compile") else [local]):
                started = [_Local(d, model, options, keep) for d in group]
                with self._lock:
                    self._workers += started
                    self._specs += group
                for w in started:
                    w.ready()
        except BaseException:
            self.close()
            raise

    def add_camera(self, camera, source, *, skip: bool = True, retry_for: float | None = None) -> int:
        """Start reading a camera, on the worker with the most room; returns its index.

        source: an address OpenCV can open (rtsp://..., a file), which needs
            opencv-python where the worker runs. It is opened by the worker, so it
            must be reachable from that machine. A worker on this machine also
            takes something it can call to get a reader, whose read() returns the
            next frame as an H x W x 3 uint8 RGB array and None at the end. That is
            sent to the worker, so it must be picklable: a function or class defined
            at the top of a module, or functools.partial of one.
        skip: when tracking falls behind the camera, take its newest frame and skip
            the rest, as a live camera needs. False tracks every frame, as for a
            recorded video.
        retry_for: a network stream or a reader that fails is opened again, waiting
            1, 2, 4 ... and at most 60 s between tries, for this many seconds; for
            ever if left out. The camera keeps its objects meanwhile.
        """
        with self._lock:
            if camera in self._cams:
                raise KeyError(f"camera {camera!r} is already added")
            i = self._place(source)
            self._workers[i].send("add_camera", camera, source, skip, retry_for)
            self._cams[camera] = _Cam(source, skip, retry_for, worker=i)
            self.home[camera] = i
            self._workers[i].cameras.add(camera)
            return i

    def remove_camera(self, camera) -> None:
        """Stop reading a camera and forget what was tracked on it."""
        with self._lock:
            cam = self._cams.pop(self._known(camera))
            self.home.pop(camera, None)
            if cam.worker is not None:
                w = self._workers[cam.worker]
                w.cameras.discard(camera)
                if w.alive:
                    w.send("remove_camera", camera)

    def start(self, camera, obj, prompt, *, frame: int | None = None) -> None:
        """Start tracking `obj` on `camera`, or start it again from a new prompt to fix
        it: a Box, Clicks or a bool mask, in the frame's own pixels.

        frame: the number of the frame the prompt was made on, from an Update. The
            worker follows the object from there through the frames it kept, and
            sends its masks for them. Left out, the object starts on the next frame
            the worker tracks.
        """
        with self._lock:
            cam = self._cams[self._known(camera)]
            cam.objects[obj] = prompt
            if cam.worker is not None:
                self._workers[cam.worker].send("start", camera, obj, prompt, frame)

    def stop(self, camera, obj=None) -> None:
        """Stop tracking `obj`; with no object, everything on the camera."""
        with self._lock:
            cam = self._cams[self._known(camera)]
            if obj is None:
                cam.objects.clear()
            else:
                cam.objects.pop(obj, None)
            if cam.worker is not None:
                self._workers[cam.worker].send("stop", camera, obj)

    def results(self, timeout: float | None = None):
        """Updates as they arrive, from every worker. With a timeout, it ends after that
        many seconds without one; it also ends once the router is closed.

        Most carry masks. Some carry a `status` instead, with `error` saying why:
        "disconnected", "reconnecting" and "reconnected" as a camera's stream drops
        and comes back; "moved" when its worker was lost and another took it over,
        its objects starting again from their last masks; "waiting" when no worker
        can take it yet; "ended" when its stream is over or retry_for ran out.
        """
        while True:
            try:
                update = self._updates.get(timeout=timeout)
            except queue.Empty:
                return
            if update is None:
                self._updates.put(None)  # for whoever reads next
                return
            yield update

    def status(self) -> dict:
        """How every camera and worker is doing, for a dashboard or a health check.

        cameras: name -> its worker, its state ("starting" until its first masks,
            "tracking", "reconnecting", "waiting" or "ended") and since when, masks
            per second and the median delay from decoding to here over the last 5 s,
            when its last frame was decoded, and its objects.
        workers: one entry each: where it is, whether it is running, how busy its
            GPU was over the last second, its cameras, and seconds since it was last heard.
        """
        now, mono = time.time(), time.monotonic()
        with self._lock:
            cameras = {}
            for name, cam in self._cams.items():
                while cam.seen and cam.seen[0][0] < now - _WINDOW:
                    cam.seen.popleft()
                cameras[name] = {
                    "worker": self._workers[cam.worker].device if cam.worker is not None else None,
                    "state": cam.state, "since": cam.since,
                    "frames_per_s": round(len(cam.seen) / _WINDOW, 1),
                    "delay_ms": round(1e3 * statistics.median(d for _, d in cam.seen), 1) if cam.seen else None,
                    "last_frame": cam.last_frame, "objects": sorted(cam.objects, key=str)}
            workers = [{"worker": w.device, "running": w.alive, "busy": round(w.busy, 2), "cameras": len(w.cameras),
                        "heard_s_ago": round(mono - w.heard, 1)} for w in self._workers]
        return {"cameras": cameras, "workers": workers}

    def close(self) -> None:
        """Lets go of every worker: this machine's stop, other machines' forget its cameras."""
        self._closing.set()
        with self._lock:
            workers = list(self._workers)
        for w in workers:
            w.close()
        if self._watcher is not None:
            self._watcher.join(timeout=5)
        self._updates.put(None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _known(self, camera):
        if camera not in self._cams:
            raise KeyError(f"no camera {camera!r}; add it with add_camera()")
        return camera

    def _place(self, source):
        """The running worker with the most room for `source`: what one camera costs it,
        from how busy it last said it was, times the cameras it would have, fewest
        cameras on a tie. A worker that has not yet reported any camera is taken to cost
        what a camera costs the others, so that a burst of new cameras is not all sent to
        it. A reader function can only go to a worker on this machine."""
        known = [w.busy / w.counted for w in self._workers if w.alive and w.counted]
        typical = sum(known) / len(known) if known else 0.0

        def load(w):
            per_camera = w.busy / w.counted if w.counted else typical
            return per_camera * (len(w.cameras) + 1), len(w.cameras)
        running = [i for i, w in enumerate(self._workers) if w.alive and (isinstance(source, str) or not w.remote)]
        if not running:
            if not isinstance(source, str) and all(w.remote for w in self._workers):
                raise TypeError("a camera on another machine is given by its address, not by a reader")
            raise RuntimeError("no running worker can take this camera")
        return min(running, key=lambda i: load(self._workers[i]))

    def _watch(self):
        """The router's own thread: moves workers' messages to results(), pings every
        worker, and deals with any that has gone quiet or away."""
        next_ping = 0.0
        while not self._closing.is_set():
            with self._lock:
                running = [w for w in self._workers if w.alive]
            if not running:
                self._closing.wait(0.5)
                continue
            for w in wait(running, timeout=0.5):
                try:
                    message = w.recv()
                    w.heard = time.monotonic()
                    self._handle(w, message)
                except EOFError:
                    self._lost(w, "it closed the connection")
                except Exception as e:
                    self._lost(w, f"{type(e).__name__}: {e}")
            if time.monotonic() >= next_ping:
                next_ping = time.monotonic() + _PING
                for w in running:
                    if not w.alive:
                        continue
                    if w.broken:
                        self._lost(w, f"a send to it failed ({w.broken})")
                        continue
                    if time.monotonic() - w.heard > _SILENCE:
                        self._lost(w, f"nothing heard from it for {_SILENCE:.0f} s")
                        continue
                    try:
                        w.send("ping")
                    except Exception as e:
                        self._lost(w, f"{type(e).__name__}: {e}")

    def _handle(self, w, message):
        kind, *rest = message
        if kind == "load":
            w.busy, w.counted = float(rest[0]), int(rest[1])
        elif kind == "updates":
            for u in rest[0]:
                if not isinstance(u, Update):
                    raise ValueError(f"a worker sent {type(u).__name__} for an update")
                self._note(u)
                self._updates.put(u)
        else:
            raise ValueError(f"a worker sent {kind!r}")

    def _note(self, u):
        """Keeps what status() reports and what a move needs: each object's latest mask."""
        with self._lock:
            cam = self._cams.get(u.camera)
            if cam is None:
                return
            if u.masks:
                for obj, mask in u.masks.items():
                    if obj in cam.objects:
                        cam.objects[obj] = mask
                if u.time is not None:
                    cam.last_frame = u.time
                    cam.seen.append((time.time(), time.time() - u.time))
                self._state(cam, "tracking")
            elif u.status:
                self._state(cam, _STATES.get(u.status, cam.state))
                level = logging.INFO if u.status == "reconnecting" else logging.WARNING
                log.log(level, "camera %s %s%s", u.camera, u.status, f": {u.error}" if u.error else "")
            elif u.error:
                log.warning("camera %s: %s", u.camera, u.error)

    def _state(self, cam, state):
        if cam.state != state:
            cam.state, cam.since = state, time.time()

    def _lost(self, w, why):
        """A worker has stopped answering: its cameras go to the others, and it is
        brought back in the background."""
        with self._lock:
            if not w.alive or self._closing.is_set():
                return
            log.error("lost the worker at %s: %s", w.device, why)
            w.alive = False
            moving = sorted(w.cameras, key=str)
            w.cameras.clear()
            for camera in moving:
                self._move(camera, f"its worker at {w.device} stopped ({why})")
            i = self._workers.index(w)
        w.close()  # outside the lock: stopping a local worker can take seconds
        threading.Thread(target=self._revive, args=(i,), daemon=True).start()

    def _move(self, camera, why):
        """Puts a camera on another worker, its objects starting from their last masks."""
        cam = self._cams[camera]
        self.home.pop(camera, None)
        cam.worker = None
        try:
            i = self._place(cam.source)
        except (RuntimeError, TypeError):
            self._event(camera, "waiting", f"{why}; waiting for a worker to take it")
            return
        w = self._workers[i]
        try:
            w.send("add_camera", camera, cam.source, cam.skip, cam.retry_for)
            for obj, prompt in cam.objects.items():
                w.send("start", camera, obj, prompt.array() if isinstance(prompt, Mask) else prompt, None)
        except Exception as e:  # that one is going too: it will be noticed, and this camera wait
            self._event(camera, "waiting", f"{why}; {w.device} could not take it either ({e})")
            return
        cam.worker, self.home[camera] = i, i
        w.cameras.add(camera)
        self._event(camera, "moved", f"{why}; now on {w.device}")

    def _event(self, camera, status, why):
        u = Update(camera, None, None, {}, error=why, status=status)
        self._note(u)
        self._updates.put(u)

    def _revive(self, i):
        """Starts worker i again, or reconnects to it, waiting 2, 4 ... and at most 60 s
        between tries, until it answers or the router closes."""
        spec, wait_s = self._specs[i], 2.0
        while not self._closing.wait(wait_s):
            try:
                if spec.partition(":")[0] in ("cpu", "cuda", "mps"):
                    w = _Local(spec, self._model, self._options, self._keep)
                    w.ready()
                else:
                    w = _Remote(spec, self._token, self._model)
            except Exception as e:
                log.info("the worker at %s is not back yet (%s); next try in %.0f s", spec, e, min(60.0, wait_s * 2))
                wait_s = min(60.0, wait_s * 2)
                continue
            with self._lock:
                if self._closing.is_set():
                    w.close()
                    return
                self._workers[i] = w
                log.warning("the worker at %s is back", spec)
                for camera, cam in sorted(self._cams.items(), key=lambda c: str(c[0])):
                    if cam.worker is None and cam.state == "waiting":
                        self._move(camera, f"{spec} is back")
            return
