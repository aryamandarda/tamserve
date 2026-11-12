"""Cameras over several GPUs, used through one object.

A Router starts a worker process per GPU (tamserve.worker) and puts each camera on
the one with the most room, where it stays: its objects' memory lives on that GPU.
Workers open and decode their own cameras, so video never passes between processes;
the router sends them commands and receives compact masks.
"""
from __future__ import annotations

import os
import queue
import threading
from multiprocessing.connection import wait

import torch

from tamserve.worker import Update, serve

_ENVIRON = threading.Lock()


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


class _Worker:
    """A worker process, as the router sees it."""

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
        self.alive, self.lock = False, threading.Lock()

    def ready(self):
        """Waits until the worker's engine is built."""
        try:
            kind, detail = self.events.recv()
        except EOFError:
            kind, detail = "error", "it stopped while starting"
        if kind != "ready":
            raise RuntimeError(f"the worker for {self.device} did not start: {detail}")
        self.alive = True

    def send(self, *command):
        with self.lock:
            if not self.alive:
                raise RuntimeError(f"the worker for {self.device} has stopped")
            self.commands.send(command)


class Router:
    """Cameras spread over this machine's GPUs, used through one object.

    workers: "local" for a worker on every GPU this process can see, or a list of
        devices, such as ["cuda:0", "cuda:2"] or ["cpu", "cpu"].
    keep: frames each camera keeps, for prompts made on a frame its worker has
        already passed (start(..., frame=n)).
    Every other argument goes to each worker's Engine, such as compile=True.

    Create it under `if __name__ == "__main__":`. Workers are new Python processes,
    and they import your main module as they start.
    """

    def __init__(self, model: str = "edgetam", workers="local", *, keep: int = 32, **options):
        if workers == "local":
            workers = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if not workers:
                raise ValueError("no GPU is visible; pass workers=['cpu'] to run on the CPU")
        self.home: dict = {}  # camera -> index of its worker
        self._updates: queue.Queue = queue.Queue()
        self._workers: list = []
        self._closing, self._reader = False, None
        try:
            # Compiling, the first worker goes alone and the others start from its compiled code.
            for group in ([workers[:1], workers[1:]] if options.get("compile") else [workers]):
                started = [_Worker(d, model, options, keep) for d in group]
                self._workers += started
                for w in started:
                    w.ready()
        except BaseException:
            self.close()
            raise
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def add_camera(self, camera, source, *, skip: bool = True) -> int:
        """Start reading a camera, on the worker with the most room; returns its index.

        source: an address OpenCV can open (rtsp://..., a file), which needs
            opencv-python; or something the worker can call to get a reader, whose
            read() returns the next frame as an H x W x 3 uint8 RGB array and None at
            the end. It is sent to the worker, so it must be picklable: a function or
            class defined at the top of a module, or functools.partial of one.
        skip: when tracking falls behind the camera, take its newest frame and skip
            the rest, as a live camera needs. False tracks every frame, as for a
            recorded video.
        """
        if camera in self.home:
            raise KeyError(f"camera {camera!r} is already added")
        i = self._place()
        self._workers[i].send("add_camera", camera, source, skip)
        self.home[camera] = i
        self._workers[i].cameras.add(camera)
        return i

    def remove_camera(self, camera) -> None:
        """Stop reading a camera and forget what was tracked on it."""
        w = self._workers[self._where(camera)]
        del self.home[camera]
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
        self._workers[self._where(camera)].send("start", camera, obj, prompt, frame)

    def stop(self, camera, obj=None) -> None:
        """Stop tracking `obj`; with no object, everything on the camera."""
        self._workers[self._where(camera)].send("stop", camera, obj)

    def results(self, timeout: float | None = None):
        """Updates as they arrive, from every worker. With a timeout, it ends after that
        many seconds without one; it also ends once the router is closed."""
        while True:
            try:
                update = self._updates.get(timeout=timeout)
            except queue.Empty:
                return
            if update is None:
                self._updates.put(None)  # for whoever reads next
                return
            yield update

    def close(self) -> None:
        """Stops every worker and waits for them to exit."""
        self._closing = True
        for w in self._workers:
            try:
                w.send("close")
            except (RuntimeError, OSError):
                pass
        for w in self._workers:
            w.process.join(timeout=10)
            if w.process.is_alive():
                w.process.terminate()
        if self._reader is not None:
            self._reader.join(timeout=5)
        else:
            self._updates.put(None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _where(self, camera):
        if camera not in self.home:
            raise KeyError(f"no camera {camera!r}; add it with add_camera()")
        return self.home[camera]

    def _place(self):
        """The running worker with the most room: how busy it last said it was, scaled
        to the cameras it has now (some may have come since), fewest cameras on a tie."""
        def load(w):
            per_camera = w.busy / w.counted if w.counted else 0.0
            return per_camera * (len(w.cameras) + 1), len(w.cameras)
        running = [i for i, w in enumerate(self._workers) if w.alive]
        if not running:
            raise RuntimeError("no worker is running")
        return min(running, key=lambda i: load(self._workers[i]))

    def _read(self):
        """Moves every worker's messages to the queue results() reads from."""
        reading = {w.events: w for w in self._workers}
        while reading:
            for conn in wait(list(reading)):
                w = reading[conn]
                try:
                    kind, *rest = conn.recv()
                except (EOFError, OSError):
                    del reading[conn]
                    with w.lock:
                        w.alive = False
                    if not self._closing:
                        for camera in list(w.cameras):
                            self._updates.put(Update(camera, None, None, {}, error="its worker stopped"))
                    continue
                if kind == "updates":
                    for u in rest[0]:
                        self._updates.put(u)
                elif kind == "load":
                    w.busy, w.counted = rest
        self._updates.put(None)
