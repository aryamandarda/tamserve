"""An Engine whose image encoder runs on the Neural Engine and whose GPU follows the objects."""
from __future__ import annotations

import threading

import numpy as np
import torch

from tamserve.apple.encoder import SIZE, NeuralEncoder
from tamserve.engine import Engine


class AppleEngine(Engine):
    """Engine, on an Apple-silicon Mac. Made by Engine(predictor) when it can be; not built directly.

    A frame handed to step() goes to the Neural Engine and the GPU waits for its
    features. A frame the engine is given early, by start_encoding() as it arrives or
    as one of step()'s `next_frames`, starts on the Neural Engine at once, while the
    GPU follows objects on the frames before, and that is where the speed is: 40
    frames a second with one camera on an M4 Max, against 22 with everything on the
    GPU. Frames from prepare() are encoded on the GPU, as stock encodes them.
    """

    def __init__(self, predictor, *, encoder=None, **options):
        options.pop("neural_engine", None)
        super().__init__(predictor, **options)
        self._neural = encoder if encoder is not None else NeuralEncoder(predictor.tamserve_model)
        # Position encodings depend on the frame size alone: the model's own, made once.
        self._use(True)
        try:
            blank = torch.zeros(3, SIZE, SIZE, dtype=torch.uint8, device=self.device)
            with torch.inference_mode():
                self._pos_enc = Engine._backbone(self, [blank])["vision_pos_enc"]
        finally:
            self._use(False)
        self._ahead: dict = {}  # camera -> [(a copy of a frame given early, what _prepared gives for it)], oldest first
        self._early = threading.Lock()  # guards _ahead: start_encoding() comes from the threads that read cameras
        self._next = None       # the `next_frames` of the step() under way, until they have been started

    def step(self, frames: dict, seeds: dict | None = None, *, next_frames: dict | None = None) -> dict:
        self._next = next_frames or {}
        try:
            return super().step(frames, seeds)
        finally:
            self._next = None

    def start_encoding(self, frames: dict) -> dict:
        return self._early_start(frames, {cam for cam, _ in list(self.tracks)})  # list(): step() may be changing it

    def drop(self, cam, obj=None) -> None:
        super().drop(cam, obj)
        if obj is None:
            with self._early:
                self._ahead.pop(cam, None)

    def _early_start(self, frames, cams):
        """Start the Neural Engine on the frames of `cams`. Returns camera -> whether its frame is encoded yet."""
        ready = {}
        for c, frame in frames.items():
            if c not in cams or isinstance(frame, tuple):
                continue
            frame = np.array(frame)  # the engine's own copy: it is compared later, and the caller may reuse its array
            started = self._start(c, frame)
            job = started[0].features
            with self._early:
                # Earlier frames of this camera whose turn has not come are passed over and never encoded.
                # The newest that is encoded stays, with any under way: a loop that is behind can take it.
                rest = [e for e in self._ahead.get(c, ()) if not e[1][0].features.cancel()]
                done = [i for i, e in enumerate(rest) if e[1][0].features.done()]
                self._ahead[c] = rest[done[-1] if done else 0:] + [(frame, started)]
            ready[c] = lambda job=job: job.done() and not job.cancelled()
        return ready

    def _prepared(self, frames, busy):
        out = {}
        for c in busy:
            frame = frames[c]
            if isinstance(frame, tuple):  # prepare() output
                out[c] = frame
                continue
            # Known by its pixels: the caller may hand over another array, or have written over the one it gave.
            # Early frames newer than this one stay: they are for later steps.
            with self._early:
                early = self._ahead.get(c, [])
                raw = np.asarray(frame).dtype == np.uint8
                at = next((i for i, e in enumerate(early) if raw and np.array_equal(e[0], frame)), None)
                started = None if at is None else early[at][1]
                del early[:0 if at is None else at + 1]
            out[c] = started or self._start(c, frame)
        if self._next is not None:  # after this step's own frames, so those are encoded first
            self._early_start(self._next, {cam for cam, _ in self.tracks} | set(busy))
            self._next = None
        return out

    def _start(self, cam, frame):
        """Send a frame to the Neural Engine; what stands for it until its features are read."""
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"camera {cam!r}: a frame is an H x W x 3 uint8 array or prepare() output")
        mark = torch.zeros(1)
        mark.features = self._neural.submit(frame)  # freed with the mark if the frame is never used
        return mark, frame.shape[:2]

    def _backbone(self, images):
        jobs = [getattr(im, "features", None) for im in images]
        rest = [im for im, job in zip(images, jobs) if job is None]
        if len(rest) == len(images):
            return super()._backbone(images)
        on_gpu = super()._backbone(rest)["backbone_fpn"] if rest else None
        kind = torch.float16 if self.bf16 and self.device.type == "mps" else torch.float32
        maps = []
        for level in range(3):
            rows, k = [], 0
            for job in jobs:
                if job is None:
                    rows.append(on_gpu[level][k:k + 1].to(kind))
                    k += 1
                else:
                    rows.append(torch.from_numpy(job.result()[level]).to(self.device, kind))
            maps.append(rows[0] if len(rows) == 1 else torch.cat(rows))
        return {"vision_features": maps[-1], "vision_pos_enc": self._pos_enc, "backbone_fpn": maps}
