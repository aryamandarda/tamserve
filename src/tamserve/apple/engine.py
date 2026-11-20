"""An Engine whose image encoder runs on the Neural Engine and whose GPU follows the objects."""
from __future__ import annotations

import numpy as np
import torch

from tamserve.apple.encoder import SIZE, NeuralEncoder
from tamserve.engine import Engine


class AppleEngine(Engine):
    """Engine, on an Apple-silicon Mac. Made by Engine(predictor) when it can be; not built directly.

    A frame handed to step() goes to the Neural Engine and the GPU waits for its
    features. Frames passed as `next_frames` start on the Neural Engine at once, while
    the GPU follows objects on the current ones, and that is where the speed is: 40
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
        self._ahead: dict = {}  # camera -> (a copy of the frame given as its next, what _prepared gives for it)
        self._next = None       # the `next_frames` of the step() under way, until they have been started

    def step(self, frames: dict, seeds: dict | None = None, *, next_frames: dict | None = None) -> dict:
        self._next = next_frames or {}
        try:
            return super().step(frames, seeds)
        finally:
            self._next = None

    def _prepared(self, frames, busy):
        out = {}
        for c in busy:
            frame = frames[c]
            if isinstance(frame, tuple):  # prepare() output
                out[c] = frame
                continue
            ahead = self._ahead.pop(c, None)
            # Known by its pixels: the caller may hand over another array, or have written over the one it gave.
            came = ahead is not None and np.asarray(frame).dtype == np.uint8 and np.array_equal(ahead[0], frame)
            out[c] = ahead[1] if came else self._start(c, frame)
        if self._next is not None:  # after this step's own frames, so those are encoded first
            followed = {cam for cam, _ in self.tracks}
            # Earlier next frames that did not come are dropped here. Each is copied: it is read after step() returns.
            self._ahead = {}
            for c, frame in self._next.items():
                if (c in followed or c in busy) and not isinstance(frame, tuple):
                    frame = np.array(frame)
                    self._ahead[c] = (frame, self._start(c, frame))
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
