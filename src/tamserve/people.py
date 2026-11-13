"""Find the people on every camera and follow them, with no prompt from you.

SAM 3, Meta's model that outlines whatever you name, looks at each camera every
`look_every` frames and outlines every person on it. Between looks the engine
carries each outline from frame to frame, every person on every camera in one
batch. A look is slow (about 65 ms on an RTX 4090, against about 4 ms to follow
a frame), which is why SAM 3 only looks now and then.

On a look, the engine first predicts where each person it is following is on
that frame, and SAM 3's outlines are matched to those predictions by overlap, so
a person keeps their id. Matched and new people then start again from SAM 3's
outlines. Someone a look no longer sees, perhaps behind somebody else, is left
out of the answer but still followed for `hold` frames, and has their id back
if a look sees them again. After that they are forgotten.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

_SAME = 0.3  # overlap (IoU) from which a found outline and a followed person are the same person


def _match(people: dict, found: list, device=None) -> dict:
    """found index -> person id, one to one, the best overlap first."""
    if not people or not found:
        return {}
    ids = list(people)
    a = torch.stack(found).to(device).flatten(1).float()
    b = torch.stack([people[t] for t in ids]).to(device).flatten(1).float()
    inter = a @ b.T  # exact: counts of pixels, whole numbers that float32 holds up to 2**24
    iou = (inter / (a.sum(1)[:, None] + b.sum(1) - inter).clamp(min=1)).cpu()
    out, taken = {}, set()
    for flat in torch.argsort(iou.flatten(), descending=True).tolist():
        i, j = divmod(flat, len(ids))
        if iou[i, j] < _SAME:
            break
        if i not in out and ids[j] not in taken:
            out[i] = ids[j]
            taken.add(ids[j])
    return out


class PeopleFinder:
    """Every person in a frame, outlined by SAM 3.

    Needs Meta's SAM 3 code (the README has the install line) and its weights,
    which are gated: accept Meta's licence for facebook/sam3 on Hugging Face and
    log in with `hf auth login`, or pass a downloaded `checkpoint`.

    threshold: the confidence, 0 to 1, below which SAM 3's outlines are ignored.
    compile: a torch.compile mode such as "default", or None. The word "person" is
        encoded once here, not on every look as SAM 3 ships (113 ms a look down to
        95), and compiling brings a look to about 65 ms, with the same outlines up
        to rounding. Compiling takes a few minutes, done here.
    """

    def __init__(self, *, checkpoint: str | None = None, threshold: float = 0.4, compile: str | None = None,
                 device=None):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Built on the CPU and moved: SAM 3's builder only knows "cuda", the first GPU.
        model = build_sam3_image_model(device="cpu", checkpoint_path=checkpoint,
                                       load_from_HF=checkpoint is None).to(self.device)
        if compile:
            # The decoder gains the most: it is hundreds of small GPU jobs, and the CPU
            # launching them was its limit (20 ms of a look).
            trunk = model.backbone.vision_backbone.trunk
            trunk.forward = torch.compile(trunk.forward, mode=compile, fullgraph=True)
            for part in (model.transformer.encoder, model.transformer.decoder, model.segmentation_head):
                part.forward = torch.compile(part.forward, mode=compile)
        self.model = model
        self.processor = Sam3Processor(model, device=self.device, confidence_threshold=threshold)
        self._words = None  # "person", encoded
        self.scores = []    # per frame of the latest call, each outline's confidence
        if compile:  # compile now, on a made-up picture, not during the first real looks
            noise = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
            for _ in range(4):
                self([noise])

    @torch.inference_mode()
    def __call__(self, frames: list) -> list:
        """frames: H x W x 3 uint8 RGB arrays. Returns, for each, a list of bool masks."""
        found, self.scores = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for frame in frames:
                state = self.processor.set_image(Image.fromarray(frame))
                if self._words is None:
                    self._words = self.model.backbone.forward_text(["person"], device=self.device)
                state["backbone_out"].update(self._words)
                state["geometric_prompt"] = self.model._get_dummy_prompt()
                state = self.processor._forward_grounding(state)
                found.append(list(state["masks"][:, 0].cpu()))
                self.scores.append(state["scores"].float().cpu().tolist())
        return found


@dataclass
class _Camera:
    people: dict = field(default_factory=dict)  # person id -> their latest mask
    unseen: dict = field(default_factory=dict)  # person id -> frames since a look last saw them
    since_look: int | None = None               # frames since the last look; None before the first
    next_id: int = 1
    size: tuple | None = None                   # (height, width) of its frames


class PeopleTracker:
    """Follow every person on every camera; call step() once per round of frames.

    engine: the Engine that follows people between looks. It belongs to the tracker:
        do not start other objects on it.
    finder: what finds the people. By default SAM 3 (PeopleFinder), compiled if the
        engine is. Anything that can be called with a list of frames and returns a
        list of bool masks for each will do; if it has `scores`, the confidence of
        each mask from its latest call, they are used as `start` describes.
    look_every: frames between a camera's looks. 15 suits rooms with a few people.
        In a crowd, people are found sooner and swapped less with a smaller number,
        at a higher cost.
    hold: frames a person the finder no longer sees is still followed, out of the
        answer, in case a look sees them again.
    start: the confidence an outline needs to start following someone new. Weaker
        outlines, down to the finder's own threshold, can only confirm a person
        already followed, after the confident ones have had first pick.
    """

    def __init__(self, engine, *, finder=None, look_every: int = 15, hold: int = 30, start: float = 0.7):
        if look_every < 1 or hold < 0:
            raise ValueError("look_every must be at least 1 and hold at least 0")
        if engine.gpu_masks:
            raise ValueError("PeopleTracker needs an engine that returns masks on the CPU (gpu_masks=False)")
        self.engine = engine
        self.finder = finder if finder is not None else PeopleFinder(
            device=engine.device, compile="default" if engine.compiled else None)
        self.look_every, self.hold, self.start = look_every, hold, start
        self.cameras: dict = {}
        self._joined = 0  # cameras that have had their first look

    def drop(self, cam) -> None:
        """Forget a camera. A camera that stops sending frames keeps its people until dropped."""
        self.cameras.pop(cam, None)
        self.engine.drop(cam)

    def step(self, frames: dict) -> dict:
        """frames: camera id -> the frame as a decoder gives it (an H x W x 3 uint8 RGB
        array). Returns camera id -> {person id: bool mask at the frame's own size}.
        Ids are whole numbers from 1, counted per camera."""
        frames = {c: np.asarray(f) for c, f in frames.items()}
        for c, f in frames.items():
            cam = self.cameras.setdefault(c, _Camera())
            if cam.size not in (None, f.shape[:2]):
                self.drop(c)  # the camera changed resolution: start it afresh
                cam = self.cameras.setdefault(c, _Camera())
            cam.size = f.shape[:2]
        looking = self._due(list(frames))
        # Uploaded and resized once: the same tensors serve the pass that follows people
        # and the pass that starts them on a look frame, so a look frame is encoded once.
        ready = self.engine._prepared(frames, [c for c in frames if self.cameras[c].people or c in looking])

        # The engine predicts this frame for every person on every camera, in one batch.
        for c, masks in self.engine.step({c: ready[c] for c in frames if self.cameras[c].people}).items():
            cam = self.cameras[c]
            for t, m in masks.items():
                cam.people[t] = m
                cam.unseen[t] += 1

        seeds = {}
        found = self.finder([frames[c] for c in looking]) if looking else []
        scores = getattr(self.finder, "scores", None) or [None] * len(found)
        for c, outlines, confidence in zip(looking, found, scores):
            cam = self.cameras[c]
            outlines, ids = self._sorted({t: m for t, m in cam.people.items() if m.any()}, outlines, confidence)
            for i, mask in enumerate(outlines):
                t = ids.get(i)
                if t is None:  # someone new
                    t, cam.next_id = cam.next_id, cam.next_id + 1
                seeds.setdefault(c, {})[t] = cam.people[t] = mask
                cam.unseen[t] = 0
            for t in [t for t, n in cam.unseen.items() if n > self.hold]:
                self.engine.drop(c, t)
                del cam.people[t], cam.unseen[t]
            if cam.since_look is None:  # first look: set this camera's later looks apart from the others'
                cam.since_look = self._joined % self.look_every
                self._joined += 1
            else:
                cam.since_look = 0
        if seeds:  # on the frame just followed, without moving anyone else forward
            self.engine.add({c: ready[c] for c in seeds}, seeds)
        out = {}
        for c in frames:
            cam = self.cameras[c]
            cam.since_look += 1
            # Only the people the latest look saw; the others keep their ids, out of sight.
            out[c] = {t: m for t, m in cam.people.items() if cam.unseen[t] < cam.since_look and m.any()}
        return out

    def _sorted(self, people, outlines, confidence):
        """(the outlines kept, their index -> the id of the person each one is). Confident
        outlines have first pick of the people being followed; the rest may only confirm
        someone left over, and are dropped otherwise."""
        device = self.engine.device
        if confidence is None:
            return outlines, _match(people, outlines, device)
        sure = [i for i, s in enumerate(confidence) if s >= self.start]
        weak = [i for i, s in enumerate(confidence) if s < self.start]
        ids = {sure[k]: t for k, t in _match(people, [outlines[i] for i in sure], device).items()}
        left = {t: m for t, m in people.items() if t not in ids.values()}
        ids.update({weak[k]: t for k, t in _match(left, [outlines[i] for i in weak], device).items()})
        kept = [i for i in range(len(outlines)) if i in ids or confidence[i] >= self.start]
        return [outlines[i] for i in kept], {k: ids[i] for k, i in enumerate(kept) if i in ids}

    def _due(self, cams):
        """Cameras that look this step. A camera's first frame is always looked at: who is
        on it should not wait their turn. After that the most overdue go first, about
        cameras / look_every per step, so looks do not all land at once."""
        new = [c for c in cams if self.cameras[c].since_look is None]
        due = [c for c in cams if (self.cameras[c].since_look or 0) >= self.look_every]
        due.sort(key=lambda c: -self.cameras[c].since_look)
        return new + due[:max(1, math.ceil(len(self.cameras) / self.look_every))]
