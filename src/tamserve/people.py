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

from dataclasses import dataclass, field

import numpy as np

from tamserve.finder import PeopleFinder
from tamserve.tracker import _due, _first_pick, _interval, _match

_SAME = 0.3  # overlap (IoU) from which a found outline and a followed person are the same person


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
        at a higher cost. A dict gives cameras their own, such as {"plaza": 1}, and
        those it does not name keep 15. It can be changed while running.
    hold: frames a person the finder no longer sees is still followed, out of the
        answer, in case a look sees them again.
    start: the confidence an outline needs to start following someone new. Weaker
        outlines, down to the finder's own threshold, can only confirm a person
        already followed, after the confident ones have had first pick.
    """

    def __init__(self, engine, *, finder=None, look_every: int | dict = 15, hold: int = 30, start: float = 0.7):
        for c in look_every if isinstance(look_every, dict) else [None]:
            _interval(look_every, c)  # raises below 1
        if hold < 0:
            raise ValueError("hold must be at least 0")
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

    def step(self, frames: dict, *, next_frames: dict | None = None) -> dict:
        """frames: camera id -> the frame as a decoder gives it (an H x W x 3 uint8 RGB
        array). next_frames: the frames the next step() will be given, where you already
        have them; a Mac starts on them early (Engine.step). Returns camera id ->
        {person id: bool mask at the frame's own size}.
        Ids are whole numbers from 1, counted per camera."""
        frames = {c: np.asarray(f) for c, f in frames.items()}
        for c, f in frames.items():
            cam = self.cameras.setdefault(c, _Camera())
            if cam.size not in (None, f.shape[:2]):
                self.drop(c)  # the camera changed resolution: start it afresh
                cam = self.cameras.setdefault(c, _Camera())
            cam.size = f.shape[:2]
        every = {c: _interval(self.look_every, c, 15) for c in self.cameras}
        looking = _due(self.cameras, list(frames), every)
        # Uploaded and resized once: the same tensors serve the pass that follows people
        # and the pass that starts them on a look frame, so a look frame is encoded once.
        ready = self.engine._prepared(frames, [c for c in frames if self.cameras[c].people or c in looking])

        # The engine predicts this frame for every person on every camera, in one batch.
        following = {c: ready[c] for c in frames if self.cameras[c].people}
        for c, masks in self.engine.step(following, next_frames=next_frames).items():
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
                cam.since_look = self._joined % every[c]
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
            return outlines, _match(people, outlines, _SAME, device)
        kept, ids = _first_pick(people, outlines, confidence, self.start, _SAME, device)
        return [outlines[i] for i in kept], ids
