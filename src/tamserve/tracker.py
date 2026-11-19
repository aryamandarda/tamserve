"""Find what you name on every camera and follow it.

You give each camera one or more prompts, such as "dog" or "the red mug". SAM 3,
Meta's model that outlines whatever you name, looks at each camera every
`look_every` frames and outlines every match. Between looks the engine carries
each outline from frame to frame, every object on every camera in one batch. A
look is slow (about 65 ms on an RTX 4090 for one prompt) and is not shared
between cameras, which is why SAM 3 only looks now and then.

On a look, the engine first predicts where each object it is following is on
that frame, and SAM 3's outlines are matched to those predictions by overlap, so
an object keeps its id. What is kept follows the rules of SAM 3's own video
tracker, read from its code:

- An outline that repeats a more confident one of the same look is dropped.
- Only a confident outline starts a new object. A weaker one can confirm an
  object already followed, and is dropped otherwise.
- A followed object starts again from SAM 3's outline only when SAM 3 is very
  sure of that outline. Otherwise it keeps the mask the engine gave it.
- A new object that most looks of its first 15 frames do not see is removed.
- After that an object stays in the answer for as long as the engine has a mask
  for it, whether the latest look saw it or not. It is forgotten once no look
  has seen it for `hold` frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import torch
from PIL import Image

from tamserve.finder import Finder

# SAM 3's own values (sam3_video_base.py and model_builder.py at commit f1647e8).
_SAME = 0.1       # overlap (IoU) from which a found outline and a followed object are the same object
_REPEAT = 0.1     # overlap from which a found outline repeats a more confident one
_RESTART = 0.8    # confidence from which a followed object starts again from the found outline
_TRIAL = 15       # frames a new object is on trial for
_MISSED = 8 / 15  # the share of a trial's looks that must miss the object for it to be removed
# Cameras that do not move (still=True).
_BLOCK = 16       # a picture is compared by the mean colour of each block of this many pixels a side
_MOVED = 8        # grey levels a block's mean colour must move by for the picture to have changed
_STILL_LOOK = 30  # frames between the looks of a camera whose picture has not changed


def _overlap(a: list, b: list, device=None) -> torch.Tensor:
    """The overlap (IoU) of every mask in a with every mask in b, len(a) x len(b), on the CPU."""
    a = torch.stack(a).to(device).flatten(1).float()
    b = torch.stack(b).to(device).flatten(1).float()
    inter = a @ b.T  # exact: counts of pixels, whole numbers that float32 holds up to 2**24
    return (inter / (a.sum(1)[:, None] + b.sum(1) - inter).clamp(min=1)).cpu()


def _match(followed: dict, found: list, least: float, device=None) -> dict:
    """found index -> id, one to one, the best overlap first, down to an overlap of `least`."""
    if not followed or not found:
        return {}
    ids = list(followed)
    iou = _overlap(found, [followed[t] for t in ids], device)
    out, taken = {}, set()
    for flat in torch.argsort(iou.flatten(), descending=True).tolist():
        i, j = divmod(flat, len(ids))
        if iou[i, j] < least:
            break
        if i not in out and ids[j] not in taken:
            out[i] = ids[j]
            taken.add(ids[j])
    return out


def _first_pick(followed: dict, outlines: list, confidence: list, start: float, least: float, device=None):
    """(the indices of the outlines kept, position among those -> the id it is). Outlines
    of at least `start` confidence have first pick of what is being followed; the rest
    may only confirm something left over, and are dropped otherwise."""
    sure = [i for i, s in enumerate(confidence) if s >= start]
    weak = [i for i, s in enumerate(confidence) if s < start]
    ids = {sure[k]: t for k, t in _match(followed, [outlines[i] for i in sure], least, device).items()}
    left = {t: m for t, m in followed.items() if t not in ids.values()}
    ids.update({weak[k]: t for k, t in _match(left, [outlines[i] for i in weak], least, device).items()})
    kept = [i for i in range(len(outlines)) if i in ids or confidence[i] >= start]
    return kept, {k: ids[i] for k, i in enumerate(kept) if i in ids}


def _thumbnail(frame) -> np.ndarray:
    """One mean colour per block of a frame: what a still camera's picture is compared
    by. About half a millisecond for a 720p frame."""
    return np.asarray(Image.fromarray(frame).reduce(_BLOCK), dtype=np.int16)


def _moved(a, b) -> bool:
    return bool((np.abs(a - b).max(-1) > _MOVED).any())


def _interval(look_every, cam, default=None):
    """Frames between a camera's looks: `look_every`, or that camera's own where it is a dict."""
    every = look_every.get(cam, default) if isinstance(look_every, dict) else look_every
    if every < 1:
        raise ValueError("look_every must be at least 1")
    return every


def _due(cameras: dict, cams: list, every: dict, changed=lambda c: True) -> list:
    """Cameras that look this step. A camera's first frame is always looked at: what is
    on it should not wait its turn. After that a camera is due `every[camera]` frames
    after its last look, and no more look in one step than those intervals call for on
    average, so looks do not all land at once. Those that look most often go first,
    then the most overdue."""
    new = [c for c in cams if cameras[c].since_look is None]
    due = [c for c in cams if (cameras[c].since_look or 0) >= every[c] and changed(c)]
    due.sort(key=lambda c: (every[c], -cameras[c].since_look))
    return new + due[:max(1, math.ceil(sum(1 / Fraction(k) for k in every.values())))]


@dataclass
class _Camera:
    objects: dict = field(default_factory=dict)  # object id -> its latest mask
    unseen: dict = field(default_factory=dict)   # object id -> frames since a look last saw it
    name: dict = field(default_factory=dict)     # object id -> the prompt it was found for
    born: dict = field(default_factory=dict)     # object id -> the frame it started on
    missed: dict = field(default_factory=dict)   # object id -> looks of its trial that did not see it
    since_look: int | None = None                # frames since the last look; None before the first
    next_id: int = 1
    frame: int = 0                               # frames this camera has sent
    size: tuple | None = None                    # (height, width) of its frames
    followed: object = None                      # still: thumbnail of the last frame its objects were followed on
    looked: object = None                        # still: thumbnail of the frame of its last look


class Tracker:
    """Follow everything you name on every camera; call step() once per round of frames.

    engine: the Engine that follows objects between looks. It belongs to the tracker:
        do not start other objects on it. A larger model follows more accurately:
        Engine(build_video_predictor("cuda", "sam2.1-large")).
    prompts: what to find. One phrase ("dog"), several (["dog", "ball"]), or a dict
        that gives each camera its own. Several prompts on a camera cost far less than
        that many cameras: its picture is encoded once for all of them. `prompts` can
        be changed while running; a camera takes its new ones up at its next look.
    finder: what finds the objects. By default SAM 3 (Finder), compiled if the engine
        is. Anything that can be called as Finder is and sets `scores` and `names` as
        it does will do.
    look_every: frames between a camera's looks. Looking more often finds things
        sooner at a higher cost; it does not raise accuracy. A dict gives cameras
        their own, such as {"dock": 1}, and those it does not name keep 3. Like
        `prompts`, it can be changed while running.
    hold: frames an object no look has seen is still followed before it is forgotten.
    start: the confidence an outline needs to start following something new. Weaker
        outlines, down to the finder's own threshold, can only confirm an object
        already followed.
    still: for cameras that do not move. A camera whose picture has not changed looks
        every 30th frame only, and its objects are not followed on frames that did not
        change: their masks stay as they were. It changes nothing on a camera where
        something always moves.
    """

    def __init__(self, engine, prompts, *, finder=None, look_every: int | dict = 3, hold: int = 300,
                 start: float = 0.7, still: bool = False):
        for c in look_every if isinstance(look_every, dict) else [None]:
            _interval(look_every, c)  # raises below 1
        if hold < 0:
            raise ValueError("hold must be at least 0")
        if engine.gpu_masks:
            raise ValueError("Tracker needs an engine that returns masks on the CPU (gpu_masks=False)")
        self.engine, self.prompts = engine, prompts
        self.finder = finder if finder is not None else Finder(
            device=engine.device, compile="default" if engine.compiled else None)
        self.look_every, self.hold, self.start, self.still = look_every, hold, start, still
        self.cameras: dict = {}
        self._joined = 0  # cameras that have had their first look

    def drop(self, cam) -> None:
        """Forget a camera. A camera that stops sending frames keeps its objects until dropped."""
        self.cameras.pop(cam, None)
        self.engine.drop(cam)

    def names(self, cam) -> dict:
        """object id -> the prompt it was found for, for everything followed on a camera."""
        return dict(self.cameras[cam].name) if cam in self.cameras else {}

    def step(self, frames: dict) -> dict:
        """frames: camera id -> the frame as a decoder gives it (an H x W x 3 uint8 RGB
        array). Returns camera id -> {object id: bool mask at the frame's own size}.
        Ids are whole numbers from 1, counted per camera; names() says what each one is."""
        frames = {c: np.asarray(f) for c, f in frames.items()}
        for c, f in frames.items():
            cam = self.cameras.setdefault(c, _Camera())
            if cam.size not in (None, f.shape[:2]):
                self.drop(c)  # the camera changed resolution: start it afresh
                cam = self.cameras.setdefault(c, _Camera())
            cam.size = f.shape[:2]
        thumbs = {c: _thumbnail(f) for c, f in frames.items()} if self.still else {}
        every = {c: _interval(self.look_every, c, 3) for c in self.cameras}
        looking = _due(self.cameras, list(frames), every,
                       lambda c: not self.still or self._changed(self.cameras[c], thumbs[c]))
        follow = [c for c in frames if self.cameras[c].objects]
        if self.still:
            for c in list(follow):
                cam = self.cameras[c]
                if cam.followed is not None and not _moved(thumbs[c], cam.followed):
                    follow.remove(c)  # nothing moved: its masks stay as they are
                    for t in cam.unseen:
                        cam.unseen[t] += 1
            for c in follow + looking:
                self.cameras[c].followed = thumbs[c]
        # Uploaded and resized once: the same tensors serve the pass that follows objects
        # and the pass that starts them on a look frame, so a look frame is encoded once.
        ready = self.engine._prepared(frames, [c for c in frames if c in follow or c in looking])

        # The engine predicts this frame for every object on every camera, in one batch.
        for c, masks in self.engine.step({c: ready[c] for c in follow}).items():
            cam = self.cameras[c]
            for t, m in masks.items():
                cam.objects[t] = m
                cam.unseen[t] += 1
        for c in frames:
            self.cameras[c].frame += 1

        seeds = {}
        wanted = [self._wanted(c) for c in looking]
        found = self.finder([frames[c] for c in looking], wanted) if looking else []
        for k, c in enumerate(looking):
            cam = self.cameras[c]
            misses = max(1, math.ceil(_MISSED * (_TRIAL // every[c])))  # looks, of a trial's
            for t in [t for t, name in cam.name.items() if name not in wanted[k]]:
                self._forget(c, t)  # the camera is no longer asked for it
            for prompt in wanted[k]:  # each prompt's outlines meet only that prompt's objects
                mine = [i for i, name in enumerate(self.finder.names[k]) if name == prompt]
                outlines, sure = self._distinct([found[k][i] for i in mine], [self.finder.scores[k][i] for i in mine])
                live = {t: m for t, m in cam.objects.items() if m.any() and cam.name[t] == prompt}
                kept, ids = _first_pick(live, outlines, sure, self.start, _SAME, self.engine.device)
                for i, j in enumerate(kept):
                    t = ids.get(i)
                    if t is None:  # something new
                        t, cam.next_id = cam.next_id, cam.next_id + 1
                        cam.name[t], cam.born[t] = prompt, cam.frame
                    if t not in cam.objects or sure[j] >= _RESTART:
                        seeds.setdefault(c, {})[t] = cam.objects[t] = outlines[j]
                    cam.unseen[t] = 0
                for t in [t for t in live if t not in ids.values() and cam.frame - cam.born[t] <= _TRIAL]:
                    cam.missed[t] = cam.missed.get(t, 0) + 1
                    if cam.missed[t] >= misses:  # a new object the looks keep missing was a false find
                        self._forget(c, t)
            for t in [t for t, n in cam.unseen.items() if n > self.hold]:
                self._forget(c, t)
            if self.still:
                cam.looked = thumbs[c]
            if cam.since_look is None:  # first look: set this camera's later looks apart from the others'
                cam.since_look = self._joined % every[c]
                self._joined += 1
            else:
                cam.since_look = 0
        if seeds:  # on the frame just followed, without moving anything else forward
            self.engine.add({c: ready[c] for c in seeds}, seeds)
        out = {}
        for c in frames:
            cam = self.cameras[c]
            cam.since_look += 1
            out[c] = {t: m for t, m in cam.objects.items() if m.any()}
        return out

    def _wanted(self, cam) -> list:
        """A camera's prompts."""
        if isinstance(self.prompts, dict) and cam not in self.prompts:
            raise KeyError(f"no prompts were given for camera {cam!r}")
        wanted = self.prompts[cam] if isinstance(self.prompts, dict) else self.prompts
        return [wanted] if isinstance(wanted, str) else list(dict.fromkeys(wanted))

    def _distinct(self, outlines, confidence):
        """The outlines without those that repeat a more confident one, and their confidences."""
        if len(outlines) < 2:
            return outlines, confidence
        iou, best = _overlap(outlines, outlines, self.engine.device), []
        for i in sorted(range(len(outlines)), key=lambda i: -confidence[i]):
            if all(iou[i, j] < _REPEAT for j in best):
                best.append(i)
        return [outlines[i] for i in sorted(best)], [confidence[i] for i in sorted(best)]

    def _forget(self, c, t) -> None:
        cam = self.cameras[c]
        self.engine.drop(c, t)
        for known in (cam.objects, cam.unseen, cam.name, cam.born, cam.missed):
            known.pop(t, None)

    def _changed(self, cam, thumb) -> bool:
        """Whether a still camera's picture changed enough since its last look to look again."""
        if cam.looked is None or cam.since_look >= _STILL_LOOK:
            return True
        if any(cam.frame - born < _TRIAL for born in cam.born.values()):
            return True  # something new is on trial, and only looks can judge it
        return _moved(thumb, cam.looked)
