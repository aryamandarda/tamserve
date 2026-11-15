"""Find what you name on every camera and follow it.

You give each camera one or more prompts, such as "dog" or "the red mug". SAM 3,
Meta's model that outlines whatever you name, looks at each camera every
`look_every` frames and outlines every match. Between looks the engine carries
each outline from frame to frame, every object on every camera in one batch. A
look is slow (about 65 ms on an RTX 4090 for one prompt, against a few ms to
follow a frame), which is why SAM 3 only looks now and then.

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

import numpy as np
import torch
from PIL import Image

# SAM 3's own values (sam3_video_base.py and model_builder.py at the commit below).
_SAME = 0.1       # overlap (IoU) from which a found outline and a followed object are the same object
_REPEAT = 0.1     # overlap from which a found outline repeats a more confident one
_RESTART = 0.8    # confidence from which a followed object starts again from the found outline
_TRIAL = 15       # frames a new object is on trial for
_MISSED = 8 / 15  # the share of a trial's looks that must miss the object for it to be removed
# Cameras that do not move (still=True).
_BLOCK = 16       # a picture is compared by the mean colour of each block of this many pixels a side
_MOVED = 8        # grey levels a block's mean colour must move by for the picture to have changed
_STILL_LOOK = 30  # frames between the looks of a camera whose picture has not changed
# The word list SAM 3 reads prompts with. Its PyPI package leaves the file out, so it is
# fetched once from Meta's repository and checked against this SHA-256.
_VOCAB = ("https://raw.githubusercontent.com/facebookresearch/sam3/f1647e8d0c1d280991f4d1b7d974c2704537fe66"
          "/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
          "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a")


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


class Finder:
    """Everything you name in a frame, outlined by SAM 3.

    Needs Meta's SAM 3 code (`pip install "tamserve[sam3]"`) and its weights,
    which are gated: accept Meta's licence for facebook/sam3 on Hugging Face and
    log in with `hf auth login`, or pass a downloaded `checkpoint`.

    threshold: the confidence, 0 to 1, below which SAM 3's outlines are ignored.
    compile: a torch.compile mode such as "default", or None. Each prompt is
        encoded once here, not on every look as SAM 3 ships (113 ms a look down to
        95), and compiling brings a look to about 65 ms, with the same outlines up
        to rounding. Compiling takes a few minutes, done here.
    """

    def __init__(self, *, checkpoint: str | None = None, threshold: float = 0.5, compile: str | None = None,
                 device=None):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        from tamserve.models import CACHE, _download

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        vocab = _download(_VOCAB[0], CACHE / "bpe_simple_vocab_16e6.txt.gz", _VOCAB[1])
        # Built on the CPU and moved: SAM 3's builder only knows "cuda", the first GPU.
        model = build_sam3_image_model(bpe_path=str(vocab), device="cpu", checkpoint_path=checkpoint,
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
        self._words = {}  # prompt -> its encoding
        self.scores = []  # per frame of the latest call, each outline's confidence
        self.names = []   # the same, the prompt each outline was found for
        if compile:  # compile now, on a made-up picture, not during the first real looks
            noise = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
            for _ in range(4):
                Finder.__call__(self, [noise], [["person"]])

    @torch.inference_mode()
    def __call__(self, frames: list, prompts: list) -> list:
        """frames: H x W x 3 uint8 RGB arrays. prompts: for each frame, the list of things
        to find in it. Returns, for each frame, a list of bool masks; `scores` and `names`
        then hold each mask's confidence and the prompt it was found for."""
        found, self.scores, self.names = [], [], []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for frame, wanted in zip(frames, prompts):
                # The picture is encoded once, whatever the number of prompts: most of a look.
                state = self.processor.set_image(Image.fromarray(frame))
                masks, scores, names = [], [], []
                for prompt in wanted:
                    if prompt not in self._words:
                        self._words[prompt] = self.model.backbone.forward_text([prompt], device=self.device)
                    state["backbone_out"].update(self._words[prompt])
                    state["geometric_prompt"] = self.model._get_dummy_prompt()
                    state = self.processor._forward_grounding(state)
                    masks += list(state["masks"][:, 0].cpu())
                    scores += state["scores"].float().cpu().tolist()
                    names += [prompt] * (len(masks) - len(names))
                found.append(masks)
                self.scores.append(scores)
                self.names.append(names)
        return found


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
        that many cameras: its picture is encoded once for all of them.
    finder: what finds the objects. By default SAM 3 (Finder), compiled if the engine
        is. Anything that can be called as Finder is and sets `scores` and `names` as
        it does will do.
    look_every: frames between a camera's looks. Looking more often finds things
        sooner at a higher cost; it does not raise accuracy.
    hold: frames an object no look has seen is still followed before it is forgotten.
    start: the confidence an outline needs to start following something new. Weaker
        outlines, down to the finder's own threshold, can only confirm an object
        already followed.
    still: for cameras that do not move. A camera whose picture has not changed looks
        every 30th frame only, and its objects are not followed on frames that did not
        change: their masks stay as they were. It changes nothing on a camera where
        something always moves.
    """

    def __init__(self, engine, prompts, *, finder=None, look_every: int = 3, hold: int = 300, start: float = 0.7,
                 still: bool = False):
        if look_every < 1 or hold < 0:
            raise ValueError("look_every must be at least 1 and hold at least 0")
        if engine.gpu_masks:
            raise ValueError("Tracker needs an engine that returns masks on the CPU (gpu_masks=False)")
        self.engine, self.prompts = engine, prompts
        self.finder = finder if finder is not None else Finder(
            device=engine.device, compile="default" if engine.compiled else None)
        self.look_every, self.hold, self.start, self.still = look_every, hold, start, still
        self._misses = max(1, math.ceil(_MISSED * (_TRIAL // look_every)))  # looks, of a trial's
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
        looking = self._due(list(frames), thumbs)
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
                    if cam.missed[t] >= self._misses:  # a new object the looks keep missing was a false find
                        self._forget(c, t)
            for t in [t for t, n in cam.unseen.items() if n > self.hold]:
                self._forget(c, t)
            if self.still:
                cam.looked = thumbs[c]
            if cam.since_look is None:  # first look: set this camera's later looks apart from the others'
                cam.since_look = self._joined % self.look_every
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

    def _due(self, cams, thumbs):
        """Cameras that look this step. A camera's first frame is always looked at: what is
        on it should not wait its turn. After that the most overdue go first, about
        cameras / look_every per step, so looks do not all land at once."""
        new = [c for c in cams if self.cameras[c].since_look is None]
        due = [c for c in cams if (self.cameras[c].since_look or 0) >= self.look_every
               and (not self.still or self._changed(self.cameras[c], thumbs[c]))]
        due.sort(key=lambda c: -self.cameras[c].since_look)
        return new + due[:max(1, math.ceil(len(self.cameras) / self.look_every))]
