"""Continuous batching for SAM 2-family trackers (EdgeTAM, SAM 2.1). Cameras and
the objects followed on them come and go independently, and each step is one
batched pass over every object on every camera with a frame waiting.

An object keeps only the memory the model reads: its seed frame, the last 6
frames' memory and the last 15 frames' object pointers, so it can be followed
indefinitely. The model code is stock. The engine calls the same encoder, memory
attention, mask decoder and memory encoder as SAM2VideoPredictor, and builds each
object's memory exactly as _prepare_memory_conditioned_features does. Objects
whose memories have the same shape share one memory-attention call; once they
are 16 frames in, that is all of them.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tamserve.reference import precision

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
# Image-encoder calls: up to _ENCODE_WHOLE frames go in one, more are cut into pieces
# of _ENCODE_BATCH. EdgeTAM's encoder does the least GPU work per frame at 4 frames
# (5% faster at 64 cameras on an RTX 4090; SAM 2.1's is within 3% either way), and its
# activations stay small. Up to 8 frames a second call costs more to launch than it saves.
_ENCODE_WHOLE, _ENCODE_BATCH = 8, 4
# Objects per pass through memory attention, the decoder and the memory encoder, whose
# buffers grow with the object count (SAM 2.1 attends over about 29,000 memory tokens
# per object). 16 is as fast as 64 on an RTX 4090, and leaves five sizes to compile
# instead of seven.
_OBJECT_BATCH = 16
# Rows up to which a compiled call is replayed as a CUDA graph: one launch instead of
# hundreds. With one or two rows launching is the limit (220 against 130 frames/s at
# one camera on an RTX 4090); with more the GPU is, and replaying costs it 4 to 8%.
_GRAPH_ROWS = 2
_GRAPH_MODES = ("reduce-overhead", "max-autotune")  # the torch.compile modes that use CUDA graphs


class Box(NamedTuple):
    """A seed: a box around the object, in the frame's own pixels."""
    x0: float
    y0: float
    x1: float
    y1: float


class Clicks(NamedTuple):
    """A seed: points in the frame's own pixels, (x, y) each. labels: 1 = on the
    object, 0 = not on it; all 1 when left out."""
    xy: list
    labels: list | None = None


def prepare(image, size: int = 1024, device=None, *, resize_on_device: bool = False):
    """A frame resized exactly as stock load_video_frames resizes it, kept as
    uint8, optionally uploaded to `device` so the step does not pay for it.
    `image` is a PIL image, an H x W x 3 uint8 RGB array (what video decoders
    give; OpenCV's are BGR and need converting), or the path of an image file.
    resize_on_device uploads the full frame and resizes it there instead, far less
    CPU work: as PIL does, bicubic, width then height, rounding in between, so
    pixels match stock's or differ by 1.
    Returns (3 x size x size tensor, (height, width))."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        with Image.open(image) as f:
            return prepare(f.convert("RGB"), size, device, resize_on_device=resize_on_device)
    img = image
    if resize_on_device:
        return prepare_batch([img], size, device)[0]
    pixels = torch.from_numpy(np.array(img.convert("RGB").resize((size, size))))
    pixels = pixels.permute(2, 0, 1).contiguous()
    return (pixels if device is None else pixels.to(device)), (img.height, img.width)


def prepare_batch(images, size: int = 1024, device=None):
    """prepare(resize_on_device=True) for many PIL frames: frames of one size are
    uploaded and resized together, 16 at a time, instead of one call per frame."""
    out, groups = [None] * len(images), {}
    for i, im in enumerate(images):
        groups.setdefault((im.height, im.width), []).append(i)
    for (h, w), idx in groups.items():
        for k in range(0, len(idx), 16):
            chunk = idx[k:k + 16]
            x = torch.from_numpy(np.stack([np.asarray(images[i].convert("RGB")) for i in chunk]))
            x = x.to(device).permute(0, 3, 1, 2).float()
            for hh in (h, size):
                x = F.interpolate(x, size=(hh, size), mode="bicubic", antialias=True, align_corners=False)
                x = x.round_().clamp_(0, 255)
            x = x.to(torch.uint8)
            for j, i in enumerate(chunk):
                out[i] = (x[j], (h, w))
    return out


def _cat_backbone(parts):
    """forward_image outputs for consecutive chunks of frames, joined along the batch."""
    cat = lambda xs: torch.cat(xs) if torch.is_tensor(xs[0]) else [cat(list(x)) for x in zip(*xs)]
    return {k: cat([p[k] for p in parts]) for k in parts[0]}


def _each(fn, x):
    """fn applied to every tensor inside nested lists, tuples and dicts."""
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, dict):
        return {k: _each(fn, v) for k, v in x.items()}
    return type(x)(_each(fn, v) for v in x) if isinstance(x, (list, tuple)) else x


def _version(t):
    """Counts in-place writes to `t`; None for inference tensors, which have no count."""
    return None if t.is_inference() else t._version


def _tokens(x):
    """Memory as (objects, tokens, channels). EdgeTAM's perceiver already gives
    tokens; SAM 2.1 gives (objects, channels, H, W) maps, flattened as stock does."""
    return x.flatten(2).permute(0, 2, 1) if x.ndim == 4 else x


@dataclass
class _Track:
    """One object's memory, one row in each tensor."""
    cond_mem: torch.Tensor  # its seed frame
    cond_ptr: torch.Tensor
    mem: deque              # recent frames, oldest first
    ptr: deque
    t: int = 0              # frames followed since the seed frame


class Engine:
    """Follow objects on many cameras at once.

    step() takes one frame per camera and returns every followed object's mask
    on it. Objects start through `seeds`, on any step, and stop through drop();
    the others on their camera carry on undisturbed.

    exact: seed each object on its own and follow a step's objects in one pass,
        as stock does, so one camera matches stock bit for bit. Otherwise objects
        that start together are seeded together and objects go 16 to a pass,
        which differs from stock only by GPU rounding.
    compile: compile the modules that follow objects with torch.compile: True for
        the fastest setup measured (tuned kernels, plus CUDA graphs for calls of one
        or two rows), or one torch.compile mode's name for every call ("default" is
        ready soonest). Every batch is then cut into power-of-two pieces, so each
        module sees only a few sizes, and all of them are compiled here so that no
        later step stalls: about 20 minutes the first time on a machine for True,
        under 3 for "default", and 1 to 2 once PyTorch has them cached on disk.
        Seeding stays uncompiled.
    gpu_masks: return masks on the GPU instead of copying them to the CPU.
    """

    def __init__(self, predictor, *, bf16: bool = True, exact: bool = False, compile: bool | str = False,
                 gpu_masks: bool = False):
        m = predictor
        # The memory layout below mirrors stock for this configuration only.
        if not (m.num_maskmem > 0 and m.use_obj_ptrs_in_encoder and m.memory_temporal_stride_for_eval == 1
                and m.use_mask_input_as_output_without_sam and not m.non_overlap_masks
                and not m.non_overlap_masks_for_mem_enc and not m.training):
            raise ValueError("unsupported predictor configuration; use build_video_predictor")
        self.model = m
        self.bf16 = bf16
        self.exact = exact
        self.gpu_masks = gpu_masks
        self.device = m.device
        self.tracks: dict = {}  # (camera id, object id) -> _Track, in the order objects started
        self._pos = None  # memory positional encoding: the same for every frame and object
        self._last = None  # the previous step's frames and their features
        # Stock divides by 255 in float64 on the CPU, then normalises in float32 on
        # the device. A 256-entry table reproduces the first step exactly; CUDA
        # divides by a scalar through its reciprocal, which rounds differently.
        self._unit = (torch.arange(256, dtype=torch.float64) / 255.0).float().to(self.device)
        self._mean = torch.tensor(_MEAN, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(_STD, device=self.device).view(3, 1, 1)
        self._blank = None  # the prompt encoder's output when there is no prompt
        # module -> the forward to follow objects with, in place of stock's
        self._fast: dict = {m.sam_prompt_encoder: self._no_prompt,
                            m.image_encoder.neck.position_encoding: self._one_pos}
        # ...and the forward each had on the instance (usually none), to put back.
        self._was: dict = {mod: vars(mod).get("forward") for mod in self._fast}
        self._attention = None  # stock's attention module, once compiled
        self.compiled = bool(compile)
        if compile:
            self._compile(compile)

    def _compile(self, mode):
        from tamserve_sam2.modeling.sam import transformer

        self._warm(2)  # uncompiled first: fills position encodings the modules cache
        # Stock pins its attention kernel through a context manager the compiler cannot
        # trace, which cuts the mask decoder into 18 graphs and memory attention into 8.
        # With stock's own fallback switch, PyTorch picks the kernel and each is one graph.
        # _use() sets the switch while the compiled modules run and puts it back after.
        self._attention, self._pinned = transformer, transformer.ALLOW_ALL_KERNELS
        torch._dynamo.config.cache_size_limit = 64
        # The mode for calls of up to _GRAPH_ROWS rows, and for larger ones.
        modes = ("max-autotune", "max-autotune-no-cudagraphs") if mode is True else (mode, mode)

        def wrap(fn, dynamic, rows):
            few = torch.compile(fn, mode=modes[0], dynamic=dynamic)
            many = few if modes[0] == modes[1] else torch.compile(fn, mode=modes[1], dynamic=dynamic)

            def run(*a, **k):
                which = rows(a, k) > _GRAPH_ROWS
                # Compact copies of the inputs: a view's layout depends on how the batch
                # was cut, and the compiler would compile again for each layout.
                out = (few, many)[which](*_each(torch.Tensor.contiguous, a), **_each(torch.Tensor.contiguous, k))
                # A CUDA graph reuses its output buffers on the next call.
                return _each(torch.Tensor.clone, out) if modes[which] in _GRAPH_MODES else out
            return run

        # How many rows a call has: the first argument's, unless the module is called by keyword.
        rows = {"memory_attention": lambda a, k: k["memory"].shape[1],
                "sam_mask_decoder": lambda a, k: k["image_embeddings"].shape[0]}
        for name in ("image_encoder", "spatial_perceiver", "memory_attention", "sam_mask_decoder", "memory_encoder"):
            mod = getattr(self.model, name, None)
            if mod is not None:
                self._was[mod] = vars(mod).get("forward")
                # Memory length changes while an object is young; batch sizes are the pieces'.
                self._fast[mod] = wrap(mod.forward, name == "memory_attention",
                                       rows.get(name, lambda a, k: a[0].shape[0]))
        k = 1
        while k <= _OBJECT_BATCH:
            self._warm(k)
            k *= 2
        torch.cuda.empty_cache()  # hand back what following 64 objects at once reserved

    def _warm(self, k):
        """Follow one object on each of k cameras of noise until their memories are full
        (so every memory length is seen at this size), then forget them."""
        n = self.model.image_size
        seed = torch.zeros(n, n, dtype=torch.bool)
        seed[n // 4:n // 2, n // 4:n // 2] = True
        frames = lambda: {i: (torch.randint(0, 256, (3, n, n), dtype=torch.uint8, device=self.device), (n, n))
                          for i in range(k)}
        keep, self.tracks = self.tracks, {}
        self.step(frames(), {i: {0: seed} for i in range(k)})
        for _ in range(self.model.max_obj_ptrs_in_encoder):
            self.step(frames())
        self.tracks, self._last = keep, None

    def _use(self, fast):
        """Point the modules at the forward objects are followed with, or back at what
        they had: the predictor is shared with stock."""
        for mod, fn in self._fast.items():
            if fast:
                mod.forward = fn
            elif self._was[mod] is None:
                vars(mod).pop("forward", None)
            else:
                mod.forward = self._was[mod]
        if self._attention is not None:
            self._attention.ALLOW_ALL_KERNELS = fast or self._pinned

    def _no_prompt(self, points, boxes, masks):
        """The prompt encoder's output for objects followed without a prompt: the same
        for each, so worked out once. Stock redoes it on every frame with masked
        writes, and each one makes the CPU wait for the GPU to finish its queue."""
        assert boxes is None and masks is None, "only objects followed without a prompt come through here"
        if self._blank is None:
            enc = self.model.sam_prompt_encoder
            self._blank = type(enc).forward(enc, (points[0][:1], points[1][:1]), None, None)
        sparse, dense = self._blank
        n = points[0].shape[0]
        return sparse.expand(n, -1, -1), dense.expand(n, -1, -1, -1)

    def _one_pos(self, x):
        """The image encoder's position encoding for one frame instead of one per
        frame. They are all the same, 88 MB a frame at 1024 x 1024, and one row is read."""
        enc = self.model.image_encoder.neck.position_encoding
        return type(enc).forward(enc, x[:1])

    def _index(self, values):
        """`values` as a tensor on the device. A plain copy to the GPU waits for
        everything queued on it; from pinned memory it does not."""
        idx = torch.tensor(values)
        if self.device.type != "cuda":
            return idx.to(self.device)
        return idx.pin_memory().to(self.device, non_blocking=True)

    def _pieces(self, n, cap):
        """Slices covering n rows, at most `cap` each; powers of two when compiled."""
        i = 0
        while i < n:
            size = min(cap, n - i)
            if self.compiled:
                size = 1 << (size.bit_length() - 1)
            yield slice(i, i + size)
            i += size

    def drop(self, cam, obj=None) -> None:
        """Stop following an object; with no object, every object on the camera."""
        for key in [k for k in self.tracks if k[0] == cam and obj in (None, k[1])]:
            del self.tracks[key]
        if not self.tracks:
            self._last = None  # free the cached features

    def step(self, frames: dict, seeds: dict | None = None) -> dict:
        """One batched pass: every object on every camera here advances one frame.

        frames: camera id -> prepare() output.
        seeds: camera id -> {object id: bool mask, Box or Clicks}: objects that
            start on this frame, or start again if they are already followed.
        Returns camera id -> {object id: bool mask at the frame's own size}.
        """
        seeds = seeds or {}
        new = {(c, o) for c in seeds for o in seeds[c]}
        return self._run(frames, seeds, [k for k in self.tracks if k[0] in frames and k not in new], frames, False)

    def add(self, frames: dict, seeds: dict) -> dict:
        """Start objects (or start them again) on frames step() has already been
        given, without advancing the others: for prompts decided after seeing a
        step's masks. Pass the same frame tensors and they are not encoded again.
        Returns the started objects' masks."""
        return self._run(frames, seeds, [], seeds, True)

    @torch.inference_mode()
    def _run(self, frames, seeds, following, cams, reuse):
        frameless = [c for c in seeds if c not in frames]
        if frameless:
            raise KeyError(f"seeds need the frame they apply to: {frameless}")
        starting = [(c, o) for c in seeds for o in seeds[c]]
        out = {c: {} for c in cams}
        live = {c for c, _ in starting + following}
        busy = [c for c in frames if c in live]
        if not busy:
            return out  # nothing to follow: the frames are not even read
        # Objects of one age side by side, so each memory-attention call is as large as it can be.
        following.sort(key=lambda k: (len(self.tracks[k].mem), len(self.tracks[k].ptr)))
        hw = {c: tuple(frames[c][1]) for c in busy}
        # Stock takes all of a video's objects in one pass, whatever their number.
        cap = len(starting) + len(following) if self.exact else _OBJECT_BATCH
        try:
            with precision(self.device.type, self.bf16):
                feats, pos, sizes, cols = self._encode([frames[c][0] for c in busy], reuse)
                col, low = dict(zip(busy, cols)), {}
                self._use(False)
                for i in range(0, len(starting), cap):
                    low.update(self._seed(starting[i:i + cap], col, feats, sizes, seeds, hw))
                self._use(True)
                for i in range(0, len(following), cap):
                    low.update(self._track(following[i:i + cap], col, feats, pos, sizes))
                return self._outputs(low, hw, out)
        finally:
            self._use(False)  # the predictor is shared: leave it as stock has it

    def _normalize(self, images):
        x = torch.stack(images).to(self.device, non_blocking=True)
        return (self._unit[x.int()] - self._mean) / self._std

    def _encode(self, images, reuse):
        """Features for `images` and each image's column in them. With `reuse`
        (add() after step()), frames the last call encoded are not encoded again;
        the version check catches a buffer torch saw being refilled since."""
        if reuse and self._last is not None:
            cols = [next((i for i, (t, v) in enumerate(self._last[0])
                          if t is im and v is not None and v == _version(im)), None)
                    for im in images]
            if None not in cols:
                return (*self._last[1:], cols)
        self._last = None  # free the previous step's features before making this one's
        self._use(True)
        cap = len(images) if len(images) <= _ENCODE_WHOLE else _ENCODE_BATCH
        parts = [self.model.forward_image(self._normalize(images[p])) for p in self._pieces(len(images), cap)]
        backbone = parts[0] if len(parts) == 1 else _cat_backbone(parts)
        _, feats, pos, sizes = self.model._prepare_backbone_features(backbone)
        # Position encodings depend only on the frame size, so every frame's are the
        # same (_one_pos): keep one copy of the level memory attention reads.
        pos = pos[-1][:, :1].clone()
        self._last = ([(im, _version(im)) for im in images], feats, pos, sizes)
        return feats, pos, sizes, list(range(len(images)))

    def _rows(self, xs, keys, col):
        """Each object's camera's column of (tokens, frames, channels) features,
        every object's tokens kept together as stock lays them out: the GPU rounds
        differently on other layouts."""
        idx = self._index([col[c] for c, _ in keys])
        return [x.permute(1, 0, 2)[idx].permute(1, 0, 2) for x in xs]

    @staticmethod
    def _high_res(feats, sizes):
        return [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(feats[:-1], sizes[:-1])]

    def _fill_holes(self, low):
        if self.model.fill_hole_area > 0:
            from tamserve_sam2.utils.misc import fill_holes_in_mask_scores
            return fill_holes_in_mask_scores(low, self.model.fill_hole_area)
        return low

    def _remember_pos(self, mem_pos):
        if self._pos is None:
            self._pos = _tokens(mem_pos[-1][0:1]).clone()

    def _outputs(self, low, hw, out):
        """Fill `out` with each object's mask at its camera's frame size. Objects on
        frames of one size are resized, and copied off the GPU, together."""
        groups: dict = {}
        for key in low:
            groups.setdefault(hw[key[0]], []).append(key)
        for size, keys in groups.items():
            for i in range(0, len(keys), _OBJECT_BATCH):
                run = keys[i:i + _OBJECT_BATCH]
                x = low[run[0]] if len(run) == 1 else torch.cat([low[k] for k in run])
                if tuple(x.shape[-2:]) != size:
                    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
                masks = x[:, 0] > 0 if self.gpu_masks else (x[:, 0] > 0).cpu()
                for j, (c, o) in enumerate(run):
                    out[c][o] = masks[j]
        return out

    def _prompt(self, seed, size):
        """A seed as stock's add_new_mask / add_new_points_or_box prepares it: a
        1 x 1 x n x n mask, or (point coordinates, point labels) at the model's size."""
        n = self.model.image_size
        if isinstance(seed, (Box, Clicks)):
            if isinstance(seed, Box):
                xy, labels = torch.tensor(seed, dtype=torch.float32).reshape(1, 2, 2), [2, 3]  # corners
            else:
                xy = torch.tensor(seed.xy, dtype=torch.float32).reshape(1, -1, 2)
                labels = seed.labels if seed.labels is not None else [1] * xy.shape[1]
            xy = xy / torch.tensor([size[1], size[0]]) * n
            return xy.to(self.device), torch.tensor(labels, dtype=torch.int32, device=self.device).reshape(1, -1)
        x = torch.as_tensor(seed)[None, None].float().to(self.device)
        if tuple(x.shape[-2:]) != (n, n):
            x = F.interpolate(x, size=(n, n), align_corners=False, mode="bilinear", antialias=True)
            x = (x >= 0.5).float()
        return x

    def _seed(self, keys, col, feats, sizes, seeds, hw):
        """Stock add_new_mask / add_new_points_or_box + the first memory encoding,
        for each starting object.

        Stock seeds each object on its own (batch 1), then encodes the seed memory
        of a video's objects together. GPU rounding depends on batch size, so with
        exact=True the engine keeps those batch sizes; otherwise objects prompted
        the same way go through each call together.
        """
        m = self.model
        n = m.image_size
        k = len(keys)
        vf = self._rows(feats, keys, col)
        hrf = self._high_res(vf, sizes)
        prompts = [self._prompt(seeds[c][o], hw[c]) for c, o in keys]

        # Masks together, and points by how many there are (a box is two).
        groups: dict = {}
        for i, p in enumerate(prompts):
            kind = i if self.exact else ("mask" if torch.is_tensor(p) else p[0].shape[1])
            groups.setdefault(kind, []).append(i)
        heads = [None] * k
        for idx in groups.values():
            run = idx == list(range(idx[0], idx[-1] + 1))  # a slice keeps stock's memory layout
            sel = slice(idx[0], idx[-1] + 1) if run else self._index(idx)
            high_res = [h[sel] for h in hrf]
            if torch.is_tensor(prompts[idx[0]]):
                pix = vf[-1][:, sel].permute(1, 2, 0).view(len(idx), m.hidden_dim, *sizes[-1])
                out = m._use_mask_as_output(pix, high_res, torch.cat([prompts[i] for i in idx]))
            else:
                pts = {"point_coords": torch.cat([prompts[i][0] for i in idx]),
                       "point_labels": torch.cat([prompts[i][1] for i in idx])}
                pix = (vf[-1][:, sel] + m.no_mem_embed).permute(1, 2, 0).view(len(idx), m.hidden_dim, *sizes[-1])
                out = m._forward_sam_heads(backbone_features=pix, point_inputs=pts, mask_inputs=None,
                                           high_res_features=high_res,
                                           multimask_output=m._use_multimask(True, pts))
            for j, i in enumerate(idx):
                heads[i] = (out[3][j:j + 1], out[5][j:j + 1], out[6][j:j + 1])
        low = self._fill_holes(torch.cat([h[0] for h in heads]))
        obj_ptr = torch.cat([h[1] for h in heads])
        score = torch.cat([h[2] for h in heads])
        high = F.interpolate(low, size=(n, n), mode="bilinear", align_corners=False)
        spans, i = [slice(0, k)], 0
        if self.exact:  # one camera's objects at a time
            spans = []
            while i < k:
                j = next((j for j in range(i, k) if keys[j][0] != keys[i][0]), k)
                spans.append(slice(i, j))
                i = j
        mem = []
        for p in spans:
            mm, mem_pos = m._encode_new_memory(
                current_vision_feats=[x[:, p] for x in vf], feat_sizes=sizes,
                pred_masks_high_res=high[p], object_score_logits=score[p], is_mask_from_pts=True)
            self._remember_pos(mem_pos)
            mem.append(mm)
        mem = _tokens(torch.cat(mem).to(torch.bfloat16))
        for i, key in enumerate(keys):
            # Stock consolidates seed pointers into a float32 buffer.
            self.tracks[key] = _Track(cond_mem=mem[i:i + 1], cond_ptr=obj_ptr[i:i + 1].float(),
                                      mem=deque(maxlen=m.num_maskmem - 1),
                                      ptr=deque(maxlen=m.max_obj_ptrs_in_encoder - 1))
        return {key: low[i:i + 1] for i, key in enumerate(keys)}

    def _track(self, keys, col, feats, pos, sizes):
        """Stock track_step for every followed object: memory attention per memory
        shape, then the mask decoder and the memory encoder over all of them."""
        m = self.model
        buckets: dict = {}
        for key in keys:
            st = self.tracks[key]
            buckets.setdefault((len(st.mem), len(st.ptr)), []).append(key)
        order = [key for b in sorted(buckets) for key in buckets[b]]
        vf, vp = self._rows(feats, order, col), pos.expand(-1, len(order), -1)

        pix, start = [], 0
        for b in sorted(buckets):
            group = buckets[b]
            pix.append(self._attend(group, vf[-1][:, start:start + len(group)],
                                    vp[:, start:start + len(group)], sizes[-1]))
            start += len(group)
        pix, hrf, parts = torch.cat(pix), self._high_res(vf, sizes), []
        for p in self._pieces(len(order), len(order)):
            _, _, _, low, high, obj_ptr, score = m._forward_sam_heads(
                backbone_features=pix[p], point_inputs=None, mask_inputs=None,
                high_res_features=[h[p] for h in hrf],
                multimask_output=m._use_multimask(False, None))
            mem, mem_pos = m._encode_new_memory(
                current_vision_feats=[x[:, p] for x in vf], feat_sizes=sizes, pred_masks_high_res=high,
                object_score_logits=score, is_mask_from_pts=False)
            parts.append((low, obj_ptr, mem))
        low, obj_ptr, mem = parts[0] if len(parts) == 1 else (torch.cat(x) for x in zip(*parts))
        mem = _tokens(mem.to(torch.bfloat16))
        low = self._fill_holes(low)
        for i, key in enumerate(order):
            st = self.tracks[key]
            st.t += 1
            st.mem.append(mem[i:i + 1])
            st.ptr.append(obj_ptr[i:i + 1])
        return {key: low[i:i + 1] for i, key in enumerate(order)}

    def _attend(self, group, curr, curr_pos, size):
        """_prepare_memory_conditioned_features for objects whose memories have the
        same shape. Pointer distances, which depend on an object's age, only matter
        when a model enables them; EdgeTAM does not (add_tpos_enc_to_obj_ptrs)."""
        from tamserve_sam2.modeling.sam2_utils import get_1d_sine_pe

        m = self.model
        sts = [self.tracks[key] for key in group]
        B, C = curr.shape[1], m.hidden_dim
        n_mem, n_ptr = len(sts[0].mem), len(sts[0].ptr)

        # Spatial memory: seed frame (t_pos 0), then recent frames oldest first;
        # frame u-k has t_pos num_maskmem-k.
        slots = [torch.cat([st.cond_mem for st in sts])]
        slots += [torch.cat([st.mem[j] for st in sts]) for j in range(n_mem)]
        t_pos = [0] + [m.num_maskmem - n_mem + j for j in range(n_mem)]
        base = self._pos.expand(B, -1, -1).permute(1, 0, 2)
        memory = [x.permute(1, 0, 2) for x in slots]
        memory_pos = [base + m.maskmem_tpos_enc[m.num_maskmem - tp - 1] for tp in t_pos]

        # Object pointers: seed frame first, then frames u-1, u-2, ... newest first.
        ptrs = [torch.cat([st.cond_ptr for st in sts])]
        ptrs += [torch.cat([st.ptr[-d] for st in sts]) for d in range(1, n_ptr + 1)]
        obj_ptrs = torch.stack(ptrs)
        if m.add_tpos_enc_to_obj_ptrs:
            # A camera has no last frame, so stock's min(num_frames, max) is the max.
            t_diff_max = m.max_obj_ptrs_in_encoder - 1
            tpos_dim = C if m.proj_tpos_enc_in_obj_ptrs else m.mem_dim
            # One projection per age, shared by the objects of that age, as stock
            # shares it across a video's objects: GPU matmul rounding depends on the row count.
            ages: dict = {}
            for st in sts:
                if st.t not in ages:
                    dist = self._index([st.t + 1] + list(range(1, n_ptr + 1)))
                    ages[st.t] = m.obj_ptr_tpos_proj(get_1d_sine_pe(dist / t_diff_max, dim=tpos_dim)).unsqueeze(1)
            obj_pos = torch.cat([ages[st.t] for st in sts], dim=1)
        else:
            obj_pos = obj_ptrs.new_zeros(len(ptrs), B, m.mem_dim)
        if m.mem_dim < C:
            obj_ptrs = obj_ptrs.reshape(-1, B, C // m.mem_dim, m.mem_dim)
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
            obj_pos = obj_pos.repeat_interleave(C // m.mem_dim, dim=0)

        memory, memory_pos = torch.cat(memory + [obj_ptrs]), torch.cat(memory_pos + [obj_pos])
        parts = [m.memory_attention(
            curr=[curr[:, p]], curr_pos=[curr_pos[:, p]], memory=memory[:, p], memory_pos=memory_pos[:, p],
            num_obj_ptr_tokens=obj_ptrs.shape[0], num_spatial_mem=len(slots)) for p in self._pieces(B, B)]
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return out.permute(1, 2, 0).view(B, C, *size)
