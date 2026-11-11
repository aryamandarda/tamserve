"""The stock predictor on one video: the reference the engine is checked against."""
from __future__ import annotations

import contextlib
import time

import torch

Masks = dict[int, dict[int, torch.Tensor]]  # frame_idx -> obj_id -> bool mask


def precision(device: str, bf16: bool = True):
    """bf16 autocast is how EdgeTAM normally runs on GPU; fp32 everywhere else."""
    if bf16 and device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def clock(device) -> float:
    """Wall time after the GPU has finished queued work."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def track(predictor, frames_dir: str, seeds: dict[int, torch.Tensor], *,
          bf16: bool = True, stats: dict | None = None) -> Masks:
    """Track `seeds` (obj_id -> bool mask, Box or Clicks on frame 0) through every frame in `frames_dir`."""
    from tamserve.engine import Box, Clicks  # the engine imports this module

    out: Masks = {}
    with torch.inference_mode(), precision(predictor.device.type, bf16):
        state = predictor.init_state(frames_dir)
        for obj_id, seed in seeds.items():
            if isinstance(seed, Box):
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=list(seed))
            elif isinstance(seed, Clicks):
                labels = seed.labels if seed.labels is not None else [1] * len(seed.xy)
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, points=seed.xy, labels=labels)
            else:
                predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=seed)
        t = clock(predictor.device)
        for f, ids, logits in predictor.propagate_in_video(state):
            out[f] = {int(i): (logits[j, 0] > 0).cpu() for j, i in enumerate(ids)}
        if stats is not None:
            stats["propagate_s"] = clock(predictor.device) - t
    return out
