"""DAVIS 2017 sequences for the benchmarks: frames, first-frame seeds and per-frame ground truth."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tamserve.reference import Masks


def names(root: Path, split: str = "val") -> list[str]:
    return (root / "ImageSets/2017" / f"{split}.txt").read_text().split()


def frames_dir(root: Path, seq: str) -> Path:
    return root / "JPEGImages/480p" / seq


def num_frames(root: Path, seq: str) -> int:
    return len(list(frames_dir(root, seq).glob("*.jpg")))


def seeds(root: Path, seq: str) -> dict[int, torch.Tensor]:
    """Every annotated object's mask on frame 0."""
    ann = np.array(Image.open(root / "Annotations/480p" / seq / "00000.png"))
    return {int(k): torch.from_numpy(ann == k) for k in np.unique(ann) if k not in (0, 255)}


def ground_truth(root: Path, seq: str, ids, frames: int | None = None) -> Masks:
    out: Masks = {}
    for p in sorted((root / "Annotations/480p" / seq).glob("*.png")):
        f = int(p.stem)
        if frames is None or f < frames:
            ann = np.array(Image.open(p))
            out[f] = {k: torch.from_numpy(ann == k) for k in ids}
    return out
