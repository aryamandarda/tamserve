"""How far one run's masks are from a reference run's, per object."""
from __future__ import annotations

import torch

from tamserve.reference import Masks


def only(masks: Masks, obj: int) -> Masks:
    """One object's masks out of a multi-object run."""
    return {f: {obj: m[obj]} for f, m in masks.items() if obj in m}


def compare(ref: Masks, cand: Masks) -> dict:
    """Mask IoU per object per frame, plus objects lost or invented.

    lost: the reference has the object on a frame and the candidate does not.
    phantom: the reference has nothing there and the candidate does.
    missing / extra: (frame, object) keys on one side only, whatever their masks.
    Frames where both are empty agree and carry no IoU, so check object_frames.
    """
    ref_keys = {(f, o) for f, objs in ref.items() for o in objs}
    cand_keys = {(f, o) for f, objs in cand.items() for o in objs}
    ious, lost, phantom = [], 0, 0
    for f, objs in ref.items():
        for o, r in objs.items():
            c = cand.get(f, {}).get(o)
            if c is None:
                c = torch.zeros_like(r)
            r_any, c_any = bool(r.any()), bool(c.any())
            lost += int(r_any and not c_any)
            phantom += int(c_any and not r_any)
            if r_any or c_any:
                ious.append(float((r & c).sum() / (r | c).sum()))
    for f, o in cand_keys - ref_keys:
        phantom += int(bool(cand[f][o].any()))
    t = torch.tensor(ious) if ious else torch.ones(1)
    return {
        "missing": len(ref_keys - cand_keys),
        "extra": len(cand_keys - ref_keys),
        "object_frames": len(ious),
        "mean_iou": float(t.mean()),
        "p1_iou": float(t.quantile(0.01)),
        "min_iou": float(t.min()),
        "lost": lost,
        "phantom": phantom,
    }
