"""The pinned, patched EdgeTAM loads and tracks two objects, and its cached code is
only imported from a folder other users cannot write to."""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from tamserve import models
from tamserve.models import build_video_predictor
from tamserve.reference import track


def test_two_objects_propagate(tmp_path):
    # Two objects is the case tamserve's perceiver fix (models.py) fixes;
    # unpatched upstream crashes here.
    rng = np.random.default_rng(0)
    for i in range(3):
        frame = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
        Image.fromarray(frame).save(tmp_path / f"{i}.jpg")
    seeds = {}
    for obj_id, (y, x) in enumerate([(8, 8), (32, 48)], start=1):
        seeds[obj_id] = torch.zeros(64, 96, dtype=torch.bool)
        seeds[obj_id][y:y + 16, x:x + 24] = True

    out = track(build_video_predictor(device="cpu"), str(tmp_path), seeds)

    assert sorted(out) == [0, 1, 2]
    assert all(sorted(objs) == [1, 2] for objs in out.values())
    assert out[2][1].shape == (64, 96)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="needs POSIX permissions")
def test_code_others_can_write_is_not_imported(tmp_path, monkeypatch):
    # A cache folder any user can write to could hold anyone's code.
    perceiver = tmp_path / models.PACKAGE / "modeling" / "perceiver.py"
    perceiver.parent.mkdir(parents=True)
    perceiver.write_text(" ".join(new for _, new in models._PERCEIVER_FIX))
    monkeypatch.setattr(models, "EDGETAM_ROOT", tmp_path)
    assert models.edgetam_root() == tmp_path
    tmp_path.chmod(0o777)
    with pytest.raises(RuntimeError, match="another user"):
        models.edgetam_root()
