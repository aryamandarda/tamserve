"""Make the Core ML encoder that encoder.py downloads. Run once per model by whoever releases tamserve,
on a Mac, with coremltools installed:

    python -m tamserve.apple.convert edgetam edgetam-encoder.mlpackage.zip

The result takes a frame's pixels (0 to 255, already 1024 x 1024) and gives the three feature
maps the model's own forward_image gives, as 16-bit numbers: the Neural Engine works in 16-bit.
Upload the file where ENCODERS says it is and put its SHA-256 there.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from tamserve.apple.encoder import SIZE
from tamserve.engine import _MEAN, _STD
from tamserve.models import build_video_predictor


def convert(model: str, out: Path) -> str:
    """Write the model's encoder to `out` (a .zip holding edgetam-encoder.mlpackage); returns its SHA-256."""
    import coremltools as ct

    predictor = build_video_predictor("cpu", model)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = predictor
            self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

        def forward(self, image):
            return tuple(self.model.forward_image((image / 255.0 - self.mean) / self.std)["backbone_fpn"])

    example = torch.rand(1, 3, SIZE, SIZE) * 255
    with torch.no_grad():
        traced = torch.jit.trace(Encoder().eval(), example, strict=False)
    package = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=example.shape, dtype=np.float32)],
                         outputs=[ct.TensorType(name=f"f{i}", dtype=np.float16) for i in range(3)],
                         convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                         minimum_deployment_target=ct.target.macOS15)
    with tempfile.TemporaryDirectory() as tmp:
        package.save(str(Path(tmp) / "edgetam-encoder.mlpackage"))
        shutil.make_archive(str(out).removesuffix(".zip"), "zip", tmp)
    return hashlib.sha256(Path(out).read_bytes()).hexdigest()


if __name__ == "__main__":
    print(convert(sys.argv[1], Path(sys.argv[2])))
