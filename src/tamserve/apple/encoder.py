"""EdgeTAM's image encoder on the Neural Engine.

The encoder is the same network with the same weights, converted once to Core ML,
Apple's format, which is the only way onto the Neural Engine (convert.py). It is
downloaded, checked against a pinned SHA-256 and compiled for this Mac on first use.

Core ML is called through PyObjC, Apple's bridge to Python, which lets go of Python's
lock while Apple's code runs. So the encoder works in a thread of this process while
the GPU follows objects in another: measured together on an M4 Max, the encoder keeps
87% of its speed alone and the GPU 98%.
"""
from __future__ import annotations

import shutil
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from tamserve.models import CACHE, _download

SIZE = 1024
# The three feature maps the encoder gives for one frame, as (channels, height, width).
MAPS = ((32, 256, 256), (64, 128, 128), (256, 64, 64))
_RELEASE = "https://github.com/aryamandarda/tamserve/releases/download/apple-encoders-1/"
# model -> (where its converted encoder is, the SHA-256 of that file)
ENCODERS = {
    "edgetam": (_RELEASE + "edgetam-encoder.mlpackage.zip",
                "0a7c62f0b772bc5f1b41e3618b52458a9c529959adcda91c5975a50f5237b753"),
}


def compiled(model: str) -> Path:
    """The model's encoder, compiled for this Mac: from the cache, or downloaded and compiled now."""
    import CoreML
    from Foundation import NSURL

    url, sha256 = ENCODERS[model]
    root = CACHE / "apple"
    done = root / f"{model}-encoder-{sha256[:12]}.mlmodelc"
    if done.exists():
        return done
    package = root / f"{model}-encoder-{sha256[:12]}.mlpackage"
    if not package.exists():
        archive = _download(url, root / f"{model}-encoder-{sha256[:12]}.zip", sha256)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root / f"{package.name}.unpacking")
        (root / f"{package.name}.unpacking" / "edgetam-encoder.mlpackage").replace(package)
        shutil.rmtree(root / f"{package.name}.unpacking", ignore_errors=True)
        archive.unlink()
    made, error = CoreML.MLModel.compileModelAtURL_error_(NSURL.fileURLWithPath_(str(package)), None)
    if made is None:
        raise RuntimeError(f"Core ML could not compile the encoder: {error}")
    shutil.move(made.path(), done)
    return done


def _resized(frame: np.ndarray) -> np.ndarray:
    """A frame at the encoder's size, resized as stock resizes it (PIL, bicubic): the same pixels."""
    return np.asarray(Image.fromarray(frame).resize((SIZE, SIZE)))


class NeuralEncoder:
    """Frames in, feature maps out, on the Neural Engine. submit() returns at once."""

    def __init__(self, model: str):
        import CoreML
        from Foundation import NSURL

        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
        self._model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            NSURL.fileURLWithPath_(str(compiled(model))), config, None)
        if self._model is None:
            raise RuntimeError(f"Core ML could not load the encoder: {error}")
        # One input, written afresh for every frame: only one frame is on the Neural Engine at a time.
        self._input, _ = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
            [1, 3, SIZE, SIZE], CoreML.MLMultiArrayDataTypeFloat32, None)
        self._pixels = np.frombuffer(self._input.dataPointer().as_buffer(3 * SIZE * SIZE * 4),
                                     dtype=np.float32).reshape(3, SIZE, SIZE)
        self._feed, _ = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {"image": CoreML.MLFeatureValue.featureValueWithMultiArray_(self._input)}, None)
        self._half = CoreML.MLMultiArrayDataTypeFloat16
        # Resizing takes several threads' worth of a busy step's frames; the Neural Engine takes one at a time.
        self._resizing = ThreadPoolExecutor(4, thread_name_prefix="tamserve-resize")
        self._encoding = ThreadPoolExecutor(1, thread_name_prefix="tamserve-neural-engine")

    def submit(self, frame: np.ndarray) -> Future:
        """Start encoding an H x W x 3 uint8 frame. The future gives its three feature maps,
        each a 1 x channels x height x width float16 array."""
        return self._encoding.submit(self._encode, self._resizing.submit(_resized, frame))

    def _encode(self, resized: Future) -> list:
        import objc

        self._pixels[:] = resized.result().transpose(2, 0, 1)
        with objc.autorelease_pool():  # threads other than the main one have no pool of their own
            out, error = self._model.predictionFromFeatures_error_(self._feed, None)
            if out is None:
                raise RuntimeError(f"the encoder failed on the Neural Engine: {error}")
            maps = []
            for i, shape in enumerate(MAPS):
                got = out.featureValueForName_(f"f{i}").multiArrayValue()
                packed = [int(np.prod(shape[k + 1:])) for k in range(len(shape))]
                if got.dataType() != self._half or [int(s) for s in got.strides()][1:] != packed:
                    raise RuntimeError("the encoder's output is not laid out as expected")
                maps.append(np.frombuffer(got.dataPointer().as_buffer(int(np.prod(shape)) * 2),
                                          dtype=np.float16).reshape(1, *shape).copy())
            return maps

    def close(self) -> None:
        self._resizing.shutdown(wait=False, cancel_futures=True)
        self._encoding.shutdown(wait=False, cancel_futures=True)
