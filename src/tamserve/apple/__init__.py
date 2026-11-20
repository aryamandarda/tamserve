"""tamserve on an Apple-silicon Mac.

A Mac's GPU is full with one camera, so sharing it between cameras adds nothing. What a
Mac has that goes unused is the Neural Engine: a second processor, built for neural
networks, that works at the same time as the GPU. Here the image encoder runs on it and
the GPU keeps the rest, following each object.

Nothing in this folder is imported on other machines. `Engine(predictor)` picks it by
itself when the predictor is on "mps"; `Engine(predictor, neural_engine=False)` does not.
"""
from __future__ import annotations

import logging
import platform
import sys

from tamserve.apple.encoder import ENCODERS, compiled
from tamserve.apple.engine import AppleEngine

__all__ = ["AppleEngine", "usable"]

_log = logging.getLogger("tamserve")
_said: set = set()


def usable(predictor) -> bool:
    """Whether this predictor's image encoder can run on the Neural Engine here. Says why not, once."""
    why = _why_not(predictor)
    if why and why not in _said:
        _said.add(why)
        _log.info("the Neural Engine is not used: %s. Everything runs on the GPU.", why)
    return why is None


def _why_not(predictor) -> str | None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return "this is not an Apple-silicon Mac"
    if int(platform.mac_ver()[0].split(".")[0] or 0) < 15:
        return "it needs macOS 15 or newer"
    model = getattr(predictor, "tamserve_model", None)
    if model not in ENCODERS:
        return f"there is no encoder for {model or 'this model'} on it yet, only for {', '.join(ENCODERS)}"
    try:
        import CoreML  # noqa: F401
    except ImportError:
        return "pyobjc-framework-CoreML is not installed (pip install pyobjc-framework-CoreML)"
    try:
        compiled(model)  # downloads it the first time
    except (OSError, RuntimeError) as e:
        return f"its encoder could not be fetched ({e})"
    return None
