"""Messages between a Router and a worker on another machine, as plain data.

Nothing that arrives is ever run: a message is a JSON description plus the raw bytes
of any arrays in it, and only the types below can be rebuilt from it. Before any
message, each side proves that it knows the shared token, without sending it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
from typing import NamedTuple

import numpy as np
import torch

from tamserve.engine import Box, Clicks

PROTOCOL = 1  # raised whenever the messages change: a router and a worker must agree on it
TOKEN_MIN = 32
_HEADER_MAX = 1 << 20    # bytes of JSON in one message
_ARRAY_MAX = 64 << 20    # bytes in one array: the mask of a 4K frame is 8 MB
_MESSAGE_MAX = 256 << 20  # bytes in one message, arrays included
_ARRAYS_MAX = 1024       # arrays in one message
_DTYPES = {"bool", "uint8", "int32", "int64", "float32", "float64"}


class Mask(NamedTuple):
    """One object's mask, stored as runs of equal pixels, row by row: `ends` holds
    where each run but the last ends, `first` is the first run's value. A few KB for
    a typical object, where the full mask of a 1080p frame is 2 MB."""
    shape: tuple        # (height, width)
    first: bool
    ends: np.ndarray
    box: tuple | None   # (left, top, right, bottom) around the object, right and bottom exclusive; None if empty
    area: int           # pixels on the object

    def array(self) -> np.ndarray:
        """The mask as an H x W bool array, True on the object."""
        h, w = self.shape
        bounds = np.concatenate(([0], self.ends, [h * w]))
        values = np.zeros(len(bounds) - 1, dtype=bool)
        values[0 if self.first else 1::2] = True
        return np.repeat(values, np.diff(bounds)).reshape(h, w)


class Update(NamedTuple):
    """One frame of one camera: the masks of the objects tracked on it, by name. An
    object started on an earlier frame also sends its masks for the frames between,
    so one frame can arrive in more than one Update."""
    camera: object
    frame: int | None     # the frame's number on its camera, counted from 0
    time: float | None    # when the frame was decoded, as time.time() gives it on the router's machine
    masks: dict
    error: str | None = None
    ended: bool = False   # the camera's stream has ended and all of it has been tracked
    status: str | None = None  # a change in how the camera is doing; error then says why (see Router.results)


_TYPES = {"Box": Box, "Clicks": Clicks, "Mask": Mask, "Update": Update}
_NAMES = {cls: name for name, cls in _TYPES.items()}


def send(conn, message) -> None:
    """Send `message`: plain values, lists, tuples, dicts, arrays and the types above."""
    blobs: list = []
    body = _to(message, blobs)
    conn.send_bytes(json.dumps({"n": len(blobs), "m": body}).encode())
    for blob in blobs:
        conn.send_bytes(blob)


def recv(conn, timeout: float | None = None):
    """The next message, rebuilt. ValueError if it is malformed, TimeoutError if none comes."""
    if timeout is not None and not conn.poll(timeout):
        raise TimeoutError("no message arrived in time")
    try:
        raw = conn.recv_bytes(_HEADER_MAX)
        header = json.loads(raw)
        n, size, blobs = header["n"], len(raw), []
        if not isinstance(n, int) or not 0 <= n <= _ARRAYS_MAX:
            raise ValueError
        for _ in range(n):
            blobs.append(conn.recv_bytes(_ARRAY_MAX))
            size += len(blobs[-1])
            if size > _MESSAGE_MAX:
                raise ValueError
        return _from(header["m"], blobs)
    except (KeyError, TypeError, ValueError, RecursionError, UnicodeDecodeError) as e:
        raise ValueError("a malformed message") from e


def _to(x, blobs):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (np.bool_, np.integer, np.floating)):
        return x.item()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.dtype.name not in _DTYPES:
            raise TypeError(f"arrays of {x.dtype} cannot be sent to another machine")
        blobs.append(np.ascontiguousarray(x).tobytes())
        return {"T": "array", "i": len(blobs) - 1, "dtype": x.dtype.name, "shape": list(x.shape)}
    if isinstance(x, tuple):
        return {"T": _NAMES.get(type(x), "tuple"), "v": [_to(v, blobs) for v in x]}
    if isinstance(x, list):
        return [_to(v, blobs) for v in x]
    if isinstance(x, dict):
        return {"T": "dict", "v": [[_to(k, blobs), _to(v, blobs)] for k, v in x.items()]}
    raise TypeError(f"a {type(x).__name__} cannot be sent to another machine")


def _from(x, blobs):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, list):
        return [_from(v, blobs) for v in x]
    kind = x["T"]
    if kind == "array":
        i, dtype, shape = x["i"], x["dtype"], x["shape"]
        if dtype not in _DTYPES or not all(isinstance(s, int) and s >= 0 for s in shape):
            raise ValueError
        a = np.frombuffer(blobs[i], dtype=dtype)
        if a.size != math.prod(shape):
            raise ValueError
        return a.reshape(shape).copy()
    values = [_from(v, blobs) for v in x["v"]]
    if kind == "dict":
        return {k: v for k, v in values}
    if kind == "tuple":
        return tuple(values)
    return _TYPES[kind](*values)


def check_token(token: str | bytes) -> bytes:
    """The token as bytes, if it is long enough to be hard to guess from a recorded handshake."""
    token = (token.encode() if isinstance(token, str) else token).strip()
    if len(token) < TOKEN_MIN:
        raise ValueError(f"the token must be at least {TOKEN_MIN} characters; make one with "
                         'python -c "import secrets; print(secrets.token_hex(32))"')
    return token


def _proof(token: bytes, role: bytes, challenge: bytes) -> bytes:
    return hmac.new(token, role + challenge, hashlib.sha256).digest()


def _exactly(conn, size, timeout):
    if not conn.poll(timeout):
        raise TimeoutError("the other side did not answer")
    data = conn.recv_bytes(size)
    if len(data) != size:
        raise PermissionError("the other side is not speaking tamserve's protocol")
    return data


def prove_router(conn, token: bytes, timeout: float = 30) -> None:
    """The router's half of the handshake: show the worker it knows the token, and
    check that the worker knows it too. Camera addresses, which can hold passwords,
    are only sent once this has passed."""
    try:
        theirs = _exactly(conn, 32, timeout)
        mine = secrets.token_bytes(32)
        conn.send_bytes(_proof(token, b"router", theirs) + mine)
        answer = _exactly(conn, 32, timeout)
    except EOFError:
        raise PermissionError("the worker closed the connection: the token is not the one it was given") from None
    if not hmac.compare_digest(answer, _proof(token, b"worker", mine)):
        raise PermissionError("the worker does not know the token")


def prove_worker(conn, token: bytes, timeout: float = 5) -> None:
    """The worker's half of the handshake."""
    mine = secrets.token_bytes(32)
    conn.send_bytes(mine)
    reply = _exactly(conn, 64, timeout)
    if not hmac.compare_digest(reply[:32], _proof(token, b"router", mine)):
        raise PermissionError("the router does not know the token")
    conn.send_bytes(_proof(token, b"worker", reply[32:]))
