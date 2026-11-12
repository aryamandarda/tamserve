import functools
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tamserve import Engine, Router, build_video_predictor
from tamserve.worker import encode
from test_engine import RED, YELLOW, _alone, _assert_tracked_like, _disc_at, _run, _video

# Two GPUs when the machine has them; otherwise both workers run on the CPU.
DEVICES = ["cuda:0", "cuda:1"] if torch.cuda.device_count() > 1 else ["cpu", "cpu"]


class Frames:
    """A folder of numbered JPEGs played once, as a camera would send them."""

    def __init__(self, folder):
        self.paths = iter(sorted(Path(folder).glob("*.jpg"), key=lambda p: int(p.stem)))

    def read(self):
        path = next(self.paths, None)
        return None if path is None else np.asarray(Image.open(path).convert("RGB"))


def _collect(router, until, got=None):
    """Updates merged as {camera: {frame: {object: bool tensor}}}, until `until(got)`."""
    got = {} if got is None else got
    for u in router.results(timeout=300):
        assert u.error is None, u.error
        for obj, mask in u.masks.items():
            got.setdefault(u.camera, {}).setdefault(u.frame, {})[obj] = torch.from_numpy(mask.array())
        if until(got):
            return got
    raise AssertionError(f"updates stopped coming: {({c: sorted(f) for c, f in got.items()})}")


def _count(got):
    return sum(len(objects) for frames in got.values() for objects in frames.values())


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_masks_travel_as_runs(device):
    rng = np.random.default_rng(0)
    blob = torch.zeros(6, 9, dtype=torch.bool)
    blob[2:5, 3:7] = True
    masks = {"empty": torch.zeros(6, 9, dtype=torch.bool), "full": torch.ones(6, 9, dtype=torch.bool),
             "noise": torch.from_numpy(rng.random((6, 9)) > 0.5), "blob": blob}
    out = encode({k: m.to(device) for k, m in masks.items()})
    for k, m in masks.items():
        assert np.array_equal(out[k].array(), m.numpy()) and out[k].area == int(m.sum())
    assert out["empty"].box is None and out["full"].box == (0, 0, 9, 6) and out["blob"].box == (3, 2, 7, 5)


def test_router_tracks_like_one_engine(tmp_path):
    # Three cameras over two workers: each must be tracked as one engine tracks it,
    # wherever it was placed, from the frame each prompt was made on.
    videos = {"a": _video(tmp_path / "a", 96, 128, 6, [RED, YELLOW], seed=0),
              "b": _video(tmp_path / "b", 80, 112, 5, [RED], seed=1),
              "c": _video(tmp_path / "c", 96, 128, 4, [YELLOW], seed=2)}
    want = dict(zip(videos, _run(Engine(build_video_predictor(DEVICES[0])), list(videos.values()), [0, 0, 0])))
    with Router(workers=DEVICES) as router:
        for name, (folder, seeds, _) in videos.items():
            router.add_camera(name, functools.partial(Frames, folder), skip=False)
            for obj, seed in seeds.items():
                router.start(name, obj, seed.numpy(), frame=0)
        assert set(router.home.values()) == {0, 1}  # spread over both workers
        with pytest.raises(KeyError):
            router.add_camera("a", functools.partial(Frames, videos["a"][0]))
        with pytest.raises(KeyError):
            router.start("nowhere", 1, videos["a"][1][1].numpy())
        got = _collect(router, lambda g: _count(g) == 6 * 2 + 5 + 4)
    for name, (_, seeds, frames) in videos.items():
        _assert_tracked_like(want[name], got[name], objects=len(seeds), frames=len(frames))


def test_a_prompt_on_a_frame_already_passed(tmp_path):
    # Yellow is pointed at on frame 2 after the worker has tracked frame 3 and beyond:
    # it must be followed from frame 2 on, as if it had started there.
    folder, seeds, frames = _video(tmp_path / "a", 96, 128, 6, [RED, YELLOW], seed=0)
    pred = build_video_predictor(DEVICES[0])
    with Router(workers=DEVICES[:1]) as router:
        router.add_camera("a", functools.partial(Frames, folder), skip=False)
        router.start("a", "red", seeds[1].numpy(), frame=0)
        got = _collect(router, lambda g: 3 in g.get("a", {}))
        router.start("a", "yellow", _disc_at(YELLOW, 2).numpy(), frame=2)
        got = _collect(router, lambda g: sum("yellow" in o for o in g["a"].values()) == 4, got)
        got = _collect(router, lambda g: all("red" in g["a"].get(f, {}) for f in range(6)), got)
    yellow = {f - 2: {1: got["a"][f]["yellow"]} for f in range(2, 6)}
    red = {f: {1: got["a"][f]["red"]} for f in range(6)}
    _assert_tracked_like(_alone(pred, frames[2:], _disc_at(YELLOW, 2)), yellow, objects=1, frames=4)
    _assert_tracked_like(_alone(pred, frames, seeds[1]), red, objects=1, frames=6)
