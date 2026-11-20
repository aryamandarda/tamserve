from concurrent.futures import Future

import numpy as np
import pytest
import torch

from tamserve import Engine, build_video_predictor, prepare
from tamserve.apple import AppleEngine

RED, YELLOW = (220, 40, 40), (240, 220, 60)
ON_A_MAC = torch.backends.mps.is_available()


@pytest.fixture(scope="module")
def pred():
    return build_video_predictor(device="cpu")


def _frames(n, h=96, w=128, seed=0):
    """A red and a yellow disc moving over a smooth background, and each disc's mask on frame 0."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 80 * xx / w, 70 + 60 * yy / h, 110 + 40 * (xx + yy) / (h + w)], -1)
    discs = [(30, 24, 12, 1, 3, RED), (66, 100, 10, -1, -2, YELLOW)]
    frames = []
    for i in range(n):
        img = base + rng.normal(0, 4, base.shape)
        for y, x, r, dy, dx, colour in discs:
            img[(yy - y - dy * i) ** 2 + (xx - x - dx * i) ** 2 <= r * r] = colour
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    seeds = {k: torch.from_numpy((yy - y) ** 2 + (xx - x) ** 2 <= r * r) for k, (y, x, r, *_) in enumerate(discs, 1)}
    return frames, seeds


class OwnEncoder:
    """Stands in for the Neural Engine: the model's own encoder, on stock's own resize, counting its frames."""

    def __init__(self, pred):
        self.engine, self.frames = Engine(pred), 0

    def submit(self, frame):
        self.frames += 1
        self.engine._use(True)
        try:
            with torch.inference_mode():
                maps = Engine._backbone(self.engine, [prepare(frame)[0]])["backbone_fpn"]
        finally:
            self.engine._use(False)
        answer = Future()
        answer.set_result([m.numpy() for m in maps])
        return answer


def _same(a, b):
    assert a.keys() == b.keys()
    for cam in a:
        assert a[cam].keys() == b[cam].keys()
        assert all(torch.equal(a[cam][k], b[cam][k]) and a[cam][k].any() for k in a[cam])


def test_with_the_models_own_features_the_masks_are_the_engines(pred):
    frames, seeds = _frames(8)
    plain, apple = Engine(pred), AppleEngine(pred, encoder=OwnEncoder(pred))
    for k, f in enumerate(frames):
        start = {"cam": seeds} if k == 0 else None
        _same(plain.step({"cam": prepare(f)}, start), apple.step({"cam": f}, start))


def test_frames_given_as_soon_are_encoded_once_and_change_no_mask(pred):
    frames, seeds = _frames(8)
    waiting, ahead = AppleEngine(pred, encoder=OwnEncoder(pred)), AppleEngine(pred, encoder=OwnEncoder(pred))
    for k, f in enumerate(frames):
        start = {"cam": seeds} if k == 0 else None
        soon = {"cam": frames[k + 1]} if k + 1 < len(frames) else None
        _same(waiting.step({"cam": f}, start), ahead.step({"cam": f}, start, soon=soon))
    assert ahead._neural.frames == waiting._neural.frames == len(frames)


def test_a_frame_other_than_the_one_promised_is_still_the_one_followed(pred):
    frames, seeds = _frames(4)
    plain, apple = Engine(pred), AppleEngine(pred, encoder=OwnEncoder(pred))
    _same(plain.step({"cam": prepare(frames[0])}, {"cam": seeds}),
          apple.step({"cam": frames[0]}, {"cam": seeds}, soon={"cam": frames[3]}))
    _same(plain.step({"cam": prepare(frames[1])}), apple.step({"cam": frames[1]}))
    assert apple._neural.frames == 3  # frames 0 and 1, and the promised one that never came


def test_an_object_can_start_on_the_frame_just_stepped(pred):
    frames, seeds = _frames(3)
    plain, apple = Engine(pred), AppleEngine(pred, encoder=OwnEncoder(pred))
    first, later = {"cam": {1: seeds[1]}}, {"cam": {2: seeds[2]}}
    _same(plain.step({"cam": prepare(frames[0])}, first), apple.step({"cam": frames[0]}, first))
    _same(plain.add({"cam": prepare(frames[0])}, later), apple.add({"cam": frames[0]}, later))
    _same(plain.step({"cam": prepare(frames[1])}), apple.step({"cam": frames[1]}))


def test_frames_from_prepare_stay_on_the_gpu_path_beside_raw_ones(pred):
    frames, seeds = _frames(4)
    plain, apple = Engine(pred), AppleEngine(pred, encoder=OwnEncoder(pred))
    for k, f in enumerate(frames):
        start = {"raw": seeds, "ready": seeds} if k == 0 else None
        _same(plain.step({"raw": prepare(f), "ready": prepare(f)}, start),
              apple.step({"raw": f, "ready": prepare(f)}, start))
    assert apple._neural.frames == len(frames)  # only the raw camera's


def test_only_an_apple_gpu_gets_the_apple_engine(pred):
    assert type(Engine(pred)) is Engine


@pytest.mark.skipif(not ON_A_MAC, reason="needs an Apple-silicon Mac")
def test_on_a_mac_the_engine_uses_the_neural_engine_unless_told_not_to():
    mac = build_video_predictor(device="mps")
    assert type(Engine(mac)) is AppleEngine
    assert type(Engine(mac, neural_engine=False)) is Engine
    assert type(Engine(mac, exact=True)) is Engine
    assert type(Engine(build_video_predictor(device="mps", model="sam2.1-tiny"))) is Engine  # no encoder for it


@pytest.mark.skipif(not ON_A_MAC, reason="needs an Apple-silicon Mac")
def test_the_neural_engine_tracks_as_the_gpu_does():
    mac = build_video_predictor(device="mps")
    frames, seeds = _frames(20)
    gpu, neural, ahead = Engine(mac, neural_engine=False), Engine(mac), Engine(mac)
    for k, f in enumerate(frames):
        start = {"cam": seeds} if k == 0 else None
        want, got = gpu.step({"cam": f}, start)["cam"], neural.step({"cam": f}, start)["cam"]
        soon = {"cam": frames[k + 1]} if k + 1 < len(frames) else None
        _same({"cam": got}, ahead.step({"cam": f}, start, soon=soon))
        for obj in (1, 2):
            both = (want[obj] & got[obj]).sum() / (want[obj] | got[obj]).sum()
            assert both > 0.9, (k, obj, float(both))
