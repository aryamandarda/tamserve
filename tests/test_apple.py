import threading
from concurrent.futures import Future

import numpy as np
import pytest
import torch

from tamserve import Engine, PeopleTracker, Tracker, build_video_predictor, prepare
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


def test_next_frames_are_encoded_once_and_change_no_mask(pred):
    frames, seeds = _frames(8)
    waiting, ahead = AppleEngine(pred, encoder=OwnEncoder(pred)), AppleEngine(pred, encoder=OwnEncoder(pred))
    for k, f in enumerate(frames):
        start = {"cam": seeds} if k == 0 else None
        # A copy: the next frame is known by its pixels, not by being the same array.
        coming = {"cam": frames[k + 1].copy()} if k + 1 < len(frames) else None
        _same(waiting.step({"cam": f}, start), ahead.step({"cam": f}, start, next_frames=coming))
    assert ahead._neural.frames == waiting._neural.frames == len(frames)


def test_a_frame_other_than_the_one_promised_is_still_the_one_followed(pred):
    frames, seeds = _frames(4)
    plain, apple = Engine(pred), AppleEngine(pred, encoder=OwnEncoder(pred))
    _same(plain.step({"cam": prepare(frames[0])}, {"cam": seeds}),
          apple.step({"cam": frames[0]}, {"cam": seeds}, next_frames={"cam": frames[3]}))
    _same(plain.step({"cam": prepare(frames[1])}), apple.step({"cam": frames[1]}))
    assert apple._neural.frames == 3  # frames 0 and 1, and the promised one that never came


def test_a_next_frame_written_over_afterwards_is_not_mistaken_for_the_new_one(pred):
    frames, seeds = _frames(3)
    plain, apple = Engine(pred), AppleEngine(pred, encoder=OwnEncoder(pred))
    buffer = frames[1].copy()  # a decoder that reuses one array for every frame
    _same(plain.step({"cam": prepare(frames[0])}, {"cam": seeds}),
          apple.step({"cam": frames[0]}, {"cam": seeds}, next_frames={"cam": buffer}))
    buffer[:] = frames[2]
    _same(plain.step({"cam": prepare(frames[2])}), apple.step({"cam": buffer}))


def test_frames_started_as_they_arrive_are_encoded_once_and_change_no_mask(pred):
    frames, seeds = _frames(8)
    waiting, early = AppleEngine(pred, encoder=OwnEncoder(pred)), AppleEngine(pred, encoder=OwnEncoder(pred))
    for k, f in enumerate(frames):
        start = {"cam": seeds} if k == 0 else None
        # As the thread that reads a camera does, with an array of its own: a frame is known by its pixels.
        reader = threading.Thread(target=early.start_encoding, args=({"cam": f.copy()},))
        reader.start()
        reader.join()
        _same(waiting.step({"cam": f}, start), early.step({"cam": f}, start))
    assert early._neural.frames == waiting._neural.frames == len(frames)


def test_a_frame_that_arrives_and_is_never_stepped_costs_only_its_encoding(pred):
    frames, seeds = _frames(5)
    plain, apple = Engine(pred), AppleEngine(pred, encoder=OwnEncoder(pred))
    _same(plain.step({"cam": prepare(frames[0])}, {"cam": seeds}), apple.step({"cam": frames[0]}, {"cam": seeds}))
    apple.start_encoding({"cam": frames[1], "idle": frames[1]})  # nothing is followed on "idle": not encoded
    apple.start_encoding({"cam": frames[2]})                     # the loop was busy, and frame 1 is passed over
    _same(plain.step({"cam": prepare(frames[2])}), apple.step({"cam": frames[2]}))
    assert apple._neural.frames == 3  # frames 0, 1 and 2
    apple.start_encoding({"cam": frames[4]})  # arrives before the step for frame 3: it stays for its own step
    _same(plain.step({"cam": prepare(frames[3])}), apple.step({"cam": frames[3]}))
    _same(plain.step({"cam": prepare(frames[4])}), apple.step({"cam": frames[4]}))
    assert apple._neural.frames == 5
    apple.start_encoding({"cam": frames[0]})
    apple.drop("cam")
    assert not apple._ahead  # a camera that is dropped leaves nothing behind


def test_a_frame_passed_over_before_its_turn_is_not_encoded(pred):
    class Queued:
        """An encoder with every frame still waiting its turn."""

        def submit(self, frame):
            return Future()

    frames, seeds = _frames(3)
    apple = AppleEngine(pred, encoder=OwnEncoder(pred))
    apple.step({"cam": frames[0]}, {"cam": seeds})
    assert apple.start_encoding({"cam": frames[1]})["cam"]()  # encoded: a loop that is behind can take it
    apple._neural = Queued()
    first = apple.start_encoding({"cam": frames[1]})["cam"]
    second = apple.start_encoding({"cam": frames[2]})["cam"]
    assert not first() and not second()
    waiting = [started[0].features for _, started in apple._ahead["cam"]]
    assert len(waiting) == 2 and waiting[0].done() and not waiting[1].done()  # the encoded one, and frame 2


class DiscFinder:
    """Stands in for SAM 3 for either tracker: outlines the red disc and the yellow one, whatever is asked for."""

    def __call__(self, frames, prompts=None):
        found = [[torch.from_numpy(np.abs(f.astype(int) - np.array(c)).sum(-1) < 60) for c in (RED, YELLOW)]
                 for f in frames]
        self.scores, self.names = [[0.9, 0.9] for _ in frames], [["disc", "disc"] for _ in frames]
        return found


@pytest.mark.parametrize("tracker", [lambda engine: Tracker(engine, "disc", finder=DiscFinder()),
                                     lambda engine: PeopleTracker(engine, finder=DiscFinder())])
def test_the_trackers_pass_next_frames_on(pred, tracker):
    frames, _ = _frames(8)
    waiting = tracker(AppleEngine(pred, encoder=OwnEncoder(pred)))
    ahead = tracker(AppleEngine(pred, encoder=OwnEncoder(pred)))
    for k, f in enumerate(frames):
        coming = {"cam": frames[k + 1]} if k + 1 < len(frames) else None
        _same(waiting.step({"cam": f}), ahead.step({"cam": f}, next_frames=coming))
    assert ahead.engine._neural.frames == waiting.engine._neural.frames == len(frames)


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
    gpu, neural, ahead, early = Engine(mac, neural_engine=False), Engine(mac), Engine(mac), Engine(mac)
    for k, f in enumerate(frames):
        start = {"cam": seeds} if k == 0 else None
        want, got = gpu.step({"cam": f}, start)["cam"], neural.step({"cam": f}, start)["cam"]
        coming = {"cam": frames[k + 1]} if k + 1 < len(frames) else None
        _same({"cam": got}, ahead.step({"cam": f}, start, next_frames=coming))
        early.start_encoding({"cam": f})
        _same({"cam": got}, early.step({"cam": f}, start))
        for obj in (1, 2):
            both = (want[obj] & got[obj]).sum() / (want[obj] | got[obj]).sum()
            assert both > 0.9, (k, obj, float(both))
