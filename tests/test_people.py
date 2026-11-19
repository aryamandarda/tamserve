import numpy as np
import pytest
import torch

from tamserve import Engine, PeopleTracker, build_video_predictor
from tamserve.tracker import _match

RED, YELLOW = (220, 40, 40), (240, 220, 60)


@pytest.fixture(scope="module")
def pred():
    return build_video_predictor(device="cpu")


def _frames(h, w, n, discs, seed=0):
    """Discs (y, x, r, dy, dx, colour, first, last) moving over a smooth background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 80 * xx / w, 70 + 60 * yy / h, 110 + 40 * (xx + yy) / (h + w)], -1)
    frames = []
    for i in range(n):
        img = base + rng.normal(0, 4, base.shape)
        for y, x, r, dy, dx, colour, first, last in discs:
            if first <= i <= last:
                img[(yy - y - dy * i) ** 2 + (xx - x - dx * i) ** 2 <= r * r] = colour
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


class ColourFinder:
    """Stands in for SAM 3: outlines red and yellow discs, and counts the frames it is given."""

    def __init__(self):
        self.calls = []

    def __call__(self, frames):
        self.calls.append(len(frames))
        out = []
        for frame in frames:
            found = []
            for colour in (RED, YELLOW):
                m = np.abs(frame.astype(int) - np.array(colour)).sum(-1) < 60
                if m.sum() > 20:
                    found.append(torch.from_numpy(m))
            out.append(found)
        return out


def _colour_of(frame, mask):
    return frame[mask.numpy()].mean(0)


class Watching(ColourFinder):
    """Also notes which cameras each look took: a camera's frames carry its number in their first pixel."""

    def __call__(self, frames, *prompts):
        self.seen = [int(f[0, 0, 0]) for f in frames]
        return super().__call__(frames, *prompts)


def _looks(tracker, frames, cams):
    """Step the tracker through the frames on cameras numbered `cams`: which of them looked on each step."""
    looks = []
    for f in frames:
        tracker.finder.seen, numbered = [], {}
        for c in cams:
            numbered[c] = f.copy()
            numbered[c][0, 0] = c
        tracker.step(numbered)
        looks.append(tracker.finder.seen)
    return looks


def _gaps(looks, cam):
    """Frames between a camera's looks, from its second on: its first two are set apart from other cameras'."""
    at = [i for i, step in enumerate(looks) if cam in step]
    return {b - a for a, b in zip(at[1:], at[2:])}


def _tracker(pred, finder, look_every, **kw):
    return PeopleTracker(Engine(pred), finder=finder, look_every=look_every, **kw)


def test_match_is_one_to_one_by_overlap():
    a = torch.zeros(8, 8, dtype=torch.bool); a[:4, :4] = True
    b = torch.zeros(8, 8, dtype=torch.bool); b[4:, 4:] = True
    found = [b.clone(), a.clone(), torch.zeros(8, 8, dtype=torch.bool)]
    assert _match({1: a, 2: b}, found, 0.3) == {0: 2, 1: 1}


def test_ids_stay_on_their_people_across_looks(pred):
    frames = _frames(96, 128, 24, [(30, 24, 12, 1, 3, RED, 0, 99), (66, 100, 10, -1, -2, YELLOW, 0, 99)])
    tracker = _tracker(pred, ColourFinder(), look_every=5)
    for f in frames:
        out = tracker.step({"cam": f})["cam"]
        assert sorted(out) == [1, 2]
        assert _colour_of(f, out[1])[0] > 150 and _colour_of(f, out[2])[1] > 150  # red, yellow


def test_people_appear_and_leave_at_looks(pred):
    # Yellow is present for frames 8-17 only; looks land every 4 frames (0, 4, 8, ...).
    frames = _frames(96, 128, 28, [(30, 24, 12, 1, 3, RED, 0, 99), (66, 100, 10, -1, -2, YELLOW, 8, 17)])
    tracker = _tracker(pred, ColourFinder(), look_every=4)
    ids = [sorted(tracker.step({"cam": f})["cam"]) for f in frames]
    assert all(i == [1] for i in ids[:8])         # yellow not there yet
    assert all(i == [1, 2] for i in ids[8:18])    # found at the look on frame 8
    assert all(i == [1] for i in ids[24:])        # not shown once a look no longer sees it


class Blinking(ColourFinder):
    """Misses yellow at one look."""

    def __call__(self, frames):
        out = super().__call__(frames)
        return [found[:1] for found in out] if len(self.calls) == 3 else out


def test_a_missed_look_keeps_the_id(pred):
    frames = _frames(96, 128, 24, [(30, 24, 12, 1, 3, RED, 0, 99), (66, 100, 10, -1, -2, YELLOW, 0, 99)])
    tracker = _tracker(pred, Blinking(), look_every=4)
    ids = [sorted(tracker.step({"cam": f})["cam"]) for f in frames]
    assert all(i == [1, 2] for i in ids[:8] + ids[12:])  # yellow keeps id 2 after the blink
    assert all(i == [1] for i in ids[8:12])               # out of the answer until a look sees it again


def test_someone_hidden_comes_back_with_their_id(pred):
    # Yellow is not drawn on frames 6-9, so the look at frame 8 cannot see it.
    frames = _frames(96, 128, 20, [(30, 24, 12, 1, 3, RED, 0, 99), (66, 100, 10, 0, 0, YELLOW, 0, 5),
                                   (66, 100, 10, 0, 0, YELLOW, 10, 99)])
    tracker = _tracker(pred, ColourFinder(), look_every=4)
    ids = [sorted(tracker.step({"cam": f})["cam"]) for f in frames]
    assert all(i == [1, 2] for i in ids[12:])  # back as id 2, not a new id


def test_someone_gone_longer_than_hold_is_forgotten(pred):
    # Yellow is gone from frame 4 to 15, longer than hold; when it returns it is a new person.
    frames = _frames(96, 128, 24, [(30, 24, 12, 0, 1, RED, 0, 99), (66, 100, 10, 0, 0, YELLOW, 0, 3),
                                   (66, 100, 10, 0, 0, YELLOW, 16, 99)])
    tracker = _tracker(pred, ColourFinder(), look_every=4, hold=4)
    ids = [sorted(tracker.step({"cam": f})["cam"]) for f in frames]
    assert ids[0] == [1, 2] and ids[-1] == [1, 3]
    assert ("cam", 2) not in tracker.engine.tracks  # no longer followed


def test_looks_are_spread_across_cameras(pred):
    finder = ColourFinder()
    tracker = _tracker(pred, finder, look_every=3)
    for f in _frames(64, 96, 9, [(20, 20, 8, 0, 2, RED, 0, 99)]):
        tracker.step({c: f for c in range(6)})
    assert finder.calls[0] == 6          # every camera's first frame is looked at
    assert max(finder.calls[1:]) <= 2    # after that, about 6 cameras / 3 frames per step
    assert sum(finder.calls) == 6 + 2 * 8


def test_each_camera_can_have_its_own_look_every(pred):
    tracker = _tracker(pred, Watching(), look_every={0: 1, 1: 4})  # camera 2 keeps 15
    looks = _looks(tracker, _frames(64, 96, 30, []), [0, 1, 2])
    assert sorted(looks[0]) == [0, 1, 2]  # every camera's first frame is looked at
    assert [_gaps(looks, c) for c in (0, 1, 2)] == [{1}, {4}, {15}]


def test_look_every_can_change_while_running(pred):
    frames = _frames(64, 96, 12, [])
    tracker = _tracker(pred, Watching(), look_every={0: 6})
    looks = _looks(tracker, frames[:4], [0])
    tracker.look_every[0] = 1
    looks += _looks(tracker, frames[4:8], [0])
    tracker.look_every = 2
    looks += _looks(tracker, frames[8:], [0])
    assert [len(step) for step in looks] == [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
    tracker.look_every = {0: 0}
    with pytest.raises(ValueError):
        tracker.step({0: frames[0]})


def test_an_empty_camera_costs_nothing_between_looks(pred):
    tracker = _tracker(pred, ColourFinder(), look_every=5)
    for f in _frames(64, 96, 6, []):
        assert tracker.step({"cam": f}) == {"cam": {}}
        assert not tracker.engine.tracks


def test_a_camera_that_changes_resolution_starts_afresh(pred):
    tracker = _tracker(pred, ColourFinder(), look_every=4)
    for f in _frames(96, 128, 3, [(30, 24, 12, 1, 3, RED, 0, 99)]):
        tracker.step({"cam": f})
    out = tracker.step({"cam": _frames(80, 112, 1, [(30, 24, 12, 1, 3, RED, 0, 99)])[0]})["cam"]
    assert list(out) == [1] and out[1].shape == (80, 112)


def test_bad_settings_are_rejected(pred):
    for kw in ({"look_every": 0}, {"look_every": {"cam": 0}}, {"hold": -1}):
        with pytest.raises(ValueError):
            PeopleTracker(Engine(pred), finder=ColourFinder(), **kw)
    with pytest.raises(ValueError):
        PeopleTracker(Engine(pred, gpu_masks=True), finder=ColourFinder())


def test_weak_finds_confirm_people_but_do_not_start_them(pred):
    # Red is found confidently, yellow weakly: yellow is never followed on weak finds alone,
    # but once it is (found confidently on the first look) weak finds keep it.
    class Scored(ColourFinder):
        def __init__(self, yellow):
            super().__init__()
            self.yellow, self.scores = yellow, []

        def __call__(self, frames):
            out = super().__call__(frames)
            self.scores = [[0.9, self.yellow(len(self.calls))][:len(found)] for found in out]
            return out

    frames = _frames(96, 128, 12, [(30, 24, 12, 0, 1, RED, 0, 99), (66, 100, 10, 0, 0, YELLOW, 0, 99)])
    never = _tracker(pred, Scored(lambda call: 0.5), look_every=2)
    first = _tracker(pred, Scored(lambda call: 0.9 if call == 1 else 0.5), look_every=2)
    for f in frames:
        a, b = never.step({"cam": f})["cam"], first.step({"cam": f})["cam"]
        assert len(a) == 1   # only red
        assert len(b) == 2   # yellow started on the confident first look and stays
