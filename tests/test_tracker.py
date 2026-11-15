import numpy as np
import pytest
import torch

from tamserve import Engine, Tracker, build_video_predictor

RED, YELLOW = (220, 40, 40), (240, 220, 60)
COLOURS = {"red": RED, "yellow": YELLOW}


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
    """Stands in for SAM 3: the prompts "red" and "yellow" outline discs of that colour.
    Counts the frames it is given, and the first look is always a confident one.

    score: its confidence in what it finds after the first look.
    twice: every outline is given twice, as SAM 3 sometimes repeats one.
    once: these prompts are answered on the first look only.
    shifted: after the first look its outlines sit this many pixels to the right.
    """

    def __init__(self, score=0.9, twice=False, once=(), shifted=0):
        self.score, self.twice, self.once, self.shifted = score, twice, once, shifted
        self.calls, self.scores, self.names = [], [], []

    def __call__(self, frames, prompts):
        first = not self.calls
        self.calls.append(len(frames))
        out, self.scores, self.names = [], [], []
        for frame, wanted in zip(frames, prompts):
            masks, names = [], []
            for prompt in wanted:
                m = np.abs(frame.astype(int) - np.array(COLOURS[prompt])).sum(-1) < 60
                if m.sum() > 20 and (first or prompt not in self.once):
                    if self.shifted and not first:
                        m = np.roll(m, self.shifted, 1)
                    masks += [torch.from_numpy(m)] * (2 if self.twice else 1)
                    names += [prompt] * (2 if self.twice else 1)
            out.append(masks)
            self.names.append(names)
            self.scores.append([0.9 if first else self.score] * len(masks))
        return out


def _colour_of(frame, mask):
    return frame[mask.numpy()].mean(0)


def _tracker(pred, finder, look_every, prompts=("red", "yellow"), **kw):
    return Tracker(Engine(pred), list(prompts), finder=finder, look_every=look_every, **kw)


TWO = [(30, 24, 12, 1, 3, RED, 0, 99), (66, 100, 10, -1, -2, YELLOW, 0, 99)]


def test_ids_stay_on_their_objects_and_names_say_what_each_is(pred):
    frames = _frames(96, 128, 24, TWO)
    tracker = _tracker(pred, ColourFinder(), look_every=5)
    for f in frames:
        out = tracker.step({"cam": f})["cam"]
        assert sorted(out) == [1, 2]
        assert _colour_of(f, out[1])[0] > 150 and _colour_of(f, out[2])[1] > 150  # red, yellow
    assert tracker.names("cam") == {1: "red", 2: "yellow"} and tracker.names("other") == {}


def test_each_camera_can_have_its_own_prompts(pred):
    finder = ColourFinder()
    tracker = Tracker(Engine(pred), {"both": ["red", "yellow"], "one": "yellow"}, finder=finder, look_every=4)
    for f in _frames(96, 128, 8, TWO):
        out = tracker.step({"both": f, "one": f})
    assert len(out["both"]) == 2 and list(tracker.names("one").values()) == ["yellow"]
    assert _colour_of(f, out["one"][1])[1] > 150
    with pytest.raises(KeyError):
        tracker.step({"unnamed": f})


def test_a_repeated_outline_does_not_start_a_second_object(pred):
    frames = _frames(96, 128, 6, TWO[:1])
    tracker = _tracker(pred, ColourFinder(twice=True), look_every=3)
    assert all(sorted(tracker.step({"cam": f})["cam"]) == [1] for f in frames)


def test_a_missed_look_does_not_hide_an_object(pred):
    # Yellow is found once and never again: one missed look of its trial leaves it in the answer.
    frames = _frames(96, 128, 10, TWO)
    tracker = _tracker(pred, ColourFinder(once=("yellow",)), look_every=5)
    assert all(sorted(tracker.step({"cam": f})["cam"]) == [1, 2] for f in frames)


def test_a_new_object_most_looks_miss_is_removed(pred):
    frames = _frames(96, 128, 16, TWO)
    tracker = _tracker(pred, ColourFinder(once=("yellow",)), look_every=5)
    shown = [sorted(tracker.step({"cam": f})["cam"]) for f in frames]
    assert shown[9] == [1, 2]                     # missed once, on frame 5: still on trial
    assert shown[10] == [1] and shown[15] == [1]  # missed by two of its trial's three looks
    assert ("cam", 2) not in tracker.engine.tracks and tracker.names("cam") == {1: "red"}


def test_an_object_no_look_sees_is_forgotten_after_hold(pred):
    # Yellow is gone after frame 3. The engine may keep a mask for it or not; either way
    # it is forgotten once no look has seen it for longer than hold.
    frames = _frames(96, 128, 44, [(30, 24, 12, 0, 1, RED, 0, 99), (66, 100, 10, 0, 0, YELLOW, 0, 3)])
    finder = ColourFinder()
    tracker = _tracker(pred, finder, look_every=1, hold=20)
    for f in frames[:20]:
        tracker.step({"cam": f})
    assert ("cam", 2) in tracker.engine.tracks  # past its trial, so missed looks do not remove it
    for f in frames[20:]:
        out = tracker.step({"cam": f})["cam"]
    assert sorted(out) == [1] and ("cam", 2) not in tracker.engine.tracks


def test_only_confident_outlines_start_objects(pred):
    frames = _frames(96, 128, 8, TWO[:1])
    weak = Tracker(Engine(pred), "red", finder=ColourFinder(), look_every=2, start=0.95)
    sure = Tracker(Engine(pred), "red", finder=ColourFinder(), look_every=2)
    for f in frames:
        assert weak.step({"cam": f})["cam"] == {} and len(sure.step({"cam": f})["cam"]) == 1


def test_an_object_restarts_only_from_an_outline_the_finder_is_sure_of(pred):
    # After the first look the finder's outlines sit 6 pixels off. Sure of them (0.9), the
    # object starts again from each one; less sure (0.75), it keeps the engine's own mask.
    frames = _frames(96, 128, 4, [(48, 64, 14, 0, 0, RED, 0, 99)])
    on_disc = {}
    for score in (0.9, 0.75):
        tracker = Tracker(Engine(pred), "red", finder=ColourFinder(score=score, shifted=6), look_every=1)
        for f in frames:
            out = tracker.step({"cam": f})["cam"]
        on_disc[score] = (np.abs(f[out[1].numpy()].astype(int) - np.array(RED)).sum(-1) < 60).mean()
    assert on_disc[0.75] > 0.95 > on_disc[0.9]


def test_looks_are_spread_across_cameras(pred):
    finder = ColourFinder()
    tracker = _tracker(pred, finder, look_every=3)
    for f in _frames(64, 96, 9, [(20, 20, 8, 0, 2, RED, 0, 99)]):
        tracker.step({c: f for c in range(6)})
    assert finder.calls[0] == 6          # every camera's first frame is looked at
    assert max(finder.calls[1:]) <= 2    # after that, about 6 cameras / 3 frames per step
    assert sum(finder.calls) == 6 + 2 * 8


def test_a_still_camera_looks_rarely_and_is_not_followed(pred):
    frames = _frames(96, 128, 40, [(30, 24, 12, 0, 0, RED, 0, 99)])
    finder = ColourFinder()
    tracker = _tracker(pred, finder, look_every=2, still=True)
    steps = []
    tracker.engine.step = lambda ready, step=tracker.engine.step: steps.append(len(ready)) or step(ready)
    outs = [tracker.step({"cam": f})["cam"] for f in frames]
    assert all(sorted(o) == [1] for o in outs)
    assert len(finder.calls) <= 9   # every frame of the object's trial, then every 30th
    assert sum(steps) <= 8          # nothing moved, so it was followed on its trial's looks at most
    assert all(torch.equal(o[1], outs[16][1]) for o in outs[16:30])


def test_a_camera_that_changes_keeps_looking(pred):
    finder = ColourFinder()
    tracker = _tracker(pred, finder, look_every=2, still=True)
    for f in _frames(96, 128, 40, [(30, 24, 12, 1, 2, RED, 0, 99)]):
        assert sorted(tracker.step({"cam": f})["cam"]) == [1]
    assert len(finder.calls) >= 18


def test_an_empty_camera_costs_nothing_between_looks(pred):
    tracker = _tracker(pred, ColourFinder(), look_every=5)
    for f in _frames(64, 96, 6, []):
        assert tracker.step({"cam": f}) == {"cam": {}}
        assert not tracker.engine.tracks


def test_a_camera_that_changes_resolution_starts_afresh(pred):
    tracker = _tracker(pred, ColourFinder(), look_every=4)
    for f in _frames(96, 128, 3, TWO[:1]):
        tracker.step({"cam": f})
    out = tracker.step({"cam": _frames(80, 112, 1, TWO[:1])[0]})["cam"]
    assert list(out) == [1] and out[1].shape == (80, 112)


def test_bad_settings_are_rejected(pred):
    for kw in ({"look_every": 0}, {"hold": -1}):
        with pytest.raises(ValueError):
            Tracker(Engine(pred), "red", finder=ColourFinder(), **kw)
    with pytest.raises(ValueError):
        Tracker(Engine(pred, gpu_masks=True), "red", finder=ColourFinder())
