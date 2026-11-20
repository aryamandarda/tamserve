import sys

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


def test_prompts_can_change_while_running(pred):
    frames = _frames(96, 128, 12, TWO)
    tracker = Tracker(Engine(pred), {"cam": "red"}, finder=ColourFinder(), look_every=3)
    for f in frames[:4]:
        assert list(tracker.step({"cam": f})["cam"]) == [1]
    tracker.prompts["cam"] = ["red", "yellow"]
    for f in frames[4:8]:
        out = tracker.step({"cam": f})["cam"]
    assert tracker.names("cam") == {1: "red", 2: "yellow"} and sorted(out) == [1, 2]
    tracker.prompts["cam"] = "yellow"
    for f in frames[8:]:
        out = tracker.step({"cam": f})["cam"]
    assert tracker.names("cam") == {2: "yellow"} and sorted(out) == [2] and ("cam", 1) not in tracker.engine.tracks


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


def test_each_camera_can_have_its_own_look_every(pred):
    tracker = _tracker(pred, Watching(), look_every={0: 1, 1: 6})  # camera 2 keeps 3
    looks = _looks(tracker, _frames(64, 96, 20, [(20, 20, 8, 0, 2, RED, 0, 99)]), [0, 1, 2])
    assert sorted(looks[0]) == [0, 1, 2]  # every camera's first frame is looked at
    assert [_gaps(looks, c) for c in (0, 1, 2)] == [{1}, {6}, {3}]


def test_a_dict_that_gives_every_camera_one_number_looks_as_that_number_does(pred):
    frames = _frames(64, 96, 12, [])
    as_number = _looks(_tracker(pred, Watching(), look_every=3), frames, range(5))
    assert as_number == _looks(_tracker(pred, Watching(), look_every=dict.fromkeys(range(5), 3)), frames, range(5))


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


def test_cameras_that_look_most_often_do_not_wait_behind_the_others(pred):
    every = {0: 1, 1: 1, **dict.fromkeys(range(2, 8), 30)}
    tracker = _tracker(pred, Watching(), look_every=every)
    frames = _frames(64, 96, 8, [])
    looks = _looks(tracker, frames[:2], range(8))
    for c in range(2, 8):
        every[c] = 3  # six cameras fall due at once
    looks += _looks(tracker, frames[2:], range(8))
    assert all(len(step) <= 4 and {0, 1} <= set(step) for step in looks[1:])  # 2 + 6 / 3 looks a step
    assert sorted(c for step in looks[2:5] for c in step if c > 1) == [2, 3, 4, 5, 6, 7]  # the six take turns


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


class _Batch(dict):
    """What the stand-in processor gives: a dict whose entries are attributes too, as transformers' is."""
    __getattr__ = dict.__getitem__

    def to(self, device):
        return self


def _stand_in_for_transformers(calls):
    """A `transformers` with a SAM 3 that outlines discs by colour and counts what it is asked to encode."""
    class Sam3Processor:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def __call__(self, images=None, text=None, return_tensors=None):
            if images is not None:
                return _Batch(pixel_values=torch.from_numpy(np.asarray(images)))
            return _Batch(input_ids=text, attention_mask=torch.ones(1))

        def post_process_instance_segmentation(self, out, threshold, mask_threshold, target_sizes):
            picture, colour = out
            m = (picture.int() - torch.tensor(COLOURS[colour])).abs().sum(-1) < 60
            assert list(m.shape) == target_sizes[0]
            found = m.sum() > 20 and 0.9 >= threshold
            return [{"masks": m[None] if found else m[:0], "scores": torch.full((int(found),), 0.9)}]

    class Sam3Model:
        dtype = torch.uint8

        @classmethod
        def from_pretrained(cls, name, torch_dtype=None):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def get_vision_features(self, pixel_values):
            calls.append("picture")
            return pixel_values

        def get_text_features(self, input_ids, attention_mask):
            calls.append(input_ids)
            return input_ids

        def __call__(self, vision_embeds, text_embeds, attention_mask):
            return vision_embeds, text_embeds

    module = type(sys)("transformers")
    module.Sam3Model, module.Sam3Processor = Sam3Model, Sam3Processor
    return module


def test_without_an_nvidia_gpu_sam3_comes_through_hugging_face(pred, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "transformers", _stand_in_for_transformers(calls))
    tracker = Tracker(Engine(pred), ["red", "yellow"], look_every=2)  # the finder it makes itself, on the CPU
    frames = _frames(96, 128, 6, TWO)
    for f in frames:
        out = tracker.step({"cam": f})["cam"]
        assert sorted(tracker.names("cam").values()) == ["red", "yellow"]
        assert all(np.abs(_colour_of(f, out[i]) - COLOURS[name]).max() < 40 for i, name in tracker.names("cam").items())
    # A picture is encoded once for both prompts, and each prompt's words once for good.
    assert calls.count("picture") == 3 and calls.count("red") == calls.count("yellow") == 1


def test_without_sam3_a_tracker_says_how_to_install_it(pred, monkeypatch):
    monkeypatch.setitem(sys.modules, "sam3", None)  # as where SAM 3 is not installed, in either form
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ModuleNotFoundError, match=r'pip install "tamserve\[sam3\]"'):
        Tracker(Engine(pred), "red")


def test_bad_settings_are_rejected(pred):
    for kw in ({"look_every": 0}, {"look_every": {"cam": 0}}, {"hold": -1}):
        with pytest.raises(ValueError):
            Tracker(Engine(pred), "red", finder=ColourFinder(), **kw)
    with pytest.raises(ValueError):
        Tracker(Engine(pred, gpu_masks=True), "red", finder=ColourFinder())
