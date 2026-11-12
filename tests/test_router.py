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


class Flaky:
    """A camera whose stream drops after `drop_after` frames and comes back where it
    was, as a live stream does; or one that never opens."""

    def __init__(self, folder, drop_after=None, never=False, bad=False):
        self.paths = sorted(Path(folder).glob("*.jpg"), key=lambda p: int(p.stem))
        self.drop_after, self.never, self.bad, self.at, self.opened = drop_after, never, bad, 0, 0

    def __call__(self):
        self.opened += 1
        if self.never:
            raise ConnectionError("no route to the camera")
        return self

    def read(self):
        if self.bad:
            return np.zeros((8, 8, 3), np.float32)
        if self.at == self.drop_after and self.opened == 1:
            raise ConnectionError("the stream stopped sending frames")
        if self.at == len(self.paths):
            return None
        self.at += 1
        return np.asarray(Image.open(self.paths[self.at - 1]).convert("RGB"))


def _statuses(router, camera, until, timeout=120):
    seen, deadline = [], time.time() + timeout
    for u in router.results(timeout=timeout):
        if u.camera == camera and u.status:
            seen.append(u.status)
            if until in seen:
                return seen
        assert time.time() < deadline
    raise AssertionError(f"no {until!r} for {camera}: {seen}")


def test_a_dropped_camera_comes_back_and_keeps_its_objects(tmp_path):
    folder, seeds, frames = _video(tmp_path / "a", 96, 128, 6, [RED], seed=0)
    want = _alone(build_video_predictor("cpu"), frames, seeds[1])
    got = {}
    with Router(workers=["cpu"]) as router:
        router.add_camera("a", Flaky(folder, drop_after=3), skip=False)
        router.start("a", 1, seeds[1].numpy(), frame=0)
        seen = []
        for u in router.results(timeout=120):
            seen += [u.status] if u.status else []
            for obj, mask in u.masks.items():
                got.setdefault(u.frame, {})[obj] = torch.from_numpy(mask.array())
            if len(got) == 6:
                break
        assert seen[:2] == ["disconnected", "reconnected"]
        assert router.status()["cameras"]["a"]["state"] in ("tracking", "ended")  # its six frames may be over
    _assert_tracked_like(want, got, objects=1, frames=6)  # the object carried on through the gap


def test_a_camera_that_never_returns_ends_once_retry_for_runs_out(tmp_path):
    folder, _, _ = _video(tmp_path / "a", 48, 64, 2, [RED], seed=0)
    with Router(workers=["cpu"]) as router:
        t = time.time()
        router.add_camera("gone", Flaky(folder, never=True), retry_for=2.5)
        seen = _statuses(router, "gone", "ended")
        assert seen[0] == "disconnected" and "reconnecting" in seen and time.time() - t < 15
        assert router.status()["cameras"]["gone"]["state"] == "ended"


def test_bad_frames_and_bad_prompts_stay_with_their_camera(tmp_path):
    folder, seeds, frames = _video(tmp_path / "a", 96, 128, 5, [RED], seed=0)
    errors, got = [], {}
    with Router(workers=["cpu"]) as router:
        router.add_camera("broken", Flaky(folder, bad=True), retry_for=0)
        router.add_camera("good", functools.partial(Frames, folder), skip=False)
        router.start("good", "red", seeds[1].numpy(), frame=0)
        router.start("good", "wrong", np.zeros((3, 3), bool), frame=0)  # not the frame's size
        for u in router.results(timeout=120):
            if u.error:
                errors.append((u.camera, u.error))
            got.update({u.frame: {1: torch.from_numpy(m.array())} for o, m in u.masks.items() if o == "red"})
            if len(got) == 5 and any(c == "broken" for c, _ in errors):
                break
    assert any(c == "good" and "could not start 'wrong'" in e for c, e in errors)
    assert any(c == "broken" and "uint8" in e for c, e in errors)
    _assert_tracked_like(_alone(build_video_predictor("cpu"), frames, seeds[1]), got, objects=1, frames=5)


def test_a_lost_worker_hands_its_cameras_over_and_comes_back(tmp_path):
    folder, seeds, _ = _video(tmp_path / "a", 96, 128, 40, [RED], seed=0)
    with Router(workers=["cpu", "cpu"]) as router:
        router.add_camera("a", functools.partial(Frames, folder), skip=False)
        router.start("a", 1, seeds[1].numpy(), frame=0)
        lost = router.home["a"]
        for u in router.results(timeout=120):
            if u.masks:
                break
        router._workers[lost].process.kill()  # as if it crashed
        seen = _statuses(router, "a", "moved")
        assert router.home["a"] != lost
        for u in router.results(timeout=120):
            if u.camera == "a" and u.masks:
                break  # tracking carries on, on the other worker, from the last mask
        deadline = time.time() + 120
        while not router.status()["workers"][lost]["running"]:
            assert time.time() < deadline, "the lost worker was never brought back"
            time.sleep(1)
        s = router.status()
        assert s["cameras"]["a"]["worker"] == "cpu" and s["cameras"]["a"]["objects"] == [1]


def test_moving_a_full_size_mask_does_not_lock_the_router(tmp_path):
    # Moving an object sends its last mask, 900 KB at 720p: more than a pipe holds. The
    # router's own thread must not wait on that send while the worker waits on it.
    folder, seeds, _ = _video(tmp_path / "a", 720, 1280, 30, [(300, 400, 120, 2, 6, (220, 40, 40))], seed=0)
    with Router(workers=["cpu", "cpu"]) as router:
        router.add_camera("big", functools.partial(Frames, folder), skip=False)
        router.start("big", 1, seeds[1].numpy(), frame=0)
        lost = router.home["big"]
        for u in router.results(timeout=120):
            if u.masks:
                break
        router._workers[lost].process.kill()
        _statuses(router, "big", "moved")
        for u in router.results(timeout=120):
            if u.camera == "big" and u.masks:
                assert u.masks[1].shape == (720, 1280)
                break


def test_a_burst_of_cameras_is_spread_even_when_one_worker_was_busy():
    # One worker last reported a camera's worth of work, the other nothing yet: eight
    # cameras added before the next report must still be split between them.
    from types import SimpleNamespace
    router = Router.__new__(Router)
    router._workers = [SimpleNamespace(alive=True, remote=False, busy=0.0, counted=0, cameras=set()),
                       SimpleNamespace(alive=True, remote=False, busy=0.15, counted=1, cameras=set())]
    for camera in range(8):
        router._workers[router._place("rtsp://camera")].cameras.add(camera)
    assert [len(w.cameras) for w in router._workers] == [4, 4]
