import numpy as np
import pytest
import torch
from PIL import Image

from tamserve.compare import compare
from tamserve.models import build_video_predictor
from tamserve.engine import Box, Clicks, Engine, prepare, prepare_batch
from tamserve.reference import track


@pytest.fixture(scope="module")
def pred():
    return build_video_predictor(device="cpu")


@pytest.fixture(scope="module")
def pred21():
    return build_video_predictor(device="cpu", model="sam2.1-tiny")


def _video(path, h, w, frames, discs, seed):
    """Solid discs moving across a smooth background, so EdgeTAM keeps tracking
    them on every frame (on pure noise the masks die after a few frames and a
    comparison would only check empty masks). Seeds are the discs on frame 0."""
    rng = np.random.default_rng(seed)
    path.mkdir()
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 80 * xx / w, 70 + 60 * yy / h, 110 + 40 * (xx + yy) / (h + w)], -1)
    for i in range(frames):
        img = base + rng.normal(0, 4, base.shape)
        for y, x, r, dy, dx, colour in discs:
            img[(yy - y - dy * i) ** 2 + (xx - x - dx * i) ** 2 <= r * r] = colour
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path / f"{i}.jpg")
    seeds = {k: torch.from_numpy((yy - y) ** 2 + (xx - x) ** 2 <= r * r)
             for k, (y, x, r, _, _, _) in enumerate(discs, start=1)}
    paths = sorted(path.glob("*.jpg"), key=lambda p: int(p.stem))
    return str(path), seeds, [prepare(p) for p in paths]


RED = (30, 24, 12, 1, 3, (220, 40, 40))
YELLOW = (66, 100, 10, -1, -2, (240, 220, 60))


def _run(engine, videos, starts):
    """Feed each video one frame per step from its start step; collect per-video masks."""
    got = [{} for _ in videos]
    for k in range(max(s + len(v[2]) for v, s in zip(videos, starts))):
        frames, seeds = {}, {}
        for i, (v, s) in enumerate(zip(videos, starts)):
            if s <= k < s + len(v[2]):
                frames[i] = v[2][k - s]
                if k == s:
                    seeds[i] = v[1]
        for i, masks in engine.step(frames, seeds).items():
            got[i][k - starts[i]] = masks
    return got


def _assert_tracked_like(solo, got, objects, frames, exact=False):
    r = compare(solo, got)
    assert r["missing"] == r["extra"] == 0
    assert r["object_frames"] == objects * frames  # every frame carried a real mask
    assert r["lost"] == r["phantom"] == 0
    assert r["min_iou"] == 1.0 if exact else r["mean_iou"] > 0.99


def test_inputs_match_stock(pred, tmp_path):
    d, _, frames = _video(tmp_path / "a", 48, 80, 2, [RED], seed=0)
    stock = pred.init_state(d)["images"]
    assert torch.equal(Engine(pred)._normalize([f for f, _ in frames]), stock)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_inputs_match_stock_on_gpu(tmp_path):
    gpu = build_video_predictor(device="cuda")
    d, _, frames = _video(tmp_path / "a", 48, 80, 2, [RED], seed=0)
    stock = gpu.init_state(d)["images"]
    assert torch.equal(Engine(gpu)._normalize([f.cuda() for f, _ in frames]), stock)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("model", ["edgetam", "sam2.1-tiny"])
def test_one_camera_equals_stock_on_gpu(model, tmp_path, monkeypatch):
    # The configuration that matters: bf16 on a GPU, same kernels as stock at batch 1.
    # exact takes a camera's objects in one pass however many a pass normally holds.
    monkeypatch.setattr("tamserve.engine._OBJECT_BATCH", 1)
    gpu = build_video_predictor(device="cuda", model=model)
    v = _video(tmp_path / "a", 96, 128, 20, [RED, YELLOW], seed=0)
    (got,) = _run(Engine(gpu, exact=True), [v], [0])
    _assert_tracked_like(track(gpu, v[0], v[1]), got, objects=2, frames=20, exact=True)


def test_one_camera_equals_stock(pred, tmp_path):
    # 20 frames: seed, memory filling up, then steady state past frame 16.
    v = _video(tmp_path / "a", 96, 128, 20, [RED, YELLOW], seed=0)
    (got,) = _run(Engine(pred, exact=True), [v], [0])
    _assert_tracked_like(track(pred, v[0], v[1]), got, objects=2, frames=20, exact=True)


def test_one_camera_equals_stock_sam21(pred21, tmp_path):
    # SAM 2.1 stores memory as feature maps and labels pointers with frame distances.
    v = _video(tmp_path / "a", 96, 128, 20, [RED, YELLOW], seed=0)
    (got,) = _run(Engine(pred21, exact=True), [v], [0])
    _assert_tracked_like(track(pred21, v[0], v[1]), got, objects=2, frames=20, exact=True)


def _prompts(discs):
    """A box around the first disc and a click on the centre of each other one."""
    (y, x, r, *_), rest = discs[0], discs[1:]
    seeds = {1: Box(x - r, y - r, x + r, y + r)}
    seeds.update({k: Clicks([(x, y)]) for k, (y, x, *_) in enumerate(rest, start=2)})
    return seeds


@pytest.mark.parametrize("model", ["edgetam", "sam2.1-tiny"])
def test_box_and_click_seeds_equal_stock(model, pred, pred21, tmp_path):
    # Seeds as a detector or a person gives them: a box, a click, and a mask together.
    p = pred if model == "edgetam" else pred21
    green = (20, 100, 9, 2, -1, (40, 200, 60))
    d, masks, frames = _video(tmp_path / "a", 96, 128, 18, [RED, YELLOW, green], seed=0)
    seeds = {**_prompts([RED, YELLOW]), 3: masks[3]}
    (got,) = _run(Engine(p, exact=True), [(d, seeds, frames)], [0])
    _assert_tracked_like(track(p, d, seeds), got, objects=3, frames=18, exact=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("model", ["edgetam", "sam2.1-tiny"])
def test_box_and_click_seeds_equal_stock_on_gpu(model, tmp_path):
    gpu = build_video_predictor(device="cuda", model=model)
    d, _, frames = _video(tmp_path / "a", 96, 128, 18, [RED, YELLOW], seed=0)
    seeds = _prompts([RED, YELLOW])
    (got,) = _run(Engine(gpu, exact=True), [(d, seeds, frames)], [0])
    _assert_tracked_like(track(gpu, d, seeds), got, objects=2, frames=18, exact=True)


def test_box_seeds_batched_match_stock(pred, tmp_path):
    # The default seeds all boxes in one pass: same tracks, up to rounding.
    d, _, frames = _video(tmp_path / "a", 96, 128, 10, [RED, YELLOW], seed=0)
    (y1, x1, r1, *_), (y2, x2, r2, *_) = RED, YELLOW
    seeds = {1: Box(x1 - r1, y1 - r1, x1 + r1, y1 + r1), 2: Box(x2 - r2, y2 - r2, x2 + r2, y2 + r2)}
    (got,) = _run(Engine(pred), [(d, seeds, frames)], [0])
    _assert_tracked_like(track(pred, d, seeds), got, objects=2, frames=10)


def test_cameras_joining_sam21(pred21, tmp_path):
    # At least 16 frames each: stock scales pointer distances by a finished video's
    # length below that, while a live camera has no length.
    # One step apart, so for the last three steps both have full memories and share
    # an attention call while their pointer distances differ.
    a = _video(tmp_path / "a", 96, 128, 20, [RED, YELLOW], seed=0)
    b = _video(tmp_path / "b", 80, 112, 19, [RED], seed=1)
    got = _run(Engine(pred21), [a, b], [0, 1])
    _assert_tracked_like(track(pred21, a[0], a[1]), got[0], objects=2, frames=20)
    _assert_tracked_like(track(pred21, b[0], b[1]), got[1], objects=1, frames=19)


def test_cameras_joining_at_different_times(pred, tmp_path):
    # Different sizes, object counts and start steps: each camera must match its
    # solo run and come back at its own size under its own ids.
    a = _video(tmp_path / "a", 96, 128, 8, [RED, YELLOW], seed=0)
    b = _video(tmp_path / "b", 80, 112, 6, [RED], seed=1)
    got = _run(Engine(pred), [a, b], [0, 3])
    _assert_tracked_like(track(pred, a[0], a[1]), got[0], objects=2, frames=8)
    _assert_tracked_like(track(pred, b[0], b[1]), got[1], objects=1, frames=6)
    assert got[1][5][1].shape == (80, 112)


def _disc_at(disc, i, h=96, w=128):
    """The disc's mask on frame i of a _video."""
    y, x, r, dy, dx, _ = disc
    yy, xx = np.mgrid[0:h, 0:w]
    return torch.from_numpy((yy - y - dy * i) ** 2 + (xx - x - dx * i) ** 2 <= r * r)


def _alone(pred, frames, seed):
    """One object followed on its own from frames[0]: {frame: {1: mask}}."""
    engine = Engine(pred)
    out = [engine.step({"cam": frames[0]}, {"cam": {1: seed}})]
    out += [engine.step({"cam": f}) for f in frames[1:]]
    return {i: o["cam"] for i, o in enumerate(out)}


def test_objects_join_and_leave_a_camera(pred, tmp_path):
    # Yellow starts on frame 3 and red stops after frame 5: each must be followed
    # as if it were alone, whatever the other does.
    _, seeds, frames = _video(tmp_path / "a", 96, 128, 9, [RED, YELLOW], seed=0)
    engine, got = Engine(pred), []
    for i, f in enumerate(frames):
        start = {0: {"red": seeds[1]}, 3: {"yellow": _disc_at(YELLOW, 3)}}.get(i)
        got.append(engine.step({"cam": f}, {"cam": start} if start else None)["cam"])
        if i == 5:
            engine.drop("cam", "red")
    assert [sorted(g) for g in got] == [["red"]] * 3 + [["red", "yellow"]] * 3 + [["yellow"]] * 3
    red = {i: {1: g["red"]} for i, g in enumerate(got[:6])}
    yellow = {i: {1: g["yellow"]} for i, g in enumerate(got[3:])}
    _assert_tracked_like(_alone(pred, frames[:6], seeds[1]), red, objects=1, frames=6)
    _assert_tracked_like(_alone(pred, frames[3:], _disc_at(YELLOW, 3)), yellow, objects=1, frames=6)


def test_more_objects_than_one_pass_holds(pred, tmp_path, monkeypatch):
    # Three objects of two ages, two to a pass: pass boundaries and mixed memory sizes.
    monkeypatch.setattr("tamserve.engine._OBJECT_BATCH", 2)
    green = (20, 100, 9, 2, -1, (40, 200, 60))
    _, seeds, frames = _video(tmp_path / "a", 96, 128, 6, [RED, YELLOW, green], seed=0)
    starts = {1: (0, seeds[1]), 2: (0, seeds[2]), 3: (2, _disc_at(green, 2))}
    engine, got = Engine(pred), []
    for i, f in enumerate(frames):
        start = {o: seed for o, (first, seed) in starts.items() if first == i}
        got.append(engine.step({"cam": f}, {"cam": start} if start else None)["cam"])
    for o, (first, seed) in starts.items():
        mine = {i: {1: g[o]} for i, g in enumerate(got[first:])}
        _assert_tracked_like(_alone(pred, frames[first:], seed), mine, objects=1, frames=6 - first)


def test_add_starts_an_object_without_advancing_the_others(pred, tmp_path):
    # A prompt decided after seeing a frame's masks: yellow starts on the frame
    # red was just followed on, and red must not see that frame twice.
    _, seeds, frames = _video(tmp_path / "a", 96, 128, 5, [RED, YELLOW], seed=0)
    engine = Engine(pred)
    got = [engine.step({"cam": frames[0]}, {"cam": {"red": seeds[1]}})["cam"], engine.step({"cam": frames[1]})["cam"]]
    added = engine.add({"cam": frames[1]}, {"cam": {"yellow": _disc_at(YELLOW, 1)}})
    assert sorted(added["cam"]) == ["yellow"]
    got[1]["yellow"] = added["cam"]["yellow"]
    got += [engine.step({"cam": f})["cam"] for f in frames[2:]]
    red = {i: {1: g["red"]} for i, g in enumerate(got)}
    yellow = {i: {1: g["yellow"]} for i, g in enumerate(got[1:])}
    _assert_tracked_like(_alone(pred, frames, seeds[1]), red, objects=1, frames=5)
    _assert_tracked_like(_alone(pred, frames[1:], _disc_at(YELLOW, 1)), yellow, objects=1, frames=4)


def test_a_new_prompt_restarts_the_object(pred, tmp_path):
    # Red is prompted again on frame 2, by step() and by add(): from there it must
    # be what a fresh start on frame 2 gives, and yellow must carry on untouched.
    _, seeds, frames = _video(tmp_path / "a", 96, 128, 6, [RED, YELLOW], seed=0)
    again = {"cam": {1: _disc_at(RED, 2)}}
    fresh = _alone(pred, frames[2:], _disc_at(RED, 2))
    for how in ("step", "add"):
        engine = Engine(pred)
        engine.step({"cam": frames[0]}, {"cam": seeds})
        engine.step({"cam": frames[1]})
        if how == "step":
            got = [engine.step({"cam": frames[2]}, again)["cam"]]
        else:
            got = [engine.step({"cam": frames[2]})["cam"]]
            got[0][1] = engine.add({"cam": frames[2]}, again)["cam"][1]
        got += [engine.step({"cam": f})["cam"] for f in frames[3:]]
        assert list(engine.tracks) == [("cam", 1), ("cam", 2)]  # red keeps its place
        _assert_tracked_like(fresh, {i: {1: g[1]} for i, g in enumerate(got)}, objects=1, frames=4)
        assert all(g[2].any() for g in got)


def test_a_refilled_frame_buffer_is_encoded_again(pred, tmp_path):
    # One buffer per camera, refilled through numpy, which torch's change counter
    # does not see: step() must encode it every time.
    _, seeds, frames = _video(tmp_path / "a", 96, 128, 2, [RED], seed=0)
    buf, size = frames[0][0].clone(), frames[0][1]
    a, b = Engine(pred), Engine(pred)
    a.step({"cam": (buf, size)}, {"cam": seeds})
    b.step({"cam": frames[0]}, {"cam": seeds})
    buf.numpy()[:] = frames[1][0].numpy()
    assert torch.equal(a.step({"cam": (buf, size)})["cam"][1], b.step({"cam": frames[1]})["cam"][1])


def test_resize_on_device_matches_stock(tmp_path):
    rng = np.random.default_rng(0)
    for h, w in [(1080, 1920), (480, 640)]:
        im = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        a, size = prepare(im)
        b, size_b = prepare(im, device="cpu", resize_on_device=True)
        assert size == size_b and (a.int() - b.int()).abs().max() <= 1
    # Batched: frames of mixed sizes, each identical to preparing it alone.
    ims = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in [(48, 80), (60, 90), (48, 80)]]
    for (a, sa), im in zip(prepare_batch(ims, device="cpu"), ims):
        b, sb = prepare(im, device="cpu", resize_on_device=True)
        assert sa == sb and torch.equal(a, b)


def test_edges(pred, tmp_path):
    _, seeds, frames = _video(tmp_path / "a", 48, 80, 2, [RED], seed=0)
    engine = Engine(pred)
    assert engine.step({"cam": frames[0]}) == {"cam": {}}               # nothing followed yet
    assert engine.step({"cam": frames[0]}, {"cam": {}}) == {"cam": {}}  # no objects to start
    with pytest.raises(KeyError):
        engine.step({}, {"cam": seeds})                                 # seeds without their frame
    engine.step({"cam": frames[0]}, {"cam": seeds})
    # A camera may change resolution: masks come back at the size of the frame they are for.
    assert engine.step({"cam": (frames[1][0], (24, 40))})["cam"][1].shape == (24, 40)
    engine.drop("cam")
    assert engine.step({"cam": frames[1]}) == {"cam": {}} and not engine.tracks
    # The predictor is shared with stock: a step must leave its modules as they were.
    for mod in (pred.sam_prompt_encoder, pred.image_encoder.neck.position_encoding):
        assert "forward" not in vars(mod)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_masks_can_stay_on_the_gpu(tmp_path):
    gpu = build_video_predictor(device="cuda")
    _, seeds, frames = _video(tmp_path / "a", 48, 80, 2, [RED], seed=0)
    a, b = Engine(gpu, gpu_masks=True), Engine(gpu)
    for i, f in enumerate(frames):
        on_gpu, on_cpu = (e.step({"cam": f}, {"cam": seeds} if i == 0 else None)["cam"][1] for e in (a, b))
        assert on_gpu.is_cuda and torch.equal(on_gpu.cpu(), on_cpu)
