import socket
import subprocess
import sys
import threading
import time
from multiprocessing import Pipe
from pathlib import Path

import numpy as np
import pytest
import torch

from tamserve import Box, Clicks, Engine, Router, build_video_predictor, wire
from tamserve.wire import Mask, Update
from test_engine import RED, YELLOW, _assert_tracked_like, _run, _video
from test_router import _collect, _count

cv2 = pytest.importorskip("cv2")
TOKEN = "a test token that is long enough"


def test_messages_travel_as_plain_data():
    a, b = Pipe()
    mask = Mask((4, 5), False, np.array([3, 7], np.int32), (3, 0, 5, 2), 4)
    wire.send(a, ("updates", [Update("door", 7, 12.5, {"courier": mask, 3: mask}),
                              Update(2, None, None, {}, error="it ended", ended=True)]))
    kind, (door, two) = wire.recv(b)
    assert kind == "updates" and door.frame == 7 and door.masks["courier"].box == (3, 0, 5, 2)
    assert np.array_equal(door.masks[3].array(), mask.array()) and two.ended and two.error == "it ended"
    for prompt in (Box(1, 2, 3.5, 4), Clicks([(1, 2), (3, 4)], [1, 0])):
        wire.send(a, ("start", "door", 1, prompt, None))
        assert wire.recv(b) == ("start", "door", 1, prompt, None)
    wire.send(a, ("start", "yard", 2, torch.eye(3, dtype=torch.bool), 5))
    got = wire.recv(b)
    assert np.array_equal(got[3], np.eye(3, dtype=bool)) and got[4] == 5
    with pytest.raises(TypeError):
        wire.send(a, ("add_camera", "door", print, True))  # code is never sent
    a.send_bytes(b'{"n": 0, "m": {"T": "posix.system", "v": ["echo hello"]}}')
    with pytest.raises(ValueError):
        wire.recv(b)  # nor rebuilt from anything but plain data and tamserve's own types


def test_the_token_is_checked_both_ways():
    for router_token, ok in ((b"k" * 32, True), (b"wrong" * 8, False)):
        a, b = Pipe()
        seen = {}

        def worker():
            try:
                wire.prove_worker(b, b"k" * 32)
                seen["worker"] = "accepted"
            except PermissionError:
                seen["worker"] = "refused"
                b.close()

        t = threading.Thread(target=worker)
        t.start()
        if ok:
            wire.prove_router(a, router_token)
        else:
            with pytest.raises(PermissionError):
                wire.prove_router(a, router_token)
        t.join()
        assert seen["worker"] == ("accepted" if ok else "refused")


def _mp4(tmp_path, name, h, w, frames, discs, seed):
    """A video file of moving discs, its seeds, and its frames as OpenCV decodes them."""
    folder, seeds, _ = _video(tmp_path / name, h, w, frames, discs, seed)
    path = str(tmp_path / f"{name}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for p in sorted(Path(folder).glob("*.jpg"), key=lambda p: int(p.stem)):
        writer.write(cv2.imread(str(p)))
    writer.release()
    capture, raw = cv2.VideoCapture(path), []
    while (frame := capture.read())[0]:
        raw.append(cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB))
    return path, seeds, raw


@pytest.fixture
def workers(tmp_path):
    """Two `tamserve worker` processes on this machine's CPU, reached over TCP as if remote."""
    token = tmp_path / "token"
    token.write_text(TOKEN)
    procs, logs, addresses = [], [], []
    for i in range(2):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        logs.append(tmp_path / f"worker{i}.log")
        procs.append(subprocess.Popen([sys.executable, "-m", "tamserve.worker", "worker", "--device", "cpu",
                                       "--listen", f"127.0.0.1:{port}", "--token-file", str(token)],
                                      stdout=logs[-1].open("w"), stderr=subprocess.STDOUT))
        addresses.append(f"127.0.0.1:{port}")
    deadline = time.time() + 120
    while not all("listening on" in log.read_text() for log in logs):
        assert time.time() < deadline and all(p.poll() is None for p in procs), [log.read_text() for log in logs]
        time.sleep(0.2)
    yield addresses
    for p in procs:
        p.terminate()
        p.wait(10)


def test_router_with_workers_on_other_machines(tmp_path, workers):
    videos = {"a": _mp4(tmp_path, "a", 96, 128, 6, [RED, YELLOW], seed=0),
              "b": _mp4(tmp_path, "b", 80, 112, 5, [RED], seed=1),
              "c": _mp4(tmp_path, "c", 96, 128, 4, [YELLOW], seed=2)}
    want = dict(zip(videos, _run(Engine(build_video_predictor("cpu")), list(videos.values()), [0, 0, 0])))
    with Router(workers=workers, token=TOKEN) as router:
        for name, (path, seeds, _) in videos.items():
            router.add_camera(name, path, skip=False)
            for obj, seed in seeds.items():
                router.start(name, obj, seed.numpy(), frame=0)
        assert set(router.home.values()) == {0, 1}  # spread over both workers
        with pytest.raises(TypeError):
            router.add_camera("d", print)  # another machine is given an address, never code
        got = _collect(router, lambda g: _count(g) == 6 * 2 + 5 + 4)
    for name, (_, seeds, raw) in videos.items():
        _assert_tracked_like(want[name], got[name], objects=len(seeds), frames=len(raw))

    # A wrong token is refused, and the workers forgot the first router: the next starts afresh.
    with pytest.raises(PermissionError):
        Router(workers=workers[:1], token="not the token they were given")
    path, seeds, raw = videos["b"]
    with Router(workers=workers[:1], token=TOKEN) as router:
        router.add_camera("b", path, skip=False)
        router.start("b", 1, seeds[1].numpy(), frame=0)
        got = _collect(router, lambda g: _count(g) == 5)
    _assert_tracked_like(want["b"], got["b"], objects=1, frames=5)
