import torch

from tamserve.compare import compare


def _mask(*boxes):
    m = torch.zeros(8, 8, dtype=torch.bool)
    for y0, y1, x0, x1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


def test_identical_runs_agree():
    ref = {0: {1: _mask((0, 4, 0, 4))}, 1: {1: _mask((1, 5, 1, 5))}}
    r = compare(ref, ref)
    assert (r["mean_iou"], r["lost"], r["phantom"]) == (1.0, 0, 0)


def test_lost_and_phantom_are_counted():
    ref = {0: {1: _mask((0, 4, 0, 4)), 2: _mask()}}
    cand = {0: {1: _mask(), 2: _mask((0, 2, 0, 2))}}
    r = compare(ref, cand)
    assert (r["lost"], r["phantom"], r["min_iou"]) == (1, 1, 0.0)


def test_missing_object_counts_as_lost():
    r = compare({0: {1: _mask((0, 4, 0, 4))}}, {0: {}})
    assert r["lost"] == 1


def test_partial_overlap():
    r = compare({0: {1: _mask((0, 4, 0, 4))}}, {0: {1: _mask((0, 4, 0, 2))}})
    assert r["mean_iou"] == 0.5


def test_one_sided_frames_and_objects_are_counted():
    ref = {0: {1: _mask((0, 4, 0, 4))}, 1: {1: _mask()}}
    cand = {0: {1: _mask((0, 4, 0, 4)), 2: _mask((4, 8, 4, 8))}}
    r = compare(ref, cand)
    assert (r["missing"], r["extra"], r["phantom"], r["object_frames"]) == (1, 1, 1, 1)
