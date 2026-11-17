"""Coarse search, on synthetic input where the answer is known exactly.

None of this touches the dataset: a known shape is placed into a known map with
a known similarity, and the search must find it back. That pins the index and
translation conventions, which is where this kind of code goes quietly wrong.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.cloud import manhattan
from align.search import (SearchGrid, aspect_agreement, free_space_map,
                          get_backend, masked_ncc, peaks, scale_candidates,
                          scale_plausibility, search, yaw_candidates)

RES = 0.05


def room_outline(w=6.0, h=4.0, step=0.02):
    """Points on the outline of a w x h rectangular room."""
    xs = np.arange(0, w, step)
    ys = np.arange(0, h, step)
    return np.concatenate([
        np.stack([xs, np.zeros_like(xs)], 1),
        np.stack([xs, np.full_like(xs, h)], 1),
        np.stack([np.zeros_like(ys), ys], 1),
        np.stack([np.full_like(ys, w), ys], 1)])


def stamp(points, origin, shape, res=RES):
    m = np.zeros(shape, bool)
    idx = np.floor((points - np.asarray(origin)) / res).astype(int)
    ok = ((idx[:, 0] >= 0) & (idx[:, 0] < shape[1])
          & (idx[:, 1] >= 0) & (idx[:, 1] < shape[0]))
    m[idx[ok, 1], idx[ok, 0]] = True
    return m


@pytest.fixture(params=["numpy", "auto"])
def backend(request):
    return get_backend(request.param)


def test_peaks_are_suppressed_locally():
    s = np.zeros((20, 20))
    s[5, 5] = 10.0
    s[5, 6] = 9.0          # adjacent, must be suppressed
    s[15, 15] = 8.0
    got = peaks(s, k=2, radius=3)
    assert [(p[1], p[2]) for p in got] == [(5, 5), (15, 15)]


def test_translation_is_recovered_exactly(backend):
    """The index convention: a known offset must come back as that offset."""
    sheet_origin = (0.0, 0.0)
    shape = (240, 360)
    room = room_outline()
    offset = np.array([4.25, 2.70])
    sheet_map = stamp(room + offset, sheet_origin, shape).astype(np.float32)

    grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([0.0]),
                      viewports=[0])
    hyps = search(room, room[::40], sheet_map, sheet_origin, RES, grid,
                  backend=backend, w_free=0.0, w_scale=0.0, top_k=1)
    assert hyps, "search returned nothing"
    best = hyps[0]
    assert abs(best.sim2.tx - offset[0]) <= RES
    assert abs(best.sim2.ty - offset[1]) <= RES


def test_ncc_alone_is_scale_biased_and_the_prior_corrects_it():
    """Documents why W_SCALE is 1.0. If someone lowers it, this fails."""
    from align.search import W_SCALE
    room = room_outline()
    true_s = 1.30
    a = np.radians(12.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    sheet = stamp((room @ R.T) * true_s + np.array([6.0, 3.0]), (0.0, 0.0),
                  (300, 420)).astype(np.float32)
    bk = get_backend("numpy")
    rows = []
    for s in scale_candidates((1.0, 1.6), steps=7):
        g = SearchGrid(scales=np.array([s]), yaws_deg=np.array([12.0]),
                       viewports=[0])
        h = search(room, room[::40], sheet, (0.0, 0.0), RES, g, backend=bk,
                   w_free=0.0, w_scale=0.0, top_k=1)[0]
        rows.append((s, h.wall, scale_plausibility(s, true_s)))

    best_ncc = max(rows, key=lambda r: r[1])[0]
    best_both = max(rows, key=lambda r: r[1] + W_SCALE * r[2])[0]
    assert best_ncc < true_s * 0.92, "expected NCC to be biased small"
    assert abs(np.log(best_both / true_s)) < 0.12, (
        f"W_SCALE={W_SCALE} does not correct the bias; best={best_both}")


def test_scale_plausibility_peaks_at_the_prior():
    assert scale_plausibility(1.5, 1.5) == pytest.approx(1.0)
    assert scale_plausibility(1.5 * np.exp(0.18), 1.5) == pytest.approx(
        np.exp(-0.5), rel=1e-6)
    assert scale_plausibility(3.0, 1.5) < scale_plausibility(1.8, 1.5)
    assert scale_plausibility(0.0, 1.5) == 0.0


def test_scale_and_yaw_are_recovered(backend):
    sheet_origin = (0.0, 0.0)
    shape = (300, 420)
    room = room_outline()
    true_s, true_yaw = 1.30, 12.0
    a = np.radians(true_yaw)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    placed = (room @ R.T) * true_s + np.array([6.0, 3.0])
    sheet_map = stamp(placed, sheet_origin, shape).astype(np.float32)

    grid = SearchGrid(scales=scale_candidates((1.0, 1.6), steps=7),
                      yaws_deg=np.arange(6.0, 19.0, 1.5), viewports=[0])
    # The scale-prior term is what corrects NCC's bias toward small templates,
    # so it has to be on: testing the wall term alone would assert a property
    # the score deliberately does not have.
    hyps = search(room, room[::40], sheet_map, sheet_origin, RES, grid,
                  backend=backend, w_free=0.0, scale_prior=true_s, top_k=1)
    best = hyps[0]
    assert abs(np.log(best.scale / true_s)) < 0.12
    assert abs(best.yaw_deg - true_yaw) <= 2.0


def test_masked_ncc_ignores_a_dense_ink_block(backend):
    """Plain correlation puts the peak on the densest region; NCC must not.

    This is the title-block failure in miniature -- it cost ~25 inches of error
    before normalisation was added.
    """
    shape = (200, 300)
    room = room_outline(w=4.0, h=3.0)
    sheet = stamp(room + np.array([1.0, 1.0]), (0.0, 0.0), shape).astype(np.float32)
    sheet[20:120, 200:290] = 1.0          # a solid block of "ink"

    grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([0.0]),
                      viewports=[0])
    hyps = search(room, room[::40], sheet, (0.0, 0.0), RES, grid,
                  backend=backend, w_free=0.0, w_scale=0.0, top_k=1)
    assert abs(hyps[0].sim2.tx - 1.0) <= 2 * RES
    assert abs(hyps[0].sim2.ty - 1.0) <= 2 * RES


def test_free_space_term_prefers_the_true_rotation(backend):
    """The 90-degree degeneracy, and the term that breaks it.

    A non-square room with a walk down its long axis: rotating by 90 degrees
    puts the walk through the walls, which the free-space term sees and the
    wall-overlap term does not.
    """
    shape = (300, 400)
    room = room_outline(w=8.0, h=3.0)
    sheet = stamp(room + np.array([3.0, 3.0]), (0.0, 0.0), shape).astype(np.float32)

    walk = np.stack([np.linspace(0.6, 7.4, 120), np.full(120, 1.5)], 1)
    bk = backend
    scores = {}
    for yaw in (0.0, 90.0):
        a = np.radians(yaw)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        lo = (room @ R.T).min(0)
        tj = stamp(walk @ R.T, lo, (int(6 / RES), int(12 / RES)))
        f = free_space_map(bk, bk.asarray(sheet),
                           bk.asarray(tj.astype(np.float32)),
                           (sheet.shape[0] + tj.shape[0],
                            sheet.shape[1] + tj.shape[1]))
        scores[yaw] = float(bk.to_numpy(f).min())
    assert scores[0.0] < scores[90.0], (
        f"walk-on-wall should be lowest at the true rotation, got {scores}")


def test_aspect_agreement_penalises_a_90_degree_flip():
    """The raw function, pinned in isolation regardless of how `search` uses it.

    `search` never calls this against the WHOLE sheet -- see the tests below --
    it crops a LOCAL window under each candidate placement first, because a
    real reference commonly spans far more than one capture's own footprint,
    and a global aspect ratio would not discriminate a locally-wrong rotation
    (that was the previous state of this function: exported and correct, but
    unwired, for exactly that reason).
    """
    room = room_outline(w=8.0, h=3.0)
    sheet = stamp(room, (0.0, 0.0), (int(4 / RES), int(9 / RES))).astype(np.float32)
    rotated = room[:, ::-1]                      # swap axes = rotate 90 degrees
    assert aspect_agreement(room, sheet, RES) > aspect_agreement(rotated, sheet, RES)


def test_aspect_agreement_penalises_wrong_scale():
    room = room_outline(w=8.0, h=3.0)
    sheet = stamp(room, (0.0, 0.0), (int(4 / RES), int(9 / RES))).astype(np.float32)
    assert aspect_agreement(room, sheet, RES) > aspect_agreement(room * 1.8, sheet, RES)


def test_search_aspect_favours_the_true_orientation(backend):
    """The LOCAL aspect term, through `search` itself rather than the bare
    function: the true orientation's own top hypothesis should score a higher
    `.aspect` than the 90-degree-rotated one's, at each one's own best peak."""
    room = room_outline(w=8.0, h=3.0)
    shape = (300, 400)
    sheet = stamp(room + np.array([5.0, 5.0]), (0.0, 0.0), shape).astype(np.float32)

    true_grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([0.0]),
                           viewports=[0])
    flipped_grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([90.0]),
                              viewports=[0])
    kw = dict(backend=backend, w_free=0.0, w_scale=0.0, w_aspect=0.0, top_k=1)
    true_hyp = search(room, room[::200], sheet, (0.0, 0.0), RES, true_grid, **kw)[0]
    flipped_hyp = search(room, room[::200], sheet, (0.0, 0.0), RES, flipped_grid, **kw)[0]
    assert true_hyp.aspect > flipped_hyp.aspect


def test_search_aspect_is_local_not_global(backend):
    """Adding it as the third score term must not resurrect the whole-sheet
    problem: a distant, differently-shaped blob of unrelated ink -- reachable
    only through a GLOBAL bounding box, never through a crop the size of this
    capture's own footprint -- must not move the aspect score at all."""
    room = room_outline(w=8.0, h=3.0)
    shape = (300, 400)
    near = stamp(room + np.array([5.0, 5.0]), (0.0, 0.0), shape).astype(np.float32)

    far = near.copy()
    far[250:295, 5:6] = 1.0            # a tall, thin, far-away, unrelated mark

    grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([0.0]), viewports=[0])
    kw = dict(backend=backend, w_free=0.0, w_scale=0.0, w_aspect=0.0, top_k=1)
    a = search(room, room[::200], near, (0.0, 0.0), RES, grid, **kw)[0]
    b = search(room, room[::200], far, (0.0, 0.0), RES, grid, **kw)[0]
    assert a.sim2.tx == pytest.approx(b.sim2.tx)
    assert a.sim2.ty == pytest.approx(b.sim2.ty)
    assert a.aspect == pytest.approx(b.aspect, abs=1e-6), (
        "a distant mark changed the LOCAL aspect score -- it is reading the "
        "whole sheet's extent, not the crop under this placement")


def test_yaw_candidates_cover_four_rotations_and_fall_back():
    yaws, fallback = yaw_candidates(0.0, 0.0, window_deg=6.0, step_deg=3.0)
    assert not fallback
    assert len({round(y % 90.0, 3) for y in yaws}) == len(yaws) // 4
    _, fallback = yaw_candidates(0.0, 0.0, strength=0.0)
    assert fallback, "a directionless map must trigger the full sweep"


def test_backends_agree():
    """torch and numpy must produce the same score map."""
    pytest.importorskip("torch")
    shape = (160, 220)
    room = room_outline(w=4.0, h=3.0)
    sheet = stamp(room + np.array([1.5, 1.0]), (0.0, 0.0), shape).astype(np.float32)
    grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([0.0, 5.0]),
                      viewports=[0])
    a = search(room, room[::40], sheet, (0.0, 0.0), RES, grid,
               backend=get_backend("numpy"), top_k=3)
    b = search(room, room[::40], sheet, (0.0, 0.0), RES, grid,
               backend=get_backend("auto"), top_k=3)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x.score - y.score) < 1e-3
        assert abs(x.sim2.tx - y.sim2.tx) < 1e-6
