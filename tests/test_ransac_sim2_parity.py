"""`ransac_sim2` was vectorised; these pin it bit-identical to the loop it replaced.

The reference below is the original implementation, verbatim, kept here because
that is the only thing the new one has to agree with. It is not a second way of
solving the problem -- if it drifts from what `align/visual.py` used to do the
test is worthless -- so it is copied, not re-derived.

Bit-identical rather than close: the counts decide WHICH hypothesis wins, and a
single flipped inlier picks a different pose, not a slightly different one.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from align.chain import _umeyama_sim2
from align.visual import ransac_sim2


def reference_ransac_sim2(src, dst, tol, bracket=None, iters=500, seed=0):
    """`align.visual.ransac_sim2` as it stood before the vectorisation."""
    src = np.asarray(src, dtype=np.float64)[:, :2]
    dst = np.asarray(dst, dtype=np.float64)[:, :2]
    n = len(src)
    if n < 3 or np.linalg.matrix_rank(src - src.mean(0)) < 2:
        return None
    rng = np.random.default_rng(seed)
    best_mask, best_n = None, 0
    for _ in range(iters):
        pick = rng.choice(n, 3, replace=False)
        if np.linalg.matrix_rank(src[pick] - src[pick].mean(0)) < 2:
            continue
        sim = _umeyama_sim2(src[pick], dst[pick])
        mask = np.linalg.norm(sim.apply(src) - dst, axis=1) <= tol
        if mask.sum() > best_n:
            best_mask, best_n = mask, int(mask.sum())
            if best_n == n:
                break
    if best_mask is None or best_n < 3:
        return None
    sim = _umeyama_sim2(src[best_mask], dst[best_mask])
    mask = np.linalg.norm(sim.apply(src) - dst, axis=1) <= tol
    if mask.sum() < 3:
        return None
    sim = _umeyama_sim2(src[mask], dst[mask])
    if bracket is not None and not (bracket[0] <= sim.s <= bracket[1]):
        return None
    return sim, mask


def make_pair(n, outlier_fraction, seed, *, degenerate=False, collinear=False):
    """A similarity with a given share of its correspondences broken."""
    rng = np.random.default_rng(seed)
    if collinear:
        line = rng.normal(size=n)
        src = np.stack([line, 2 * line + 1], axis=1)
    else:
        src = rng.normal(size=(n, 2)) * 10
    if degenerate:
        src[:] = src[0]
    theta, scale = rng.uniform(-np.pi, np.pi), float(np.exp(rng.normal() * 0.3))
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    dst = (src @ rotation.T) * scale + rng.normal(size=2) * 5
    broken = int(n * outlier_fraction)
    if broken:
        dst[:broken] += rng.normal(size=(broken, 2)) * 50
    return src, dst


def assert_identical(got, want):
    assert (got is None) == (want is None)
    if want is None:
        return
    (sim, mask), (reference_sim, reference_mask) = got, want
    assert np.array_equal(mask, reference_mask)
    assert (sim.s, sim.theta, sim.tx, sim.ty) == (reference_sim.s, reference_sim.theta,
                                                  reference_sim.tx, reference_sim.ty)


# n = 3 is the minimum a hypothesis can be drawn from; 80 is what the keypoint
# matcher leaves per window and 512 what the dense one does.
@pytest.mark.parametrize("n", [3, 4, 5, 12, 20, 80, 200, 512])
@pytest.mark.parametrize("outlier_fraction", [0.0, 0.2, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_the_loop(n, outlier_fraction, seed):
    src, dst = make_pair(n, outlier_fraction, seed)
    for tolerance, bracket in itertools.product([0.5, 5.0], [None, (0.5, 2.0)]):
        assert_identical(ransac_sim2(src, dst, tolerance, bracket=bracket, iters=300, seed=0),
                         reference_ransac_sim2(src, dst, tolerance, bracket=bracket,
                                               iters=300, seed=0))


@pytest.mark.parametrize("n", [3, 5, 40])
@pytest.mark.parametrize("kind", ["degenerate", "collinear"])
def test_matches_the_loop_on_rank_deficient_input(n, kind):
    """Every triple degenerate: the loop `continue`d, the vectorised form counts -1."""
    src, dst = make_pair(n, 0.3, 0, degenerate=(kind == "degenerate"),
                         collinear=(kind == "collinear"))
    assert_identical(ransac_sim2(src, dst, 0.5, iters=300, seed=0),
                     reference_ransac_sim2(src, dst, 0.5, iters=300, seed=0))


def test_matches_the_loop_across_chunk_boundaries():
    """2000 iterations is what inference runs, and spans several chunks."""
    src, dst = make_pair(200, 0.4, 5)
    assert_identical(ransac_sim2(src, dst, 0.5, iters=2000, seed=0),
                     reference_ransac_sim2(src, dst, 0.5, iters=2000, seed=0))


def test_chunking_does_not_change_the_winner():
    """A budget small enough to force one hypothesis per chunk must not move it."""
    from align.visual import _sim2_hypothesis_counts
    src, dst = make_pair(120, 0.4, 3)
    rng = np.random.default_rng(0)
    picks = np.array([rng.choice(len(src), 3, replace=False) for _ in range(500)])
    whole = _sim2_hypothesis_counts(src, dst, picks, 0.5)
    split = _sim2_hypothesis_counts(src, dst, picks, 0.5, budget=1)
    assert np.array_equal(whole, split)


def test_cached_triplets_are_the_sequential_draw():
    """The cache must hand back the sequence, not merely a valid sample.

    The hypotheses are what the search explores, so a different sequence is a
    different search -- it would still return a pose, which is exactly what
    makes it worth pinning.
    """
    from align.visual import _triplet_draws
    for n in (12, 80, 512):
        rng = np.random.default_rng(0)
        expected = np.array([rng.choice(n, 3, replace=False) for _ in range(2000)])
        assert np.array_equal(_triplet_draws(n, 2000, 0), expected)


def test_cached_triplets_are_not_writable():
    """Shared between every window of every capture -- nothing may write through it."""
    from align.visual import _triplet_draws
    with pytest.raises(ValueError):
        _triplet_draws(40, 50, 0)[0, 0] = 1


def test_a_second_call_reuses_the_same_draw():
    from align.visual import _triplet_draws
    assert _triplet_draws(64, 100, 0) is _triplet_draws(64, 100, 0)
    assert _triplet_draws(64, 100, 1) is not _triplet_draws(64, 100, 0)
