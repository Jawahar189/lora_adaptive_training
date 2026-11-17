"""Segment-based reranking, on synthetic input.

The point of this module is that the correlation score cannot rank: with scale
and yaw held at ground truth, the true translation sat at the 94th, 98th and
19.5th percentile of the NCC surface on three real splits. These tests check
that counting explained wall segments does what correlation could not --
prefer the pose that actually matches.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.frames import Sim2
from align.rerank import (InlierScorer, _placed_bounds, build_scorer,
                          count_inliers, rerank)
from align.search import Hypothesis


class FakeSheet:
    """A SheetFeatures stand-in: rerank only ever reads `wall`."""

    def __init__(self, wall):
        self.wall = wall
        self.ppi = 100.0


class FakeGeom:
    """paper <-> pixel for a synthetic sheet, y-flipped like the real one."""

    def __init__(self, paper_w, paper_h, wall):
        self.paper_w, self.paper_h = paper_w, paper_h
        self.img_h, self.img_w = wall.shape
        self.ppi = self.img_w / paper_w

    def paper_to_pixel(self, xy):
        p = np.asarray(xy, dtype=np.float64)
        return np.stack([p[:, 0] / self.paper_w * self.img_w,
                         (1 - p[:, 1] / self.paper_h) * self.img_h], axis=1)


def room(w=8.0, h=5.0, step=0.01):
    xs, ys = np.arange(0, w, step), np.arange(0, h, step)
    return np.concatenate([
        np.stack([xs, np.zeros_like(xs)], 1),
        np.stack([xs, np.full_like(xs, h)], 1),
        np.stack([np.zeros_like(ys), ys], 1),
        np.stack([np.full_like(ys, w), ys], 1)])


def sheet_with(points, paper=(20.0, 12.0), ppi=100):
    w, h = paper
    wall = np.zeros((int(h * ppi), int(w * ppi)), bool)
    px = (np.asarray(points)[:, 0] * ppi).astype(int)
    py = ((h - np.asarray(points)[:, 1]) * ppi).astype(int)
    ok = (px >= 0) & (px < wall.shape[1]) & (py >= 0) & (py < wall.shape[0])
    wall[py[ok], px[ok]] = True
    for dy in (-1, 0, 1):                     # walls are drawn with some width
        for dx in (-1, 0, 1):
            wall[np.clip(py[ok] + dy, 0, wall.shape[0] - 1),
                 np.clip(px[ok] + dx, 0, wall.shape[1] - 1)] = True
    return FakeSheet(wall), FakeGeom(w, h, wall)


MPI = 3.0


def test_correct_pose_has_the_most_inliers():
    cloud = room()
    feats, geom = sheet_with(cloud + np.array([4.0, 3.0]))
    good = Sim2(s=1.0, theta=0.0, tx=4.0, ty=3.0)
    bad = Sim2(s=1.0, theta=0.0, tx=12.0, ty=7.0)
    n_good, _ = count_inliers(cloud, good, feats, geom, MPI)
    n_bad, _ = count_inliers(cloud, bad, feats, geom, MPI)
    assert n_good > 0
    assert n_good > n_bad, f"correct pose {n_good} should beat displaced {n_bad}"


def test_a_rotated_pose_is_rejected():
    """The 90-degree failure the correlation kept making."""
    cloud = room()
    feats, geom = sheet_with(cloud + np.array([4.0, 3.0]))
    upright = Sim2(s=1.0, theta=0.0, tx=4.0, ty=3.0)
    turned = Sim2(s=1.0, theta=np.pi / 2, tx=9.0, ty=3.0)
    assert (count_inliers(cloud, upright, feats, geom, MPI)[0]
            > count_inliers(cloud, turned, feats, geom, MPI)[0])


def test_rerank_promotes_the_correct_pose_from_a_bad_ordering():
    """The whole point: correlation ranks it last, segments rank it first."""
    cloud = room()
    feats, geom = sheet_with(cloud + np.array([4.0, 3.0]))
    correct = Sim2(s=1.0, theta=0.0, tx=4.0, ty=3.0)
    decoys = [Sim2(s=1.0, theta=0.0, tx=4.0 + d, ty=3.0 + d)
              for d in (3.0, 5.0, 7.0)]

    def hyp(sim, score):
        return Hypothesis(sim2=sim, score=score, wall=score, free=0.0,
                          scale_prior=1.0, scale=sim.s,
                          yaw_deg=float(np.degrees(sim.theta)), viewport=0)

    # correlation puts the correct pose LAST
    hyps = [hyp(s, 1.0 - 0.01 * i) for i, s in enumerate(decoys)]
    hyps.append(hyp(correct, 0.5))
    ranked = rerank(hyps, {0: cloud}, feats, geom, MPI)
    assert ranked[0].sim2.tx == pytest.approx(4.0), (
        f"expected the correct pose first, got tx={ranked[0].sim2.tx} "
        f"with inliers {[h.inliers for h in ranked]}")
    assert ranked[0].rank == 0


def test_rerank_falls_back_to_score_when_nothing_matches():
    """An empty sheet gives every pose zero inliers; order must still be sane."""
    cloud = room()
    feats, geom = sheet_with(np.zeros((1, 2)))
    hyps = [Hypothesis(sim2=Sim2(1.0, 0.0, float(t), 0.0), score=float(s),
                       wall=0.0, free=0.0, scale_prior=1.0, scale=1.0,
                       yaw_deg=0.0, viewport=0)
            for t, s in ((2.0, 0.3), (5.0, 0.9))]
    ranked = rerank(hyps, {0: cloud}, feats, geom, MPI)
    assert all(h.inliers == 0 for h in ranked)
    assert ranked[0].score == pytest.approx(0.9), "ties fall back to score"


# -- InlierScorer: the same measure, prepared once -------------------------
#
# A speedup that changes the answer is a different algorithm, so these pin the
# prepared scorer to `count_inliers`, which stays the definition.


def _poses():
    return [Sim2(s=s, theta=np.radians(a), tx=4.0 + dx, ty=3.0 + dy)
            for s, a, dx, dy in [(1.0, 0, 0, 0), (1.0, 0, 0.4, -0.3),
                                 (1.0, 0, 2.0, 1.0), (1.0, 12, 0, 0),
                                 (1.1, 0, 0, 0), (0.9, 5, -1.0, 0.5),
                                 (1.0, 90, 5.0, -1.5)]]


def test_scorer_on_the_references_own_grid_is_exact():
    """Prepared over one pose's bounds, the scorer IS count_inliers.

    Nothing is approximated here -- same resolution, same origin, same shape --
    so any difference at all is an arithmetic bug, not a trade-off. This is
    `segment_mode="pose"`, which reproduces the reference by construction; the
    shipped default is "cloud", checked separately below.
    """
    cloud = room()
    feats, geom = sheet_with(cloud + np.array([4.0, 3.0]))
    for sim in _poses():
        bounds = _placed_bounds(sim.apply(cloud))
        scorer = InlierScorer(cloud, feats, geom, MPI, bounds,
                              segment_mode="pose")
        assert scorer.score(sim) == count_inliers(cloud, sim, feats, geom, MPI), (
            f"pose s={sim.s} yaw={np.degrees(sim.theta):.0f} "
            f"t=({sim.tx}, {sim.ty}) diverges on its own grid")


def test_scorer_on_a_shared_grid_matches_within_a_segment_or_two():
    """Prepared over ALL the poses, it may differ by a segment or two.

    One grid cannot share every pose's origin, so plan line-work lands on
    slightly different cells and segments sitting exactly on the tolerance can
    fall either way. Measured on real sheets: 60% identical, none off by more
    than 2. Anything larger means the grid is misregistered, which is what the
    padding in `sheet.resample_to` exists to prevent.
    """
    cloud = room()
    feats, geom = sheet_with(cloud + np.array([4.0, 3.0]))
    poses = _poses()

    class H:
        def __init__(self, sim):
            self.sim2 = sim
    scorer = build_scorer([H(s) for s in poses], cloud, feats, geom, MPI,
                          segment_mode="pose")

    for sim in poses:
        ref = count_inliers(cloud, sim, feats, geom, MPI)[0]
        got = scorer.score(sim)[0]
        assert abs(got - ref) <= 2, (
            f"pose yaw={np.degrees(sim.theta):.0f} t=({sim.tx}, {sim.ty}): "
            f"scorer {got} vs reference {ref}")


def test_segments_depend_only_on_yaw_and_scale():
    """Why one LSD run serves every translation.

    `count_inliers` rasterizes over bounds derived from the placed cloud, so a
    translation moves the raster origin by exactly as much as the points and
    leaves the mask -- and LSD's output -- untouched. The cache key is built on
    this; if it stops holding, the cache silently returns the wrong segments.
    """
    cloud = room()
    feats, geom = sheet_with(cloud + np.array([4.0, 3.0]))
    base = Sim2(s=1.0, theta=np.radians(17), tx=4.0, ty=3.0)
    scorer = InlierScorer(cloud, feats, geom, MPI,
                          _placed_bounds(base.apply(cloud)),
                          segment_mode="pose")
    segs = scorer.segments_for(base)
    for dx, dy in [(3.0, -2.0), (-1.5, 0.25), (7.0, 7.0)]:
        moved = Sim2(s=base.s, theta=base.theta, tx=base.tx + dx, ty=base.ty + dy)
        assert np.array_equal(scorer.segments_for(moved), segs)
    # ...and a different yaw really does produce a different set
    turned = Sim2(s=1.0, theta=np.radians(40), tx=4.0, ty=3.0)
    assert not np.array_equal(scorer.segments_for(turned), segs)


def test_score_all_matches_scoring_one_at_a_time():
    """The vectorised path and the single-pose path are the same measure."""
    cloud = room()
    feats, geom = sheet_with(cloud + np.array([4.0, 3.0]))

    def hyp(sim):
        return Hypothesis(sim2=sim, score=0.0, wall=0.0, free=0.0,
                          scale_prior=1.0, scale=sim.s,
                          yaw_deg=float(np.degrees(sim.theta)), viewport=0)

    hyps = [hyp(s) for s in _poses()]
    scorer = build_scorer(hyps, cloud, feats, geom, MPI)
    one = [scorer.score(h.sim2) for h in hyps]
    scorer.score_all(hyps, chunk=2)            # chunked, to exercise the split
    assert [(h.inliers, h.n_segments) for h in hyps] == one


def test_resample_to_pads_rather_than_stretches():
    """Bounds hanging off the sheet must not rescale the plan to fit.

    A cloud placed near an edge asks for paper that is not there. Stretching
    the clipped crop over the full grid misregisters the plan by however much
    was missing -- which is exactly what made the shared grid disagree with the
    reference by up to 27 segments before this was fixed.
    """
    from align.sheet import resample_to

    cloud = room()
    feats, geom = sheet_with(cloud + np.array([4.0, 3.0]))
    res = 0.02
    # A window entirely inside the sheet, then the same window with blank paper
    # appended on the left. The wall must not move within the shared part.
    inside = (4.0, 3.0, 12.0, 8.0)
    grown = (0.0, 3.0, 12.0, 8.0)
    a = resample_to(feats, geom, inside, res,
                    (int(np.ceil(5.0 / res)), int(np.ceil(8.0 / res))))
    b = resample_to(feats, geom, grown, res,
                    (int(np.ceil(5.0 / res)), int(np.ceil(12.0 / res))))
    off = int(round(4.0 / res))
    overlap = b[:, off:off + a.shape[1]]
    agree = (a == overlap).mean()
    assert agree > 0.99, f"window grew and the plan shifted ({agree:.3f} agree)"


def test_cloud_mode_runs_one_detection_and_still_prefers_the_correct_pose():
    """The shipped default: one segment set for the split, carried to each pose.

    It cannot be checked against `count_inliers` pose by pose -- LSD on a
    rotated raster is genuinely not a rotation of LSD on an unrotated one --
    so what is pinned here is that it detects ONCE and that the ranking it
    produces still separates a correct pose from a displaced one.
    """
    cloud = room()
    feats, geom = sheet_with(cloud + np.array([4.0, 3.0]))
    correct = Sim2(s=1.0, theta=0.0, tx=4.0, ty=3.0)
    wrong = Sim2(s=1.0, theta=0.0, tx=11.0, ty=7.0)

    class H:
        def __init__(self, sim):
            self.sim2 = sim

    scorer = build_scorer([H(correct), H(wrong)], cloud, feats, geom, MPI,
                          segment_mode="cloud")
    good, n = scorer.score(correct)
    bad, _ = scorer.score(wrong)
    assert len(scorer._cache) == 1, "cloud mode must detect segments once"
    assert n > 0 and good > bad, f"correct {good} should beat displaced {bad}"
    # ...and the count is stable across poses, which is the property "pose"
    # mode lacks: there the denominator moves with raster aliasing.
    assert scorer.score(wrong)[1] == n
