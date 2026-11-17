"""align.visual without a network: the yaw trick, the walk filter, the fit."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import visual
from align.frames import Sim2


def _fake_embed(imgs):
    """A stand-in backbone: per-column brightness histogram, so rolling the
    image permutes the features -- the property the pooling must survive."""
    cls = imgs.mean(axis=(1, 2)) / 255.0
    patches = imgs.mean(axis=3).reshape(len(imgs), imgs.shape[1] * imgs.shape[2] // 4, 4) / 255.0
    return visual.gem_pool(cls, patches)


def test_roll_is_a_yaw_and_a_rolled_prior_matches_the_unrolled_query():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (1, 8, 16, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (1, 8, 16, 3), dtype=np.uint8)
    q = _fake_embed(a)
    pool = np.stack([_fake_embed(visual.roll_yaw(x, k, 4))[0]
                     for x in (a, b) for k in range(4)]).reshape(2, 4, -1)
    assert visual.roll_yaw(a, 2, 4).shape == a.shape
    assert np.array_equal(visual.roll_yaw(a, 4, 4), a)          # a full turn
    # the query is `a` seen a quarter turn later; the best roll of `a` wins
    q_turned = _fake_embed(visual.roll_yaw(a, 1, 4))
    idx, sims = visual.retrieve(q_turned, pool, top_k=2)
    assert idx[0, 0] == 0 and sims[0, 0] > sims[0, 1]
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)


def test_ransac_recovers_a_similarity_through_outliers():
    rng = np.random.default_rng(1)
    src = rng.uniform(-10, 10, (30, 2))
    truth = Sim2(s=0.7, theta=0.4, tx=12.0, ty=-3.0)
    dst = truth.apply(src) + rng.normal(0, 0.05, (30, 2))
    bad = rng.choice(30, 12, replace=False)
    dst[bad] += rng.uniform(-20, 20, (12, 2))
    fit = visual.ransac_sim2(src, dst, tol=0.5, bracket=(0.5, 1.0))
    sim, mask = fit
    assert abs(sim.s - 0.7) < 0.02 and abs(sim.theta - 0.4) < 0.02
    assert np.hypot(sim.tx - 12.0, sim.ty + 3.0) < 0.3
    assert mask.sum() >= 18 and not mask[bad].any()
    assert visual.ransac_sim2(src[:2], dst[:2], tol=0.5) is None
    line = np.c_[np.linspace(0, 1, 10), np.zeros(10)]
    assert visual.ransac_sim2(line, line, tol=0.5) is None
    assert visual.ransac_sim2(src, dst, tol=0.5, bracket=(1.5, 2.0)) is None


def test_consistent_anchors_keep_the_walk_and_drop_the_lone_jump():
    rng = np.random.default_rng(2)
    src = np.c_[np.arange(10.0), np.zeros(10)]           # a straight walk, 1 unit steps
    scale = 2.0
    truth = scale * src + np.array([100.0, 50.0])
    decoy = rng.uniform(-500, 500, (10, 2))               # k=2: right place, random place
    cand = np.stack([truth, decoy], axis=1)
    cand[5, 0] = [999.0, 999.0]                            # camera 5's top-1 is a lone jump
    anchors = dict(visual.consistent_anchors(src, cand, scale, radius=0.5, window=2))
    assert 5 not in anchors, "nothing on the walk corroborates the jump"
    assert all(anchors[i] == 0 for i in anchors)
    assert len(anchors) >= 8


def test_pose_from_retrieval_picks_the_view_the_cameras_agree_on():
    rng = np.random.default_rng(3)
    positions = np.c_[rng.uniform(0, 30, 40), rng.uniform(0, 12, 40), np.zeros(40)]
    truth = Sim2(s=0.5, theta=0.3, tx=40.0, ty=-5.0)
    R0 = np.eye(3)
    R1 = np.diag([1.0, -1.0, 1.0])                       # the mirrored gravity guess
    views = {0: dict(R=R0), 1: dict(R=R1)}
    dst = truth.apply(positions[:, :2])
    # the prior pool: the true places plus decoys; descriptors are one-hot so
    # retrieval is exact, with 30% of cameras retrieving a decoy
    pool_xy = np.vstack([dst, rng.uniform(-100, 100, (40, 2))])
    D = 80
    pool = np.eye(D)[:, None, :]                          # (80, 1, D)
    q = np.eye(D)[:40].copy()
    bad = rng.choice(40, 12, replace=False)
    q[bad] = np.eye(D)[40 + bad]
    fit = visual.pose_from_retrieval(views, positions, q, pool, pool_xy,
                                     brackets={0: (0.5, (0.3, 0.8)), 1: (0.5, (0.3, 0.8))},
                                     top_k=1, radius=2.0, tol=0.5)
    assert fit["up_index"] == 0 and fit["n_inliers"] == 28
    assert abs(fit["sim2"].s - 0.5) < 0.01 and abs(fit["sim2"].theta - 0.3) < 0.01
    assert fit["inlier_frac"] == 28 / 40


def test_one_unfittable_triple_does_not_take_the_whole_fit_down():
    """A NaN in the matches is one dead hypothesis, not a dead capture.

    The counts are computed for every three-point sample at once, and numpy
    raises `LinAlgError` for the WHOLE batch when one matrix will not
    decompose. `bevaligner.evaluate` catches at the CAPTURE, so one bad
    triple used to be written as `error:` on every split of that capture --
    22 of them in the run this test was written for.
    """
    rng = np.random.default_rng(3)
    src = rng.uniform(-10, 10, (40, 2))
    truth = Sim2(s=1.3, theta=-0.6, tx=-4.0, ty=7.0)
    dst = truth.apply(src)
    clean = visual.ransac_sim2(src, dst, tol=0.2)
    assert clean is not None

    for spoiled in (np.nan, np.inf):
        poisoned = dst.copy()
        poisoned[7] = spoiled                      # one match, every triple using it
        fit = visual.ransac_sim2(src, poisoned, tol=0.2)
        assert fit is not None, f"{spoiled} ended the fit"
        sim, mask = fit
        assert abs(sim.s - 1.3) < 0.02 and abs(sim.theta + 0.6) < 0.02
        assert not mask[7] and mask.sum() >= 38

    # and the clean path is untouched: identical hypotheses, identical winner
    again = visual.ransac_sim2(src, dst, tol=0.2)
    assert np.array_equal(again[1], clean[1]) and again[0].s == clean[0].s
