import numpy as np

from align.frames import Sim2
from bevaligner.inference import fit_planar_pose


def test_fit_planar_pose_recovers_similarity_with_outliers():
    rng = np.random.default_rng(0)
    source = rng.uniform(-10, 10, size=(200, 2))
    truth = Sim2(s=2.1, theta=0.9, tx=30.0, ty=-12.0)
    target = truth.apply(source) + rng.normal(scale=0.02, size=source.shape)
    outliers = rng.random(200) < 0.3
    target[outliers] = rng.uniform(-50, 50, size=(outliers.sum(), 2))
    metres_per_inch = 2.5
    fit = fit_planar_pose(source, target, metres_per_inch, tolerance_m=0.5,
                          num_iterations=1000)
    assert fit is not None
    sim2, inlier_mask = fit
    assert abs(sim2.s - truth.s) < 0.02 and abs(sim2.theta - truth.theta) < 0.01
    assert np.allclose([sim2.tx, sim2.ty], [truth.tx, truth.ty], atol=0.1)
    assert inlier_mask.sum() >= (~outliers).sum() * 0.95


def test_fit_planar_pose_rejects_a_mirror():
    rng = np.random.default_rng(1)
    source = rng.uniform(-10, 10, size=(100, 2))
    truth = Sim2(s=1.0, theta=0.3, tx=0.0, ty=0.0)
    mirrored = truth.apply(source) * [1, -1]
    fit = fit_planar_pose(source, mirrored, 1.0, tolerance_m=0.2, num_iterations=500,
                          min_inliers=12)
    assert fit is None or fit[1].sum() < 20
