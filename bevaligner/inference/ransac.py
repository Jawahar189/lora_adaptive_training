"""The planar similarity most correspondences agree on.

`align.visual.ransac_sim2` fits three-point Umeyama hypotheses with a proper
rotation, so a mirrored set of matches -- the wrong up sign -- cannot be
explained by any hypothesis and scores few inliers. That is the mechanism by
which the up-sign choice is made downstream, by inlier count.
"""
from __future__ import annotations

import numpy as np

from ..utils import geometry


def fit_planar_pose(source_world_xy: np.ndarray, target_paper_xy: np.ndarray,
                    metres_per_inch: float, tolerance_m: float = 0.5,
                    num_iterations: int = 2000, seed: int = 0, min_inliers: int = 12,
                    scale_bracket=None, tolerance_inches: float | None = None):
    """Sim2 taking gravity-up cloud xy onto paper inches, or None.

    Returns (Sim2, inlier mask). The tolerance is given in metres and converted
    with the sheet scale so it means the same thing on every structure;
    `tolerance_inches` overrides it when the caller knows the matcher's own
    resolution is coarser than that.
    """
    if len(source_world_xy) < max(3, min_inliers):
        return None
    if tolerance_inches is None:
        tolerance_inches = tolerance_m / metres_per_inch
    fit = geometry.ransac_sim2(np.asarray(source_world_xy), np.asarray(target_paper_xy),
                               tolerance_inches, bracket=scale_bracket,
                               iters=num_iterations, seed=seed)
    if fit is None:
        return None
    sim2, inlier_mask = fit
    if inlier_mask.sum() < min_inliers or not np.isfinite(sim2.s) or sim2.s <= 0:
        return None
    return sim2, inlier_mask
