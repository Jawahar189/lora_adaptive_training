"""Throw away the correspondences no single planar motion can explain.

The matcher proposes a point for every query and is never asked whether it
believes any of them. `matching.py` filters twice -- the top half by softmax
peak, then a soft mutual-nearest-neighbour check -- and both read one match at a
time. Neither asks the question that matters: do these matches *agree with each
other*? A window of the prior that happens to be feature-rich collects a crowd
of individually-confident matches pointing in unrelated directions, and the only
thing standing between that crowd and a winning inlier count is
`align.visual.ransac_sim2` -- three points, 2000 fixed iterations, no local
optimisation, scored by raw inlier count.

So COLMAP's LO-RANSAC is put in FRONT of that fit. It re-estimates on the inlier
set every time it improves a model, stops adaptively on a confidence target
rather than a fixed trial count, and is the estimator a decade of structure
from motion has been debugged against. What comes back is only ever a subset:
the pose is still the Sim2 the existing fitter produces.

WHY THIS IS A PRE-FILTER AND NOT A REPLACEMENT. Both pictures are orthographic
rasters of the same horizontal plane, so the true mapping between them has four
degrees of freedom -- scale, rotation, two translations. COLMAP's 2D estimator
fits six, and the two it adds are exactly the two that must not be free:

    source -> mirrored source, 200 points, no noise
    estimate_affine2d_robust -> 200/200 inliers, det = -6.25

A reflection is a perfect fit. But `ransac_sim2` rejects mirrored matches by
construction -- "a mirrored set of matches, the wrong up sign, cannot be
explained by any hypothesis and scores few inliers. That is the mechanism by
which the up-sign choice is made downstream, by inlier count." Handing the
gravity up-sign decision to an estimator that cannot see a reflection would
break it silently, on captures that already have no other way to be told apart.

Hence `_similarity_gate`: the fitted linear part must be a scaled rotation, up
to `max_anisotropy`. A reflection (det <= 0) or a shear is evidence the
estimator explained the scene with freedom the geometry does not have, so its
inlier set is not trustworthy.

ABSTAIN, DO NOT VETO. When the gate fails, this returns None -- "no opinion,
use every match" -- and never an all-False mask. The difference is not
cosmetic. A window's matches are mostly outliers until the right window comes
along, so a robust fit over them can return anything; measured on a capture
that places at 0.55 m with verification off, the coarse window's affine came
back at anisotropy 1.337. Vetoing that window left the capture with no pose at
all. Twelve of seventeen captures were lost that way, three of them good ones.

Abstaining is safe precisely because this is a pre-filter: whatever it declines
to judge goes to `ransac_sim2` exactly as it did before this module existed,
and that is the estimator which rejects a mirror by construction. Verification
may remove the matches it can positively explain away. It may not decide that a
window is hopeless -- the Sim2 fit is the judge, and it is better at it.

MEASURED: "homography" abstains on essentially every window, so it is a no-op.
Its eight degrees of freedom are unconstrained by a walk's narrow band of
points, and the projective part runs away -- anisotropy 5.3 to 26.9 and a
denominator swinging by 0.16 to 2.11 across the data on the windows sampled.
That is the gate working, not failing: a homography is the wrong model for two
orthographic rasters of one plane. The mode is kept because the measurement is
worth being able to repeat (`scripts/e10_colmap_verification.py`), not because
it is a candidate for the default.

pycolmap 4.2.1 returns a dict or None. Keys, confirmed against the installed
build rather than the published docs, which do not state them:

    estimate_affine2d_robust  -> {"tgt_from_src": (2, 3), "num_inliers": int,
                                  "inlier_mask": (m,) bool}
    estimate_homography_matrix-> {"H": (3, 3), "num_inliers": int,
                                  "inlier_mask": (m,) bool}
"""
from __future__ import annotations

import numpy as np

try:
    import pycolmap
    HAVE_PYCOLMAP = True
except ImportError:                                  # pragma: no cover
    pycolmap = None
    HAVE_PYCOLMAP = False

MODES = ("none", "affine", "homography")

# A scaled rotation has both singular values equal. This is how far apart they
# may drift before the fit is called a shear: 15%, which is far looser than
# raster aliasing can produce and far tighter than the 1.56 a 0.45 shear gives.
MAX_ANISOTROPY = 1.15

# A homography between two orthographic rasters is an affine one, so its bottom
# row is (0, 0, 1). `||(h20, h21)|| * source extent` is how much the projective
# denominator varies from one side of the data to the other -- dimensionless, so
# one threshold covers every window size. 5% is slack for fitting noise and far
# below what a genuinely projective fit produces.
MAX_PERSPECTIVE = 0.05


def _ransac_options(tolerance: float, seed: int, min_trials: int, max_trials: int):
    """COLMAP's RANSAC knobs, with `max_error` in the TARGET's own units.

    `estimate_affine2d_robust` estimates `tgt_from_src`, so its residual is
    measured in the target frame -- paper inches here, which is the frame
    `propose._match_and_fit` already computes its tolerance in. Nothing needs
    converting; the number means the same thing to both estimators.
    """
    return pycolmap.RANSACOptions(
        max_error=float(tolerance), min_inlier_ratio=0.01, confidence=0.9999,
        min_num_trials=int(min_trials), max_num_trials=int(max_trials),
        random_seed=int(seed))


def _similarity_gate(linear: np.ndarray, max_anisotropy: float) -> bool:
    """Is this 2x2 a scaled rotation -- proper, and the same scale both ways?"""
    if not np.all(np.isfinite(linear)):
        return False
    if np.linalg.det(linear) <= 0:            # a reflection: the wrong up sign
        return False
    high, low = np.linalg.svd(linear, compute_uv=False)
    if low <= 1e-12:                          # degenerate: a line, not a plane
        return False
    return bool(high / low <= max_anisotropy)


def verify_matches(source_xy, target_xy, tolerance: float, *, mode: str = "affine",
                   max_anisotropy: float = MAX_ANISOTROPY, seed: int = 0,
                   min_trials: int = 100, max_trials: int = 10000):
    """Which of these correspondences one planar motion can explain.

    Returns a boolean mask over the inputs, or None when no verification was
    performed -- `mode="none"`, pycolmap missing, or too few points to fit.
    None and an all-True mask are deliberately different: the caller reports
    "not verified" rather than "verified, everything passed".

    It never returns an all-False mask. A reflection, a shear, a runaway
    projective fit or an estimator that fails outright all mean the same thing
    -- no model worth filtering by -- and all abstain, leaving the Sim2 fit the
    full match set it would have had anyway.
    """
    if mode == "none" or not HAVE_PYCOLMAP:
        return None
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    source = np.ascontiguousarray(np.asarray(source_xy, dtype=np.float64)[:, :2])
    target = np.ascontiguousarray(np.asarray(target_xy, dtype=np.float64)[:, :2])
    if len(source) != len(target):
        raise ValueError(f"{len(source)} source points against {len(target)} target")
    # Three points fix a similarity and four a homography; below that the
    # estimator cannot be wrong, so there is nothing to verify.
    if len(source) < (4 if mode == "homography" else 3):
        return None
    if not (np.all(np.isfinite(source)) and np.all(np.isfinite(target))):
        return None

    options = _ransac_options(tolerance, seed, min_trials, max_trials)
    if mode == "affine":
        result = pycolmap.estimate_affine2d_robust(source, target, options)
        linear = None if result is None else np.asarray(result["tgt_from_src"])[:, :2]
    else:
        result = pycolmap.estimate_homography_matrix(source, target, options)
        if result is None:
            linear = None
        else:
            matrix = np.asarray(result["H"])
            if abs(matrix[2, 2]) < 1e-12:
                return None
            matrix = matrix / matrix[2, 2]
            # How much the projective denominator swings across the data. The
            # source's own extent makes it dimensionless, so the threshold is
            # the same number on a 10-inch window and a 10,000-inch one.
            extent = float(np.ptp(source, axis=0).max())
            if np.linalg.norm(matrix[2, :2]) * extent > MAX_PERSPECTIVE:
                return None
            linear = matrix[:2, :2]
    if result is None or linear is None:
        return None
    if not _similarity_gate(linear, max_anisotropy):
        return None
    return np.asarray(result["inlier_mask"], dtype=bool)
