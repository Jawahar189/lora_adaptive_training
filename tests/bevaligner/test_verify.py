"""COLMAP geometric verification: what it removes, and what it must not.

`estimate_affine2d_robust` fits six degrees of freedom to a four-degree-of-
freedom problem, and the two it adds -- reflection and shear -- are freedom the
geometry does not have. Measured on the installed pycolmap, a mirrored set of
200 exact correspondences scores 200/200 inliers with det = -6.25, so an
ungated inlier set cannot be trusted.

The gate therefore makes the verifier ABSTAIN, and the abstention tests are the
point of this file. Returning an all-False mask instead -- vetoing the window --
cost twelve of seventeen captures their pose on the val split, three of them
good ones, because a robust fit over a window that is mostly outliers can
return anything. What the verifier declines to judge goes to `ransac_sim2`
untouched, and that is the estimator which rejects a mirror by construction
(`test_inference.py::test_fit_planar_pose_rejects_a_mirror`).
"""
import numpy as np
import pytest

from align.frames import Sim2
from bevaligner.inference import verify as verify_module
from bevaligner.inference.verify import HAVE_PYCOLMAP, verify_matches

needs_pycolmap = pytest.mark.skipif(not HAVE_PYCOLMAP, reason="pycolmap not installed")

TRUTH = Sim2(s=2.5, theta=0.7, tx=10.0, ty=-5.0)


def _pair(num=200, seed=0):
    rng = np.random.default_rng(seed)
    source = rng.uniform(-50, 50, size=(num, 2))
    return rng, source, TRUTH.apply(source)


@needs_pycolmap
@pytest.mark.parametrize("mode", ["affine", "homography"])
def test_outliers_are_removed_and_true_matches_kept(mode):
    rng, source, target = _pair()
    target = target + rng.normal(scale=0.01, size=target.shape)
    outlier = np.zeros(len(source), bool)
    outlier[:60] = True
    target[outlier] = rng.uniform(-200, 200, size=(60, 2))

    kept = verify_matches(source, target, 1.0, mode=mode)

    assert kept is not None
    assert kept[~outlier].all(), "a true correspondence was thrown away"
    assert not kept[outlier].any(), "an outlier survived verification"


@needs_pycolmap
@pytest.mark.parametrize("mode", ["affine", "homography"])
def test_a_mirror_makes_the_verifier_abstain(mode):
    """A reflection fits perfectly, so its inlier set must not be believed --
    but abstaining, not vetoing, is what keeps the up-sign decision where it
    already works: in the Sim2 fit."""
    _rng, source, target = _pair()
    assert verify_matches(source, target * [1, -1], 1.0, mode=mode) is None


@needs_pycolmap
def test_the_sim2_fit_still_rejects_the_mirror_the_verifier_abstained_on():
    """The invariant the gate exists to protect, end to end."""
    from bevaligner.inference.ransac import fit_planar_pose
    _rng, source, target = _pair(num=100)
    mirrored = target * [1, -1]
    assert verify_matches(source, mirrored, 1.0, mode="affine") is None
    fit = _fit_on(_matches(source, mirrored), None, 1.0, _options(), 1.0)
    assert fit is None or fit[1].sum() < 20


@needs_pycolmap
@pytest.mark.parametrize("mode", ["affine", "homography"])
def test_a_shear_makes_the_verifier_abstain(mode):
    """Shear is freedom two orthographic rasters of one plane do not have."""
    _rng, source, target = _pair()
    sheared = target @ np.array([[1.0, 0.45], [0.0, 1.0]]).T
    assert verify_matches(source, sheared, 1.0, mode=mode) is None


@needs_pycolmap
@pytest.mark.parametrize("mode", ["affine", "homography"])
def test_an_all_false_mask_is_never_returned(mode):
    """Vetoing a window is not the verifier's call to make. Anything it cannot
    positively explain is an abstention, so RANSAC keeps every match."""
    rng, source, target = _pair()
    for label, bad in [("mirror", target * [1, -1]),
                       ("shear", target @ np.array([[1.0, 0.45], [0.0, 1.0]]).T),
                       ("noise", rng.uniform(-500, 500, size=target.shape))]:
        kept = verify_matches(source, bad, 1.0, mode=mode)
        assert kept is None or kept.any(), f"{label} was vetoed, not abstained on"


@needs_pycolmap
def test_a_clean_similarity_passes_whole():
    _rng, source, target = _pair()
    kept = verify_matches(source, target, 1.0, mode="affine")
    assert kept is not None and kept.all()


@needs_pycolmap
def test_collinear_points_make_the_verifier_abstain():
    """A line fixes no planar motion; the low singular value catches it."""
    source = np.stack([np.linspace(-50, 50, 80), np.zeros(80)], axis=1)
    assert verify_matches(source, TRUTH.apply(source), 1.0, mode="affine") is None


@pytest.mark.parametrize("mode", ["none", "affine", "homography"])
def test_none_means_not_verified_not_everything_passed(mode):
    """None and an all-True mask are different answers, and callers report the
    difference: `num_verified` is None when verification did not run."""
    _rng, source, target = _pair(num=2)
    assert verify_matches(source, target, 1.0, mode=mode) is None


def test_mode_none_never_touches_pycolmap():
    _rng, source, target = _pair()
    assert verify_matches(source, target, 1.0, mode="none") is None


def test_non_finite_input_is_declined():
    _rng, source, target = _pair(num=20)
    assert verify_matches(np.full_like(source, np.nan), target, 1.0, mode="affine") is None


def test_unknown_mode_is_an_error():
    _rng, source, target = _pair(num=20)
    with pytest.raises(ValueError, match="mode must be one of"):
        verify_matches(source, target, 1.0, mode="essential")


def test_mismatched_lengths_are_an_error():
    _rng, source, target = _pair(num=20)
    with pytest.raises(ValueError, match="against"):
        verify_matches(source, target[:10], 1.0, mode="affine")


def test_without_pycolmap_verification_is_skipped(monkeypatch):
    """Absent the wheel the pipeline runs exactly as it did before."""
    monkeypatch.setattr(verify_module, "HAVE_PYCOLMAP", False)
    _rng, source, target = _pair()
    assert verify_matches(source, target, 1.0, mode="affine") is None


# --- the wiring in propose.py ---------------------------------------------
#
# `_fit_on` runs RANSAC over the verified subset but has to hand back a mask
# over every match, because `num_matches`, the inlier fraction and the drawing
# all index the full set.

from bevaligner.inference.matching import Matches  # noqa: E402
from bevaligner.inference.propose import (PoseProposalOptions, _fit_on,  # noqa: E402
                                            _verified_count)


def _matches(source, target):
    return Matches(source_world_xy=source, target_paper_xy=target,
                   target_crop_px=target.copy(), confidence=np.ones(len(source)),
                   query_px=source.copy(), num_queries=len(source))


def _options(**kwargs):
    return PoseProposalOptions(tolerance_m=0.5, ransac_iterations=500,
                               min_inliers=12, seed=0, **kwargs)


def test_fit_on_grows_the_mask_back_to_every_match():
    rng, source, target = _pair(num=120, seed=3)
    target = target + rng.normal(scale=0.01, size=target.shape)
    verified = np.zeros(len(source), bool)
    verified[::2] = True                      # only the even matches survive

    fit = _fit_on(_matches(source, target), verified, 1.0, _options(), 1.0)

    assert fit is not None
    _sim2, inlier_mask = fit
    assert inlier_mask.shape == (len(source),), "mask must index every match"
    assert not inlier_mask[~verified].any(), "a match RANSAC never saw is an inlier"
    assert inlier_mask[verified].sum() > 50


def test_fit_on_without_verification_is_the_old_path_exactly():
    """`geometric_verification="none"` must not perturb a single bit."""
    rng, source, target = _pair(num=150, seed=4)
    target = target + rng.normal(scale=0.01, size=target.shape)
    target[:40] = rng.uniform(-200, 200, size=(40, 2))
    matches = _matches(source, target)

    from bevaligner.inference.ransac import fit_planar_pose
    expected = fit_planar_pose(source, target, 1.0, 0.5, 500, 0, 12, tolerance_inches=1.0)
    actual = _fit_on(matches, None, 1.0, _options(), 1.0)

    assert (expected is None) == (actual is None)
    assert np.array_equal(expected[1], actual[1])
    assert expected[0].s == actual[0].s and expected[0].theta == actual[0].theta


def test_fit_on_returns_none_when_the_survivors_cannot_support_a_fit():
    rng, source, target = _pair(num=80, seed=5)
    verified = np.zeros(len(source), bool)
    verified[:4] = True                       # below min_inliers=12
    assert _fit_on(_matches(source, target), verified, 1.0, _options(), 1.0) is None


def test_verified_count_distinguishes_skipped_from_empty():
    assert _verified_count(None) is None
    assert _verified_count(np.zeros(10, bool)) == 0
    assert _verified_count(np.ones(10, bool)) == 10


def test_verification_defaults_to_off():
    """It changes results, so it ships behind a flag until measured."""
    assert PoseProposalOptions().geometric_verification == "none"
