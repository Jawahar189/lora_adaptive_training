"""The current-side floor-fit gate and correction, on synthetic FloorFit values.

Pure-function tests: no point cloud, no disk. `usable_floor` and
`corrected_floor_height` never see a cloud, only the numbers a fit already
produced, so the fixtures here are FloorFit instances built by hand.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import align.gravity as gravity
from align.gravity import (CAMERA_HEIGHT_M, FloorFit, GravityFit, corrected_floor_height,
                           usable_floor)


def _fit(height=0.7, mad=0.02, beyond=0.005):
    return FloorFit(height=height, mad=mad, n_used=100,
                    per_camera=np.full(100, height), beyond=beyond)


def _resolve(monkeypatch, fit_for_up, fit_for_down):
    """`resolve_up_sign` calls `floor_below_trajectory` once per candidate, in
    the order (up, -up) -- stub it to hand back chosen fits without needing
    real geometry that produces exact beyond/rel_mad values.

    It returns `(up, ambiguous, floor, sign_fits)`; `sign_fits` carries both
    candidates' fits so nothing downstream has to compute one again."""
    calls = iter([fit_for_up, fit_for_down])
    monkeypatch.setattr(gravity, "floor_below_trajectory",
                        lambda *a, **k: next(calls))
    up = np.array([0.0, 0.0, 1.0])
    return gravity.resolve_up_sign(np.zeros((1, 3)), np.zeros((1, 3)), up), up


def test_resolve_up_sign_uses_beyond_when_it_discriminates(monkeypatch):
    fit_up = _fit(beyond=0.01)
    fit_down = _fit(beyond=0.05)          # gap 0.04 >= SIGN_TIE_TOL
    (u, ambiguous, f, _fits), up = _resolve(monkeypatch, fit_up, fit_down)
    assert not ambiguous
    assert f is fit_up
    assert np.array_equal(u, up)


def test_resolve_up_sign_falls_back_to_rel_mad_on_a_beyond_tie(monkeypatch):
    fit_up = _fit(height=1.0, mad=0.05, beyond=0.001)     # rel_mad 0.05
    fit_down = _fit(height=1.0, mad=0.02, beyond=0.002)   # rel_mad 0.02
    # beyond gap 0.001 < SIGN_TIE_TOL; rel_mad gap 0.03 >= REL_MAD_TIE_TOL
    (u, ambiguous, f, _fits), up = _resolve(monkeypatch, fit_up, fit_down)
    assert not ambiguous
    assert f is fit_down, "lower rel_mad should win once beyond can't decide"
    assert np.array_equal(u, -up)


def test_resolve_up_sign_is_ambiguous_when_both_tie(monkeypatch):
    fit_up = _fit(height=1.0, mad=0.05, beyond=0.001)     # rel_mad 0.05
    fit_down = _fit(height=1.0, mad=0.055, beyond=0.002)  # rel_mad 0.055
    # beyond gap 0.001 < SIGN_TIE_TOL; rel_mad gap 0.005 < REL_MAD_TIE_TOL
    (u, ambiguous, f, _fits), up = _resolve(monkeypatch, fit_up, fit_down)
    assert ambiguous
    assert f is fit_up
    assert np.array_equal(u, up)


def test_resolve_up_sign_takes_the_only_finite_candidate(monkeypatch):
    fit_up = _fit()
    fit_up.height = float("nan")
    fit_down = _fit(beyond=0.01)
    (u, ambiguous, f, _fits), up = _resolve(monkeypatch, fit_up, fit_down)
    assert not ambiguous
    assert f is fit_down
    assert np.array_equal(u, -up)


def test_usable_floor_accepts_a_tight_fit():
    assert usable_floor(_fit(mad=0.02)) is None


def test_usable_floor_rejects_high_rel_mad():
    why = usable_floor(_fit(height=0.7, mad=0.6))       # rel_mad ~0.86
    assert why is not None
    assert "rel_mad" in why


def test_usable_floor_rejects_nan_or_nonpositive_height():
    assert usable_floor(_fit(height=float("nan"))) == "no floor mode"
    assert usable_floor(_fit(height=0.0)) == "no floor mode"
    assert usable_floor(_fit(height=-1.0)) == "no floor mode"


def test_usable_floor_threshold_is_configurable():
    fit = _fit(height=0.7, mad=0.4)                     # rel_mad ~0.57
    assert usable_floor(fit, max_rel_mad=0.50) is not None
    assert usable_floor(fit, max_rel_mad=0.90) is None


def test_corrected_floor_height_matches_the_known_rig():
    fit = _fit(height=4.0)                              # a wildly wrong fit
    scale = 0.6
    mpi = 3.0
    corrected = corrected_floor_height(fit, scale, mpi)
    assert corrected is not None
    assert corrected.height * scale * mpi == pytest.approx(CAMERA_HEIGHT_M)


def test_corrected_floor_height_carries_the_rest_of_the_fit_unchanged():
    fit = _fit(height=4.0, mad=0.33, beyond=0.02)
    corrected = corrected_floor_height(fit, 0.6, 3.0)
    assert corrected.mad == fit.mad
    assert corrected.n_used == fit.n_used
    assert corrected.beyond == fit.beyond
    assert corrected.per_camera is fit.per_camera


@pytest.mark.parametrize("scale,mpi", [
    (0.0, 3.0), (float("nan"), 3.0), (-1.0, 3.0),
    (0.6, 0.0), (0.6, float("nan")), (0.6, -1.0),
])
def test_corrected_floor_height_is_none_without_a_usable_scale(scale, mpi):
    assert corrected_floor_height(_fit(), scale, mpi) is None


def test_resolve_up_sign_hands_back_both_candidates_fits(monkeypatch):
    """Both signs are fitted to choose between them; both must come back.

    `make_gravity_view` builds a view for one of these signs, and fitting a
    floor means a kd-tree over the whole cloud -- so a fit that is computed and
    then dropped is the most expensive thing a capture repeats.
    """
    fit_up = _fit(beyond=0.01)
    fit_down = _fit(beyond=0.05)
    (_u, _ambiguous, _f, sign_fits), up = _resolve(monkeypatch, fit_up, fit_down)
    assert [fit for _candidate, fit in sign_fits] == [fit_up, fit_down]
    assert np.array_equal(sign_fits[0][0], up)
    assert np.array_equal(sign_fits[1][0], -up)


def test_gravity_fit_looks_its_floor_up_by_sign():
    up = np.array([0.0, 0.0, 1.0])
    fit_up, fit_down = _fit(beyond=0.01), _fit(beyond=0.05)
    gravity_fit = GravityFit(up=up, R=np.eye(3), planarity=0.0, method="stub",
                             floor=fit_up, sign_fits=((up, fit_up), (-up, fit_down)))
    assert gravity_fit.floor_for(up) is fit_up
    assert gravity_fit.floor_for(-up) is fit_down
    assert gravity_fit.floor_for(np.array([1.0, 0.0, 0.0])) is None
