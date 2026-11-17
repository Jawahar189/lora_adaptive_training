"""The current-side floor-fit correction, end to end against a real reference.

`stage_floor_correct` (align/runner.py) is the actual fix for the documented
"current-side floor fit" bug (docs/PRIOR_CAPTURE.md): a floor fit that is
wrong but internally coherent survives `stage_cloud`'s dispersion gate and only
shows up as an implausible rig height after a scale exists to check it
against. These tests build a real `align.reference.Reference` (the same way
tests/test_reference.py does, via a fake prior source) and a real current-side
`view`, then drive `stage_search`/`stage_pose`/`stage_floor_correct` for real.

The first pass is not obtained by hoping the FFT search happens to wander into
an implausible result -- that is emergent, search-quality behaviour already
covered by tests/test_search.py, not what's under test here. Instead a
`Hypothesis` standing in for "whatever the first pass produced" is supplied
directly, with a floor fit known to be wrong by construction. That is a
faithful stand-in: `stage_floor_correct` only ever reads `run.hypotheses[0]`
and `run.geometry["views"][ui]["floor"]`, exactly what a real first pass would
have left behind.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import reference as R
from align.cloud import wall_band
from align.frames import Sim2
from align.gravity import CAMERA_HEIGHT_M, FloorFit
from align.io.cases import Case, Structure
from align.priors import PriorSplit
from align.runner import Params, RealityRun, stage_floor_correct, stage_pose
from align.search import Hypothesis, get_backend


class FakeRef:
    """Mirrors tests/test_reference.py's FakeRef: align.reference.build's only
    route to data is with_clouds()/points()/trajectory()."""

    def __init__(self, clouds, cams, key="fake"):
        self._clouds, self._cams, self.key = clouds, cams, key

    def with_clouds(self):
        return tuple(PriorSplit(reality_id=r, capture_id="CAPp",
                                structure_key="data/STR1", capture_date="2026-01-01",
                                rlt_dir=Path("/tmp") / r,
                                cloud_path=Path("/tmp") / r / "point_cloud.ply",
                                shots_path=Path("/tmp") / r / "shots.geojson")
                     for r in self._clouds)

    def points(self, reality_id, stride=1):
        return self._clouds[reality_id]

    def trajectory(self, reality_id):
        class T:
            positions = self._cams[reality_id]
            names = [f"IMG_{i:05d}.JPG" for i in range(len(self._cams[reality_id]))]
        return T()


def _l_shaped(rig_in, n=1500, seed=0):
    """An L of wall at a known height above a flat floor, plus the walk.

    Same shape as tests/test_reference.py's fixture, scaled down 5x (an 8-inch
    arm rather than 40) purely so the search raster this drives through
    align.runner.stage_search stays small enough for a unit test to run in
    well under a second -- the wrong-height scenario below widens the wall
    band enough to sweep in the floor slab too, and this is the axis that
    determines FFT cost, not fidelity to the bug being tested.
    """
    rng = np.random.default_rng(seed)
    along = rng.uniform(0, 8, n)
    z = rng.uniform(-0.6 * rig_in, 0.1 * rig_in, n)
    arm_a = np.c_[along, np.zeros(n), z]
    arm_b = np.c_[np.zeros(n), along * 0.5, z]
    wall = np.vstack([arm_a, arm_b])
    wall[:, 2] += rig_in
    cams = np.c_[np.linspace(0.4, 7.6, 20), np.full(20, 1.2), np.full(20, rig_in)]
    floor = np.c_[rng.uniform(0, 8, 800), rng.uniform(0, 4, 800),
                  np.zeros(800)]
    return np.vstack([wall, floor]), cams


SEARCH_PARAMS = dict(search_cell_m=0.25, max_search_px=500, scale_steps=5,
                     yaw_window_deg=12.0, yaw_step_deg=4.0,
                     top_k_per_hypothesis=3)


RIG_IN = CAMERA_HEIGHT_M / 3.0     # 3 m per paper inch, so the rig is known
MPI = 3.0
WRONG_FACTOR = 2.0                 # implies a ~4.2 m rig: outside RIG_PLAUSIBLE_M


def _structure(design_tm_scale=MPI):
    dtm = np.eye(4)
    dtm[:3, :3] *= design_tm_scale
    return Structure(structure_id="STR1", project_id="P1", root="data",
                     dir=Path("unused"), sheet_path=Path("unused"),
                     paper_w=1.0, paper_h=1.0, design_tm=dtm, status="ok")


def _case(structure):
    return Case(reality_id="RLT_cur", capture_id="CAPcur", project_id="P1",
               structure=structure, rlt_dir=Path("unused"),
               cloud_path=Path("unused"), shots_path=Path("unused"))


@pytest.fixture
def backend():
    return get_backend("numpy")


@pytest.fixture
def ref_map():
    """A real Reference, built the same way as the prior side's own tests."""
    pts, cams = _l_shaped(RIG_IN)
    fake = FakeRef({"RLT_a": pts}, {"RLT_a": cams})
    r = R.build(fake, _structure(), stride=1, cell_m=0.10)
    assert r is not None
    return r


def _run_with_view(ref_map, floor_height):
    """A RealityRun with one gravity-hypothesis view, cut at `floor_height`,
    over the SAME geometry the reference was built from (so the true scale
    between the two is exactly 1.0 and a correct floor recovers a real match)."""
    pts, cams = _l_shaped(RIG_IN)
    floor = FloorFit(height=floor_height, mad=0.05 * floor_height, n_used=60,
                     per_camera=np.full(60, floor_height), beyond=0.005)
    band = wall_band(pts, cams, floor)
    assert band.sum() > 0
    view = dict(R=np.eye(3), floor=floor, wall_xy=pts[band][:, :2],
               traj_xy=cams[:, :2], cam_z=cams[:, 2])
    run = RealityRun(case=_case(_structure()), reference=ref_map)
    run.geometry.update(views={0: view}, pc_xyz=pts, n_points=len(pts))
    return run


def _first_pass_hypothesis(scale, up_index=0, score=1.0, rank=0):
    return Hypothesis(sim2=Sim2(s=scale, theta=0.0, tx=0.0, ty=0.0), score=score,
                      wall=1.0, free=1.0, scale_prior=scale, scale=scale,
                      yaw_deg=0.0, viewport=0, up_index=up_index, rank=rank)


def _add_second_view(run):
    """A second up_index view, identical geometry -- these tests only exercise
    the sign-margin arithmetic in stage_pose, not the gravity/floor fitting
    itself, so the second view need only be a valid compose_pose input."""
    run.geometry["views"][1] = dict(run.geometry["views"][0])


def test_a_plausible_first_pass_is_not_flagged(ref_map):
    run = _run_with_view(ref_map, RIG_IN)
    run.hypotheses = [_first_pass_hypothesis(1.0)]
    stage_pose(run, Params())
    assert run.context["rig_height_m"] == pytest.approx(CAMERA_HEIGHT_M, rel=0.02)
    assert not run.context["rig_implausible"]


def test_an_implausible_first_pass_is_corrected(ref_map, backend):
    run = _run_with_view(ref_map, RIG_IN * WRONG_FACTOR)
    run.hypotheses = [_first_pass_hypothesis(1.0)]
    p = Params(**SEARCH_PARAMS)
    stage_pose(run, p)
    assert run.context["rig_implausible"], "fixture should start implausible"
    pre_rig = run.context["rig_height_m"]
    pre_floor_height = run.geometry["views"][0]["floor"].height

    ok = stage_floor_correct(run, p, backend=backend)

    assert ok
    assert run.context["floor_fit_corrected"] is True
    assert run.geometry["views"][0]["floor"].height != pre_floor_height
    # The corrected height is exactly what corrected_floor_height would produce
    # for the (unchanged) first-pass scale and the reference's own mpi.
    assert run.geometry["views"][0]["floor"].height == pytest.approx(
        CAMERA_HEIGHT_M / (1.0 * ref_map.metres_per_inch))
    assert run.context["rig_height_m"] != pre_rig
    assert not run.context["rig_implausible"], (
        "the corrected geometry matches the reference almost exactly, so the "
        "re-run search should land close to the known-good scale")


def test_correction_leaves_a_result_alone_when_the_band_collapses(ref_map, backend, monkeypatch):
    """The documented limit: if the corrected height itself is unusable, the
    first-pass result is kept rather than replaced with a worse one."""
    import align.runner as runner_mod

    run = _run_with_view(ref_map, RIG_IN * WRONG_FACTOR)
    run.hypotheses = [_first_pass_hypothesis(1.0)]
    p = Params()
    stage_pose(run, p)
    pre_hyps = run.hypotheses
    pre_status = run.context.copy()
    pre_floor = run.geometry["views"][0]["floor"]
    pre_wall_xy = run.geometry["views"][0]["wall_xy"]

    monkeypatch.setattr(
        runner_mod, "corrected_floor_height",
        lambda floor, scale, mpi, camera_height_m=CAMERA_HEIGHT_M:
            FloorFit(height=1e-9, mad=0.0, n_used=0, per_camera=np.array([]), beyond=0.0))

    ok = stage_floor_correct(run, p, backend=backend)

    assert not ok
    assert run.context.get("floor_fit_corrected") is not True
    assert run.geometry["views"][0]["floor"] is pre_floor
    assert run.geometry["views"][0]["wall_xy"] is pre_wall_xy
    assert run.hypotheses is pre_hyps
    assert run.context["rig_height_m"] == pre_status["rig_height_m"]


def test_a_failed_retry_search_is_rolled_back(ref_map, backend, monkeypatch):
    """If the corrected band itself searches to nothing, the first-pass result
    is restored rather than left half-overwritten. Bounds the retry to one
    attempt: stage_search is called exactly once more, never in a loop."""
    import align.runner as runner_mod

    run = _run_with_view(ref_map, RIG_IN * WRONG_FACTOR)
    run.hypotheses = [_first_pass_hypothesis(1.0)]
    p = Params()
    stage_pose(run, p)
    pre_hyps, pre_status_keys = run.hypotheses, set(run.context)
    pre_floor = run.geometry["views"][0]["floor"]

    calls = []

    def fake_search(r, params, backend=None):
        calls.append(1)
        r.status = "no hypotheses"
        return False

    monkeypatch.setattr(runner_mod, "stage_search", fake_search)

    ok = stage_floor_correct(run, p, backend=backend)

    assert not ok
    assert len(calls) == 1, "the retry must not loop"
    assert run.hypotheses is pre_hyps
    assert run.geometry["views"][0]["floor"] is pre_floor
    assert run.context.get("floor_fit_corrected") is not True
    assert set(run.context) == pre_status_keys, "context restored, not merely left stale"


def test_a_thin_sign_margin_is_flagged_uncertain(ref_map):
    run = _run_with_view(ref_map, RIG_IN)
    _add_second_view(run)
    run.context["sign_ambiguous"] = True
    run.hypotheses = [_first_pass_hypothesis(1.0, up_index=0, score=1.40),
                      _first_pass_hypothesis(1.0, up_index=1, score=1.35)]
    stage_pose(run, Params())
    assert run.context["sign_margin"] == pytest.approx(0.05)
    assert run.context["sign_uncertain"] is True


def test_a_wide_sign_margin_is_not_flagged(ref_map):
    run = _run_with_view(ref_map, RIG_IN)
    _add_second_view(run)
    run.context["sign_ambiguous"] = True
    run.hypotheses = [_first_pass_hypothesis(1.0, up_index=0, score=1.40),
                      _first_pass_hypothesis(1.0, up_index=1, score=1.10)]
    stage_pose(run, Params())
    assert run.context["sign_margin"] == pytest.approx(0.30)
    assert run.context["sign_uncertain"] is False


def test_sign_margin_is_nan_when_not_ambiguous(ref_map):
    """Never fires for an ordinary single-view reality, however close the
    (nonexistent) rival's score would have to be."""
    run = _run_with_view(ref_map, RIG_IN)
    run.context["sign_ambiguous"] = False
    run.hypotheses = [_first_pass_hypothesis(1.0, up_index=0, score=1.40)]
    stage_pose(run, Params())
    assert np.isnan(run.context["sign_margin"])
    assert run.context["sign_uncertain"] is False


def test_sign_margin_is_nan_with_only_one_surviving_view(ref_map):
    """Flagged sign_ambiguous pre-search, but one view's own band/dispersion
    gate dropped it before stage_search ever ran -- nothing to compare."""
    run = _run_with_view(ref_map, RIG_IN)
    run.context["sign_ambiguous"] = True
    run.hypotheses = [_first_pass_hypothesis(1.0, up_index=0, score=1.40)]
    stage_pose(run, Params())
    assert np.isnan(run.context["sign_margin"])
    assert run.context["sign_uncertain"] is False
