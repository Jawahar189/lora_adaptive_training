"""Identity and round-trip tests. These must pass before any algorithm exists.

The only test file that reads the real dataset. It is loaded through
`_all_cases()` rather than at import: a module-level load runs during
COLLECTION, so a missing data root turned every other test file in the suite
into a collection error rather than a skip. The dataset lives outside the repo,
so that is the normal case on any other machine.
"""
import functools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.eval.metrics import evaluate
from align.frames import (SheetGeometry, compose_tm, decompose_tm, is_identity,
                          is_similarity, target_to_paper)
from align.groundtruth import gt_trajectory, load_tm
from align.io import datasets as datasets_mod
from align.io.cases import CaseIndex
from align.io.shots import load_shots



@functools.lru_cache(maxsize=1)
def _all_cases() -> tuple:
    try:
        spec = datasets_mod.from_env()
        return () if spec is None else tuple(
            CaseIndex.load_dataset(spec).cases)
    except (FileNotFoundError, OSError, KeyError):
        return ()


ALL = _all_cases()
_SKIP = pytest.mark.skip(reason=f"set ${datasets_mod.ENV_CURRENT} to run against a dataset")
CASES = list(ALL) or [pytest.param(None, marks=_SKIP)]


def _ids(c):
    return getattr(c, "reality_id", "no-dataset")


def _eval_gt(case):
    """Evaluate a case's ground truth against itself."""
    tm = load_tm(case.rlt_dir)
    return evaluate(tm, tm, gt_trajectory(case.rlt_dir).positions,
                    load_shots(case.rlt_dir).positions,
                    case.structure.design_tm, case.structure.geometry(),
                    case.reality_id, case.structure.key, case.structure.status)


def test_paper_pixel_roundtrip():
    g = SheetGeometry(36, 24, 7200, 4800)
    g.check()
    p = np.array([[0.0, 0.0], [36.0, 24.0], [12.5, 7.25]])
    assert np.allclose(g.pixel_to_paper(g.paper_to_pixel(p)), p)
    # paper origin is bottom-left; pixel origin is top-left
    assert np.allclose(g.paper_to_pixel(np.array([[0.0, 24.0]])), [[0.0, 0.0]])
    assert np.allclose(g.paper_to_pixel(np.array([[0.0, 0.0]])), [[0.0, 4800.0]])


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_tm_is_similarity(case):
    assert is_similarity(load_tm(case.rlt_dir), tol=1e-6)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_sheet_geometry_consistent(case):
    case.structure.geometry()      # raises on aspect / anisotropy mismatch


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_shots_round_trips_through_tm(case):
    """tm @ shots positions must reproduce images_full.

    This used to assert that `inv(tm) @ images_full`, pushed back through tm,
    returned images_full -- i.e. that np.linalg.inv works. The trajectory now
    comes from production's own shots.geojson, so the same assertion has become
    a real check on the DATA: that the poses the solver consumes and the poses
    it is scored against are the same poses in two frames.
    """
    tm = load_tm(case.rlt_dir)
    sfm, gt = load_shots(case.rlt_dir), gt_trajectory(case.rlt_dir)
    assert sfm.names == gt.names, "shots.geojson and images_full disagree on images"
    back = sfm.positions @ tm[:3, :3].T + tm[:3, 3]
    scale = max(1.0, np.abs(gt.positions).max())
    assert np.abs(back - gt.positions).max() / scale < 1e-9


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_compose_decompose_roundtrip(case):
    tm = load_tm(case.rlt_dir)
    dtm = case.structure.design_tm
    assert np.allclose(compose_tm(decompose_tm(tm, dtm), dtm), tm, rtol=0, atol=1e-9)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_identity_evaluation_is_perfect(case):
    """Feeding ground truth through the evaluator must score zero error in every
    frame. This is what catches a y-flip or a frame-composition bug."""
    tm = load_tm(case.rlt_dir)
    m = _eval_gt(case)
    assert m.pixel.median < 1e-6
    assert m.pixel.max < 1e-5
    assert m.paper.max < 1e-9
    assert m.scale_log_err < 1e-12
    assert m.yaw_err_deg < 1e-9
    assert m.centroid_err_px < 1e-5
    if m.georeferenced:
        assert m.tpr_1m == 1.0
        assert m.utm is not None and m.utm.max < 1e-6
        assert m.centroid_err_utm < 1e-6


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_utm_only_when_georeferenced(case):
    """A structure whose design transform is absent or identity has no
    georeference, so UTM metres must be withheld rather than reported in the
    paper frame's units."""
    m = _eval_gt(case)
    georef = not is_identity(case.structure.design_tm)
    assert m.georeferenced is georef
    assert m.pixel is not None, "pixel error is always reported"
    if georef:
        assert m.utm is not None and m.utm.unit == "m"
        assert m.tpr_1m is not None and m.metres_per_inch is not None
    else:
        assert m.utm is None, "no UTM error without a georeference"
        assert m.tpr_1m is None and m.tpr_1m_2d is None
        assert m.centroid_err_utm is None and m.metres_per_inch is None


@pytest.mark.skipif(not ALL, reason="no dataset in the environment")
def test_pixel_error_scales_with_sheet_ppi():
    """Pixel error must use each sheet's own ppi, not a constant: the dataset
    spans 87.9 to 600 ppi, so a shared constant would be up to 7x wrong."""
    case = ALL[0]
    st = case.structure
    tm = load_tm(case.rlt_dir)
    sfm = load_shots(case.rlt_dir).positions
    gt = gt_trajectory(case.rlt_dir).positions
    # Shift the prediction by a known distance in paper inches.
    shift_in = 0.25
    bumped = tm.copy()
    d = st.design_tm if st.design_tm is not None else np.eye(4)
    bumped[:3, 3] = bumped[:3, 3] + d[:3, :3] @ np.array([shift_in, 0.0, 0.0])
    m = evaluate(bumped, tm, gt, sfm, st.design_tm, st.geometry())
    assert abs(m.pixel.median - shift_in * st.geometry().ppi) < 1e-3
    assert abs(m.paper.median - shift_in) < 1e-6


@pytest.mark.skipif(not ALL, reason="no dataset in the environment")
def test_gt_trajectory_lands_inside_the_sheet():
    """Ground-truth trajectories, mapped to pixels, must fall on the sheet.
    A y-flip error would put roughly half of them outside.

    Three splits genuinely overrun the sheet edge (the walk continues past the
    drawn extent), so a small allowance is real rather than a bug -- but a
    y-flip would show up as ~50% outside, far beyond this tolerance.
    """
    bad = []
    for case in ALL:
        g = case.structure.geometry()
        paper_xy = target_to_paper(gt_trajectory(case.rlt_dir).positions,
                                   case.structure.design_tm)[:, :2]
        px = g.paper_to_pixel(paper_xy)
        inside = ((px[:, 0] >= 0) & (px[:, 0] < g.img_w)
                  & (px[:, 1] >= 0) & (px[:, 1] < g.img_h)).mean()
        if inside < 0.30:
            bad.append((case.reality_id, round(float(inside), 3)))
    assert not bad, f"trajectories mostly outside the sheet: {bad}"
