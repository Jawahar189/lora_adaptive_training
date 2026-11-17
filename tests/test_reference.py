"""The prior reference, on synthetic input where the answer is known exactly.

Nothing here reads the dataset. The point is the conventions: a reference that
is flipped, or scaled wrong, still RUNS -- the search returns a pose, the
reranker agrees with it, and only the final transform is wrong -- so the
conventions have to be pinned by construction rather than noticed in a metric.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import reference as R
from align.gravity import CAMERA_HEIGHT_M
from align.priors import PriorSplit
from align.sheet import resample_to


class FakeRef:
    """A PriorReference's duck type, over points held in memory.

    `align.reference.build` only ever calls these four, which is itself worth
    pinning: the builder has no other route to the disk, so it cannot reach a
    file the guard has not checked.
    """

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


def _l_shaped(rig_in, n=6000, seed=0):
    """An L of wall at a known height above a flat floor, plus the walk."""
    rng = np.random.default_rng(seed)
    along = rng.uniform(0, 40, n)
    z = rng.uniform(-0.6 * rig_in, 0.1 * rig_in, n)      # inside the wall band
    arm_a = np.c_[along, np.zeros(n), z]                  # along +x
    arm_b = np.c_[np.zeros(n), along * 0.5, z]            # along +y, shorter
    wall = np.vstack([arm_a, arm_b])
    wall[:, 2] += rig_in                                  # floor at 0, cameras at rig
    cams = np.c_[np.linspace(2, 38, 60), np.full(60, 6.0), np.full(60, rig_in)]
    floor = np.c_[rng.uniform(0, 40, 4000), rng.uniform(0, 20, 4000),
                  np.zeros(4000)]
    return np.vstack([wall, floor]), cams


class _Structure:
    def __init__(self, design_tm=None):
        self.design_tm = design_tm


@pytest.fixture
def ref_map():
    rig = 2.11 / 3.0                                      # inches, at 3 m/inch
    pts, cams = _l_shaped(rig)
    fake = FakeRef({"RLT_a": pts}, {"RLT_a": cams})
    r = R.build(fake, _Structure(), stride=1, cell_m=0.10)
    assert r is not None
    return r


def test_builds_a_sheet_shaped_reference(ref_map):
    assert ref_map.feats.wall.shape == ref_map.free.wall.shape
    assert ref_map.geom.img_w == ref_map.feats.wall.shape[1]
    assert ref_map.geom.img_h == ref_map.feats.wall.shape[0]
    ref_map.geom.check()                                  # aspect and ppi agree
    assert ref_map.feats.ppi == pytest.approx(1.0 / ref_map.res)
    assert ref_map.wall_cells > 0


def test_wall_is_stored_y_down(ref_map):
    """The flip that fails silently.

    `sheet.resample_to` crops through `paper_to_pixel` (which flips) and then
    flips back, so `feats.wall` must be y-DOWN for the round trip to land on the
    raster the points actually made. Store it y-up and everything still runs --
    the search just returns a pose mirrored about the horizontal axis.

    The L is asymmetric on purpose: a symmetric reference cannot detect this.
    """
    h, w = ref_map.feats.wall.shape
    bounds = (0.0, 0.0, w * ref_map.res, h * ref_map.res)
    back = resample_to(ref_map.feats, ref_map.geom, bounds, ref_map.res, (h, w))
    y_up = np.flipud(ref_map.feats.wall)
    assert back.sum() > 0
    agree = (back == y_up).mean()
    flipped = (back == ref_map.feats.wall).mean()
    assert agree > 0.99, f"round trip disagrees with the y-up raster ({agree:.3f})"
    assert agree > flipped, "the reference is stored upside down"


def test_long_arm_of_the_l_runs_along_x(ref_map):
    """A transposed raster would pass the flip test and still be wrong."""
    y_up = np.flipud(ref_map.feats.wall)
    rows, cols = np.nonzero(y_up)
    assert cols.ptp() > rows.ptp(), "the 40-inch arm should be the wide one"
    # The arms meet at the origin corner, which is row 0 / col 0 in a y-up grid.
    assert rows.min() < 0.2 * y_up.shape[0]
    assert cols.min() < 0.2 * y_up.shape[1]


def test_coverage_marks_the_walk_and_not_the_wall(ref_map):
    assert ref_map.free.wall.sum() > ref_map.feats.wall.sum()
    assert not (ref_map.free.wall & ref_map.feats.wall).any()


def test_origin_maps_local_back_to_paper(ref_map):
    local = np.array([[0.0, 0.0], [1.0, 2.0]])
    assert np.allclose(ref_map.to_paper(local), local + ref_map.origin)


def test_scale_comes_from_the_design_transform_when_there_is_one():
    dtm = np.eye(4) * 1.0
    dtm[:3, :3] *= 7.5                       # 7.5 metres per paper inch
    mpi, source = R.metres_per_inch_for(_Structure(dtm), camera_height_in=0.28)
    assert mpi == pytest.approx(7.5)
    assert source == "design_tm"


def test_scale_falls_back_to_the_prior_rig_height():
    mpi, source = R.metres_per_inch_for(_Structure(None),
                                        camera_height_in=CAMERA_HEIGHT_M / 3.0)
    assert mpi == pytest.approx(3.0)
    assert source == "prior_camera_height"


def test_scale_is_flagged_when_it_is_a_guess():
    mpi, source = R.metres_per_inch_for(_Structure(None), camera_height_in=float("nan"))
    assert source == "fallback"
    assert mpi == R.FALLBACK_METRES_PER_INCH


def _cluttered(rig):
    """An L plus a dense slab just under the lens: the histogram prefers the slab."""
    pts, cams = _l_shaped(rig)
    rng = np.random.default_rng(1)
    clutter = np.c_[rng.uniform(0, 40, 30000), rng.uniform(0, 20, 30000),
                    np.full(30000, rig * 0.93)]
    return np.vstack([pts, clutter]), cams


def test_an_implausible_floor_is_corrected_not_dropped():
    """A floor fitted at a fraction of the rig height cuts the band to a sliver.

    Measured over 98 georeferenced prior splits, 23 land outside 1.2-3.5 m and
    every one is too LOW, so this is the common case rather than the exotic one.
    Dropping them would cost a quarter of the reference.
    """
    rig = 2.11 / 3.0
    dtm = np.eye(4)
    dtm[:3, :3] *= 3.0                        # 3 m per inch, so the rig is known
    pts, cams = _cluttered(rig)
    r = R.build(FakeRef({"RLT_a": pts}, {"RLT_a": cams}), _Structure(dtm),
                stride=1, cell_m=0.10)
    assert r is not None
    assert r.prior_reality_ids == ("RLT_a",), "the split should be kept, not dropped"
    assert r.camera_height_in == pytest.approx(rig, rel=0.02)
    assert any("known rig height" in w for w in r.warnings), r.warnings


def test_a_bad_floor_is_undetectable_without_a_known_scale():
    """The limit of the correction, pinned so it is not mistaken for a bug.

    Correcting a floor fit needs an expected height. Where the sheet carries no
    design transform the scale is DERIVED from that same fit, so a lone prior
    split has nothing to check itself against and the bad height stands. Two or
    more splits recover the majority; one does not. This is the paper-frame
    half of the dataset, so it bounds what the reference can be trusted to do.
    """
    rig = 2.11 / 3.0
    pts, cams = _cluttered(rig)
    r = R.build(FakeRef({"RLT_a": pts}, {"RLT_a": cams}), _Structure(None),
                stride=1, cell_m=0.10)
    assert r is not None
    assert r.scale_source == "prior_camera_height"
    assert r.camera_height_in < 0.5 * rig, "the bad fit should stand, undetected"
    assert not any("known rig height" in w for w in r.warnings)


def test_a_reference_with_no_usable_split_is_none():
    fake = FakeRef({}, {})
    assert R.build(fake, _Structure(), stride=1) is None
