"""`prior_bev.mosaic_latest`: each cell keeps the newest date's walls."""
import numpy as np

from bevaligner.data.prior_bev import MOSAIC_MIN_POINTS, PriorSplit, mosaic_latest


def split(reality_id, date, xy):
    xy = np.asarray(xy, dtype=float)
    return PriorSplit(reality_id=reality_id, capture_id=f"SNP{date}", capture_date=date,
                      band_paper_xy=xy, band_relative_height=np.zeros(len(xy)),
                      camera_paper_xy=xy[:3], camera_paper_z_median=0.0,
                      camera_names=(), rig_height_in=1.0)


CELL = 10.0     # the grid starts at the lowest point, so clusters sit well inside cells


def cell(k, n=MOSAIC_MIN_POINTS * 2, seed=0):
    """`n` points inside grid cell `k` along x (clusters 2 cells apart never touch)."""
    rng = np.random.default_rng(seed)
    x0 = 2 * CELL * k
    return np.vstack([[x0 + 1.0, 1.0],
                      rng.uniform([x0 + 1.0, 1.0], [x0 + 4.0, 4.0], (n - 1, 2))])


def test_the_newest_date_owns_a_shared_cell():
    old = split("old", "2026-08-01", np.vstack([cell(0), cell(1)]))
    new = split("new", "2026-08-20", cell(1, seed=1))
    out = {s.reality_id: s for s in mosaic_latest([old, new], cell_in=CELL)}
    assert len(out["new"].band_paper_xy) == len(new.band_paper_xy)
    assert (out["old"].band_paper_xy[:, 0] < CELL).all()         # only its own cell is left


def test_splits_of_one_date_are_never_cut_against_each_other():
    a = split("a", "2026-08-20", cell(0))
    b = split("b", "2026-08-20", cell(0, seed=1))
    old = split("old", "2026-08-01", cell(0, seed=2))
    out = {s.reality_id: s for s in mosaic_latest([a, b, old], cell_in=CELL)}
    assert len(out["a"].band_paper_xy) == len(a.band_paper_xy)
    assert len(out["b"].band_paper_xy) == len(b.band_paper_xy)
    assert "old" not in out


def test_a_cell_too_sparse_for_the_newest_date_goes_to_an_older_one():
    sparse = split("new", "2026-08-20", cell(0, n=MOSAIC_MIN_POINTS - 1))
    dense = split("old", "2026-08-01", cell(0, seed=1))
    out = {s.reality_id: s for s in mosaic_latest([sparse, dense], cell_in=CELL)}
    assert "new" not in out and len(out["old"].band_paper_xy) == len(dense.band_paper_xy)


def test_one_date_passes_through():
    splits = [split("a", "2026-08-20", cell(0)), split("b", "2026-08-20", cell(1))]
    assert mosaic_latest(splits, cell_in=CELL) == splits
