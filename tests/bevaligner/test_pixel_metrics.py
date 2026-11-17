"""The placement error measured on the drawing, in its own pixels.

Synthetic throughout: a fake structure with a known `SheetGeometry` and a known
`design_tm`, so the expected pixel distance can be written down rather than
read off a run. Needs no cache, no data and no GPU.

The chain under test is `align.eval.metrics.image_errors`, which this package
reuses rather than re-deriving:

    target frame -> paper inches -> floormap pixels

and the point of these tests is that the two ends of it line up: an offset of
one inch on the paper must come out as exactly `ppi` pixels.
"""
import csv
import math

import numpy as np
import pytest

from align.frames import SheetGeometry
from bevaligner.utils.metrics import (CSV_EXTRA_COLUMNS, METRE_COLUMNS,
                                        PIXEL_COLUMNS, SheetResultWriter,
                                        metre_aliases, pixel_metrics,
                                        summarise_pixels)

PPI = 150.0
PAPER_W, PAPER_H = 42.0, 30.0


class _Structure:
    """Enough of `align.io.cases.Structure` for the metric."""

    key = "data_local/STR1"

    def __init__(self, ppi=PPI, design_tm=None, raises=False):
        self.design_tm = design_tm
        self._raises = raises
        self._geom = SheetGeometry(PAPER_W, PAPER_H,
                                   int(PAPER_W * ppi), int(PAPER_H * ppi))

    def geometry(self):
        if self._raises:
            raise ValueError("aspect mismatch: image 1.18519 vs paper 1.41513")
        return self._geom


def _identity_tm():
    return np.eye(4)


def _shifted_tm(dx_inches, dy_inches=0.0):
    """ply -> paper, displaced by a known number of paper inches."""
    tm = np.eye(4)
    tm[0, 3] = dx_inches
    tm[1, 3] = dy_inches
    return tm


def _cameras(n=25, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(5.0, 25.0, size=(n, 2))
    return np.column_stack([xy, np.zeros(n)])


def test_a_known_inch_offset_becomes_exactly_that_many_pixels():
    """One paper inch is `ppi` pixels. If this drifts the chain is wrong."""
    gt = _cameras()
    out = pixel_metrics(_shifted_tm(1.0), gt, gt, _Structure())
    assert out["rmse_px"] == pytest.approx(PPI, abs=0.01)
    assert out["median_px"] == pytest.approx(PPI, abs=0.01)
    assert out["max_px"] == pytest.approx(PPI, abs=0.01)


def test_a_diagonal_offset_uses_the_euclidean_distance():
    gt = _cameras()
    out = pixel_metrics(_shifted_tm(3.0, 4.0), gt, gt, _Structure())
    assert out["rmse_px"] == pytest.approx(5.0 * PPI, abs=0.01)


def test_a_perfect_placement_is_zero_pixels():
    gt = _cameras()
    out = pixel_metrics(_identity_tm(), gt, gt, _Structure())
    assert out["rmse_px"] == 0.0
    assert out["tpr@5px"] == 1.0 and out["tpr@100px"] == 1.0


def test_ppi_scales_the_answer():
    """The sheet's own resolution is the whole conversion, so doubling it
    doubles the error. This is why `sheet_ppi` is reported beside it."""
    gt = _cameras()
    coarse = pixel_metrics(_shifted_tm(1.0), gt, gt, _Structure(ppi=150.0))
    fine = pixel_metrics(_shifted_tm(1.0), gt, gt, _Structure(ppi=300.0))
    assert fine["rmse_px"] == pytest.approx(2 * coarse["rmse_px"], rel=1e-6)
    assert coarse["sheet_ppi"] == 150.0 and fine["sheet_ppi"] == 300.0


def test_rmse_is_not_the_mean():
    """RMSE, so a photograph in the wrong room is not averaged away."""
    gt = np.array([[10.0, 10.0, 0.0], [11.0, 10.0, 0.0]])
    sfm = np.array([[10.0, 10.0, 0.0], [11.0 - 10.0, 10.0, 0.0]])  # one off by 10in
    out = pixel_metrics(_identity_tm(), sfm, gt, _Structure())
    errors = np.array([0.0, 10.0 * PPI])
    assert out["rmse_px"] == pytest.approx(math.sqrt((errors ** 2).mean()), abs=0.01)
    assert out["rmse_px"] > errors.mean()


def test_thresholds_count_the_images_inside_them():
    gt = _cameras(n=4)
    sfm = gt.copy()
    sfm[:2, 0] += 1.0                       # two images one inch (150 px) out
    out = pixel_metrics(_identity_tm(), sfm, gt, _Structure())
    assert out["n_images_px"] == 4
    assert out["tpr@100px"] == 0.5          # the two unmoved ones
    assert out["tpr@5px"] == 0.5


# --- the structures that have no georeference ------------------------------

def test_a_structure_without_a_design_tm_is_still_measured():
    """194 of 1,676 validation rows are `no_world_transform`. Their target
    frame IS paper inches, `target_to_paper` is the identity, and pixels are
    the one unit that works for them and for the georeferenced rows alike."""
    gt = _cameras()
    out = pixel_metrics(_shifted_tm(2.0), gt, gt, _Structure(design_tm=None))
    assert out["rmse_px"] == pytest.approx(2 * PPI, abs=0.01)


def test_a_georeferenced_structure_divides_by_the_design_scale():
    """With a design_tm the target frame is metres, so a metre of error is
    `ppi / metres_per_inch` pixels, not `ppi`."""
    metres_per_inch = 4.0
    design_tm = np.eye(4)
    design_tm[:3, :3] *= metres_per_inch
    gt = _cameras() * metres_per_inch
    tm = np.eye(4)
    tm[0, 3] = metres_per_inch              # one paper inch, expressed in metres
    out = pixel_metrics(tm, gt / metres_per_inch * metres_per_inch, gt,
                        _Structure(design_tm=design_tm))
    assert out["rmse_px"] == pytest.approx(PPI, rel=1e-6)


# --- when it cannot be measured --------------------------------------------

def test_a_sheet_that_fails_its_aspect_check_is_declined_not_raised():
    """One validation structure fails `SheetGeometry.check`. A wrong ppi would
    be worse than none, and the capture's row must still be written."""
    gt = _cameras()
    assert pixel_metrics(_identity_tm(), gt, gt, _Structure(raises=True)) == {}


def test_no_ground_truth_is_declined():
    assert pixel_metrics(_identity_tm(), np.zeros((0, 3)), np.zeros((0, 3)),
                         _Structure()) == {}
    assert pixel_metrics(_identity_tm(), None, None, _Structure()) == {}


# --- reporting --------------------------------------------------------------

def test_the_csv_carries_both_units(tmp_path):
    """`align.eval.reality.COLUMNS` drops unknown keys, so without the subclass
    these land in the JSONL only."""
    writer = SheetResultWriter(tmp_path / "run")
    # read before close: the header must be on disk as soon as the run starts
    header = next(csv.reader(open(writer.paths["csv"])))
    writer.close()
    for column in CSV_EXTRA_COLUMNS:
        assert column in header, f"{column} missing from the CSV header"
    assert "rmse_xy" in header, "the base columns must survive"
    # the point of the metre aliases: both units readable side by side
    assert {"rmse_m", "rmse_px"} <= set(header)


# --- the metres, under a name that says metres -----------------------------

def test_metre_aliases_copy_the_values_without_recomputing():
    source = {"rmse_xy": 1.5, "median_xy": 1.2, "p90_xy": 2.0, "max_xy": 3.0,
              "tpr@1.0": 0.5, "rmse_z": 0.4}
    assert metre_aliases(source) == {"rmse_m": 1.5, "median_m": 1.2,
                                     "p90_m": 2.0, "max_m": 3.0}


def test_metre_aliases_skip_what_is_not_there():
    """An unscored capture has no metrics to alias, and must not gain empty ones."""
    assert metre_aliases({}) == {}
    assert metre_aliases({"rmse_xy": 1.5}) == {"rmse_m": 1.5}


def test_every_metre_column_has_a_pixel_counterpart():
    """The two blocks are meant to be read as a pair."""
    for metre in METRE_COLUMNS:
        assert metre.replace("_m", "_px") in PIXEL_COLUMNS


def test_a_row_round_trips_through_the_writer(tmp_path):
    class _Case:
        reality_id, capture_id = "RLT1", "SNP1"
        structure = _Structure()
        structure.key = "data_local/STR1"
        structure.georeferenced = False

    writer = SheetResultWriter(tmp_path / "run")
    writer.write(_Case(), {"predicted": dict(rmse_xy=1.0, rmse_m=1.0, rmse_px=42.5,
                                             sheet_ppi=150.0, tm=np.eye(4))})
    writer.close()
    row = next(csv.DictReader(open(writer.paths["csv"])))
    assert row["rmse_px"] == "42.5" and row["sheet_ppi"] == "150.0"
    assert row["rmse_m"] == "1.0" and row["rmse_xy"] == "1.0"


def test_the_summary_warns_when_ppi_is_not_one_number():
    mixed = [{"rmse_px": 10.0, "sheet_ppi": 60.0, "tpr@50px": 1.0},
             {"rmse_px": 90.0, "sheet_ppi": 600.0, "tpr@50px": 0.0}]
    text = summarise_pixels(mixed)
    assert "60..600" in text and "not the same length" in text
    same = [{"rmse_px": 10.0, "sheet_ppi": 150.0, "tpr@50px": 1.0}]
    assert "150 throughout" in summarise_pixels(same)


def test_the_summary_says_so_when_nothing_was_measured():
    assert "no realities scored" in summarise_pixels([{"rmse_xy": 1.0}])
