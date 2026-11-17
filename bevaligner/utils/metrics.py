"""Scores, in the repo's reporting format plus the paper's PCK."""
from __future__ import annotations

import numpy as np
import torch

from align.eval.metrics import THRESHOLDS_PX, image_errors
from align.eval.reality import (COLUMNS, RIG_PLAUSIBLE_M, THRESHOLDS,  # noqa: F401
                                ResultWriter, reality_metrics, rig_height_m,
                                summarise)

# `reality_metrics` names its horizontal error `*_xy`, which says the error is
# horizontal but never says the unit. These are the same numbers under names
# that do, so a reader scanning the CSV can pair them with the pixel columns.
METRE_ALIASES = {"rmse_xy": "rmse_m", "median_xy": "median_m",
                 "p90_xy": "p90_m", "max_xy": "max_m"}
METRE_COLUMNS = tuple(METRE_ALIASES.values())

# The per-image error measured again on the drawing, in its own pixels.
PIXEL_COLUMNS = ("sheet_ppi", "n_images_px", "rmse_px", "median_px", "p90_px",
                 "max_px", *[f"tpr@{t:g}px" for t in THRESHOLDS_PX])

# What `SheetResultWriter` adds to the CSV, metres first so the two units read
# as a pair.
#
# `confidence` is last and is on EVERY learned-path run, chained or not, unlike
# the gate columns below: how likely a pose is to be right is a property of the
# pose, not of whether the run happened to chain. It is the one column here that
# needs no ground truth, so it is also the only quality signal a production run
# has. See `bevaligner/inference/confidence.py`.
CSV_EXTRA_COLUMNS = (*METRE_COLUMNS, *PIXEL_COLUMNS, "confidence")

# How a chained run reached each pose, and how firmly. Added to the CSV only
# under --chain, so a run without it is byte-identical to one from before any of
# this existed -- the preservation guarantee made structural rather than
# promised. `chain_mode` is on EVERY row of a chained run, including the
# single-split captures that were never chained, so the two kinds of run can
# never be read as each other.
CHAIN_COLUMNS = ("chain_mode", "chain_how", "chain_rung", "chain_seeded_from",
                 "chain_reach", "chain_pass", "chain_junction",
                 "chain_prior_radius_in", "chain_prior_centre_x",
                 "chain_prior_centre_y", "chain_attempt", "chain_winner_from_prior",
                 "chain_n_shared", "chain_fit_residual_in",
                 "chain_junction_realised_in", "chain_junction_realised_m", "gate_passed", "gate_reason",
                 "gate_margin", "gate_inlier_frac", "gate_rival_distance_m")


def percent_correct_keypoints(predicted: torch.Tensor, target: torch.Tensor,
                              thresholds, weights=None) -> dict[str, float]:
    """Fraction of correspondences within each threshold (same units as the
    coordinates). `weights` optionally weights by confidence."""
    distance = (predicted - target).norm(dim=-1)
    scores = {}
    for threshold in thresholds:
        hit = (distance <= threshold).float()
        if weights is None:
            scores[f"pck@{threshold:g}"] = float(hit.mean())
        else:
            scores[f"pck@{threshold:g}"] = float(
                (hit * weights).sum() / weights.sum().clamp_min(1e-9))
    return scores


def horizontal_errors_m(transform_matrix, sfm_positions, gt_positions,
                        metres_per_inch, georeferenced) -> np.ndarray:
    """Per-image planar error in metres, the quantity `reality_metrics` summarises."""
    from align.frames import apply as apply_transform
    predicted = apply_transform(np.asarray(transform_matrix),
                                np.asarray(sfm_positions, dtype=np.float64))
    delta = predicted - np.asarray(gt_positions, dtype=np.float64)
    if not georeferenced:
        delta = delta * metres_per_inch
    return np.linalg.norm(delta[:, :2], axis=1)


def metre_aliases(metrics: dict) -> dict:
    """`reality_metrics`'s horizontal error, under names that state the unit.

    Same numbers as `rmse_xy` and friends -- nothing is recomputed. They exist
    so the CSV pairs readably with the pixel columns, where `rmse_px` says its
    unit and `rmse_xy` does not.

    READ `scale_source` BEFORE TRUSTING THESE AS METRES. `reality_metrics`
    reaches metres by multiplying the paper-inch error by `metres_per_inch`, and
    on a structure with no georeference that factor is an estimate rather than a
    measurement -- `scale_source` is then `e1` or `fallback`, and `fallback` is a
    hardcoded 3.0. 194 of the 1,676 validation rows are in that state, so a
    `rmse_m` there is paper inches times a constant, not a surveyed distance.
    `align.eval.metrics` withholds metres entirely in that case, on the grounds
    that it would be "a number in fake units"; this column keeps the number
    `reality_metrics` already reports and names it honestly instead, because
    dropping it would change the metric runs are compared on. The pixel columns
    beside it are real for every structure.
    """
    return {alias: metrics[name] for name, alias in METRE_ALIASES.items()
            if metrics.get(name) is not None}


def pixel_metrics(transform_matrix, sfm_positions, gt_positions, structure) -> dict:
    """The per-image error again, measured on the drawing in its own pixels.

    `reality_metrics` answers "how many metres out", which is the number runs
    are compared on. This answers the question a reviewer actually asks, which
    is about a picture: with the capture drawn on `floormap.png`, how far from
    its true place does each photograph land?

    Both sides are carried into the pixel frame before anything is measured --
    ground truth from `images_full.json` and the prediction from this pose --
    by `align.eval.metrics.image_errors`, which is the repo's one definition of
    that chain and is already what `scripts/image_errors.py` reports:

        target frame  --target_to_paper(design_tm)-->  paper inches
                      --SheetGeometry.paper_to_pixel-->  floormap pixels

    The target frame is UTM metres on a georeferenced structure and paper
    inches on one without a `design_tm`, where `target_to_paper` is the
    identity. Both end up in pixels, which is the point: it is the one unit
    every structure can be scored in, georeferenced or not.

    NOT COMPARABLE BETWEEN STRUCTURES. `ppi` is a property of the sheet, and it
    runs 60.4 to 600.1 across the 90 validation structures while
    `metres_per_inch` runs 2.36 to 30.4 -- so a metre of error is about 12 px
    on one structure and 123 on another. Right for reading one capture against
    its own drawing; wrong to pool. `rmse_xy` stays the metric runs are
    compared on.

    Returns {} when the sheet cannot be measured -- no floor plan on disk, no
    paper size, or a sheet whose aspect ratio disagrees with its paper size
    (`SheetGeometry.check`, which one validation structure fails: image 1.185
    against paper 1.415). An empty column means NOT MEASURABLE, never zero
    error, and the rest of the row is written as usual.
    """
    if gt_positions is None or not len(gt_positions):
        return {}
    try:
        geometry = structure.geometry()
        # `image_errors` takes gt_tm for symmetry with `evaluate` and never
        # reads it; the ground truth it uses is `gt_positions`, already in the
        # target frame.
        _paper, pixels, _utm, _pred = image_errors(
            transform_matrix, None, gt_positions, sfm_positions,
            structure.design_tm, geometry)
    except Exception:                                      # noqa: BLE001
        return {}                                          # see the docstring
    if not len(pixels):
        return {}
    out = {"sheet_ppi": round(float(geometry.ppi), 2),
           "n_images_px": int(len(pixels)),
           # RMSE, not the mean: the question is how far a photograph lands
           # from its place, and squaring keeps the ones that land in the wrong
           # room from being averaged away by the ones that do not.
           "rmse_px": round(float(np.sqrt((pixels ** 2).mean())), 2),
           "median_px": round(float(np.median(pixels)), 2),
           "p90_px": round(float(np.percentile(pixels, 90)), 2),
           "max_px": round(float(pixels.max()), 2)}
    for threshold in THRESHOLDS_PX:
        out[f"tpr@{threshold:g}px"] = round(float((pixels <= threshold).mean()), 4)
    return out


def summarise_pixels(rows) -> str:
    """The pixel block for the end of a run, beside `summarise`'s metres.

    Prints the `ppi` spread of the very rows it pooled, because pooling pixels
    across structures mixes units that differ by about 100x (see
    `pixel_metrics`). A reader who sees one ppi has a figure that means
    something; a reader who sees 60..600 has been told not to trust the median.
    """
    scored = [r for r in rows if r.get("rmse_px") is not None]
    if not scored:
        return "  no realities scored in pixels"
    rmse = np.array([r["rmse_px"] for r in scored], dtype=float)
    ppi = np.array([r["sheet_ppi"] for r in scored], dtype=float)
    lines = [f"  pixels on the drawing, n={len(scored)}"]
    lines.append(f"    rmse_px  median {np.median(rmse):8.1f} px  "
                 f"p90 {np.percentile(rmse, 90):9.1f} px")
    for threshold in THRESHOLDS_PX:
        key = f"tpr@{threshold:g}px"
        values = np.array([r[key] for r in scored if r.get(key) is not None], dtype=float)
        if len(values):
            lines.append(f"    {key:<10} mean {values.mean():.3f}   "
                         f"median {np.median(values):.3f}")
    if ppi.min() == ppi.max():
        lines.append(f"    ppi {ppi.min():.0f} throughout")
    else:
        lines.append(f"    ppi {ppi.min():.0f}..{ppi.max():.0f} across these "
                     f"structures -- a pixel is not the same length in each, so "
                     f"read this per capture, not pooled")
    return "\n".join(lines)


class SheetResultWriter(ResultWriter):
    """`ResultWriter` with the floor-plan pixel columns on the CSV.

    `align.eval.reality.COLUMNS` is a fixed tuple and the writer drops anything
    outside it (`extrasaction="ignore"`), so a new metric reaches the JSONL but
    not the CSV -- and the CSV is the reporting format. Subclassed rather than
    widening COLUMNS because that tuple is also the classical pipeline's output
    contract, and `align/` is not this branch's to change.

    The header the base class just wrote is rewound and rewritten; nothing else
    has reached the file yet, so there is nothing to lose.
    """

    EXTRA_COLUMNS = CSV_EXTRA_COLUMNS
    # The confidence travels with the transform, not just with the report: the
    # predictions JSONL is the file a consumer actually reads, and a transform
    # with no idea how much to believe it is what this column exists to fix.
    PREDICTION_EXTRAS = ("confidence",)

    def __init__(self, stem, thresholds=THRESHOLDS, extra_columns=None):
        import csv as _csv
        super().__init__(stem, thresholds)
        if extra_columns:
            self.EXTRA_COLUMNS = (*CSV_EXTRA_COLUMNS, *extra_columns)
        self._csvf.seek(0)
        self._csvf.truncate()
        self._csv = _csv.DictWriter(
            self._csvf, fieldnames=[*COLUMNS, *self.EXTRA_COLUMNS],
            extrasaction="ignore")
        self._csv.writeheader()
        # Flushed here for the same reason the base class flushes per reality:
        # a run that dies before its first capture should leave a readable file
        # with a header, not an empty one.
        self._csvf.flush()


def summarise_confidence(rows) -> str:
    """One line on the distribution of `confidence`, for the end of a run.

    Thresholds at 0.5 and 0.85 because those are the two the calibration table
    in `inference/confidence.py` supports: 0.85 and above is right about 92% of
    the time on both runs it was measured over, and 0.5 and above about 70%.
    Rows with no confidence -- an abstention, or a pose carried over by a
    transfer, which no search scored -- are counted as unscored rather than as
    zeroes.
    """
    scored = [float(r["confidence"]) for r in rows if r.get("confidence") is not None]
    if not scored:
        return "  confidence: none scored"
    scored_array = np.array(scored)
    return ("  confidence: median {:.2f}   >= 0.50: {}/{}   >= 0.85: {}/{}"
            .format(np.median(scored_array), int((scored_array >= 0.5).sum()),
                    len(rows), int((scored_array >= 0.85).sum()), len(rows)))


# How each pose was actually arrived at. Without this the headline number cannot
# be read: a chained run that improves because its seeds happened to land on
# easy captures looks exactly like one that improves because the chain works.
CHAIN_BUCKETS = ("seed (geo)", "seed (oracle)", "seed (search)", "seed (fallback)",
                 "transferred_verified", "transferred_unverified", "chained",
                 "widened", "recovered", "independent", "best_effort")


def _reach_label(row) -> str:
    """`chained` splits apart by whether the anchor was next door.

    Without this the bucket fuses two populations with different expected
    error -- a split leaning on its neighbour and one reaching back past a
    failure -- and the run cannot be read.
    """
    reach = row.get("chain_reach")
    try:
        reach = int(reach)
    except (TypeError, ValueError):
        return ""
    return "" if reach <= 1 else " (reached)"


def _bucket(how: str) -> str:
    """The row's `chain_how`, reduced to one of `CHAIN_BUCKETS`."""
    how = (how or "").strip()
    for name in CHAIN_BUCKETS:
        if how == name or how.startswith(name):
            return name
    return "independent" if how.startswith("independent") else "other"


def summarise_chain(rows) -> str:
    """Per-bucket counts and accuracy for a chained run.

    Reads the same `tpr@1.0 > 0.5` that `summarise` calls a placed reality, so
    the buckets add up to the headline rather than measuring something adjacent.
    Rows with no score -- an unscored run -- report counts only.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for row in rows:
        name = _bucket(row.get("chain_how"))
        if name in ("chained", "transferred_verified", "widened", "recovered"):
            name += _reach_label(row)
        buckets[name].append(row)
    if not buckets:
        return "no chained rows"
    lines = ["  how                      n   over 0.5   rmse_xy median   gated"]
    order = [name for base in (*CHAIN_BUCKETS, "other")
             for name in (base, base + " (reached)") if name in buckets]
    for name in order:
        group = buckets[name]
        scored = [r["tpr@1.0"] for r in group if r.get("tpr@1.0") is not None]
        errors = [r["rmse_xy"] for r in group if r.get("rmse_xy") is not None]
        gated = sum(1 for r in group if r.get("gate_passed"))
        over = (f"{sum(1 for t in scored if t > 0.5):>4}/{len(scored):<4}"
                if scored else "    -     ")
        median = f"{np.median(errors):>8.2f} m" if errors else "         -"
        lines.append(f"  {name:<22} {len(group):>4}   {over}  {median}"
                     f"      {gated:>4}")
    return "\n".join(lines)
