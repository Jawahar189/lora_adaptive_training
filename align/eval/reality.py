"""Per-reality accuracy in metres: the format results are reported in.

One row per reality, and every run writes it. Two things are persisted together
and neither is useful without the other:

  the transform   16 floats, the actual answer. Kept so a threshold change, a
                  new metric or a re-analysis costs nothing instead of a
                  seventy-minute re-solve.
  the metrics     RMSE split into plan and height, and the TPR curve.

Error is HORIZONTAL. `rmse_xy` and `tpr@X` both measure distance in plan only,
and height is reported beside them as `rmse_z` rather than folded in.

That split is not tidiness. The search solves a planar similarity and never
touches height at all; z is set afterwards by levelling this capture against
the prior. A combined 3-D number would let a bad floor fit masquerade as a bad
match -- and it did: at the oracle pose, where the horizontal fit is exact by
construction, a 3-D rmse still read as high as 2 m on nothing but the height
estimate. `rmse_xy` measures the match, `rmse_z` measures the floor fit, and
neither is charged for the other.

`tpr@X` is the fraction of a reality's photographs landing within X metres in
plan. Reported across 0.1 to 2.0 m because a single threshold hides the shape:
a pose can be metre-accurate on every image, or right on half of them and wild
on the rest, and those are different failures.

Metres throughout. Where the structure is georeferenced the target frame is
already metres; where it is not it is paper inches, carried through the sheet
scale -- so `scale_source` travels with every row, because a distance derived
through an estimated scale is an estimate.
"""
from __future__ import annotations

import numpy as np

from align.frames import apply as apply4
from align.frames import compose_tm

THRESHOLDS = (0.1, 0.5, 1.0, 1.5, 2.0)

#: Written by every scorer, in this order.
COLUMNS = ("reality_id", "structure", "capture_id", "pose", "status",
           "georeferenced", "scale_source", "metres_per_inch", "n_images",
           "rmse_xy", "rmse_z", "median_xy", "p90_xy", "max_xy",
           *[f"tpr@{t}" for t in THRESHOLDS],
           "rig_height_m", "rig_implausible", "floor_fit_corrected",
           "yaw_fallback", "sign_margin", "sign_uncertain", "overlap_5m")

# The monopod, and the range a fitted height has to fall in to be believable.
# Outside it the floor fit found something that is not the floor -- a stairwell,
# an open edge, the ground outside -- which corrupts the wall band and the scale
# prior as well as the height, so it is recorded rather than silently used.
RIG_PLAUSIBLE_M = (1.2, 3.5)


def level_tz(sim2, reference, cam_z_up, floor_height) -> float:
    """Height for a pose the planar search never solved for.

    Matches this capture's camera heights to the PRIOR's, rather than landing
    its floor on the prior's floor. Both are legitimate -- the same rig walked
    both -- but floor-landing compounds two independent floor fits, and matching
    cameras uses one. Measured over 14 realities at the oracle pose, where the
    horizontal fit is exact and any residual is purely this estimator:
    floor-landing gave a median rmse_z of 0.150 m, camera-matching 0.077 m.
    """
    prior_z = reference.prior_camera_z
    if prior_z is None or not len(prior_z):        # nothing to level against
        return float(reference.floor_z_in
                     - sim2.s * (float(np.mean(cam_z_up)) - floor_height))
    return float(np.median(prior_z) - sim2.s * float(np.median(cam_z_up)))


def compose_pose(sim2, tilt_R, reference, cam_z_up, floor_height,
                 design_tm) -> np.ndarray:
    """A planar pose plus a gravity tilt, into the full ply -> target transform.

    The local solve frame is the reference's raster with its origin at (0, 0);
    `reference.origin` puts it back on the sheet. Without the height term every
    georeferenced prediction would sit at paper z = 0, about a hundred metres
    under the building, and every distance metric would be measuring that.
    """
    M = sim2.to_sim3(tilt_R=tilt_R,
                     tz=level_tz(sim2, reference, cam_z_up, floor_height))
    M[0, 3] += reference.origin[0]
    M[1, 3] += reference.origin[1]
    return compose_tm(M, design_tm)


def reality_metrics(tm, sfm_positions, gt_positions, metres_per_inch,
                    georeferenced, thresholds=THRESHOLDS) -> dict:
    """Per-image error for one reality, summarised. Metres, horizontal."""
    pred = apply4(np.asarray(tm), np.asarray(sfm_positions, dtype=np.float64))
    d = pred - np.asarray(gt_positions, dtype=np.float64)
    if not georeferenced:
        d = d * metres_per_inch          # the target frame is paper inches
    dxy = np.linalg.norm(d[:, :2], axis=1)
    dz = np.abs(d[:, 2])
    out = {
        "n_images": int(len(dxy)),
        "rmse_xy": float(np.sqrt((dxy ** 2).mean())),
        "rmse_z": float(np.sqrt((dz ** 2).mean())),
        "median_xy": float(np.median(dxy)),
        "p90_xy": float(np.percentile(dxy, 90)),
        "max_xy": float(dxy.max()),
    }
    for t in thresholds:
        out[f"tpr@{t}"] = float((dxy <= t).mean())
    return out


def rig_height_m(floor_height, scale, metres_per_inch) -> float:
    """This capture's implied camera height. It should be about 2.11 m."""
    return float(floor_height * scale * metres_per_inch)


def summarise(rows, thresholds=THRESHOLDS) -> str:
    """A fixed block of text for the end of a run, so runs stay comparable."""
    if not rows:
        return "  no realities scored"
    lines = [f"  n={len(rows)} realities, "
             f"{sum(r.get('n_images', 0) for r in rows)} images"]
    for k in ("rmse_xy", "rmse_z"):
        v = np.array([r[k] for r in rows if r.get(k) is not None])
        lines.append(f"    {k:8s} median {np.median(v):8.3f} m   "
                     f"p90 {np.percentile(v, 90):9.3f} m")
    for t in thresholds:
        v = np.array([r[f"tpr@{t}"] for r in rows if r.get(f"tpr@{t}") is not None])
        lines.append(f"    tpr@{t:<4} mean {v.mean():.3f}   median {np.median(v):.3f}   "
                     f"realities over 0.5: {(v > 0.5).sum():3d}/{len(v)}")
    return "\n".join(lines)


class ResultWriter:
    """Writes the transform and the metrics for every reality, always, as it goes.

    Three files, opened once and flushed per reality so a run that dies at
    reality 90 leaves 89 usable results rather than nothing:

      <stem>.jsonl          everything, including the 16 floats of each pose
      <stem>.csv            one row per (reality, pose), the reporting format
      <stem>_predictions.jsonl
                            {"reality_id", "tm"} plus `PREDICTION_EXTRAS` --
                            the shape the existing evaluation scripts already
                            consume, which read by key and ignore the rest

    A reality that cannot be solved still gets a row, carrying its reason. An
    abstention is a result; dropping it would quietly shrink the denominator and
    flatter every average.
    """

    #: Per-pose keys carried in the predictions JSONL beside the transform.
    #: Empty here on purpose: that file's shape is the classical pipeline's
    #: output contract, so a subclass computing something new widens it for its
    #: own runs rather than for everybody's.
    PREDICTION_EXTRAS: tuple = ()

    def __init__(self, stem, thresholds=THRESHOLDS):
        import csv as _csv
        from pathlib import Path as _Path
        stem = _Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds
        self.paths = dict(jsonl=stem.with_suffix(".jsonl"),
                          csv=stem.with_suffix(".csv"),
                          predictions=stem.with_name(stem.name + "_predictions.jsonl"))
        self._jsonl = open(self.paths["jsonl"], "w")
        self._pred = open(self.paths["predictions"], "w")
        self._csvf = open(self.paths["csv"], "w", newline="")
        self._csv = _csv.DictWriter(self._csvf, fieldnames=list(COLUMNS),
                                    extrasaction="ignore")
        self._csv.writeheader()
        self.rows = []

    def write(self, case, poses: dict, status: str = "ok", **context) -> None:
        """`poses` maps a name to {"tm": 4x4, **metrics}. May be empty."""
        import json as _json
        import numpy as _np
        base = dict(reality_id=case.reality_id, structure=case.structure.key,
                    capture_id=case.capture_id,
                    georeferenced=bool(case.structure.georeferenced),
                    status=status, **context)
        record = dict(base, poses={})
        for name, pose in (poses or {}).items():
            tm = _np.asarray(pose["tm"], dtype=float).reshape(4, 4)
            metrics = {k: v for k, v in pose.items() if k != "tm"}
            record["poses"][name] = dict(metrics, tm=[float(x) for x in tm.ravel()])
            self._csv.writerow(dict(base, pose=name, **metrics))
            if name == "predicted":
                prediction = {"reality_id": case.reality_id,
                              "tm": [float(x) for x in tm.ravel()]}
                # A key the pose never set, or set to None, is left out rather
                # than written as null: the readers all test `is None` on what
                # they need, and an absent field reads as "not measured".
                prediction.update({key: metrics[key] for key in self.PREDICTION_EXTRAS
                                   if metrics.get(key) is not None})
                self._pred.write(_json.dumps(prediction) + "\n")
                self.rows.append(dict(base, **metrics))
        if not poses:
            self._csv.writerow(dict(base, pose="none"))
        self._jsonl.write(_json.dumps(record) + "\n")
        for f in (self._jsonl, self._pred, self._csvf):
            f.flush()

    def close(self) -> None:
        for f in (self._jsonl, self._pred, self._csvf):
            f.close()

    def report(self) -> str:
        return summarise(self.rows, self.thresholds)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
