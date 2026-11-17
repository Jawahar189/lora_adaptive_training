"""Metrics. Reads ground truth via align.groundtruth only.

Image-to-image error is reported in two frames:

  pixels  -- always. Positions are carried to paper inches and then to floormap
             pixels with the sheet's own ppi, which varies 87.9..600 across the
             37 structures, so pixel error is not derivable from inch error
             without the per-sheet scale.

  UTM m   -- only where the structure carries a real design transform. Where
             design_tm is absent or identity the target frame IS the paper, and
             there is no georeference to express metres against; those cases
             report pixel error alone rather than a number in fake units.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from align.frames import (decompose_tm, design_scale, is_identity, sim3_scale,
                          sim3_yaw, target_to_paper)

THRESHOLDS_M = (0.25, 0.5, 1.0, 2.0, 5.0)
THRESHOLDS_PX = (5.0, 10.0, 25.0, 50.0, 100.0)


@dataclass
class ErrorStats:
    """Per-image distances summarised. `unit` names what the numbers are."""
    unit: str
    n: int
    median: float
    mean: float
    p90: float
    max: float
    within: dict = field(default_factory=dict)

    @classmethod
    def of(cls, d: np.ndarray, unit: str, thresholds) -> "ErrorStats":
        d = np.asarray(d, dtype=np.float64)
        return cls(unit=unit, n=int(len(d)), median=float(np.median(d)),
                   mean=float(d.mean()), p90=float(np.percentile(d, 90)),
                   max=float(d.max()),
                   within={str(t): float((d <= t).mean()) for t in thresholds})


@dataclass
class CaseMetrics:
    reality_id: str
    structure: str
    status: str
    n_images: int
    georeferenced: bool

    pixel: ErrorStats                    # always present
    utm: ErrorStats | None               # None when design_tm is identity/absent
    paper: ErrorStats                    # inches, the frame we actually solve in

    tpr_1m: float | None                 # None without a georeference
    tpr_1m_2d: float | None
    scale_log_err: float
    yaw_err_deg: float
    yaw_err_mod90_deg: float
    centroid_err_px: float
    centroid_err_utm: float | None
    ppi: float
    metres_per_inch: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def image_errors(pred_tm, gt_tm, gt_positions, sfm_positions, design_tm, geometry):
    """Per-image error vectors in paper inches, floormap pixels, and UTM metres.

    Returns (paper_dist, pixel_dist, utm_dist_or_None). The UTM component is
    withheld rather than faked when the structure has no georeference.
    """
    pred_tm = np.asarray(pred_tm, dtype=np.float64)
    sfm = np.asarray(sfm_positions, dtype=np.float64)
    gt_pos = np.asarray(gt_positions, dtype=np.float64)

    pred_pos = sfm @ pred_tm[:3, :3].T + pred_tm[:3, 3]

    # Paper inches -- the frame the solver works in, and the bridge to pixels.
    pred_paper = target_to_paper(pred_pos, design_tm)
    gt_paper = target_to_paper(gt_pos, design_tm)
    d_paper = np.linalg.norm(pred_paper[:, :2] - gt_paper[:, :2], axis=1)

    # Floormap pixels -- what a reviewer sees when the capture is plotted.
    pred_px = geometry.paper_to_pixel(pred_paper[:, :2])
    gt_px = geometry.paper_to_pixel(gt_paper[:, :2])
    d_px = np.linalg.norm(pred_px - gt_px, axis=1)

    # UTM metres -- only where a real design transform exists. When it does the
    # target frame already IS UTM, so this is the raw 3-D distance.
    d_utm = None
    if not is_identity(design_tm):
        d_utm = np.linalg.norm(pred_pos - gt_pos, axis=1)

    return d_paper, d_px, d_utm, pred_pos


def evaluate(pred_tm, gt_tm, gt_positions, sfm_positions, design_tm, geometry,
             reality_id="", structure="", status="") -> CaseMetrics:
    gt_tm = np.asarray(gt_tm, dtype=np.float64)
    pred_tm = np.asarray(pred_tm, dtype=np.float64)
    gt_pos = np.asarray(gt_positions, dtype=np.float64)

    d_paper, d_px, d_utm, pred_pos = image_errors(
        pred_tm, gt_tm, gt_positions, sfm_positions, design_tm, geometry)

    georef = not is_identity(design_tm)
    # Compare the ply->paper parts so scale and yaw are frame-independent.
    P = decompose_tm(pred_tm, design_tm)
    G = decompose_tm(gt_tm, design_tm)
    dyaw = np.degrees(sim3_yaw(P) - sim3_yaw(G))
    dyaw = (dyaw + 180.0) % 360.0 - 180.0

    px_per_inch = geometry.ppi
    centroid_px = float(np.linalg.norm(
        geometry.paper_to_pixel(target_to_paper(pred_pos, design_tm)[:, :2]).mean(0)
        - geometry.paper_to_pixel(target_to_paper(gt_pos, design_tm)[:, :2]).mean(0)))

    return CaseMetrics(
        reality_id=reality_id, structure=structure, status=status,
        n_images=len(d_px), georeferenced=georef,
        pixel=ErrorStats.of(d_px, "px", THRESHOLDS_PX),
        utm=ErrorStats.of(d_utm, "m", THRESHOLDS_M) if d_utm is not None else None,
        paper=ErrorStats.of(d_paper, "in", tuple(t / px_per_inch for t in THRESHOLDS_PX)),
        tpr_1m=float((d_utm <= 1.0).mean()) if d_utm is not None else None,
        tpr_1m_2d=float((np.linalg.norm(
            (pred_pos - gt_pos)[:, :2], axis=1) <= 1.0).mean())
        if d_utm is not None else None,
        scale_log_err=float(abs(np.log(sim3_scale(P) / sim3_scale(G)))),
        yaw_err_deg=float(abs(dyaw)),
        yaw_err_mod90_deg=float(abs((dyaw + 45.0) % 90.0 - 45.0)),
        centroid_err_px=centroid_px,
        # Never ||t_pred - t_gt||: the raw translation column is entangled with
        # scale and rotation about a UTM origin 250 km away.
        centroid_err_utm=float(np.linalg.norm(pred_pos.mean(0) - gt_pos.mean(0)))
        if d_utm is not None else None,
        ppi=float(px_per_inch),
        metres_per_inch=float(design_scale(design_tm)) if georef else None,
    )
