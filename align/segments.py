"""Line segments, and how well a cloud's wall segments are explained by a plan.

The oracle-inlier test (E2) asks a question that has nothing to do with search
quality: placed at the ground-truth pose, how much of the cloud's wall geometry
actually has a counterpart on the sheet? A split that scores zero here cannot be
solved by any geometric method and belongs in the abstain bucket permanently.

Matching is done against a distance transform plus an orientation map rather
than against extracted plan segments: a wall drawn as one long line and seen by
the camera as three occluded fragments should count as three matches, not fail
a one-to-one assignment.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Raster:
    mask: np.ndarray      # bool
    origin: np.ndarray    # (x, y) in inches of pixel (0, 0)
    res: float            # inches per pixel

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy)[:, :2] - self.origin) / self.res


def rasterize(xy: np.ndarray, res: float, pad: float = 0.5,
              bounds: tuple[float, float, float, float] | None = None,
              min_count: int = 1) -> Raster:
    """Top-down occupancy of (N,2) points, `res` inches per pixel."""
    p = np.asarray(xy, dtype=np.float64)[:, :2]
    if bounds is None:
        lo = p.min(0) - pad
        hi = p.max(0) + pad
    else:
        lo = np.array(bounds[:2], dtype=np.float64)
        hi = np.array(bounds[2:], dtype=np.float64)
    w = max(int(np.ceil((hi[0] - lo[0]) / res)), 1)
    h = max(int(np.ceil((hi[1] - lo[1]) / res)), 1)
    ix = ((p[:, 0] - lo[0]) / res).astype(np.int64)
    iy = ((p[:, 1] - lo[1]) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    counts = np.zeros((h, w), dtype=np.int32)
    np.add.at(counts, (iy[ok], ix[ok]), 1)
    return Raster(mask=counts >= min_count, origin=lo, res=res)


def detect_segments(mask: np.ndarray, min_len_px: float = 8.0) -> np.ndarray:
    """(N, 4) array of x0, y0, x1, y1 in pixel coordinates."""
    img = (mask.astype(np.uint8)) * 255
    if img.size == 0 or not mask.any():
        return np.zeros((0, 4))
    lsd = cv2.createLineSegmentDetector()
    lines, _, _, _ = lsd.detect(img)
    if lines is None or len(lines) == 0:
        return np.zeros((0, 4))
    seg = lines.reshape(-1, 4)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return seg[length >= min_len_px]


def orientation_map(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Per-pixel dominant line direction in radians, from the structure tensor.

    Returned modulo pi: a wall has a direction, not a sense.
    """
    f = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    k = (0, 0)
    jxx = cv2.GaussianBlur(gx * gx, k, 2 * sigma)
    jyy = cv2.GaussianBlur(gy * gy, k, 2 * sigma)
    jxy = cv2.GaussianBlur(gx * gy, k, 2 * sigma)
    # Principal gradient direction; the line runs perpendicular to it.
    grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    return (grad + np.pi / 2) % np.pi


def _sample_along(seg: np.ndarray, n: int = 12) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[None, :, None]
    a = seg[:, None, :2]
    b = seg[:, None, 2:]
    return a + (b - a) * t                         # (S, n, 2)


def match_inliers(cloud_segs: np.ndarray, plan_mask: np.ndarray,
                  tol_px: float, ang_tol_deg: float = 5.0,
                  min_frac: float = 0.7, n_samples: int = 12
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Which cloud segments are explained by the plan.

    A segment is an inlier when at least `min_frac` of the points sampled along
    it lie within `tol_px` of plan line-work whose local direction agrees to
    `ang_tol_deg`.

    Returns (inlier mask over segments, per-segment supported fraction).
    """
    if len(cloud_segs) == 0:
        return np.zeros(0, bool), np.zeros(0)
    solid = plan_mask.astype(np.uint8)
    if not solid.any():
        return np.zeros(len(cloud_segs), bool), np.zeros(len(cloud_segs))
    dist = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 3)
    ang = orientation_map(plan_mask)

    pts = _sample_along(cloud_segs, n_samples)                 # (S, n, 2)
    h, w = plan_mask.shape
    px = np.clip(pts[..., 0].astype(np.int64), 0, w - 1)
    py = np.clip(pts[..., 1].astype(np.int64), 0, h - 1)
    inside = ((pts[..., 0] >= 0) & (pts[..., 0] < w)
              & (pts[..., 1] >= 0) & (pts[..., 1] < h))

    d = dist[py, px]
    seg_ang = np.arctan2(cloud_segs[:, 3] - cloud_segs[:, 1],
                         cloud_segs[:, 2] - cloud_segs[:, 0]) % np.pi
    da = np.abs(ang[py, px] - seg_ang[:, None])
    da = np.minimum(da, np.pi - da)                            # direction, not sense

    ok = inside & (d <= tol_px) & (da <= np.radians(ang_tol_deg))
    frac = ok.mean(axis=1)
    return frac >= min_frac, frac
