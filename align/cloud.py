"""Point cloud -> top-down wall raster, as separable stages.

Each function here is a pure stage: inputs in, result out, no hidden state and
no knowledge of ground truth. `align.pipeline` wires them together for one
reality; tests call them directly on synthetic input.

The banding step used to be copy-pasted into two scripts, where a single
reality's cloud was read four times and its floor fitted four times across the
stage-major drivers. It lives here now.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from align.gravity import FloorFit, trajectory_extent
from align.io.ply import PointCloud, read_ply
from align.segments import Raster, rasterize

# Band limits as multiples of the camera's height above the floor. Below the
# floor is nothing; a quarter-height above the camera is where useful wall
# still is without reaching for ceilings and roof structure.
BAND_LO = -0.85
BAND_HI = 0.25

MIN_BAND_POINTS = 200


def load(cloud_path, stride: int = 3, min_source_frames: int = 2) -> PointCloud:
    """Read a split's cloud.

    `source_frame_count >= 2` is a cheap multi-view-confirmed filter;
    `mean_confidence` and `quarantined_count` are constant across every cloud in
    the dataset and carry nothing.
    """
    return read_ply(cloud_path, stride=stride, min_source_frames=min_source_frames)


def wall_band(points_up: np.ndarray, traj_up: np.ndarray,
              floor: FloorFit) -> np.ndarray:
    """Boolean mask selecting the wall band, relative to the LOCAL floor.

    Each point is banded against its nearest camera rather than a global height,
    so a walk that climbs stairs or crosses levels does not smear the band into
    a slab that spans both.
    """
    P = np.asarray(points_up, dtype=np.float64)
    C = np.asarray(traj_up, dtype=np.float64)
    h = float(floor.height)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("floor height is not usable; cannot band")
    _, nearest = cKDTree(C[:, :2]).query(P[:, :2])
    rel = P[:, 2] - C[nearest, 2]
    return (rel > BAND_LO * h) & (rel < BAND_HI * h)


def wall_raster(points_xy: np.ndarray, res: float,
                bounds: tuple[float, float, float, float] | None = None,
                pad: float = 0.5, min_count: int = 1) -> Raster:
    """Top-down occupancy of the banded points, `res` units per pixel."""
    return rasterize(points_xy, res=res, pad=pad, bounds=bounds,
                     min_count=min_count)


def raster_bounds(points_xy: np.ndarray, res: float, pad: float = 0.5,
                  max_px: int = 4000) -> tuple[tuple, float]:
    """Bounds and a possibly-coarsened resolution that keeps the raster bounded.

    The site-plan structure sits at 78.5 m per inch, where a tolerance-derived
    resolution would ask for a raster tens of thousands of pixels across.
    """
    p = np.asarray(points_xy, dtype=np.float64)[:, :2]
    lo = p.min(0) - pad
    hi = p.max(0) + pad
    span = float(np.max(hi - lo))
    if span / res > max_px:
        res = span / max_px
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])), float(res)


@dataclass
class Manhattan:
    angle: float        # radians in [0, pi/2); the dominant direction mod 90 deg
    strength: float     # 0..1; low means no dominant direction exists

    @property
    def degrees(self) -> float:
        return float(np.degrees(self.angle))


def manhattan(mask: np.ndarray, sigma: float = 1.0,
              percentile: float = 97.0) -> Manhattan:
    """Dominant line direction of a binary map, folded modulo 90 degrees.

    Buildings and sheets are overwhelmingly rectilinear, so the strongest
    gradient direction modulo 90 degrees pins yaw to four candidates instead of
    a full sweep. Uses the fourth-harmonic circular mean, which is the standard
    way to average directions with 90-degree symmetry.
    """
    import cv2
    m = np.asarray(mask)
    if m.dtype != np.float32:
        m = m.astype(np.float32)
    if not m.any():
        return Manhattan(angle=0.0, strength=0.0)
    f = cv2.GaussianBlur(m, (0, 0), sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    keep = mag > np.percentile(mag, percentile)
    if keep.sum() < 50:
        return Manhattan(angle=0.0, strength=0.0)
    w = mag[keep]
    theta = np.arctan2(gy[keep], gx[keep]) % np.pi
    c = float(np.sum(w * np.cos(4 * theta)))
    s = float(np.sum(w * np.sin(4 * theta)))
    ang = (np.arctan2(s, c) / 4.0) % (np.pi / 2)
    return Manhattan(angle=float(ang), strength=float(np.hypot(c, s) / w.sum()))
