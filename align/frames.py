"""Coordinate-frame algebra. Every conversion between frames lives here.

Frames
------
ply    -- the arbitrary SfM/panovggt frame a split is reconstructed in.
paper  -- the architectural sheet, in INCHES, origin bottom-left, y up.
pixel  -- floormap.png, origin top-left, y DOWN.
utm    -- projected metres (the CRS named in geo_first_point.json).

`tm` (the thing we solve for) maps ply -> utm when the structure has a
design_tm, and ply -> paper when it does not. We always solve in `paper` and
compose to `utm` only at the very end.

The pixel mapping is taken from production's own plotting code:
    px = paper_x / paper_w * img_w
    py = (1 - paper_y / paper_h) * img_h
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# 4-DOF planar similarity: the actual search space
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Sim2:
    s: float       # uniform scale
    theta: float   # yaw, radians, counter-clockwise
    tx: float
    ty: float

    @property
    def R(self) -> np.ndarray:
        c, sn = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -sn], [sn, c]])

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return self.s * (np.asarray(xy)[:, :2] @ self.R.T) + np.array([self.tx, self.ty])

    def to_sim3(self, tilt_R: np.ndarray | None = None, tz: float = 0.0) -> np.ndarray:
        """Compose into a 4x4. `tilt_R` maps the ply frame to gravity-up first."""
        c, sn = np.cos(self.theta), np.sin(self.theta)
        Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        R = Rz if tilt_R is None else Rz @ tilt_R
        M = np.eye(4)
        M[:3, :3] = self.s * R
        M[:3, 3] = [self.tx, self.ty, tz]
        return M


def sim3_scale(M: np.ndarray) -> float:
    return float(np.linalg.norm(M[:3, 0]))


def sim3_yaw(M: np.ndarray) -> float:
    """Yaw in radians of the 4x4's rotation part."""
    return float(np.arctan2(M[1, 0], M[0, 0]))


def is_similarity(M: np.ndarray, tol: float = 1e-6) -> bool:
    A = np.asarray(M)[:3, :3]
    s = np.linalg.norm(A, axis=0)
    if s.max() / s.min() - 1.0 > tol:
        return False
    Q = A / s
    return bool(np.abs(Q.T @ Q - np.eye(3)).max() <= tol)


def is_identity(M: np.ndarray | None, tol: float = 1e-9) -> bool:
    """True when a design transform carries no georeference.

    Either it is absent, or it is literally the identity -- both mean the target
    frame IS the paper, so there are no UTM metres to report.
    """
    return M is None or bool(np.allclose(np.asarray(M), np.eye(4), atol=tol))


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to (N,3) or (N,2) points; 2-D input is treated as z=0."""
    p = np.asarray(pts, dtype=np.float64)
    if p.shape[1] == 2:
        p = np.c_[p, np.zeros(len(p))]
    return p @ M[:3, :3].T + M[:3, 3]


# --------------------------------------------------------------------------
# paper <-> pixel
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SheetGeometry:
    paper_w: float   # inches
    paper_h: float
    img_w: int       # pixels
    img_h: int

    @property
    def ppi(self) -> float:
        """Pixels per inch. Measured values across the 37 structures:
        87.9, 150, 200, 300, 600 -- never hardcode."""
        return self.img_w / self.paper_w

    def check(self, tol: float = 3e-3) -> None:
        """Fail loudly rather than silently skew if the aspect ratio disagrees."""
        ar_img = self.img_w / self.img_h
        ar_paper = self.paper_w / self.paper_h
        if abs(ar_img - ar_paper) / ar_paper > tol:
            raise ValueError(
                f"aspect mismatch: image {ar_img:.5f} vs paper {ar_paper:.5f}")
        ppi_y = self.img_h / self.paper_h
        if abs(ppi_y - self.ppi) / self.ppi > tol:
            raise ValueError(f"anisotropic ppi: {self.ppi:.2f} x vs {ppi_y:.2f} y")

    def paper_to_pixel(self, paper_xy: np.ndarray) -> np.ndarray:
        p = np.asarray(paper_xy, dtype=np.float64)
        px = p[:, 0] / self.paper_w * self.img_w
        py = (1.0 - p[:, 1] / self.paper_h) * self.img_h
        return np.stack([px, py], axis=1)

    def pixel_to_paper(self, pix_xy: np.ndarray) -> np.ndarray:
        q = np.asarray(pix_xy, dtype=np.float64)
        x = q[:, 0] / self.img_w * self.paper_w
        y = (1.0 - q[:, 1] / self.img_h) * self.paper_h
        return np.stack([x, y], axis=1)


# --------------------------------------------------------------------------
# design_tm: paper inches -> UTM
# --------------------------------------------------------------------------
def load_design_tm(path) -> np.ndarray:
    """Accepts both shapes present in the dataset: {"tm":[16],"offset":[3]}
    (10 structures) and a bare list of 16 floats (8 structures).

    The `offset` field is NOT applied -- design_tm maps to absolute UTM.
    """
    raw = json.loads(Path(path).read_text())
    flat = raw["tm"] if isinstance(raw, dict) else raw
    M = np.asarray(flat, dtype=np.float64).reshape(4, 4)
    if not is_similarity(M, tol=1e-6):
        raise ValueError(f"{path}: design_tm is not a similarity")
    return M


def design_scale(design_tm: np.ndarray) -> float:
    """Metres per paper inch. Observed range 2.44 .. 78.5."""
    return sim3_scale(design_tm)


def design_z_datum(design_tm: np.ndarray) -> float:
    """The sheet's z datum. Takes 8 distinct values across the 18 files
    (64.12, 68.69, 100.0, 103.0, 106.03, 108.0, 109.0, 115.38) -- read it,
    never assume 100."""
    return float(design_tm[2, 3])


# --------------------------------------------------------------------------
# final composition
# --------------------------------------------------------------------------
def compose_tm(ply_to_paper: np.ndarray, design_tm: np.ndarray | None) -> np.ndarray:
    """The predicted `tm`: ply -> UTM if the structure is georeferenced,
    otherwise ply -> paper inches."""
    return ply_to_paper if design_tm is None else design_tm @ ply_to_paper


def decompose_tm(tm: np.ndarray, design_tm: np.ndarray | None) -> np.ndarray:
    """Inverse of compose_tm: recover the ply -> paper part."""
    return tm if design_tm is None else np.linalg.inv(design_tm) @ tm


def target_to_paper(pts: np.ndarray, design_tm: np.ndarray | None) -> np.ndarray:
    """Map target-frame points (UTM metres, or already paper inches) to paper."""
    if design_tm is None:
        return np.asarray(pts, dtype=np.float64)
    return apply(np.linalg.inv(design_tm), pts)
