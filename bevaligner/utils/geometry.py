"""Frames and fitting, straight from `align`.

`align.frames` is the single source of truth for the point-cloud -> target ->
paper -> pixel chain and nothing here re-derives it. The only additions are the
two affines that chain uses in image space.
"""
from __future__ import annotations

import numpy as np

from align.chain import _umeyama_sim2 as umeyama_sim2  # noqa: F401
from align.frames import (Sim2, SheetGeometry, apply, compose_tm,  # noqa: F401
                          decompose_tm, design_scale, sim3_scale, sim3_yaw,
                          target_to_paper)
from align.visual import ransac_sim2  # noqa: F401

from . import affine


def paper_to_sheet_px_affine(sheet_geometry: SheetGeometry) -> np.ndarray:
    """Paper inches (origin bottom-left, y up) -> sheet pixels (y down).
    Identical to `SheetGeometry.paper_to_pixel`."""
    return np.array([[sheet_geometry.img_w / sheet_geometry.paper_w, 0.0, 0.0],
                     [0.0, -sheet_geometry.img_h / sheet_geometry.paper_h,
                      float(sheet_geometry.img_h)]])


def sheet_px_to_paper_affine(sheet_geometry: SheetGeometry) -> np.ndarray:
    return affine.invert(paper_to_sheet_px_affine(sheet_geometry))


def pose_transform_matrix(sim2: Sim2, rotation_to_gravity_up: np.ndarray,
                          camera_z_gravity_up: np.ndarray,
                          camera_height_above_floor: float,
                          design_tm: np.ndarray | None) -> np.ndarray:
    """The full point-cloud -> target 4x4 for a planar pose.

    The floor of this capture lands on paper z = 0. Cameras sit
    `camera_height_above_floor` above the floor in gravity-up cloud units, so
    the floor is at `median(camera_z) - camera_height_above_floor` and `tz`
    cancels it, scaled. Metrics are horizontal; this only keeps georeferenced
    predictions from sitting a hundred metres under the building.
    """
    floor_z = float(np.median(camera_z_gravity_up)) - float(camera_height_above_floor)
    translation_z = -sim2.s * floor_z
    return compose_tm(sim2.to_sim3(tilt_R=rotation_to_gravity_up, tz=translation_z), design_tm)
