"""Gravity and the wall band, through `align.gravity` and `align.cloud`.

One `GravityView` per up-vector hypothesis: the cloud rotated so gravity is -z,
its wall-band points, and each point's height relative to its nearest camera as
a fraction of the camera height. The band is stored wide so a caller can
tighten it (the solver's default band, or a jittered one) without touching
the cloud again.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from align.cloud import BAND_HI, BAND_LO, MIN_BAND_POINTS  # noqa: F401
from align.gravity import (FloorFit, GravityFit, _rotation_to_z,  # noqa: F401
                           assumed_z_up, estimate_gravity, floor_below_trajectory, known_up)

WIDE_BAND = (-0.95, 0.35)     # what the cache keeps; the default band sits inside


@dataclass
class GravityView:
    """The cloud seen from one gravity hypothesis, ready to render."""

    up_index: int
    rotation_to_gravity_up: np.ndarray   # 3x3, point cloud -> gravity-up
    band_xy: np.ndarray                  # (N, 2) float32, wide band, gravity-up xy
    # (N,) float32, (z - z_of_nearest_camera) / camera_height_above_floor
    band_relative_height: np.ndarray
    band_camera_index: np.ndarray        # (N,) int32, index of the nearest camera
    camera_xy: np.ndarray                # (M, 2) float64
    camera_z: np.ndarray                 # (M,) float64, gravity-up
    camera_height_above_floor: float     # cloud units
    floor_relative_mad: float

    def select_band_points(self, low: float = BAND_LO, high: float = BAND_HI) -> np.ndarray:
        """Band points inside (low, high) x camera height."""
        inside = (self.band_relative_height > low) & (self.band_relative_height < high)
        return self.band_xy[inside]


def rotations_per_hypothesis(gravity_fit: GravityFit) -> dict[int, np.ndarray]:
    return {up_index: _rotation_to_z(up)
            for up_index, up in enumerate(gravity_fit.alternatives())}


def band_points_with_nearest_camera(cloud_gravity_up: np.ndarray,
                                    cameras_gravity_up: np.ndarray,
                                    floor: FloorFit, low: float, high: float):
    """Points banded against their nearest camera, as `align.cloud.wall_band`
    does, returning the relative height and camera index as well."""
    points = np.asarray(cloud_gravity_up, dtype=np.float64)
    cameras = np.asarray(cameras_gravity_up, dtype=np.float64)
    camera_height = float(floor.height)
    _, nearest = cKDTree(cameras[:, :2]).query(points[:, :2])
    relative_height = (points[:, 2] - cameras[nearest, 2]) / camera_height
    inside = (relative_height > low) & (relative_height < high)
    return points[inside][:, :2], relative_height[inside], nearest[inside]


def make_gravity_view(cloud_xyz: np.ndarray, camera_positions_xyz: np.ndarray, up_index: int,
                      rotation_to_gravity_up: np.ndarray,
                      wide_band=WIDE_BAND,
                      floor: FloorFit | None = None) -> GravityView | str:
    """The cloud in one gravity frame, or a string saying why it is unusable.

    `floor` is the fit for this hypothesis when the caller already has one --
    `estimate_gravity` fits every candidate sign to choose between them, so
    for a hypothesis it considered the answer is known before this is called.
    """
    cloud_up = cloud_xyz @ rotation_to_gravity_up.T
    cameras_up = camera_positions_xyz @ rotation_to_gravity_up.T
    if floor is None:
        floor = floor_below_trajectory(cloud_up, cameras_up)
    if not np.isfinite(floor.height) or floor.height <= 0:
        return "no_floor"
    band_xy, relative_height, camera_index = band_points_with_nearest_camera(
        cloud_up, cameras_up, floor, *wide_band)
    in_default_band = (relative_height > BAND_LO) & (relative_height < BAND_HI)
    if in_default_band.sum() < MIN_BAND_POINTS:
        return "thin_band"
    return GravityView(
        up_index=up_index,
        rotation_to_gravity_up=np.asarray(rotation_to_gravity_up, dtype=np.float64),
        band_xy=band_xy.astype(np.float32),
        band_relative_height=relative_height.astype(np.float32),
        band_camera_index=camera_index.astype(np.int32),
        camera_xy=cameras_up[:, :2], camera_z=cameras_up[:, 2],
        camera_height_above_floor=float(floor.height),
        floor_relative_mad=float(floor.rel_mad))


def rebanded_view(cloud_xyz: np.ndarray, camera_positions_xyz: np.ndarray,
                  up_index: int, rotation_to_gravity_up: np.ndarray,
                  camera_height: float, floor_relative_mad: float = 0.0,
                  wide_band=WIDE_BAND) -> GravityView | str:
    """`view` re-cut at a substituted camera height, in the same gravity frame.

    The floor histogram latches onto stacked material or a scaffold deck often
    enough to matter, and always LOW, which cuts the wall band to a slab. The
    height is the one thing about a walk that is known in advance -- a 360 rig
    on a monopod -- so where the fit is implausible the known height is
    substituted and the band re-cut from the cloud. `camera_height` is in cloud
    units, like `FloorFit.height`.

    Takes the rotation rather than a `GravityView`, because the split that most
    needs this is the one that produced no view at all: `no_floor` means the
    histogram found nothing, which is exactly when a known height is the only
    thing left to band against.

    Banding stays per-nearest-camera, as `align.cloud.wall_band` does, so a ramp
    still does not smear the band into a wedge.
    """
    if not np.isfinite(camera_height) or camera_height <= 0:
        return "no_floor"
    rotation = np.asarray(rotation_to_gravity_up, dtype=np.float64)
    cloud_up = cloud_xyz @ rotation.T
    cameras_up = camera_positions_xyz @ rotation.T
    floor = FloorFit(height=float(camera_height),
                     mad=float(floor_relative_mad) * float(camera_height),
                     n_used=len(cameras_up), per_camera=np.zeros(0))
    band_xy, relative_height, camera_index = band_points_with_nearest_camera(
        cloud_up, cameras_up, floor, *wide_band)
    in_default_band = (relative_height > BAND_LO) & (relative_height < BAND_HI)
    if in_default_band.sum() < MIN_BAND_POINTS:
        return "thin_band"
    return GravityView(
        up_index=int(up_index), rotation_to_gravity_up=rotation,
        band_xy=band_xy.astype(np.float32),
        band_relative_height=relative_height.astype(np.float32),
        band_camera_index=camera_index.astype(np.int32),
        camera_xy=cameras_up[:, :2], camera_z=cameras_up[:, 2],
        camera_height_above_floor=float(camera_height),
        floor_relative_mad=float(floor_relative_mad))


def metric_scale_views(cloud_xyz: np.ndarray, camera_positions_xyz: np.ndarray,
                       gravity_fit: GravityFit, views: dict, rig_height_m: float,
                       metres_per_unit: float) -> dict:
    """`views` re-cut at the camera height a metric scale implies, in place of the floor fit's.

    A metric scale of the cloud -- metres per unit, from the capture's IMU -- and the rig's height
    in metres give the camera's height in cloud units without looking for a floor, so a split the
    floor fit found nothing under, or latched onto a slab or a deck for, is banded at the height the
    walk was actually shot at. Each hypothesis is re-cut there (`rebanded_view`); one the metric
    height leaves too thin a band for keeps what it had. A scale that is not a positive number
    leaves `views` as they are.

    On the 34 jobs of the 2-3 Nov report holding the 114 splits with no floor (624 realities), with
    the IMU's scale: within 2 m 322 -> 393 realities (86 fixed, 15 broken), 51 of the 114 with no
    floor among them.
    """
    height = (float(rig_height_m) / float(metres_per_unit)
              if np.isfinite(metres_per_unit) and metres_per_unit > 0 else float("nan"))
    if not np.isfinite(height) or height <= 0:
        return views
    out = {}
    for up_index, rotation in rotations_per_hypothesis(gravity_fit).items():
        view = rebanded_view(cloud_xyz, camera_positions_xyz, up_index, rotation, height)
        out[up_index] = view if isinstance(view, GravityView) else views.get(up_index, view)
    return out


def gravity_views(cloud_xyz: np.ndarray, camera_positions_xyz: np.ndarray, up: np.ndarray | None = None):
    """Every gravity hypothesis of a cloud:
    `(GravityFit, {up_index: GravityView | reason})`.

    `up`, when the caller knows it (a registered prior's, from its tm.json), is used as it is (one view).
    ASSUME_Z_UP=1 in the environment skips the estimate and takes the cloud's +z as up (one view),
    for clouds stored z-up already; preprocess and infer both come through here."""
    if up is not None:
        gravity_fit = known_up(cloud_xyz, camera_positions_xyz, up)
    elif os.environ.get("ASSUME_Z_UP", "") not in ("", "0"):
        gravity_fit = assumed_z_up(cloud_xyz, camera_positions_xyz)
    else:
        gravity_fit = estimate_gravity(cloud_xyz, camera_positions_xyz)
    ups = gravity_fit.alternatives()
    views = {up_index: make_gravity_view(cloud_xyz, camera_positions_xyz, up_index, rotation,
                                         floor=gravity_fit.floor_for(ups[up_index]))
             for up_index, rotation in rotations_per_hypothesis(gravity_fit).items()}
    return gravity_fit, views
