"""Augmentation, every op keeping its affine exact.

Geometric ops happen *before* rendering, on the points (subset, band, crop
extent), so the map is re-counted rather than resampled; a quarter turn is the
one pixel op and it is lossless. That now holds on BOTH sides: the target is
rendered from the prior's points into a chosen window
(`prior_bev.render_target`), so there is no crop-of-a-crop resample anywhere.
The photometric jitter is written out by hand because torchvision is absent.

No flips anywhere: a mirror is the failure mode the up-sign ambiguity
produces, and the network must learn to reject it, not to shrug at it.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from ..utils import affine
from ..utils.gravity import BAND_HI, BAND_LO
from . import density


# ----------------------------------------------------------------------------
# density side (points)

def wall_band_mask(record, rng: np.random.Generator | None,
                   band_low_range, band_high_range) -> np.ndarray:
    """Wide-band points narrowed to a (jittered) band. Returns a bool mask."""
    if rng is None:
        low, high = BAND_LO, BAND_HI
    else:
        low, high = rng.uniform(*band_low_range), rng.uniform(*band_high_range)
    return (record.band_relative_height > low) & (record.band_relative_height < high)


def subsample_points(record, mask: np.ndarray, rng: np.random.Generator,
                     camera_window_probability: float, camera_window_min_fraction: float,
                     point_keep_fraction, overlap_bias: float = 0.0) -> np.ndarray:
    """Mimic a partial capture: keep the points seen from a contiguous run of
    cameras, or a random fraction of all points.

    `overlap_bias` is the probability that the camera run is centred on a camera
    the PRIOR also walked past. A walk overlaps its prior only in part on about
    half the dataset, and the part outside has no target to match to: biasing
    the window into the overlap turns those rows from near-empty pairs into
    useful ones, which is why the overlap gate can sit as low as 0.2.
    """
    num_cameras = len(record.camera_xy)
    if num_cameras >= 4 and rng.random() < camera_window_probability:
        window = int(math.ceil(rng.uniform(camera_window_min_fraction, 1.0) * num_cameras))
        overlapping = np.flatnonzero(record.overlap_mask) \
            if len(record.overlap_mask) == num_cameras else np.zeros(0, dtype=int)
        if len(overlapping) and rng.random() < overlap_bias:
            centre = int(rng.choice(overlapping))
            first = int(np.clip(centre - window // 2, 0, num_cameras - window))
        else:
            first = int(rng.integers(0, num_cameras - window + 1))
        return mask & (record.band_camera_index >= first) & \
            (record.band_camera_index < first + window)
    keep_fraction = rng.uniform(*point_keep_fraction)
    return mask & (rng.random(len(mask)) < keep_fraction)


def random_sub_extent(extent, rng: np.random.Generator, scale_range):
    """A random sub-square of a square extent, side in `scale_range` x the span."""
    lo_x, lo_y, hi_x, hi_y = extent
    span = hi_x - lo_x
    side = span * rng.uniform(*scale_range)
    offset_x = rng.uniform(0, span - side)
    offset_y = rng.uniform(0, span - side)
    return (lo_x + offset_x, lo_y + offset_y, lo_x + offset_x + side, lo_y + offset_y + side)


def rotate_density_map(density_map: density.DensityMap, turns: int) -> density.DensityMap:
    size = density_map.size
    image = np.ascontiguousarray(np.rot90(density_map.image, turns))
    rotated_affine = affine.compose(
        density_map.density_px_to_world_xy,
        affine.invert(affine.rotation_quarter_turns(turns, size, size)))
    return density.DensityMap(image=image, density_px_to_world_xy=rotated_affine,
                              world_units_per_pixel=density_map.world_units_per_pixel,
                              extent=density_map.extent,
                              num_points=density_map.num_points)


def draw_rotation(rng: np.random.Generator | None, degrees: float, pivot) -> np.ndarray | None:
    """A random proper rotation about `pivot`, or None when the knob is off.

    None rather than the identity, for two reasons. Composing an identity still
    copies every point, which is the whole cost of the op; and more importantly
    a disabled knob must draw NOTHING from the generator -- one extra draw
    shifts every later one and no earlier run would reproduce.

    The pivot only keeps the rotated coordinates in the same numeric range as
    the originals. Which point it is cannot change the picture, because the
    extent is re-squared around whatever comes out.
    """
    if rng is None or degrees <= 0:
        return None
    return affine.rotation_about(math.radians(float(rng.uniform(-degrees, degrees))),
                                 float(pivot[0]), float(pivot[1]))


def render_record_density(record, size: int, rng: np.random.Generator | None, data_config,
                          min_points: int = 200) -> density.DensityMap | None:
    """The training (rng given) or evaluation (rng None) density map of a record."""
    mask = wall_band_mask(record, rng, data_config.band_low_range, data_config.band_high_range)
    if rng is not None:
        mask = subsample_points(record, mask, rng, data_config.camera_window_probability,
                                data_config.camera_window_min_fraction,
                                data_config.point_keep_fraction,
                                data_config.camera_window_overlap_bias)
    points = record.band_xy[mask]
    if len(points) < min_points:
        return None
    if rng is None:
        return density.render_density_map(points, size)
    # Continuous yaw, on the POINTS and BEFORE the extent. Nothing is lost and
    # nothing is padded, because `square_extent` is recomputed around the
    # rotated cloud, so the square is the rotated cloud's own. The other order
    # -- square first, then turn -- would clip the corners of a square sized
    # for a cloud that has moved, and a clipped corner renders as a zero count,
    # which reads as "no wall here" and is a lie.
    rotation = draw_rotation(rng, data_config.density_rotation_degrees,
                             np.asarray(points, dtype=np.float64)[:, :2].mean(0))
    if rotation is not None:
        points = affine.apply(rotation, points)
    extent = random_sub_extent(density.square_extent(points), rng,
                               data_config.random_crop_scale)
    gamma = float(rng.choice(data_config.density_gammas))
    min_count = int(rng.choice(data_config.density_min_counts))
    density_map = density.render_density_map(points, size, extent=extent, gamma=gamma,
                                             min_count=min_count)
    if density_map.image.max() <= 0:
        return None
    if rotation is not None:
        # The invariant: the op writes itself into `density_px_to_world_xy`
        # here, before `dataset.py` chains it into `density_to_crop`.
        # `render_density_map` returned density px -> ROTATED cloud xy; the
        # inverse takes it the rest of the way back to the cloud the ground
        # truth is quoted in.
        density_map = dataclasses.replace(
            density_map,
            density_px_to_world_xy=affine.compose(affine.invert(rotation),
                                                  density_map.density_px_to_world_xy))
    return rotate_density_map(density_map, int(rng.integers(0, 4)))


# ----------------------------------------------------------------------------
# photometric, torch, (3,H,W) in [0, 1]

_LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def jitter_brightness_contrast(image: torch.Tensor, rng, brightness: float,
                               contrast: float) -> torch.Tensor:
    if brightness > 0:
        image = image * _uniform(rng, max(0.0, 1 - brightness), 1 + brightness)
    if contrast > 0:
        mean_luma = (image * _LUMA).sum(0, keepdim=True).mean()
        image = (image - mean_luma) * _uniform(rng, max(0.0, 1 - contrast),
                                               1 + contrast) + mean_luma
    return image.clamp(0, 1)

