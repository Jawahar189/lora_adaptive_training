"""2x3 planar affines, the one currency every image-space transform is quoted in.

A transform `A` maps `[u, v]` to `A[:, :2] @ [u, v] + A[:, 2]`. Pixel `(row i,
col j)` covers `[j, j+1) x [i, i+1)` and its centre is `(j + 0.5, i + 0.5)`,
which is what `F.grid_sample(align_corners=False)` assumes when the normalised
coordinate is `2u / W - 1`.

The transforms this package chains together are named after the two frames
they connect, always source-to-target:

    density_px_to_world_xy    density-map pixel -> gravity-up point-cloud xy
    sheet_px_to_crop_px       floor-plan sheet pixel -> crop pixel
    paper_to_sheet_px         paper inches -> floor-plan sheet pixel

Every augmentation composes the op's inverse onto the stored affine so the
ground-truth chain stays exact to floating point, never approximated.
"""
from __future__ import annotations

import numpy as np


def identity() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def as_affine(transform) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (3, 3):
        matrix = matrix[:2]
    if matrix.shape != (2, 3):
        raise ValueError(f"expected a 2x3 affine, got {matrix.shape}")
    return matrix


def compose(second, first) -> np.ndarray:
    """`second` after `first`: the transform that applies `first`, then `second`."""
    first, second = as_affine(first), as_affine(second)
    composed = np.empty((2, 3))
    composed[:, :2] = second[:, :2] @ first[:, :2]
    composed[:, 2] = second[:, :2] @ first[:, 2] + second[:, 2]
    return composed


def chain(*transforms) -> np.ndarray:
    """`chain(A, B, C)` applies A first, then B, then C."""
    chained = identity()
    for transform in transforms:
        chained = compose(transform, chained)
    return chained


def invert(transform) -> np.ndarray:
    matrix = as_affine(transform)
    inverse_linear = np.linalg.inv(matrix[:, :2])
    inverted = np.empty((2, 3))
    inverted[:, :2] = inverse_linear
    inverted[:, 2] = -inverse_linear @ matrix[:, 2]
    return inverted


def apply(transform, points) -> np.ndarray:
    """(N,2) points through the affine. Accepts (N,3) and ignores the rest."""
    matrix = as_affine(transform)
    pts = np.asarray(points, dtype=np.float64)
    single_point = pts.ndim == 1
    pts = np.atleast_2d(pts)[:, :2]
    out = pts @ matrix[:, :2].T + matrix[:, 2]
    return out[0] if single_point else out


def scaling(scale_x: float, scale_y: float | None = None) -> np.ndarray:
    scale_y = scale_x if scale_y is None else scale_y
    return np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0]])


def translation(offset_x: float, offset_y: float) -> np.ndarray:
    return np.array([[1.0, 0.0, offset_x], [0.0, 1.0, offset_y]])


def crop_to_origin(x0: float, y0: float) -> np.ndarray:
    """Coordinates of the parent image -> coordinates of a crop whose top-left
    corner sits at `(x0, y0)` in the parent."""
    return translation(-x0, -y0)


def rotation_quarter_turns(turns: int, width: int, height: int) -> np.ndarray:
    """Coordinates of a (height, width) image -> coordinates in `np.rot90(img, turns)`.

    `np.rot90` turns counter-clockwise; one turn takes input `(u, v)` to
    `(v, W - u)` in an image that is now `(W, H)`. Further turns compose with
    the swapped dimensions.
    """
    rotation = identity()
    current_w, current_h = width, height
    for _ in range(turns % 4):
        step = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, float(current_w)]])
        rotation = compose(step, rotation)
        current_w, current_h = current_h, current_w
    return rotation


def rotation_about(radians: float, pivot_x: float = 0.0, pivot_y: float = 0.0) -> np.ndarray:
    """A proper rotation by `radians` about `(pivot_x, pivot_y)`.

    The same handedness as `from_sim2`: it takes (1, 0) towards (cos, sin), so
    an angle composed here reads the same way as a `Sim2.theta`.

    Unlike `rotation_quarter_turns` this is a WORLD-space rotation, for points
    on their way into a count and never for the count itself -- an arbitrary
    angle on a density map would have to interpolate, and an interpolated count
    is not a count. The determinant is +1 by construction, so nothing it
    touches can become a mirror.
    """
    cos_a, sin_a = float(np.cos(radians)), float(np.sin(radians))
    linear = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pivot = np.array([float(pivot_x), float(pivot_y)])
    offset = pivot - linear @ pivot
    return np.array([[cos_a, -sin_a, offset[0]], [sin_a, cos_a, offset[1]]])


def determinant(transform) -> float:
    """Sign tells chirality: negative means the transform mirrors."""
    return float(np.linalg.det(as_affine(transform)[:, :2]))


def isotropic_scale(transform) -> float:
    """Isotropic scale of a similarity (geometric mean of the singular values
    for anything that is not quite one)."""
    singular_values = np.linalg.svd(as_affine(transform)[:, :2], compute_uv=False)
    return float(np.sqrt(singular_values[0] * singular_values[1]))


def from_sim2(sim2) -> np.ndarray:
    """`align.frames.Sim2` -> 2x3. Uses the same convention as `Sim2.apply`."""
    cos_theta, sin_theta = np.cos(sim2.theta), np.sin(sim2.theta)
    scale = sim2.s
    return np.array([[scale * cos_theta, -scale * sin_theta, sim2.tx],
                     [scale * sin_theta, scale * cos_theta, sim2.ty]])


def to_sim2(transform):
    """2x3 similarity -> `align.frames.Sim2`. Raises if it mirrors."""
    from align.frames import Sim2
    matrix = as_affine(transform)
    if determinant(matrix) <= 0:
        raise ValueError("a mirrored affine is not a Sim2")
    return Sim2(s=isotropic_scale(matrix),
                theta=float(np.arctan2(matrix[1, 0], matrix[0, 0])),
                tx=float(matrix[0, 2]), ty=float(matrix[1, 2]))
