"""The density map: a gravity-aligned cloud seen from above, as an image.

Points in the wall band are counted into a square grid, gamma-compressed and
normalised. That is the paper's floor-plan proxy, and it is rendered **y-down**
like any image, so that the map and the floor-plan sheet differ by a proper
rotation and never by a mirror (`align.segments.rasterize` is y-up; do not mix
the two).

Ground-truth free: this module knows nothing about where the cloud belongs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DensityMap:
    image: np.ndarray                  # (S, S) float32 in [0, 1], row 0 at the largest y
    density_px_to_world_xy: np.ndarray  # 2x3: density pixel -> gravity-up cloud xy
    world_units_per_pixel: float
    # lo_x, lo_y, hi_x, hi_y in gravity-up cloud xy
    extent: tuple[float, float, float, float]
    num_points: int
    # Built on first use and kept: the search draws queries from this same map
    # once per window, and the distribution is a million entries wide.
    _query_cdf: np.ndarray | None = field(default=None, init=False, repr=False,
                                          compare=False)

    @property
    def query_cdf(self) -> np.ndarray:
        """`query_cdf(self.image)`, computed once."""
        if self._query_cdf is None:
            self._query_cdf = query_cdf(self.image)
        return self._query_cdf

    @property
    def size(self) -> int:
        return int(self.image.shape[0])

    @property
    def world_xy_to_density_px(self) -> np.ndarray:
        from ..utils import affine
        return affine.invert(self.density_px_to_world_xy)


def square_extent(xy: np.ndarray, pad_fraction: float = 0.05,
                  trim_fraction: float = 0.025) -> tuple[float, float, float, float]:
    """A square extent around the cloud, trimmed of horizontal outliers.

    `trim_fraction` is the percentile clipped at each end (the paper's
    rho_xz = 2.5%), `pad_fraction` the margin on each side of the trimmed span.
    The square is centred on the trimmed box so a long thin walk sits in the
    middle of the map with blank space either side, not pressed against one edge.
    """
    points = np.asarray(xy, dtype=np.float64)[:, :2]
    if trim_fraction > 0:
        low = np.percentile(points, 100 * trim_fraction, axis=0)
        high = np.percentile(points, 100 * (1 - trim_fraction), axis=0)
    else:
        low, high = points.min(0), points.max(0)
    centre = (low + high) / 2
    span = float((high - low).max()) * (1 + 2 * pad_fraction)
    span = max(span, 1e-6) * (1 + 1e-9)     # the extreme points stay strictly inside
    half_span = span / 2
    return (float(centre[0] - half_span), float(centre[1] - half_span),
            float(centre[0] + half_span), float(centre[1] + half_span))


def render_density_map(band_xy: np.ndarray, size: int, *,
                       extent: tuple[float, float, float, float] | None = None,
                       pad_fraction: float = 0.05, trim_fraction: float = 0.025,
                       gamma: float = 0.5, min_count: int = 1) -> DensityMap:
    """Count band points into a `size` x `size` grid, gamma-correct, normalise.

    `extent` (lo_x, lo_y, hi_x, hi_y) fixes the area covered, for crops that
    must be re-rendered exactly; otherwise `square_extent` picks it. The extent
    must be square (both axes share one resolution) and is made so if it is not.
    """
    points = np.asarray(band_xy, dtype=np.float64)[:, :2]
    if extent is None:
        extent = square_extent(points, pad_fraction=pad_fraction, trim_fraction=trim_fraction)
    lo_x, lo_y, hi_x, hi_y = extent
    span = max(hi_x - lo_x, hi_y - lo_y)
    hi_x, hi_y = lo_x + span, lo_y + span
    units_per_pixel = span / size

    col = np.floor((points[:, 0] - lo_x) / units_per_pixel).astype(np.int64)
    row_from_bottom = np.floor((points[:, 1] - lo_y) / units_per_pixel).astype(np.int64)
    # a point sitting exactly on the upper edge belongs to the last cell
    tolerance = 1e-9 * span
    col[(col == size) & (points[:, 0] <= hi_x + tolerance)] = size - 1
    row_from_bottom[(row_from_bottom == size)
                    & (points[:, 1] <= hi_y + tolerance)] = size - 1
    inside = ((col >= 0) & (col < size)
              & (row_from_bottom >= 0) & (row_from_bottom < size))
    row = size - 1 - row_from_bottom[inside]                 # y-down image
    # `np.bincount` over the flattened cell index, not `np.add.at`: the same
    # integer counting, several times faster. The whole prior is binned once per
    # window, so this is the inner loop of the search.
    counts = np.bincount(row * size + col[inside],
                         minlength=size * size).reshape(size, size).astype(np.int32)
    if min_count > 1:
        counts[counts < min_count] = 0

    if gamma != 1.0:
        # Raising a million cells costs more than everything else in this
        # function put together, and a cell holds a small integer count: raise
        # each DISTINCT count once and index the result back. Elementwise the
        # same call on the same values, so the image is identical.
        table = np.power(np.arange(int(counts.max()) + 1, dtype=np.float32), gamma,
                         dtype=np.float32)
        image = table[counts]
    else:
        image = counts.astype(np.float32)
    peak = float(image.max())
    if peak > 0:
        image /= peak

    density_px_to_world_xy = np.array([[units_per_pixel, 0.0, lo_x],
                                       [0.0, -units_per_pixel, hi_y]])
    return DensityMap(image=image, density_px_to_world_xy=density_px_to_world_xy,
                      world_units_per_pixel=units_per_pixel,
                      extent=(lo_x, lo_y, hi_x, hi_y), num_points=int(inside.sum()))


def query_cdf(image: np.ndarray) -> np.ndarray:
    """The normalised cumulative density `sample_query_pixels` draws against.

    Split out because the QUERY side's map is fixed for a whole gravity
    hypothesis while the search runs a window at a time, so building this once
    saves rebuilding a million-element distribution for every window. It is
    spelled exactly as `Generator.choice` spells it internally -- normalise,
    `cumsum`, divide by the last entry -- so drawing against it is not merely
    equivalent to the old `choice(..., p=...)` call but identical to it.
    """
    flat = np.asarray(image, dtype=np.float64).ravel()
    total_mass = flat.sum()
    if total_mass <= 0:
        raise ValueError("cannot sample queries from an empty density map")
    cdf = (flat / total_mass).cumsum()
    cdf /= cdf[-1]
    return cdf


def sample_query_pixels(image: np.ndarray, num_queries: int, rng: np.random.Generator,
                        jitter: bool = True, cdf: np.ndarray | None = None) -> np.ndarray:
    """`num_queries` continuous pixel positions drawn with probability
    proportional to density.

    Returns (num_queries, 2) as (u, v) = (col, row). Sub-pixel jitter spreads
    the queries inside each cell so the bilinear feature sample is not always
    at the centre.

    `cdf` is `query_cdf(image)` when the caller already holds it; it draws the
    same pixels and consumes the same randomness either way.
    """
    cdf = query_cdf(image) if cdf is None else cdf
    flat_index = cdf.searchsorted(rng.random(num_queries), side="right")
    rows, cols = np.divmod(flat_index, image.shape[1])
    offset = rng.random((num_queries, 2)) if jitter else np.full((num_queries, 2), 0.5)
    return np.stack([cols + offset[:, 0], rows + offset[:, 1]], axis=1)
