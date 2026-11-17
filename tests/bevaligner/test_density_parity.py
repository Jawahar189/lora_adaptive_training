"""`density.py` was made faster three ways; these pin all three bit-identical.

Each reference below is the code as it stood before the change, verbatim. The
density map is the encoder's input, so a single flipped low bit in it moves
every feature, every match and finally the pose -- "close enough" is not a
property worth testing here.
"""
from __future__ import annotations

import numpy as np
import pytest

from bevaligner.data.density import (query_cdf, render_density_map,
                                       sample_query_pixels)


def reference_sample_query_pixels(image, num_queries, rng, jitter=True):
    """The `Generator.choice(..., p=...)` call that `query_cdf` replaced."""
    flat = np.asarray(image, dtype=np.float64).ravel()
    total_mass = flat.sum()
    if total_mass <= 0:
        raise ValueError("cannot sample queries from an empty density map")
    flat_index = rng.choice(flat.size, size=num_queries, replace=True, p=flat / total_mass)
    rows, cols = np.divmod(flat_index, image.shape[1])
    offset = rng.random((num_queries, 2)) if jitter else np.full((num_queries, 2), 0.5)
    return np.stack([cols + offset[:, 0], rows + offset[:, 1]], axis=1)


def reference_counts_and_image(points, size, extent, gamma=0.5, min_count=1):
    """`np.add.at` scatter and whole-array `np.power`, as they were."""
    points = np.asarray(points, dtype=np.float64)[:, :2]
    lo_x, lo_y, hi_x, hi_y = extent
    span = max(hi_x - lo_x, hi_y - lo_y)
    hi_x, hi_y = lo_x + span, lo_y + span
    units_per_pixel = span / size
    col = np.floor((points[:, 0] - lo_x) / units_per_pixel).astype(np.int64)
    row_from_bottom = np.floor((points[:, 1] - lo_y) / units_per_pixel).astype(np.int64)
    tolerance = 1e-9 * span
    col[(col == size) & (points[:, 0] <= hi_x + tolerance)] = size - 1
    row_from_bottom[(row_from_bottom == size)
                    & (points[:, 1] <= hi_y + tolerance)] = size - 1
    inside = ((col >= 0) & (col < size)
              & (row_from_bottom >= 0) & (row_from_bottom < size))
    row = size - 1 - row_from_bottom[inside]
    counts = np.zeros((size, size), dtype=np.int32)
    np.add.at(counts, (row, col[inside]), 1)
    if min_count > 1:
        counts[counts < min_count] = 0
    image = counts.astype(np.float32)
    if gamma != 1.0:
        image = np.power(image, gamma, dtype=np.float32)
    peak = float(image.max())
    if peak > 0:
        image /= peak
    return image


def density_image(shape, seed, sparsity):
    rng = np.random.default_rng(seed)
    image = np.abs(rng.normal(size=shape)).astype(np.float32)
    image[rng.random(shape) < sparsity] = 0
    return image


def scattered_points(n, seed):
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return (np.random.default_rng(seed).normal(size=(n, 2)) * 100).astype(np.float32)


# 1024 is the eval resolution; the sparse cases are what a thin walk looks like.
@pytest.mark.parametrize("shape", [(64, 64), (256, 256), (1024, 1024), (128, 256)])
@pytest.mark.parametrize("sparsity", [0.0, 0.9, 0.99])
@pytest.mark.parametrize("num_queries", [16, 1024, 2048])
def test_sample_query_pixels_draws_what_choice_drew(shape, sparsity, num_queries):
    image = density_image(shape, 0, sparsity)
    if image.sum() <= 0:
        pytest.skip("no mass to draw from")
    want = reference_sample_query_pixels(image, num_queries, np.random.default_rng(5))
    got = sample_query_pixels(image, num_queries, np.random.default_rng(5))
    assert np.array_equal(got, want)


def test_passing_the_cached_cdf_changes_nothing():
    """The cache must be an optimisation, not a second sampler."""
    image = density_image((512, 512), 1, 0.8)
    cached = query_cdf(image)
    assert np.array_equal(
        sample_query_pixels(image, 2048, np.random.default_rng(3), cdf=cached),
        sample_query_pixels(image, 2048, np.random.default_rng(3)))


def test_sample_query_pixels_still_refuses_an_empty_map():
    with pytest.raises(ValueError):
        sample_query_pixels(np.zeros((32, 32), dtype=np.float32), 8,
                            np.random.default_rng(0))


def test_density_map_caches_its_cdf():
    density_map = render_density_map(scattered_points(5000, 0), 128)
    assert density_map.query_cdf is density_map.query_cdf
    assert np.array_equal(density_map.query_cdf, query_cdf(density_map.image))


@pytest.mark.parametrize("n", [0, 1, 1000, 50_000, 280_000])
@pytest.mark.parametrize("size", [64, 256])
@pytest.mark.parametrize("gamma,min_count", [(0.5, 1), (1.0, 1), (0.7, 3)])
def test_render_density_map_matches_add_at_and_power(n, size, gamma, min_count):
    points = scattered_points(n, 7)
    extent = (-50.0, -50.0, 50.0, 50.0)
    got = render_density_map(points, size, extent=extent, gamma=gamma,
                             min_count=min_count)
    want = reference_counts_and_image(points, size, extent, gamma=gamma,
                                      min_count=min_count)
    assert np.array_equal(got.image.view(np.uint32), want.view(np.uint32))


def test_render_density_map_survives_a_saturated_cell():
    """Every point in one cell: the gamma table is indexed at its largest count."""
    points = np.zeros((10_000, 2), dtype=np.float32)
    got = render_density_map(points, 64, extent=(-1.0, -1.0, 1.0, 1.0))
    want = reference_counts_and_image(points, 64, (-1.0, -1.0, 1.0, 1.0))
    assert np.array_equal(got.image.view(np.uint32), want.view(np.uint32))


# ----------------------------------------------------------------------------
# the window tiling, which used to sweep the whole band once per grid cell

def reference_tiled_windows(pool, side, options):
    """`_tiled_windows` as it stood: a pass over every point, per cell."""
    lo_x, lo_y, hi_x, hi_y = pool.extent
    side = float(min(max(side, 1e-6), max(hi_x - lo_x, 1e-6)))
    step = max(side * options.window_stride_fraction, 1e-9)
    xs = np.arange(lo_x, max(hi_x - side, lo_x) + step, step)
    ys = np.arange(lo_y, max(hi_y - side, lo_y) + step, step)
    points = pool.band_paper_xy
    windows = []
    for x in xs:
        for y in ys:
            inside = ((points[:, 0] >= x) & (points[:, 0] < x + side)
                      & (points[:, 1] >= y) & (points[:, 1] < y + side))
            count = int(inside.sum())
            if count:
                windows.append((count, (float(x), float(y),
                                        float(x + side), float(y + side))))
    if not windows:
        return [pool.extent]
    windows.sort(key=lambda item: -item[0])
    return [window for _count, window in windows[:options.max_windows]]


class _Options:
    def __init__(self, max_windows=128, window_stride_fraction=0.5):
        self.max_windows = max_windows
        self.window_stride_fraction = window_stride_fraction


class _Pool:
    def __init__(self, points, extent):
        self.band_paper_xy, self.extent = points, extent


def _tiling_case(seed, n, snap=True):
    """Points, with a tenth of them snapped onto a lattice.

    The snapped tenth is the point of the fixture: it puts many coordinates
    exactly ON a window edge, which is where a `>=`/`<` pair and a
    `searchsorted` would part company if they were going to.
    """
    rng = np.random.default_rng(seed)
    if n == 0:
        points = np.zeros((0, 2))
        return _Pool(points, (-1.0, -1.0, 1.0, 1.0))
    points = rng.normal(size=(n, 2)) * 200
    if snap:
        points[:n // 10] = np.round(points[:n // 10] / 10) * 10
    extent = (float(points[:, 0].min()), float(points[:, 1].min()),
              float(points[:, 0].max()), float(points[:, 1].max()))
    return _Pool(points, extent)


# A stride other than 0.5 is the case where windows do NOT align to a lattice,
# which is why the counts are found by slicing rather than by binning.
@pytest.mark.parametrize("seed,n,side,stride", [
    (0, 50_000, 80.0, 0.5), (1, 200_000, 40.0, 0.5), (2, 10_000, 300.0, 0.5),
    (3, 100_000, 55.0, 0.3), (4, 5_000, 20.0, 0.7), (5, 0, 50.0, 0.5)])
def test_tiled_windows_matches_the_sweep(seed, n, side, stride):
    from bevaligner.inference.propose import _tiled_windows
    pool = _tiling_case(seed, n)
    options = _Options(128, stride)
    assert _tiled_windows(pool, side, options) == reference_tiled_windows(
        pool, side, options)


def test_tiled_windows_keeps_the_tie_order():
    """Equal counts must stay in the x-major order the sweep appended them in.

    `window_index` reaches the result row and `scripts/e9_window_oracle.py`
    indexes straight into this list, so the ORDER is part of the output.
    """
    from bevaligner.inference.propose import _tiled_windows
    # a uniform lattice: every window holds exactly the same count
    grid = np.stack(np.meshgrid(np.arange(40.0), np.arange(40.0)), -1).reshape(-1, 2)
    pool = _Pool(grid, (0.0, 0.0, 39.0, 39.0))
    options = _Options(16, 0.5)
    assert _tiled_windows(pool, 10.0, options) == reference_tiled_windows(
        pool, 10.0, options)
