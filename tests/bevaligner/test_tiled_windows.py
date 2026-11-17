"""`propose._tiled_windows`: the summed-area count against the definition it replaces."""
from types import SimpleNamespace

import numpy as np
import pytest

from bevaligner.inference.propose import _tiled_windows


def brute_force(pool, side, options):
    """Every window's count by the half-open test itself, x-major, richest first."""
    lo_x, lo_y, hi_x, hi_y = pool.extent
    side = float(min(max(side, 1e-6), max(hi_x - lo_x, 1e-6)))
    step = max(side * options.window_stride_fraction, 1e-9)
    xs = np.arange(lo_x, max(hi_x - side, lo_x) + step, step)
    ys = np.arange(lo_y, max(hi_y - side, lo_y) + step, step)
    p = pool.band_paper_xy
    windows = []
    for x in xs:
        for y in ys:
            count = int(((p[:, 0] >= x) & (p[:, 0] < x + side)
                         & (p[:, 1] >= y) & (p[:, 1] < y + side)).sum())
            if count:
                windows.append((count, (float(x), float(y), float(x + side), float(y + side))))
    if not windows:
        return [pool.extent]
    windows.sort(key=lambda item: -item[0])
    return [w for _, w in windows[:options.max_windows]]


def pool_of(points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    extent = (tuple(points.min(0)) + tuple(points.max(0))) if len(points) else (0.0, 0.0, 10.0, 10.0)
    return SimpleNamespace(extent=extent, band_paper_xy=points)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("stride", [0.5, 0.37, 0.25, 1.0])
def test_matches_the_definition(seed, stride):
    rng = np.random.default_rng(seed)
    clusters = rng.uniform(0, 300, (8, 2))
    points = np.vstack([c + rng.normal(0, 6, (400, 2)) for c in clusters])
    points = np.round(points, 1)               # ties on window edges, on purpose
    pool = pool_of(points)
    for side in (5.0, 23.7, 60.0, 400.0):
        options = SimpleNamespace(window_stride_fraction=stride, max_windows=128)
        assert _tiled_windows(pool, side, options) == brute_force(pool, side, options)


def test_points_exactly_on_window_edges():
    pool = pool_of([[x, y] for x in range(0, 21, 5) for y in range(0, 21, 5)] * 3)
    for side in (5.0, 10.0):
        options = SimpleNamespace(window_stride_fraction=0.5, max_windows=1000)
        assert _tiled_windows(pool, side, options) == brute_force(pool, side, options)


def test_no_points_and_no_windows():
    pool = SimpleNamespace(extent=(0.0, 0.0, 10.0, 10.0), band_paper_xy=np.zeros((0, 2)))
    options = SimpleNamespace(window_stride_fraction=0.5, max_windows=128)
    assert _tiled_windows(pool, 4.0, options) == [pool.extent]
    pool = pool_of([[1.0, 1.0], [2.0, 2.0]])
    assert _tiled_windows(pool, 4.0, SimpleNamespace(window_stride_fraction=0.5, max_windows=0)) == []
