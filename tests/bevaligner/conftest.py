"""Synthetic prior pools, so the affine and transform tests need no cache."""
import numpy as np
import pytest

from bevaligner.data import prior_bev


def make_split(reality_id="RLTsynthetic", *, band_paper_xy=None, camera_paper_xy=None,
               rig_height_in=2.11, camera_paper_z_median=100.0, names=None):
    band = (np.zeros((0, 2)) if band_paper_xy is None
            else np.asarray(band_paper_xy, dtype=np.float32))
    cameras = (np.zeros((0, 2)) if camera_paper_xy is None
               else np.asarray(camera_paper_xy, dtype=np.float64))
    return prior_bev.PriorSplit(
        reality_id=reality_id, capture_id="CAPsynthetic", capture_date="2026-09-01",
        band_paper_xy=band,
        # mid-band, so the default (unjittered) narrowing keeps every point
        band_relative_height=np.zeros(len(band), dtype=np.float32),
        camera_paper_xy=cameras, camera_paper_z_median=camera_paper_z_median,
        camera_names=tuple(names or ()), rig_height_in=rig_height_in)


def make_pool(band_paper_xy, camera_paper_xy, **kwargs):
    split = make_split(band_paper_xy=band_paper_xy, camera_paper_xy=camera_paper_xy,
                       **kwargs)
    return prior_bev.PriorPool(key="synthetic", splits=(split,))


@pytest.fixture
def pool_factory():
    return make_pool
