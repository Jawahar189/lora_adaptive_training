"""wall_band's reaction to a floor-height correction, on synthetic points.

A wrong floor height doesn't just mis-score a reality: it slices the wall band
at the wrong z, which is the actual mechanism behind the current-side
floor-fit bug (docs/PRIOR_CAPTURE.md). This pins that a corrected FloorFit
moves the band the way the physics says it should, independent of anything
runner-level.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.cloud import wall_band
from align.gravity import FloorFit

CAM_Z = 5.0        # every camera sits at this height, up-frame
TRUE_RIG = 5.0      # the real camera-to-floor drop
WRONG_RIG = 2.0      # what a fit that latched onto a mezzanine would report


def _cams(n=20):
    return np.c_[np.linspace(0, 10, n), np.zeros(n), np.full(n, CAM_Z)]


def _wall_points(n=400):
    """Points spread over the true wall band (z in (0.75, 6.25) at rig=5)."""
    z = np.linspace(0.6, 6.4, n)
    x = np.linspace(0, 10, n)
    return np.c_[x, np.zeros(n), z]


def _floor(height):
    return FloorFit(height=height, mad=0.02, n_used=50,
                    per_camera=np.full(50, height), beyond=0.005)


def test_a_wrong_floor_height_cuts_the_band_too_high():
    """The classic current-side failure: fit locks onto a surface well below
    the true floor, so `floor.height` (the reported drop) is too large --
    which narrows the retained slab to a strip near the top of the true wall."""
    P, C = _wall_points(), _cams()
    wrong = wall_band(P, C, _floor(WRONG_RIG))
    correct = wall_band(P, C, _floor(TRUE_RIG))
    assert correct.sum() > wrong.sum()


def test_correcting_the_floor_height_recovers_excluded_wall_points():
    P, C = _wall_points(), _cams()
    probe = np.array([[5.0, 0.0, 1.0]])          # rel = -4.0 under either band
    wrong = wall_band(probe, C, _floor(WRONG_RIG))    # band (-1.7, 0.5): excluded
    correct = wall_band(probe, C, _floor(TRUE_RIG))   # band (-4.25, 1.25): included
    assert not wrong[0]
    assert correct[0]


def test_wall_band_raises_on_an_unusable_floor():
    P, C = _wall_points(), _cams()
    with pytest.raises(ValueError):
        wall_band(P, C, _floor(float("nan")))
    with pytest.raises(ValueError):
        wall_band(P, C, _floor(0.0))
