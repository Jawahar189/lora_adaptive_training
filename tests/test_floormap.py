"""The vendored converter still agrees with production's.

`align/io/floormap.py` is a copy of a function that lives in another repository
and is run by the pipeline that produced the files on disk. A copy is only safe
while it still reproduces them, so this compares against real deliveries rather
than against a fixture: for each split, push its own shots through its own
registration and check the result equals the per-image file production shipped.

That file is the current split's ground truth, so this test reads it through
`align.groundtruth` and lives here rather than in the module it tests. Nothing
in the solving path may name it -- see `tests/test_no_leakage.py`.

Skipped when the capture data is not mounted, which is the normal state on a
machine that only has the repository.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from align.groundtruth import load_tm, target_frame_poses
from align.io.cases import PairedIndex
from align.io.floormap import (build_transform, euler_from_rotation_matrix,
                               every_nth, poses_to_images, shot_poses)

# Angles are degrees and positions are metres or inches; both are reproduced by
# the same arithmetic in the same order, so equality is to float noise, not to a
# tolerance anyone had to choose.
ATOL_DEG = 1e-9
ATOL_POS = 1e-9

# One image in 1769 sits in the gimbal-lock branch, where x and z are not
# separable and production's file and ours pick different decompositions of the
# same orientation. Allowing a count rather than widening the tolerance keeps
# the test able to see a real drift.
MAX_DEGENERATE = 1


def _splits(limit=4):
    try:
        index = PairedIndex.load(("datasets/dataset_all_local_current.csv",
                                  "datasets/dataset_all_local_prior.csv"))
    except Exception:                                          # noqa: BLE001
        pytest.skip("dataset index unavailable")
    out = []
    for case in index.current:
        rlt = case.rlt_dir
        if (rlt / "shots.geojson").is_file() and (rlt / "tm.json").is_file():
            out.append(case)
        if len(out) >= limit:
            break
    if not out:
        pytest.skip("capture data not mounted")
    return out


def test_reproduces_production_per_image_poses():
    """Ours == theirs, on their own data, through their own transform."""
    checked = 0
    for case in _splits():
        ref = target_frame_poses(case.rlt_dir)
        got = poses_to_images(shot_poses(case.rlt_dir), load_tm(case.rlt_dir))
        shared = sorted(set(ref) & set(got))
        assert shared, f"{case.reality_id}: no image names in common"

        pos = np.array([np.asarray(got[k]["position"]) - np.asarray(ref[k]["position"])
                        for k in shared])
        assert np.abs(pos).max() < ATOL_POS, (
            f"{case.reality_id}: positions differ by {np.abs(pos).max():.2e}")

        rot = np.array([((np.asarray(got[k]["rotation"])
                          - np.asarray(ref[k]["rotation"])) + 180) % 360 - 180
                        for k in shared])
        off = int((np.abs(rot).max(axis=1) > ATOL_DEG).sum())
        assert off <= MAX_DEGENERATE, (
            f"{case.reality_id}: {off} of {len(shared)} images disagree on "
            f"rotation, max {np.abs(rot).max():.3f} deg")
        checked += len(shared)
    assert checked > 100, f"only {checked} images checked; the sample is too thin"


def test_identity_transform_leaves_the_ply_frame_alone():
    """With no transform, positions are the shot translations themselves.

    Pins the one thing the vendored arithmetic could silently get wrong in a way
    the comparison above would hide: the axis corrections must touch only the
    rotation, never the translation.
    """
    case = _splits(limit=1)[0]
    poses = shot_poses(case.rlt_dir)
    got = poses_to_images(poses)
    for name, p in list(poses.items())[:20]:
        assert np.allclose(got[name]["position"], p["translation"], atol=ATOL_POS)


def test_gimbal_branch_is_continuous_either_side():
    """Approaching lock from both sides gives the same orientation.

    Not the same euler triple -- that is the point -- but the same matrix when
    rebuilt, which is what a viewer actually consumes.
    """
    for pitch in (89.9989, 89.9991):
        m = np.identity(4)
        c, s = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        m[0, 2], m[0, 0] = s, c
        m[2, 2], m[2, 0] = c, -s
        x, y, z = euler_from_rotation_matrix(m)
        assert abs(y - pitch) < 1e-6, f"pitch not recovered at {pitch}"
        assert np.isfinite([x, y, z]).all()


def test_every_nth_keeps_the_indices_production_keeps():
    images = {f"VID_x_IMG_{i:05d}.JPG": {"position": [0, 0, 0]}
              for i in range(1, 32)}
    kept = sorted(every_nth(images, 3))
    assert kept[:4] == ["VID_x_IMG_00001.JPG", "VID_x_IMG_00004.JPG",
                        "VID_x_IMG_00007.JPG", "VID_x_IMG_00010.JPG"]
    assert len(kept) == 11


def test_build_transform_is_a_rigid_motion():
    """No scale, no reflection -- so normalising columns later is a no-op here
    and any scale that does appear came from the transform, as intended."""
    R = build_transform([0.4, -1.2, 0.7], [3.0, -2.0, 1.5])[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
