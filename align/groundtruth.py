"""THE LEAKAGE BOUNDARY: the CURRENT split's own answer.

This module and nothing else may open the current split's tm.json or
images_full.json. Both are ground truth: `tm.json` is the transform being
solved for, and `images_full.json` holds camera positions ALREADY IN THE TARGET
FRAME.

There is no longer any path from here into solver inputs. The solver used to
take its trajectory from `simulate_sfm_trajectory` -- inv(tm) @ images_full, a
sanctioned stand-in for the poses SfM supplies in production -- and now reads
those poses directly from `shots.geojson` via `align.io.shots`. The two agree to
1.2e-9 m across all 273 splits, so nothing downstream changed; what changed is
that this module is now read only by the evaluator.

A PRIOR capture's tm.json is a different thing entirely -- the prior is already
registered in production, so its transform is a legitimate input. It is read by
`align.priors`, which is constructed so that it cannot reach the current split.

Positions only, for the solver's purposes: gravity, scale, Manhattan and
chaining all work from positions alone, and `gt_trajectory` surfaces nothing
else.

The rotations were long left alone because the euler convention was
undocumented and a wrong guess fails silently. It is now pinned -- see
`align/io/floormap.py`, which reproduces this file from shots and a transform
-- so `target_frame_poses` hands back the raw record for the one caller that
has to compare against it. It is still ground truth and still evaluation-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The trajectory type lives outside this module so the solver can hold one
# without importing the boundary.
from align.io.traj import Trajectory, image_index as _img_index


def load_tm(rlt_dir) -> np.ndarray:
    raw = json.loads((Path(rlt_dir) / "tm.json").read_text())
    return np.asarray(raw["tm"], dtype=np.float64).reshape(4, 4)


def is_manually_registered(rlt_dir) -> bool:
    """6 splits carry "reg_tool": true. They were registered by hand, so they
    score tpr=1.0 against their own tm tautologically."""
    raw = json.loads((Path(rlt_dir) / "tm.json").read_text())
    return bool(raw.get("reg_tool", False))


def target_frame_poses(rlt_dir) -> dict:
    """The per-image record as delivered: position AND rotation, target frame.

    Ground truth, and the only reason it exists is to check that the converter
    in `align.io.floormap` still reproduces it. Nothing that solves may call it.
    """
    return json.loads((Path(rlt_dir) / "images_full.json").read_text())


def gt_trajectory(rlt_dir) -> Trajectory:
    """Camera positions in the TARGET frame. Ground truth -- evaluation only."""
    raw = json.loads((Path(rlt_dir) / "images_full.json").read_text())
    names = sorted(raw, key=_img_index)
    pos = np.array([raw[n]["position"] for n in names], dtype=np.float64)
    return Trajectory(names, np.array([_img_index(n) for n in names]), pos, "target")
