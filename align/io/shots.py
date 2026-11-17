"""Camera poses in the ply frame, from production's own SfM record.

`shots.geojson` is what the reconstruction stage emits: one feature per
photograph, carrying the pose in the arbitrary frame the point cloud is
reconstructed in. That is exactly the input the solver is entitled to, and it
is a real file rather than a reconstruction of one -- which matters, because
the trajectory used to be synthesized by un-applying the ground-truth transform
to the target-frame poses. Measured over all 273 splits in the two dataset
indexes, the two agree to a maximum RMS of 1.2e-9 m, so nothing about the
solver's behaviour changes; what changes is that the solver no longer has to
reach through ground truth to get its own input.

Two quirks:

  * `geometry.coordinates` is EMPTY on every feature -- the position lives in
    `properties.translation`, not in the geometry, so this cannot be read with
    a generic GeoJSON reader;
  * features are not stored in image order, so they are sorted by image index
    here, the same way the target-frame loader sorts.

Positions only, for the reason given in `align.io.traj`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from align.io.traj import Trajectory, image_index

SHOTS_NAME = "shots.geojson"


def shots_path(rlt_dir) -> Path:
    return Path(rlt_dir) / SHOTS_NAME


def load_shots(rlt_dir) -> Trajectory:
    """Camera positions in the ply frame."""
    raw = json.loads(shots_path(rlt_dir).read_text())
    rows = []
    for feat in raw.get("features", []):
        props = feat.get("properties") or {}
        name, xyz = props.get("filename"), props.get("translation")
        if not name or xyz is None:
            continue
        rows.append((image_index(name), name, xyz))
    rows.sort(key=lambda r: (r[0], r[1]))
    if not rows:
        raise ValueError(f"{shots_path(rlt_dir)}: no features carry a translation")
    names = [r[1] for r in rows]
    return Trajectory(names,
                      np.array([r[0] for r in rows]),
                      np.array([r[2] for r in rows], dtype=np.float64),
                      "ply")
