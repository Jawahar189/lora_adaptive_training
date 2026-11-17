"""SfM poses plus a transform, into the floormap viewer's per-image JSON.

Vendored from `image_registration_on_existing_map/src/utils/colmap2floormap.py`
— production's own converter — rather than imported, so this pipeline does not
depend on a sibling checkout staying where it is. `tests/test_floormap.py` pins
it against production's output on real splits; if the upstream convention ever
changes, that test is what notices.

The convention is not guessable and is reproduced here exactly:

  1. `rotation` is an axis-angle vector whose axis is NEGATED before use;
  2. two fixed corrections follow, Rz(pi) then Ry(pi/2), which carry the
     reconstruction's axes onto the viewer's;
  3. the transform is applied on the LEFT, so the same function serves any
     ply -> target transform — production's registration, or ours;
  4. the three columns are normalised to drop scale before reading angles off,
     because the transform carries one and euler angles must not;
  5. angles come out in degrees, three.js `Euler` order "XYZ".

Verified against production's own per-image files on 5 splits covering 1769
images: positions exact, angles equal to 1e-12 degrees. The sixth split differs
on one image of 522, which sits in the gimbal-lock branch (pitch 89.78 deg,
where `x` and `z` trade against each other and the cutoff below picks the other
decomposition). Same orientation, different triple — a degeneracy of euler
angles, not a disagreement about the convention.

Positions here are the honest output of the transform. Production's own
localisation file overrides z with a constant, which is a viewer convention
rather than geometry; `override_z` exists for callers that want to match it and
is off by default.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from align.io.shots import shots_path
from align.io.traj import image_index as _index

# Two constants of the viewer's frame, not of any capture.
_RZ_PI = math.pi
_RY_HALF_PI = math.pi / 2


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues, as a 4x4. Spelled out rather than taken from scipy so the
    vendored arithmetic is byte-comparable with upstream."""
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    m = np.identity(4)
    m[0, 0] = t * x * x + c
    m[0, 1] = t * x * y - s * z
    m[0, 2] = t * x * z + s * y
    m[1, 0] = t * x * y + s * z
    m[1, 1] = t * y * y + c
    m[1, 2] = t * y * z - s * x
    m[2, 0] = t * x * z - s * y
    m[2, 1] = t * y * z + s * x
    m[2, 2] = t * z * z + c
    return m


def build_transform(rotation, translation) -> np.ndarray:
    """One shot's pose as a 4x4 in the ply frame, in the viewer's axes."""
    axis = -np.asarray(rotation, dtype=np.float64)
    angle = float(np.linalg.norm(axis))
    if angle:
        axis = axis / angle
    m = np.identity(4)
    m[:3, 3] = np.asarray(translation, dtype=np.float64)
    m = m @ _axis_angle_matrix(axis, angle)
    m = m @ _axis_angle_matrix(np.array([0.0, 0.0, 1.0]), _RZ_PI)
    m = m @ _axis_angle_matrix(np.array([0.0, 1.0, 0.0]), _RY_HALF_PI)
    return m


def euler_from_rotation_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Degrees, three.js `Euler` order "XYZ".

    The branch guards gimbal lock: at |m02| -> 1 the x and z angles are no
    longer separable and only their sum is determined.
    """
    y = math.degrees(math.asin(max(-1.0, min(1.0, float(m[0, 2])))))
    if abs(m[0, 2]) < 0.99999:
        x = math.degrees(math.atan2(-m[1, 2], m[2, 2]))
        z = math.degrees(math.atan2(-m[0, 1], m[0, 0]))
    else:
        x = math.degrees(math.atan2(m[2, 1], m[1, 1]))
        z = 0.0
    return x, y, z


def shot_poses(rlt_dir) -> dict[str, dict]:
    """Every photograph's pose in the ply frame, keyed by filename.

    `align.io.shots` reads the same file for positions only; rotations are
    needed here and nowhere in the solver, which is why they are read separately
    rather than widened into `Trajectory`.
    """
    raw = json.loads(Path(shots_path(rlt_dir)).read_text())
    out = {}
    for feat in raw.get("features", []):
        props = feat.get("properties") or {}
        name, xyz, rot = (props.get("filename"), props.get("translation"),
                          props.get("rotation"))
        if not name or xyz is None or rot is None:
            continue
        out[name] = {"translation": xyz, "rotation": rot}
    if not out:
        raise ValueError(f"{shots_path(rlt_dir)}: no features carry a pose")
    return out


def poses_to_images(poses: dict, tm: np.ndarray | None = None,
                    override_z: float | None = None) -> dict:
    """`{filename: {"position": [x, y, z], "rotation": [x, y, z]}}`.

    `tm` is any ply -> target transform; passing ours is what makes this a
    prediction rather than a re-export.
    """
    T = None if tm is None else np.asarray(tm, dtype=np.float64).reshape(4, 4)
    out = {}
    for name in sorted(poses, key=lambda n: (_index(n), n)):
        p = poses[name]
        m = build_transform(p["rotation"], p["translation"])
        if T is not None:
            m = T @ m
        for col in range(3):                      # drop the transform's scale
            s = np.linalg.norm(m[:3, col])
            if s:
                m[:3, col] /= s
        x, y, z = euler_from_rotation_matrix(m)
        pos_z = m[2, 3] if override_z is None else float(override_z)
        out[name.split("/")[-1]] = {
            "position": [float(m[0, 3]), float(m[1, 3]), float(pos_z)],
            "rotation": [x, y, z],
        }
    return out


def every_nth(images: dict, n: int, remainder: int = 1) -> dict:
    """The regular subset production's localisation file carries.

    Measured on the delivered files, the kept indices are 1, 4, 7, ... — so the
    remainder is 1, not 0. Upstream's own `main()` passes 0 and would keep a
    different third.
    """
    return {k: v for k, v in images.items() if _index(k) % n == remainder}
