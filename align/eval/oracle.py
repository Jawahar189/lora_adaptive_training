"""The pose a perfect matcher would return, for experiments that need to cheat.

Oracle helpers: they read ground truth, so they live under `align/eval` with the
metrics and are invisible to the solver. An experiment imports these to place a
split where it belongs and then asks a question about the SCORE -- where the
true pose ranks, whether the enumeration contains it -- without a search in the
loop that could be blamed for the answer.

The target is a 4-DOF planar similarity, not the ground-truth 4x4. That is what
`align.search` can emit: a Sim2 applied after a gravity tilt. Scoring against
the full transform would charge the search for a tilt it never had the freedom
to produce.
"""
from __future__ import annotations

import numpy as np

from align import cloud as cloudmod
from align.chain import _umeyama_sim2
from align.frames import target_to_paper
from align.gravity import _rotation_to_z, estimate_gravity, floor_below_trajectory
from align.groundtruth import gt_trajectory


def gt_local_xy(case, ref_map):
    """Ground-truth camera positions in the reference's local solve frame."""
    gt_paper = target_to_paper(gt_trajectory(case.rlt_dir).positions,
                               case.structure.design_tm)[:, :2]
    return gt_paper - ref_map.origin


def best_planar_fit(gt_local, rotations, traj_ply):
    """The closest 4-DOF pose, over gravity frames the caller already has.

    Takes the rotations rather than a cloud, so a caller that has already
    estimated gravity does not pay for it twice -- which is the whole reason the
    runner exists. `rotations` maps up-index to its 3x3.

    Returns (residual_in, up_index, R, Sim2) or None.
    """
    best = None
    for ui, Rm in rotations.items():
        C = traj_ply @ Rm.T
        sim2 = _umeyama_sim2(C[:, :2], gt_local)
        if sim2 is None:
            continue
        resid = float(np.linalg.norm(sim2.apply(C[:, :2]) - gt_local, axis=1).mean())
        if best is None or resid < best[0]:
            best = (resid, ui, Rm, sim2)
    return best


def true_placement(case, ref_map, cloud_xyz, traj_ply):
    """Best planar fit from the ply frame to the truth, in the local frame.

    Both gravity hypotheses are tried and the closer-fitting one wins -- that is
    the up-sign an oracle would have picked, and 49 of 132 splits are ambiguous
    enough that picking one blind is wrong about half the time.

    Returns ((residual_in, up_index, R, Sim2), GravityFit, gt_local_xy).
    """
    gt_local = gt_local_xy(case, ref_map)
    grav = estimate_gravity(cloud_xyz, traj_ply)
    rotations = {ui: _rotation_to_z(up)
                 for ui, up in enumerate(grav.alternatives())}
    return best_planar_fit(gt_local, rotations, traj_ply), grav, gt_local


def wall_band_local(cloud_xyz, traj_ply, Rm):
    """The current split's wall band and walk, in one gravity frame.

    Returns (wall_xy, traj_xy, FloorFit) or (None, None, floor) when the split
    has no usable floor or too thin a band.
    """
    P, C = cloud_xyz @ Rm.T, traj_ply @ Rm.T
    floor = floor_below_trajectory(P, C)
    if not np.isfinite(floor.height) or floor.height <= 0:
        return None, None, floor
    band = cloudmod.wall_band(P, C, floor)
    if band.sum() < cloudmod.MIN_BAND_POINTS:
        return None, None, floor
    return P[band][:, :2], C[:, :2], floor
