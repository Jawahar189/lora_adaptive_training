"""Scoring one reality's run. Lives under align/eval because it reads the truth.

Kept out of `align.runner` deliberately. The runner orchestrates the SOLVING
stages and must not be able to reach the evaluator -- that was the exemption
`align.pipeline` used to hold, and removing it was the point of the boundary
re-cut. So the runner takes a scorer as an argument and defaults it to this one;
a run with `scoring=False` never imports this module at all.

Everything here happens after the pose exists and feeds nothing back.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from align.eval.oracle import best_planar_fit
from align.eval.reality import compose_pose, reality_metrics
from align.frames import target_to_paper
from align.groundtruth import gt_trajectory


def score_run(run, p) -> bool:
    """Score the pose against ground truth. Strictly after; feeds nothing back.

    Reads the truth ONCE and derives everything from it: the metrics, the local
    frame the oracle fits in, and the coverage figure. The prior's cameras come
    off the reference rather than from disk, and the oracle reuses the gravity
    frames stage_cloud already produced -- so scoring costs one file read, not a
    second pass over the capture.
    """
    t0 = time.time()
    case, ref_map = run.case, run.reference
    traj = run.geometry["trajectory"]
    gt = gt_trajectory(case.rlt_dir)
    order = {n: k for k, n in enumerate(traj.names)}
    shared = [n for n in gt.names if n in order]
    if len(shared) < 2:
        return run.note("score", False, t0, "no images in common with the truth")
    sfm = traj.positions[[order[n] for n in shared]]
    gt_index = {n: k for k, n in enumerate(gt.names)}
    gtp = np.asarray(gt.positions)[[gt_index[n] for n in shared]]
    mpi, geo = ref_map.metres_per_inch, case.structure.georeferenced

    # One conversion of the truth into the plan, reused three ways.
    gt_paper = target_to_paper(gtp, case.structure.design_tm)[:, :2]
    gt_local = gt_paper - ref_map.origin

    prior_cams = ref_map.prior_camera_xy
    if prior_cams is not None and len(prior_cams):
        d, _ = cKDTree(prior_cams).query(gt_paper)
        run.context["overlap_5m"] = float((d * mpi <= p.overlap_radius_m).mean())

    for pose in run.poses.values():
        pose.update(reality_metrics(pose["tm"], sfm, gtp, mpi, geo,
                                    thresholds=p.thresholds))

    if p.score_oracle:
        views = run.geometry["views"]
        sfm_rot = {ui: v["R"] for ui, v in views.items()}
        best = best_planar_fit(gt_local, sfm_rot, sfm)
        if best is not None:
            _r, ui, Rm, true_sim2 = best
            v = views[ui]
            tm = compose_pose(true_sim2, Rm, ref_map, v["cam_z"],
                              v["floor"].height, case.structure.design_tm)
            run.poses["oracle"] = dict(
                tm=tm, **reality_metrics(tm, sfm, gtp, mpi, geo,
                                         thresholds=p.thresholds))
            run.geometry["true_sim2"] = true_sim2
            run.geometry["oracle_up_index"] = ui
            run.geometry["truth_sfm"] = sfm
            run.geometry["truth_gtp"] = gtp
            # Is the chosen pose the right one, by the same definition the
            # scale/yaw experiment used? Position, heading and scale all have to
            # agree -- a pose can be a metre away and still be facing backwards.
            chosen = run.hypotheses[0].sim2
            dpos = float(np.linalg.norm(
                chosen.apply(run.geometry["predicted_view"]["traj_xy"])
                - true_sim2.apply(v["traj_xy"]), axis=1).mean()) * mpi
            dyaw = abs(((np.degrees(chosen.theta - true_sim2.theta) + 180)
                        % 360) - 180)
            ratio = chosen.s / true_sim2.s
            run.context.update(
                pose_error_m=dpos, yaw_error_deg=float(dyaw),
                scale_ratio=float(ratio),
                top1_correct=bool(dpos <= p.pos_tol_m and dyaw <= p.yaw_tol_deg
                                  and 1 / p.scale_tol <= ratio <= p.scale_tol))
    run.geometry["gt_local"] = gt_local
    # The same truth in the frame the sheet plot draws in. Kept here rather than
    # re-derived there so the answer is still read exactly once per run.
    run.geometry["gt_target"] = gtp
    m = run.poses["predicted"]
    return run.note("score", True, t0,
                    f"rmse_xy {m['rmse_xy']:.2f} m, tpr@1 {m['tpr@1.0']:.2f}")


