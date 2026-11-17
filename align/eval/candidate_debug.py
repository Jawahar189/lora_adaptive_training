"""Where truth sits among the candidates the search actually evaluated.

Two failure modes look identical from the outside -- the answer is wrong --
but call for different fixes:

  ENUMERATION -- the true (scale, yaw, translation) was never proposed. No
      amount of reranking recovers it; the grid itself has to change.
  RANKING     -- it was proposed, but the score function ranked something
      else higher. The grid is fine; the score isn't.

`gt_probe_row` tells them apart for one reality: it checks whether the grid's
own scale bracket and yaw window reach the truth, then probes what the SAME
score function (`align.search.search`) makes of the true pose itself, snapped
to the nearest grid-representable scale and yaw. `hypothesis_metrics` gives
the raw material for the second question -- every evaluated candidate's own
distance from truth, not just the winner's.

Kept out of `align.runner` for the same reason `align.eval.scoring` is: this
reads ground truth, transitively, through `run.geometry["true_sim2"]`. A run
with `debug_candidates=False` never imports this module.
"""
from __future__ import annotations

import numpy as np

from align.eval.reality import compose_pose, reality_metrics
from align.search import SearchGrid, TranslationPrior, search

PROBE_THRESHOLDS = (1.0, 1.5)


def hypothesis_metrics(run) -> list[dict]:
    """One dict per `run.hypotheses[i]`, its own distance from ground truth.

    Goes through `compose_pose` + `reality_metrics`, the same pipeline that
    produces `rmse_xy`/`tpr@X` for the `predicted`/`oracle` poses elsewhere in
    a run, so these numbers are directly comparable -- not a cheaper
    approximation in the search's own local 2D frame.
    """
    if "true_sim2" not in run.geometry:
        return []
    ref_map = run.reference
    views = run.geometry["views"]
    design_tm = run.case.structure.design_tm
    mpi, geo = ref_map.metres_per_inch, run.case.structure.georeferenced
    sfm, gtp = run.geometry["truth_sfm"], run.geometry["truth_gtp"]

    out = []
    for h in run.hypotheses:
        v = views[h.up_index]
        tm = compose_pose(h.sim2, v["R"], ref_map, v["cam_z"], v["floor"].height,
                          design_tm)
        m = reality_metrics(tm, sfm, gtp, mpi, geo, thresholds=PROBE_THRESHOLDS)
        out.append(dict(
            reality_id=run.case.reality_id, rank=h.rank, score=h.score,
            scale=h.scale, yaw_deg=h.yaw_deg, up_index=h.up_index,
            tx=h.sim2.tx, ty=h.sim2.ty, wall=h.wall, free=h.free,
            scale_prior=h.scale_prior, aspect=h.aspect,
            inliers=h.inliers, n_segments=h.n_segments,
            rmse_xy=m["rmse_xy"], **{"tpr@1.0": m["tpr@1.0"],
                                     "tpr@1.5": m["tpr@1.5"]}))
    return out


def _nearest(values: np.ndarray, target: float, dist) -> tuple[float, float]:
    """The candidate value closest to `target`, and that distance."""
    d = dist(values, target)
    i = int(np.argmin(d))
    return float(values[i]), float(d[i])


def gt_probe_row(run, p, hyp_metrics: list[dict]) -> dict:
    """The enumeration/ranking verdict for one reality.

    `hyp_metrics` is `hypothesis_metrics(run)`'s own output, reused so
    `winner_rmse_xy`/`winner_tpr@1.0` don't get recomputed.
    """
    row = dict.fromkeys(
        ["up_index", "true_scale", "true_yaw_deg", "grid_scale_lo",
         "grid_scale_hi", "nearest_grid_scale", "scale_in_bracket",
         "yaw_window_deg", "nearest_grid_yaw", "yaw_gap_deg", "yaw_reachable",
         "gt_score", "gt_wall", "gt_free", "gt_scale_prior", "gt_aspect",
         "gt_rmse_xy", "gt_tpr@1.0", "gt_tpr@1.5",
         "winner_score", "winner_rmse_xy", "winner_tpr@1.0", "score_gap",
         "visual_pose_err_m", "visual_yaw_err_deg", "visual_scale_ratio",
         "visual_correct", "visual_inlier_frac"])
    row["reality_id"] = run.case.reality_id
    row["status"] = run.status

    fit = run.geometry.get("visual")
    if fit is not None and "true_sim2" in run.geometry:
        true_sim2 = run.geometry["true_sim2"]
        v = run.geometry["views"][fit["up_index"]]
        vt = run.geometry["views"][run.geometry["oracle_up_index"]]
        dpos = float(np.linalg.norm(fit["sim2"].apply(v["traj_xy"])
                                    - true_sim2.apply(vt["traj_xy"]), axis=1).mean())
        dpos *= run.reference.metres_per_inch
        dyaw = abs(((np.degrees(fit["sim2"].theta - true_sim2.theta) + 180) % 360) - 180)
        ratio = fit["sim2"].s / true_sim2.s
        row.update(visual_pose_err_m=dpos, visual_yaw_err_deg=float(dyaw),
                   visual_scale_ratio=float(ratio),
                   visual_correct=bool(dpos <= p.pos_tol_m and dyaw <= p.yaw_tol_deg
                                       and 1 / p.scale_tol <= ratio <= p.scale_tol),
                   visual_inlier_frac=fit["inlier_frac"])
    # an accepted visual pose never searched, so there is no grid to probe
    if (run.status != "ok" or "true_sim2" not in run.geometry
            or "sheet_map" not in run.geometry
            or not run.hypotheses or not hyp_metrics):
        return row

    ui = run.geometry["oracle_up_index"]
    true_sim2 = run.geometry["true_sim2"]
    same_ui = [h for h in run.hypotheses if h.up_index == ui]
    if not same_ui:
        return row

    true_scale = true_sim2.s
    true_yaw_deg = float(np.degrees(true_sim2.theta)) % 360.0
    scales = np.array([h.scale for h in same_ui])
    yaws = np.array([h.yaw_deg for h in same_ui])

    grid_scale_lo, grid_scale_hi = float(scales.min()), float(scales.max())
    nearest_grid_scale, _ = _nearest(
        scales, true_scale, lambda v, t: np.abs(np.log(v / t)))
    nearest_grid_yaw, yaw_gap_deg = _nearest(
        yaws, true_yaw_deg, lambda v, t: np.abs((v - t + 180) % 360 - 180))
    yaw_window_deg = p.yaw_window_deg

    row.update(
        up_index=ui, true_scale=true_scale, true_yaw_deg=true_yaw_deg,
        grid_scale_lo=grid_scale_lo, grid_scale_hi=grid_scale_hi,
        nearest_grid_scale=nearest_grid_scale,
        scale_in_bracket=bool(grid_scale_lo <= true_scale <= grid_scale_hi),
        yaw_window_deg=yaw_window_deg, nearest_grid_yaw=nearest_grid_yaw,
        yaw_gap_deg=yaw_gap_deg,
        yaw_reachable=bool(yaw_gap_deg <= yaw_window_deg))

    # The probe: the SAME score function, pinned to the nearest grid scale and
    # yaw, biased onto the true translation via a disc one search-cell wide --
    # small enough that landing off-truth costs only sub-cell error, not
    # "wherever the correlation surface happens to peak".
    v = run.geometry["views"][ui]
    sheet_map, free_map = run.geometry["sheet_map"], run.geometry["free_map"]
    sres = run.geometry["search_res"]
    grid = SearchGrid(scales=np.array([nearest_grid_scale]),
                      yaws_deg=np.array([nearest_grid_yaw]), viewports=[0])
    prior = TranslationPrior(centre=np.array([true_sim2.tx, true_sim2.ty]),
                             radius=sres, ref=np.array([[0.0, 0.0]]))
    probe = search(v["wall_xy"], v["traj_xy"], sheet_map, (0.0, 0.0), sres,
                   grid, up_index=ui, w_wall=p.w_wall, w_free=p.w_free,
                   w_scale=p.w_scale, w_aspect=p.w_aspect,
                   scale_prior=v["scale_prior"], prior=prior,
                   interior=free_map, top_k=1)

    winner = hyp_metrics[0]
    row.update(winner_score=run.hypotheses[0].score,
              winner_rmse_xy=winner["rmse_xy"],
              **{"winner_tpr@1.0": winner["tpr@1.0"]})

    if probe:
        ph = probe[0]
        # `search` folds the translation prior's soft bonus into `score`, which
        # the winner never had. Rebuild the score from its terms so the gap is
        # the score function's own verdict, not the prior's thumb on the scale.
        ph.score = (p.w_wall * ph.wall - p.w_free * ph.free
                    + p.w_scale * ph.scale_prior
                    + (p.w_aspect * ph.aspect if np.isfinite(ph.aspect) else 0.0))
        ref_map = run.reference
        design_tm = run.case.structure.design_tm
        mpi, geo = ref_map.metres_per_inch, run.case.structure.georeferenced
        tm = compose_pose(ph.sim2, v["R"], ref_map, v["cam_z"],
                          v["floor"].height, design_tm)
        m = reality_metrics(tm, run.geometry["truth_sfm"],
                            run.geometry["truth_gtp"], mpi, geo,
                            thresholds=PROBE_THRESHOLDS)
        row.update(
            gt_score=ph.score, gt_wall=ph.wall, gt_free=ph.free,
            gt_scale_prior=ph.scale_prior, gt_aspect=ph.aspect,
            gt_rmse_xy=m["rmse_xy"],
            **{"gt_tpr@1.0": m["tpr@1.0"], "gt_tpr@1.5": m["tpr@1.5"]})
        row["score_gap"] = ph.score - row["winner_score"]

    return row
