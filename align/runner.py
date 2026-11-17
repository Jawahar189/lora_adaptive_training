"""Every stage for one reality, driven from Python rather than from a shell.

The experiment scripts are stage-major: each loops over all 131 realities doing
one thing, which is the right shape for measuring a stage and the wrong shape
for running the pipeline. A capture's reference gets rebuilt, its cloud reread
and its floor refitted once per stage, nothing can be stepped through in a
debugger, and chaining the stages means a shell script passing JSONL between
processes.

This is the other axis. One reality goes in, every stage runs against it in
order, and one row comes out:

    params = Params(current_csv=[...], prior_csv=[...])
    for row in run(params):
        ...

`Params` carries every knob any stage takes, so a sweep is a field rather than
an edit. Nothing here shells out; `scripts/run_alignment.py` is a thin argparse
over these same fields.

Ground truth is read only by the stages that score, always after the pose exists,
and never reaches the solver -- `scoring=False` turns all of it off and the
pipeline still produces a transform.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from align import cloud as cloudmod
from align import reference as refmod
from align import rerank as rerankmod
from align import search as searchmod
from align import visual as visualmod
from align.eval.reality import (RIG_PLAUSIBLE_M, THRESHOLDS, compose_pose,
                                reality_metrics, rig_height_m)
from align.gravity import (MAX_REL_MAD, SIGN_MARGIN_MIN, _rotation_to_z,
                           corrected_floor_height, estimate_gravity,
                           floor_below_trajectory, relative_scale_prior,
                           usable_floor)
from align.io import datasets as datasets_mod
from align.io import floormap
from align.io.cases import PairedIndex
from align.io.pairing import prior_reference_for
from align.io.ply import read_ply
from align.io.shots import load_shots
from align.sheet import resample_to


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
@dataclass
class Params:
    """Every knob, grouped by the stage that reads it.

    Defaults are the measured ones -- see `docs/PRIOR_CAPTURE.md` for where each
    number came from. Changing one here changes it for the whole run; changing
    one that affects the reference raster also needs `reference_force`, or the
    cache hands back a reference built with the old value.
    """

    # -- data ---------------------------------------------------------------
    current_csv: Sequence[str] = ()
    prior_csv: Sequence[str] = ()
    realities: Sequence[str] = ()        # explicit ids; empty means all
    limit: int | None = None
    repo_root: str | None = None

    # -- stage: reference ---------------------------------------------------
    reference_stride: int = refmod.MERGE_STRIDE
    reference_cell_m: float = refmod.CELL_M
    reference_pad_m: float = refmod.PAD_M
    coverage_dilate_m: float = refmod.COVERAGE_DILATE_M
    level_gap_m: float = refmod.LEVEL_GAP_M
    consistency_tol_m: float = refmod.CONSISTENCY_TOL_M
    max_rel_mad: float = refmod.MAX_REL_MAD
    height_band: tuple[float, float] = refmod.HEIGHT_BAND
    max_reference_px: int = refmod.MAX_REF_PX
    min_wall_cells: int = refmod.MIN_WALL_CELLS
    reference_cache: str | None = str(refmod.CACHE_DIR)
    reference_force: bool = False

    # -- stage: current cloud ----------------------------------------------
    cloud_stride: int = 3
    min_source_frames: int = 2
    cloud_max_rel_mad: float = MAX_REL_MAD
    correct_floor_fit: bool = True
    sign_margin_min: float = SIGN_MARGIN_MIN

    # -- stage: search ------------------------------------------------------
    search_cell_m: float = 0.20          # align.pipeline's COARSE_CELL_M
    max_search_px: int = 4000
    yaw_window_deg: float = 10.0
    yaw_step_deg: float = searchmod.YAW_STEP_DEG
    yaw_min_strength: float = 0.08
    scale_steps: int = 9
    # Measured on the 100_prod prior-capture set (94 scored) over 40 random
    # 50/50 splits, selecting on the train half only: top-1 within 2 m went
    # 20 -> 31 of 94 (test halves 10.6 -> 14.9 of 47, train-test gap 0.25).
    # The scale prior at 1.0 was over-penalising true poses at the bracket
    # edge; the aspect term, off in align.search, favours them.
    w_wall: float = searchmod.W_WALL
    w_free: float = 0.1                  # 0.5 is the sheet default
    w_scale: float = 0.25                # align.search's own default is 1.0
    w_aspect: float = 0.25               # ...and 0.0
    top_k_per_hypothesis: int = searchmod.TOP_K_PER_HYPOTHESIS
    backend: str = "auto"
    both_gravity_hypotheses: bool = True
    rerank: bool = False                 # segment-inlier re-sort of every proposal

    # -- visual prior: where the panoramas say the cameras are ---------------
    # Measured (scripts/e8_visual_retrieval.py, 100_prod): a pose fitted to
    # retrieved prior cameras is right for 33/94 realities against the
    # search's 30, and the two disagree usefully (either: 39). The RANSAC
    # inlier fraction separates them: >= 0.5 is right 4 times in 5. In the
    # runner, with the correlation refining the fit: top-1 31 vs 30 of 94,
    # realities with tpr@1.0 > 0.95 18 vs 10, median rmse_xy 9.3 vs 21.8 m;
    # hold-out 10 vs 9 of 30. The gain is precision, on the accepted path.
    visual_prior: bool = False
    visual_model: str = visualmod.DEFAULT_MODEL
    visual_cache: str = "cache/desc"
    visual_top_k: int = 5
    visual_radius_m: float = 5.0         # walk-consistency tolerance
    visual_accept_frac: float = 0.5      # take the pose outright above this...
    visual_min_inliers: int = 3          # ...else offer it to the search as a candidate
    visual_workers: int = 16

    # -- stage: scoring -----------------------------------------------------
    scoring: bool = True                 # False solves and never reads truth
    thresholds: Sequence[float] = THRESHOLDS
    pos_tol_m: float = 2.0
    yaw_tol_deg: float = 10.0
    scale_tol: float = 1.15
    overlap_radius_m: float = 5.0
    score_oracle: bool = True            # the best planar pose, for the ceiling

    # -- stage: visualisation ----------------------------------------------
    visualise: bool = False
    viz_dir: str = "out/run/viz"
    viz_width: int = 1100
    viz_top_k: int = 5

    # -- stage: per-image poses --------------------------------------------
    # `images_name` defaults to a name of our own rather than production's.
    # Nothing under align/ may name that file -- tests/test_no_leakage.py holds
    # the line whether the reference is a read or a write -- and the guard is
    # worth more than the convenience. `scripts/run_alignment.py` sits outside
    # the scanned tree and defaults the flag to production's name, so the CLI
    # still writes what a consumer expects.
    write_images: bool = False
    images_dir: str = ""                 # default: <stem>_images/
    images_name: str = "images_predicted.json"
    images_every_nth: int = 1            # 3 keeps the subset production ships
    images_z: float | None = None        # a float pins z, as the viewer does
    images_plot: bool = True
    images_plot_gt: bool = False         # needs scoring; silently absent without
    images_plot_name: str = "floormap.png"
    images_plot_width: int = 1800

    # -- output -------------------------------------------------------------
    stem: str = "out/run/alignment"
    html_report: bool = False            # write <stem>.html: a filterable gallery
    debug_candidates: bool = False       # write <stem>_gt_probe/_grid_hypotheses.csv
    verbose: bool = True

    def dataset(self):
        """The CSVs this run reads: the fields, else the environment.

        Never a remembered path -- a run always states which data it ran on, in
        its own arguments or in the environment it inherited.
        """
        if self.current_csv or self.prior_csv:
            return datasets_mod.resolve(self.current_csv or None,
                                        self.prior_csv or None)
        spec = datasets_mod.from_env()
        if spec is None:
            raise ValueError(
                "no dataset: set current_csv/prior_csv, or "
                f"${datasets_mod.ENV_CURRENT} and ${datasets_mod.ENV_PRIOR}")
        return spec

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------
# per-reality stages
# --------------------------------------------------------------------------
@dataclass
class Stage:
    """One stage's outcome, so a row can say where it stopped and why."""
    name: str
    ok: bool
    seconds: float
    detail: str = ""


@dataclass
class RealityRun:
    """Everything one reality produced. One of these per row."""
    case: object
    stages: list[Stage] = field(default_factory=list)
    status: str = "ok"
    reference: object | None = None
    prior: object | None = None
    hypotheses: list = field(default_factory=list)
    poses: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)   # intermediates, for viz

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def note(self, name, ok, t0, detail=""):
        self.stages.append(Stage(name, ok, time.time() - t0, detail))
        return ok


def stage_reference(run: RealityRun, index, p: Params) -> bool:
    """Build (or load) the prior capture's BEV for this reality."""
    t0 = time.time()
    ref = prior_reference_for(run.case, index)
    if ref is None:
        run.status = "no prior cloud"
        return run.note("reference", False, t0, run.status)
    run.prior = ref
    kw = dict(stride=p.reference_stride, cell_m=p.reference_cell_m,
              pad_m=p.reference_pad_m, coverage_dilate_m=p.coverage_dilate_m,
              level_gap_m=p.level_gap_m, consistency_tol_m=p.consistency_tol_m,
              max_rel_mad=p.max_rel_mad, height_band=p.height_band,
              max_ref_px=p.max_reference_px, min_wall_cells=p.min_wall_cells)
    if p.reference_cache:
        ref_map = refmod.build_cached(ref, run.case.structure,
                                      cache_dir=p.reference_cache,
                                      force=p.reference_force, **kw)
    else:
        ref_map = refmod.build(ref, run.case.structure, **kw)
    if ref_map is None:
        run.status = "no usable prior split"
        return run.note("reference", False, t0, run.status)
    run.reference = ref_map
    run.context.update(metres_per_inch=ref_map.metres_per_inch,
                       scale_source=ref_map.scale_source,
                       prior_splits=len(ref_map.prior_reality_ids),
                       prior_dropped=len(ref_map.dropped),
                       wall_cells=ref_map.wall_cells,
                       same_day=ref.same_day)
    return run.note("reference", True, t0,
                    f"{len(ref_map.prior_reality_ids)} splits, "
                    f"{ref_map.wall_cells} wall cells")


def stage_cloud(run: RealityRun, p: Params) -> bool:
    """Read the current capture, fit gravity, and cut its wall band.

    Both gravity hypotheses are carried when the floor/ceiling test ties -- an
    inverted up-vector mirrors the bird's-eye view, and a mirrored building
    matches nothing at any pose.
    """
    t0 = time.time()
    case = run.case
    if case.cloud_path is None:
        run.status = "no point cloud"
        return run.note("cloud", False, t0, run.status)
    pc = read_ply(case.cloud_path, stride=p.cloud_stride,
                  min_source_frames=p.min_source_frames)
    traj = load_shots(case.rlt_dir)
    grav = estimate_gravity(pc.xyz, traj.positions)
    ups = grav.alternatives() if p.both_gravity_hypotheses else grav.alternatives()[:1]

    views = {}
    for ui, up in enumerate(ups):
        Rm = _rotation_to_z(up)
        P, C = pc.xyz @ Rm.T, traj.positions @ Rm.T
        floor = floor_below_trajectory(P, C)
        if not np.isfinite(floor.height) or floor.height <= 0:
            continue
        if usable_floor(floor, p.cloud_max_rel_mad):
            continue
        band = cloudmod.wall_band(P, C, floor)
        if band.sum() < cloudmod.MIN_BAND_POINTS:
            continue
        views[ui] = dict(R=Rm, floor=floor, wall_xy=P[band][:, :2],
                         traj_xy=C[:, :2], cam_z=C[:, 2])
    if not views:
        run.status = "no usable wall band"
        return run.note("cloud", False, t0, run.status)
    run.geometry.update(views=views, trajectory=traj, n_points=len(pc.xyz),
                        pc_xyz=pc.xyz)
    run.context["sign_ambiguous"] = bool(grav.sign_ambiguous)
    return run.note("cloud", True, t0,
                    f"{len(pc.xyz):,} points, {len(views)} gravity view(s)")


def search_rasters(ref_map, p: Params):
    """The reference resampled to the search cell.

    Stored at 0.10 m so the finer stages have something to pool; correlating
    there is four times the work for no gain, and the FFT is padded by the
    template on every side so the cost is superlinear.
    """
    h, w = ref_map.feats.wall.shape
    bounds = (0.0, 0.0, w * ref_map.res, h * ref_map.res)
    span = max(bounds[2], bounds[3])
    res = max(p.search_cell_m / ref_map.metres_per_inch,
              span / p.max_search_px, ref_map.res)
    shape = (max(int(bounds[3] / res), 1), max(int(bounds[2] / res), 1))
    return (resample_to(ref_map.feats, ref_map.geom, bounds, res, shape).astype(np.float32),
            resample_to(ref_map.free, ref_map.geom, bounds, res, shape).astype(np.float32),
            res)


def stage_visual(run: RealityRun, p: Params, backbone) -> bool:
    """Ask the panoramas where the cameras are, before any correlation.

    Describes the split's images and its prior pool's (cached per split),
    retrieves each current camera's look-alike prior camera, and fits the
    planar pose those places imply (`align.visual.pose_from_retrieval`). The
    result waits in `run.geometry["visual"]`; `run_reality` decides whether
    it replaces the search or merely joins its candidates.
    """
    t0 = time.time()
    case, ref, ref_map = run.case, run.prior, run.reference
    traj = run.geometry["trajectory"]
    q = visualmod.describe(case.rlt_dir, traj.names, backbone, p.visual_cache,
                           case.reality_id, rolls=1, workers=p.visual_workers)[:, 0]
    pool, pool_xy = [], []
    for rid in ref_map.prior_reality_ids:
        t = ref.trajectory(rid)
        pool.append(visualmod.describe(ref.split_dir(rid), t.names, backbone,
                                       p.visual_cache, rid, rolls=visualmod.ROLLS,
                                       workers=p.visual_workers))
        pool_xy.append(t.positions[:, :2] - ref_map.origin)
    if not pool:
        return run.note("visual", False, t0, "no prior images")
    pool, pool_xy = np.concatenate(pool), np.concatenate(pool_xy)
    # The rig-height bracket is wrong for 21 of 94 realities (the true scale
    # sits outside it), and retrieved places are quantised to the prior's
    # cameras, so the fitted scale is only coarse. The check here is sanity
    # against a degenerate fit -- retrievals collapsing onto one prior camera
    # fit a Sim2 that shrinks the walk to a point -- not the rig prior.
    brackets, s0s = {}, {}
    for ui, v in run.geometry["views"].items():
        s0, _bracket = relative_scale_prior(v["floor"], ref_map.camera_height_in)
        s0s[ui] = s0
        brackets[ui] = (s0, (s0 / 2.0, s0 * 2.0) if np.isfinite(s0) else None)
    mpi = ref_map.metres_per_inch
    fit = visualmod.pose_from_retrieval(
        run.geometry["views"], traj.positions, q, pool, pool_xy, brackets,
        top_k=p.visual_top_k, radius=p.visual_radius_m / mpi, tol=2.0 / mpi)
    if fit is not None:
        fit["scale_prior"] = s0s.get(fit["up_index"], float("nan"))
    run.geometry["visual"] = fit
    run.context.update(
        visual_n_query=len(q), visual_n_pool=len(pool),
        visual_inlier_frac=fit["inlier_frac"] if fit else 0.0,
        visual_n_inliers=fit["n_inliers"] if fit else 0,
        visual_n_anchors=fit["n_anchors"] if fit else 0,
        visual_fit=fit["how"] if fit else None)
    if fit is None:
        return run.note("visual", False, t0, f"{len(q)} images, no pose from {len(pool)} prior images")
    return run.note("visual", True, t0,
                    f"{fit['n_inliers']}/{len(q)} cameras agree ({fit['how']}, view {fit['up_index']})")


VISUAL_SCALE_SPAN = 0.35      # log-scale half-width searched around the visual pose
VISUAL_YAW_SPAN_DEG = 9.0     # ...and heading half-width, at the search's own step


def visual_local_search(run: RealityRun, p: Params, backend=None) -> list:
    """The panoramas propose, the correlation refines.

    Retrieved places are the PRIOR's camera positions, so the pose fitted to
    them is right to a metre or two and to ~15% in scale -- a good place to
    look, not a final answer. Search the same wall/free score over a small
    neighbourhood of it: scales within VISUAL_SCALE_SPAN, headings within
    VISUAL_YAW_SPAN_DEG, translations keeping the walk's centre inside a
    `visual_radius_m` disc. The candidates come back scored exactly like the
    global search's, so they can replace it or compete with it.
    """
    fit = run.geometry["visual"]
    ref_map = run.reference
    if "sheet_map" not in run.geometry:
        sheet_map, free_map, sres = search_rasters(ref_map, p)
        run.geometry.update(sheet_map=sheet_map, free_map=free_map, search_res=sres)
    v = run.geometry["views"][fit["up_index"]]
    sim, mpi = fit["sim2"], ref_map.metres_per_inch
    s0 = fit.get("scale_prior", float("nan"))
    if "scale_prior" not in v and np.isfinite(s0):
        v["scale_prior"] = s0
    yaw0 = float(np.degrees(sim.theta)) % 360.0
    grid = searchmod.SearchGrid(
        scales=sim.s * np.exp(np.linspace(-VISUAL_SCALE_SPAN, VISUAL_SCALE_SPAN, 7)),
        yaws_deg=(yaw0 + np.arange(-VISUAL_YAW_SPAN_DEG, VISUAL_YAW_SPAN_DEG + 1e-9,
                                   p.yaw_step_deg)) % 360.0,
        viewports=[0])
    centroid = np.asarray(v["traj_xy"]).mean(axis=0, keepdims=True)
    prior = searchmod.TranslationPrior(centre=sim.apply(centroid)[0],
                                       radius=p.visual_radius_m / mpi,
                                       ref=centroid, soft_weight=0.0)
    hyps = searchmod.search(
        v["wall_xy"], v["traj_xy"], run.geometry["sheet_map"], (0.0, 0.0),
        run.geometry["search_res"], grid, backend=backend,
        # the panoramas' scale is the prior here, not the rig's: the rig
        # bracket is wrong for 21 of 94 realities, and inside this small
        # neighbourhood the wall term decides between nearby scales
        scale_prior=sim.s, up_index=fit["up_index"],
        w_wall=p.w_wall, w_free=p.w_free, w_scale=p.w_scale, w_aspect=p.w_aspect,
        interior=run.geometry["free_map"], prior=prior, top_k=p.top_k_per_hypothesis)
    for h in hyps:
        h.visual = True
    return hyps


def stage_search(run: RealityRun, p: Params, backend=None) -> bool:
    """Propose poses: every scale and heading, every translation by FFT."""
    t0 = time.time()
    ref_map = run.reference
    sheet_map, free_map, sres = search_rasters(ref_map, p)
    run.geometry.update(sheet_map=sheet_map, free_map=free_map, search_res=sres)
    bk = backend or searchmod.get_backend(p.backend)
    sm = cloudmod.manhattan(sheet_map)
    cell = p.search_cell_m / ref_map.metres_per_inch

    hyps, fallbacks = [], []
    for ui, v in run.geometry["views"].items():
        s0, bracket = relative_scale_prior(v["floor"], ref_map.camera_height_in)
        if not np.isfinite(s0):
            continue
        cm = cloudmod.manhattan(
            cloudmod.wall_raster(v["wall_xy"] * s0, res=cell).mask)
        yaws, fallback = searchmod.yaw_candidates(
            cm.angle, sm.angle, window_deg=p.yaw_window_deg,
            step_deg=p.yaw_step_deg, strength=min(cm.strength, sm.strength),
            min_strength=p.yaw_min_strength)
        fallbacks.append(fallback)
        grid = searchmod.SearchGrid(
            scales=searchmod.scale_candidates(bracket, steps=p.scale_steps),
            yaws_deg=yaws, viewports=[0], fallback_yaw=fallback)
        v["scale_prior"] = s0
        hyps += searchmod.search(
            v["wall_xy"], v["traj_xy"], sheet_map, (0.0, 0.0), sres, grid,
            backend=bk, scale_prior=s0, up_index=ui, w_wall=p.w_wall,
            w_free=p.w_free, w_scale=p.w_scale, w_aspect=p.w_aspect,
            interior=free_map, top_k=p.top_k_per_hypothesis)
    if not hyps:
        run.status = "no hypotheses"
        return run.note("search", False, t0, run.status)
    hyps.sort(key=lambda h: -h.score)
    for i, h in enumerate(hyps):
        h.rank = i
    if p.rerank:
        # Correlation proposed; segment matching decides -- over EVERY
        # proposal, because the right pose is routinely hundreds deep in the
        # correlation ranking (see align.rerank's docstring).
        hyps = rerankmod.rerank(
            hyps, {ui: v["wall_xy"] for ui, v in run.geometry["views"].items()},
            ref_map.feats, ref_map.geom, ref_map.metres_per_inch)
    run.hypotheses = hyps
    run.context["yaw_fallback"] = bool(fallbacks and all(fallbacks))
    run.context["n_hypotheses"] = len(hyps)
    return run.note("search", True, t0,
                    f"{len(hyps)} hypotheses" + (", reranked" if p.rerank else ""))


def stage_pose(run: RealityRun, p: Params) -> bool:
    """Compose the winning planar pose into a full transform."""
    t0 = time.time()
    best = run.hypotheses[0]
    v = run.geometry["views"][best.up_index]
    tm = compose_pose(best.sim2, v["R"], run.reference, v["cam_z"],
                      v["floor"].height, run.case.structure.design_tm)
    run.poses["predicted"] = {"tm": tm}
    run.geometry["predicted_view"] = v
    run.context["rig_height_m"] = rig_height_m(
        v["floor"].height, best.sim2.s, run.reference.metres_per_inch)
    run.context["rig_implausible"] = not (
        RIG_PLAUSIBLE_M[0] <= run.context["rig_height_m"] <= RIG_PLAUSIBLE_M[1])

    sign_margin = float("nan")
    if run.context.get("sign_ambiguous"):
        by_ui: dict[int, float] = {}
        for h in run.hypotheses:
            if h.up_index not in by_ui or h.score > by_ui[h.up_index]:
                by_ui[h.up_index] = h.score
        if len(by_ui) >= 2:
            top, second = sorted(by_ui.values(), reverse=True)[:2]
            sign_margin = top - second
    run.context["sign_margin"] = sign_margin
    run.context["sign_uncertain"] = bool(
        run.context.get("sign_ambiguous") and np.isfinite(sign_margin)
        and sign_margin < p.sign_margin_min)

    return run.note("pose", True, t0,
                    f"scale {best.sim2.s:.4f}, yaw {best.yaw_deg:.1f} deg")


def stage_floor_correct(run: RealityRun, p: Params, backend=None) -> bool:
    """Retry once against the known rig height when the first pass implies an
    implausible one.

    `stage_cloud`'s dispersion gate catches an internally-inconsistent fit, but
    a coherent fit on the wrong surface (a stairwell, an atrium, a level below)
    passes it and only shows up once a scale exists to check the implied rig
    height against -- which is only available after `stage_search` runs. This
    inverts that scale to correct the floor height the same way the prior side
    substitutes a known height for an implausible one (align.reference.build),
    re-cuts the wall band, and re-searches exactly once. Unconditional: it does
    not re-check whether the retry actually fixed the implausibility, mirroring
    the prior side's substitute-and-move-on (a case that cannot be corrected is
    left as the first pass produced it, still flagged `rig_implausible`).
    """
    t0 = time.time()
    best = run.hypotheses[0]
    ui = best.up_index
    v = run.geometry["views"][ui]
    corrected = corrected_floor_height(v["floor"], best.sim2.s,
                                       run.reference.metres_per_inch)
    if corrected is None:
        return run.note("floor_correct", False, t0, "no scale to correct against")
    P = run.geometry["pc_xyz"] @ v["R"].T
    C = np.column_stack([v["traj_xy"], v["cam_z"]])
    band = cloudmod.wall_band(P, C, corrected)
    if band.sum() < cloudmod.MIN_BAND_POINTS:
        return run.note("floor_correct", False, t0,
                        f"corrected band too thin ({band.sum()} points)")

    pre_rig = run.context.get("rig_height_m")
    pre_floor, pre_wall_xy = v["floor"], v["wall_xy"]
    pre_hyps, pre_status = run.hypotheses, run.status
    pre_context = dict(run.context)
    v["floor"], v["wall_xy"] = corrected, P[band][:, :2]

    if not stage_search(run, p, backend=backend):
        # The corrected band gave no hypotheses at all: worse than the first
        # pass, which we know produced one. Put everything back rather than
        # replace a working (if implausible) result with a failed one.
        v["floor"], v["wall_xy"] = pre_floor, pre_wall_xy
        run.hypotheses, run.status, run.context = pre_hyps, pre_status, pre_context
        return run.note("floor_correct", False, t0, "corrected band: no hypotheses")

    stage_pose(run, p)
    run.context["floor_fit_corrected"] = True
    return run.note("floor_correct", True, t0,
                    f"rig {pre_rig:.2f} m -> {run.context['rig_height_m']:.2f} m")


def stage_images(run: RealityRun, p: Params) -> bool:
    """The per-image poses the predicted transform implies, written and drawn.

    The transform is the answer; this is the answer in the shape the floormap
    viewer consumes -- one record per photograph, position and heading, in the
    target frame. It is a pure function of the pose and the shots, so it reads
    nothing new and cannot affect anything upstream of it.

    Runs after scoring rather than immediately after the pose so that the
    ground-truth overlay costs no second read of the truth: `score_run` has
    already converted it into the plan, and this borrows that. With
    `scoring=False` there is simply no overlay, which is the correct behaviour
    for a deployment run rather than a limitation of it.
    """
    t0 = time.time()
    tm = run.poses["predicted"]["tm"]
    dest = Path(p.images_dir or f"{p.stem}_images") / run.case.reality_id
    try:
        poses = floormap.shot_poses(run.case.rlt_dir)
        images = floormap.poses_to_images(poses, tm, override_z=p.images_z)
        if p.images_every_nth > 1:
            images = floormap.every_nth(images, p.images_every_nth)
    except Exception as exc:                                  # noqa: BLE001
        return run.note("images", False, t0, repr(exc)[:120])
    if not images:
        return run.note("images", False, t0, "no shots carried a pose")

    dest.mkdir(parents=True, exist_ok=True)
    path = dest / p.images_name
    path.write_text(json.dumps(images, indent=2))
    run.context["images_path"] = str(path)
    run.context["images_n"] = len(images)

    drew = ""
    if p.images_plot:
        gt = run.geometry.get("gt_target") if p.images_plot_gt else None
        if p.images_plot_gt and gt is None:
            drew = ", no truth to draw"
        try:
            from align import viz_floorplan
            viz_floorplan.render(run.case, images, dest / p.images_plot_name,
                                 gt_target_xyz=gt, width=p.images_plot_width)
            drew = f", plotted{drew}"
        except Exception as exc:                              # noqa: BLE001
            drew = f", plot failed: {exc!r}"[:90]
    return run.note("images", True, t0, f"{len(images)} images{drew}")


def default_scorer():
    """The scorer, imported only when a run actually scores.

    Deferred so that `scoring=False` never pulls the evaluator into the process,
    and so this module cannot reach ground truth by importing it at the top.
    """
    from align.eval.scoring import score_run
    return score_run


def run_reality(case, index, p: Params, backend=None, scorer=None,
                backbone=None) -> RealityRun:
    """Every solving stage for one reality, then scoring if it was asked for.

    Stops at the first stage that cannot proceed, leaving the run carrying the
    reason. `scorer` is injected rather than imported: the solving path has no
    route to the evaluator, which is what the leakage boundary rests on.

    With `visual_prior` the panoramas are asked first: a pose most cameras
    agree on (`visual_accept_frac`) replaces the search; a weaker one joins
    the search's candidates on equal terms; none leaves the search alone.
    """
    run = RealityRun(case=case)
    if not stage_reference(run, index, p):
        return run
    if not stage_cloud(run, p):
        return run
    fit = None
    if p.visual_prior and backbone is not None:
        try:
            stage_visual(run, p, backbone)
        except Exception as exc:                          # noqa: BLE001
            run.note("visual", False, time.time(), repr(exc)[:120])
        fit = run.geometry.get("visual")
    local = []
    if fit is not None and fit["n_inliers"] >= p.visual_min_inliers:
        t0 = time.time()
        local = visual_local_search(run, p, backend=backend)
        run.note("visual_search", bool(local), t0,
                 f"{len(local)} candidates around the panoramas' pose")
    if local and fit["inlier_frac"] >= p.visual_accept_frac:
        # enough cameras agree: the neighbourhood IS the search
        local.sort(key=lambda h: -h.score)
        for i, h in enumerate(local):
            h.rank = i
        run.hypotheses = local
        run.context.update(n_hypotheses=len(local), visual_how="accepted",
                           visual_rank=0)
        run.note("search", True, time.time(),
                 f"skipped: {len(local)} candidates from the panoramas' neighbourhood")
    else:
        if not stage_search(run, p, backend=backend):
            return run
        if local:
            merged = sorted(run.hypotheses + local, key=lambda h: -h.score)
            for i, h in enumerate(merged):
                h.rank = i
            run.hypotheses = merged
            run.context.update(n_hypotheses=len(merged), visual_how="injected",
                               visual_rank=min(h.rank for h in local))
        elif p.visual_prior:
            run.context["visual_how"] = "none"
    stage_pose(run, p)
    run.context.setdefault("floor_fit_corrected", False)
    if p.correct_floor_fit and run.context.get("rig_implausible"):
        stage_floor_correct(run, p, backend=backend)
    if p.scoring:
        score = scorer or default_scorer()
        try:
            score(run, p)
        except Exception as exc:                          # noqa: BLE001
            run.note("score", False, time.time(), repr(exc)[:120])
    if p.write_images:
        stage_images(run, p)
    return run


# --------------------------------------------------------------------------
# the row-major loop
# --------------------------------------------------------------------------
def run(p: Params, on_reality: Callable[[RealityRun], None] | None = None,
        scorer: Callable | None = None) -> Iterator[RealityRun]:
    """Yield one finished RealityRun per row, in index order."""
    index = PairedIndex.load(p.dataset(), p.repo_root)
    cases = list(index.current)
    if p.realities:
        wanted = set(p.realities)
        cases = [c for c in cases if c.reality_id in wanted]
        missing = wanted - {c.reality_id for c in cases}
        if missing:
            raise KeyError(f"unknown reality ids: {sorted(missing)}")
    if p.limit:
        cases = cases[: p.limit]
    backend = searchmod.get_backend(p.backend)
    scorer = scorer or (default_scorer() if p.scoring else None)
    backbone = visualmod.Backbone(p.visual_model) if p.visual_prior else None

    for i, case in enumerate(cases, 1):
        t0 = time.time()
        try:
            r = run_reality(case, index, p, backend=backend,
                            scorer=scorer, backbone=backbone)
        except Exception as exc:                          # noqa: BLE001
            r = RealityRun(case=case, status=f"error: {exc!r}"[:160])
        if p.verbose:
            m = r.poses.get("predicted", {})
            tail = (f"rmse_xy {m['rmse_xy']:7.2f} m  tpr@1 {m['tpr@1.0']:.2f}"
                    if "rmse_xy" in m else r.status)
            if r.context.get("visual_how"):
                tail += (f"  [visual {r.context['visual_how']}, "
                         f"{r.context.get('visual_inlier_frac', 0):.2f} agree]")
            print(f"[{i:3d}/{len(cases)}] {case.reality_id}  "
                  f"{time.time() - t0:5.1f}s  {tail}", flush=True)
        if on_reality is not None:
            on_reality(r)
        yield r
