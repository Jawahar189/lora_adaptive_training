"""Run every stage, reality by reality. The one entry point.

Give it a current index and the prior index it references, and it builds each
capture's reference, cuts its wall band, searches, composes a transform, scores
it and draws it -- one reality at a time, in one process. Nothing shells out;
`align.runner` holds the stages as functions and this is argparse over its
`Params`, so the same run is reproducible from Python:

    from align.runner import Params, run
    for r in run(Params(current_csv=["cur.csv"], prior_csv=["prior.csv"])):
        print(r.case.reality_id, r.poses["predicted"]["rmse_xy"])

Every knob any stage takes is a flag here, grouped by the stage that reads it.
Defaults are the measured ones; `docs/PRIOR_CAPTURE.md` says where each came
from.

Output is ROW-MAJOR: one row per reality, carrying its prior-side and
current-side facts side by side, so a result can be read without joining
anything. The transform is kept with the metrics.

  python scripts/run_alignment.py --current-csv CUR.csv --prior-csv PRIOR.csv
  python scripts/run_alignment.py --limit 8 --visualise
  python scripts/run_alignment.py --reality RLT1788173473510 --visualise --no-cache
  python scripts/run_alignment.py --w-free 0 0.25 0.5     # sweep, one run each
  python scripts/run_alignment.py --visualise --html-report   # + browsable gallery
  python scripts/run_alignment.py --reality X --debug-candidates  # enum vs rank
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import reference as refmod
from align import visual as visualmod
from align import viz_prior
from align import viz_report
from align.eval import candidate_debug
from align.eval.reality import COLUMNS, summarise
from align.io import datasets as datasets_mod
from align.runner import Params, run

# One row per reality: the prior side, the current side, and the outcome.
ROW_COLUMNS = [
    "reality_id", "capture_id", "structure", "root", "status", "georeferenced",
    # prior side
    "prior_capture_ids", "prior_splits", "prior_dropped", "prior_wall_cells",
    "prior_same_day", "prior_capture_date", "overlap_5m",
    # current side
    "capture_date", "n_images", "rig_height_m", "rig_implausible",
    "floor_fit_corrected",
    "metres_per_inch", "scale_source", "sign_ambiguous", "sign_margin",
    "sign_uncertain", "yaw_fallback",
    "n_hypotheses", "scale", "yaw_deg", "score",
    "top1_correct", "pose_error_m", "yaw_error_deg", "scale_ratio",
    # outcome
    "rmse_xy", "rmse_z", "median_xy", "p90_xy", "max_xy",
    "tpr@0.1", "tpr@0.5", "tpr@1.0", "tpr@1.5", "tpr@2.0",
    # the ceiling, for reading the row above honestly
    "oracle_rmse_xy", "oracle_rmse_z", "oracle_tpr@1.0",
    "seconds", "stages",
    # --visual-prior only: what the panoramas said
    "visual_how", "visual_fit", "visual_inlier_frac", "visual_n_inliers",
    "visual_n_anchors", "visual_rank", "visual_n_pool",
]


def row_for(r) -> dict:
    """Flatten one RealityRun into a row."""
    c, ctx = r.case, r.context
    pred = r.poses.get("predicted", {})
    orc = r.poses.get("oracle", {})
    best = r.hypotheses[0] if r.hypotheses else None
    row = dict(
        reality_id=c.reality_id, capture_id=c.capture_id,
        structure=c.structure.structure_id, root=c.structure.root,
        status=r.status, georeferenced=c.structure.georeferenced,
        prior_capture_ids=" ".join(c.prior_capture_ids),
        prior_capture_date=c.prior_capture_date, capture_date=c.capture_date,
        prior_splits=ctx.get("prior_splits"), prior_dropped=ctx.get("prior_dropped"),
        prior_wall_cells=ctx.get("wall_cells"), prior_same_day=ctx.get("same_day"),
        overlap_5m=ctx.get("overlap_5m"), n_images=pred.get("n_images"),
        rig_height_m=ctx.get("rig_height_m"),
        rig_implausible=ctx.get("rig_implausible"),
        floor_fit_corrected=ctx.get("floor_fit_corrected"),
        metres_per_inch=ctx.get("metres_per_inch"),
        scale_source=ctx.get("scale_source"),
        sign_ambiguous=ctx.get("sign_ambiguous"),
        sign_margin=ctx.get("sign_margin"),
        sign_uncertain=ctx.get("sign_uncertain"),
        yaw_fallback=ctx.get("yaw_fallback"),
        n_hypotheses=ctx.get("n_hypotheses"),
        scale=None if best is None else best.sim2.s,
        yaw_deg=None if best is None else best.yaw_deg,
        score=None if best is None else best.score,
        top1_correct=ctx.get("top1_correct"),
        pose_error_m=ctx.get("pose_error_m"),
        yaw_error_deg=ctx.get("yaw_error_deg"),
        scale_ratio=ctx.get("scale_ratio"),
        seconds=round(sum(s.seconds for s in r.stages), 1),
        stages=" ".join(f"{s.name}{'' if s.ok else '!'}" for s in r.stages),
        oracle_rmse_xy=orc.get("rmse_xy"), oracle_rmse_z=orc.get("rmse_z"),
        **{"oracle_tpr@1.0": orc.get("tpr@1.0")},
        **{k: ctx.get(k) for k in (
            "visual_how", "visual_fit", "visual_inlier_frac", "visual_n_inliers",
            "visual_n_anchors", "visual_rank", "visual_n_pool")},
    )
    for k in COLUMNS:
        if k in pred:
            row[k] = pred[k]
    return row


class Writer:
    """Row-major CSV, the transforms, and the full record -- flushed per reality."""

    def __init__(self, stem):
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        # `with_suffix` truncates at the stem's OWN last dot, which a swept
        # weight (e.g. "_free0.25") already has -- it silently collapsed every
        # sweep combination sharing a prefix before that dot onto one file.
        self.paths = dict(rows=stem.with_name(stem.name + ".csv"),
                          detail=stem.with_name(stem.name + ".jsonl"),
                          transforms=stem.with_name(stem.name + "_transforms.jsonl"))
        self._csvf = open(self.paths["rows"], "w", newline="")
        self._csv = csv.DictWriter(self._csvf, fieldnames=ROW_COLUMNS,
                                   extrasaction="ignore")
        self._csv.writeheader()
        self._jf = open(self.paths["detail"], "w")
        self._tf = open(self.paths["transforms"], "w")
        self.rows = []

    def write(self, r):
        row = row_for(r)
        self._csv.writerow(row)
        self.rows.append(row)
        poses = {n: {k: v for k, v in p.items() if k != "tm"}
                 for n, p in r.poses.items()}
        self._jf.write(json.dumps(dict(
            row, poses=poses, warnings=getattr(r.reference, "warnings", []),
            dropped=[list(d) for d in getattr(r.reference, "dropped", [])],
            prior_reality_ids=list(getattr(r.reference, "prior_reality_ids", ())),
            stage_seconds={s.name: round(s.seconds, 2) for s in r.stages},
        )) + "\n")
        for name, pose in r.poses.items():
            self._tf.write(json.dumps({
                "reality_id": r.case.reality_id, "pose": name,
                "tm": [float(x) for x in np.asarray(pose["tm"]).ravel()]}) + "\n")
        for f in (self._csvf, self._jf, self._tf):
            f.flush()

    def close(self):
        for f in (self._csvf, self._jf, self._tf):
            f.close()


GT_PROBE_COLUMNS = [
    "reality_id", "status", "up_index", "scale_in_bracket", "true_scale",
    "grid_scale_lo", "grid_scale_hi", "nearest_grid_scale", "yaw_reachable",
    "true_yaw_deg", "yaw_window_deg", "nearest_grid_yaw", "yaw_gap_deg",
    "gt_score", "gt_wall", "gt_free", "gt_scale_prior", "gt_aspect",
    "gt_rmse_xy", "gt_tpr@1.0", "gt_tpr@1.5",
    "winner_score", "winner_rmse_xy", "winner_tpr@1.0", "score_gap",
    "visual_pose_err_m", "visual_yaw_err_deg", "visual_scale_ratio",
    "visual_correct", "visual_inlier_frac",
]
GRID_HYPOTHESES_COLUMNS = [
    "reality_id", "rank", "score", "scale", "yaw_deg", "up_index", "tx", "ty",
    "wall", "free", "scale_prior", "aspect", "inliers", "n_segments",
    "tpr@1.0", "tpr@1.5", "rmse_xy",
]


class DebugWriter:
    """Per-candidate ground-truth distance, and the enumeration/ranking probe.

    Opt-in (`--debug-candidates`): a reality's candidate set can be thousands
    of hypotheses, so this is a separate pair of files rather than something
    folded into `Writer`'s per-reality row.
    """

    def __init__(self, stem):
        stem = Path(stem)
        self.paths = dict(
            gt_probe=stem.with_name(stem.name + "_gt_probe.csv"),
            grid_hypotheses=stem.with_name(stem.name + "_grid_hypotheses.csv"))
        self._pf = open(self.paths["gt_probe"], "w", newline="")
        self._probe = csv.DictWriter(self._pf, fieldnames=GT_PROBE_COLUMNS,
                                     extrasaction="ignore")
        self._probe.writeheader()
        self._hf = open(self.paths["grid_hypotheses"], "w", newline="")
        self._hyps = csv.DictWriter(self._hf, fieldnames=GRID_HYPOTHESES_COLUMNS,
                                    extrasaction="ignore")
        self._hyps.writeheader()

    def write(self, r, p):
        hyp_metrics = candidate_debug.hypothesis_metrics(r)
        self._hyps.writerows(hyp_metrics)
        self._probe.writerow(candidate_debug.gt_probe_row(r, p, hyp_metrics))
        self._pf.flush()
        self._hf.flush()

    def close(self):
        self._pf.close()
        self._hf.close()


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets_mod.add_arguments(ap)
    g = ap.add_argument_group("selection")
    g.add_argument("--reality", nargs="+", default=None,
                   help="only these reality ids")
    g.add_argument("--limit", type=int, default=None,
                   help="only the first N realities")

    g = ap.add_argument_group("stage: reference (the prior BEV)")
    g.add_argument("--reference-stride", type=int, default=refmod.MERGE_STRIDE,
                   help="keep every Nth point of each prior cloud")
    g.add_argument("--reference-cell-m", type=float, default=refmod.CELL_M,
                   help="raster cell in metres")
    g.add_argument("--reference-pad-m", type=float, default=refmod.PAD_M,
                   help="margin around the merged extent")
    g.add_argument("--coverage-dilate-m", type=float,
                   default=refmod.COVERAGE_DILATE_M,
                   help="how wide the prior's walk is dilated into a corridor")
    g.add_argument("--level-gap-m", type=float, default=refmod.LEVEL_GAP_M,
                   help="floor-height gap that separates two storeys")
    g.add_argument("--consistency-tol-m", type=float,
                   default=refmod.CONSISTENCY_TOL_M,
                   help="drop a prior split disagreeing with a sibling by more")
    g.add_argument("--max-rel-mad", type=float, default=refmod.MAX_REL_MAD,
                   help="reject a floor fit more dispersed than this")
    g.add_argument("--height-band", type=float, nargs=2,
                   default=list(refmod.HEIGHT_BAND), metavar=("LO", "HI"),
                   help="fraction of the expected rig height still believable")
    g.add_argument("--max-reference-px", type=int, default=refmod.MAX_REF_PX,
                   help="cap on the reference raster's largest dimension")
    g.add_argument("--min-wall-cells", type=int, default=refmod.MIN_WALL_CELLS,
                   help="below this the reference is flagged too thin")
    g.add_argument("--no-cache", action="store_true",
                   help="build every reference in memory, cache nothing")
    g.add_argument("--reference-force", action="store_true",
                   help="rebuild even where a cached reference exists; needed "
                        "when sweeping anything but stride and cell size")

    g = ap.add_argument_group("stage: current cloud")
    g.add_argument("--cloud-stride", type=int, default=3,
                   help="keep every Nth point of the current cloud")
    g.add_argument("--min-source-frames", type=int, default=2,
                   help="drop points seen by fewer source frames")
    g.add_argument("--single-gravity", action="store_true",
                   help="carry only the better up-vector; the default carries "
                        "both where the floor/ceiling test ties")

    g = ap.add_argument_group("stage: search")
    g.add_argument("--search-cell-m", type=float, default=0.20,
                   help="correlation cell in metres")
    g.add_argument("--max-search-px", type=int, default=4000,
                   help="cap on the correlated raster's largest dimension")
    g.add_argument("--yaw-window-deg", type=float, default=10.0,
                   help="half-width swept around each Manhattan direction")
    g.add_argument("--yaw-step-deg", type=float, default=3.0,
                   help="spacing of heading candidates")
    g.add_argument("--yaw-min-strength", type=float, default=0.08,
                   help="below this the Manhattan estimate is unusable and the "
                        "search sweeps the full circle")
    g.add_argument("--scale-steps", type=int, default=9,
                   help="scale candidates across the rig-height bracket")
    g.add_argument("--w-wall", type=float, nargs="+", default=[1.0],
                   help="weight of wall agreement; several values sweep")
    g.add_argument("--w-free", type=float, nargs="+", default=[0.1],
                   help="weight of the coverage term; several values sweep")
    g.add_argument("--w-scale", type=float, nargs="+", default=[0.25],
                   help="weight of scale plausibility; several values sweep")
    g.add_argument("--w-aspect", type=float, nargs="+", default=[0.25],
                   help="weight of local oriented-bbox aspect agreement "
                        "(breaks 90-degree-rotation ties); several values sweep")
    g.add_argument("--top-k", type=int, default=5,
                   help="peaks kept per scale/heading combination")
    g.add_argument("--backend", choices=["auto", "numpy"], default="auto")
    g.add_argument("--rerank", action="store_true",
                   help="re-sort every proposal by wall-segment inlier count "
                        "(align.rerank), correlation score as tie-break -- "
                        "the 'dispose' half the correlation search proposes for")

    g = ap.add_argument_group("visual prior: where the panoramas say the cameras are")
    g.add_argument("--visual-prior", action="store_true",
                   help="describe the split's panoramas and its prior pool's "
                        "with DINOv2 (align.visual), retrieve each camera's "
                        "look-alike prior camera and fit the pose those places "
                        "imply; measured 33/94 correct on its own vs the search's "
                        "30, and right 4 times in 5 when >= half the cameras agree")
    g.add_argument("--visual-model", default=visualmod.DEFAULT_MODEL)
    g.add_argument("--visual-cache", default="cache/desc",
                   help="per-split descriptor cache; the first run over a "
                        "dataset pays ~1 h of extraction, later ones seconds")
    g.add_argument("--visual-top-k", type=int, default=5)
    g.add_argument("--visual-accept-frac", type=float, default=0.5,
                   help="take the panoramas' pose outright when this fraction "
                        "of cameras agree with it; below, it competes as a "
                        "candidate in the search's ranking")
    g.add_argument("--visual-min-inliers", type=int, default=3)
    g.add_argument("--visual-radius-m", type=float, default=5.0)
    g.add_argument("--visual-workers", type=int, default=16)

    g = ap.add_argument_group("stage: scoring")
    g.add_argument("--no-score", action="store_true",
                   help="solve and write transforms without reading truth")
    g.add_argument("--no-oracle", action="store_true",
                   help="skip the best-planar-pose ceiling")
    g.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.1, 0.5, 1.0, 1.5, 2.0],
                   help="metres, for tpr@X")
    g.add_argument("--overlap-radius-m", type=float, default=5.0,
                   help="a camera counts as covered within this of a prior one")
    g.add_argument("--pos-tol-m", type=float, default=2.0,
                   help="a pose counts as correct within this of truth")
    g.add_argument("--yaw-tol-deg", type=float, default=10.0,
                   help="...and within this heading error")
    g.add_argument("--scale-tol", type=float, default=1.15,
                   help="...and this scale ratio")
    g.add_argument("--debug-candidates", action="store_true",
                   help="write <stem>_gt_probe.csv (per-reality: does the "
                        "grid's scale bracket/yaw window reach truth, and "
                        "what does the score function make of the true pose "
                        "itself) and <stem>_grid_hypotheses.csv (every "
                        "evaluated candidate's own distance from truth) -- "
                        "separates a search-enumeration miss from a "
                        "ranking/scoring miss. Needs --no-oracle off "
                        "(the default); output can be large over a full "
                        "batch, pairs well with --reality")

    g = ap.add_argument_group("stage: visualisation")
    g.add_argument("--visualise", action="store_true",
                   help="render the reference, the placement and the runners-up")
    g.add_argument("--viz-dir", default=None,
                   help="default: <stem>_viz/")
    g.add_argument("--viz-width", type=int, default=1100)
    g.add_argument("--viz-top-k", type=int, default=5)

    # The per-image file production's viewer consumes. The default name lives
    # here rather than in align/ because nothing under align/ may name it --
    # tests/test_no_leakage.py forbids the string outright, and this tree is
    # outside the scan it walks.
    g = ap.add_argument_group("stage: per-image poses")
    g.add_argument("--images", action="store_true",
                   help="write each photograph's predicted pose, and plot it")
    g.add_argument("--images-dir", default=None,
                   help="default: <stem>_images/<reality_id>/")
    g.add_argument("--images-name", default="images.json",
                   help="filename written per reality")
    g.add_argument("--images-every-nth", type=int, default=1,
                   help="3 keeps indices 1, 4, 7 ... the subset production ships")
    g.add_argument("--images-z", type=float, default=None,
                   help="pin every z to this, as the viewer's own file does "
                        "(104.6 on every delivered split); default is the solved z")
    g.add_argument("--no-images-plot", action="store_true",
                   help="write the JSON but skip the floormap picture")
    g.add_argument("--images-gt", action="store_true",
                   help="draw ground truth beneath; needs scoring, and is "
                        "silently absent under --no-score")
    g.add_argument("--images-plot-width", type=int, default=1800)

    g = ap.add_argument_group("output")
    g.add_argument("--stem", default="out/run/alignment",
                   help="path prefix for .csv, .jsonl and _transforms.jsonl")
    g.add_argument("--html-report", action="store_true",
                   help="write <stem>.html: a filterable, sortable gallery of "
                        "every reality's metrics and viz images, all shown by "
                        "default (needs --visualise for the images)")
    g.add_argument("--quiet", action="store_true")
    return ap


def params_from(args, w_wall, w_free, w_scale, w_aspect, stem) -> Params:
    return Params(
        current_csv=tuple(args.current_csv or ()),
        prior_csv=tuple(args.prior_csv or ()),
        realities=tuple(args.reality or ()), limit=args.limit,
        reference_stride=args.reference_stride,
        reference_cell_m=args.reference_cell_m,
        reference_pad_m=args.reference_pad_m,
        coverage_dilate_m=args.coverage_dilate_m,
        level_gap_m=args.level_gap_m,
        consistency_tol_m=args.consistency_tol_m,
        max_rel_mad=args.max_rel_mad, height_band=tuple(args.height_band),
        max_reference_px=args.max_reference_px,
        min_wall_cells=args.min_wall_cells,
        reference_cache=None if args.no_cache else str(refmod.CACHE_DIR),
        reference_force=args.reference_force,
        cloud_stride=args.cloud_stride, min_source_frames=args.min_source_frames,
        both_gravity_hypotheses=not args.single_gravity,
        search_cell_m=args.search_cell_m, max_search_px=args.max_search_px,
        yaw_window_deg=args.yaw_window_deg, yaw_step_deg=args.yaw_step_deg,
        yaw_min_strength=args.yaw_min_strength, scale_steps=args.scale_steps,
        w_wall=w_wall, w_free=w_free, w_scale=w_scale, w_aspect=w_aspect,
        top_k_per_hypothesis=args.top_k, backend=args.backend,
        rerank=args.rerank,
        visual_prior=args.visual_prior, visual_model=args.visual_model,
        visual_cache=args.visual_cache, visual_top_k=args.visual_top_k,
        visual_accept_frac=args.visual_accept_frac,
        visual_min_inliers=args.visual_min_inliers,
        visual_radius_m=args.visual_radius_m, visual_workers=args.visual_workers,
        scoring=not args.no_score, score_oracle=not args.no_oracle,
        thresholds=tuple(args.thresholds),
        overlap_radius_m=args.overlap_radius_m, pos_tol_m=args.pos_tol_m,
        yaw_tol_deg=args.yaw_tol_deg, scale_tol=args.scale_tol,
        debug_candidates=args.debug_candidates,
        visualise=args.visualise,
        viz_dir=args.viz_dir or f"{stem}_viz",
        viz_width=args.viz_width, viz_top_k=args.viz_top_k,
        write_images=args.images,
        images_dir=args.images_dir or f"{stem}_images",
        images_name=args.images_name,
        images_every_nth=args.images_every_nth,
        images_z=args.images_z,
        images_plot=not args.no_images_plot,
        images_plot_gt=args.images_gt,
        images_plot_width=args.images_plot_width,
        stem=stem, html_report=args.html_report, verbose=not args.quiet)


def one_run(p: Params) -> Writer:
    out = Writer(p.stem)
    debug_out = DebugWriter(p.stem) if p.debug_candidates else None
    t0 = time.time()
    for r in run(p):
        out.write(r)
        if debug_out is not None:
            debug_out.write(r, p)
        if p.visualise and r.hypotheses:
            try:
                viz_prior.render(r, p.viz_dir, top_k=p.viz_top_k,
                                 width=p.viz_width)
            except Exception as exc:                      # noqa: BLE001
                print(f"   viz failed: {exc!r}"[:140], flush=True)
    out.close()
    if debug_out is not None:
        debug_out.close()
    scored = [r for r in out.rows if r.get("rmse_xy") is not None]
    print(f"\n{len(out.rows)} realities in {time.time() - t0:.0f}s   "
          f"({len(scored)} scored)")
    if scored:
        print(summarise(scored, p.thresholds))
        graded = [r for r in out.rows if r.get("top1_correct") is not None]
        if graded:
            n = sum(1 for r in graded if r["top1_correct"])
            print(f"    top-1 correct  {n}/{len(graded)} "
                  f"({100 * n / len(graded):.0f}%)  "
                  f"within {p.pos_tol_m:g} m, {p.yaw_tol_deg:g} deg, "
                  f"{p.scale_tol:g}x scale")
        orc = [r for r in out.rows if r.get("oracle_rmse_xy") is not None]
        if orc:
            v = np.array([r["oracle_rmse_xy"] for r in orc])
            print(f"    ceiling  median oracle rmse_xy {np.median(v):.3f} m "
                  f"over {len(orc)} realities")
    for why in sorted({r["status"] for r in out.rows if r["status"] != "ok"}):
        n = sum(1 for r in out.rows if r["status"] == why)
        print(f"    abstained: {n:3d}  {why}")
    if p.visual_prior:
        for how in sorted({r.get("visual_how") for r in out.rows if r.get("visual_how")}):
            rows = [r for r in out.rows if r.get("visual_how") == how]
            n_ok = sum(1 for r in rows if r.get("top1_correct"))
            print(f"    visual {how:9s} {len(rows):3d}  top-1 correct {n_ok:3d}")
    if p.html_report:
        report_path = Path(p.stem).with_name(Path(p.stem).name + ".html")
        viz_report.write(out.paths["detail"], p.viz_dir, report_path)
        out.paths["report"] = report_path
    for name, path in out.paths.items():
        print(f"  {name:11s} {path}")
    if debug_out is not None:
        for name, path in debug_out.paths.items():
            print(f"  {name:11s} {path}")
    if p.visualise:
        print(f"  {'viz':11s} {p.viz_dir}/<reality_id>/")
    if p.write_images:
        print(f"  {'images':11s} {p.images_dir}/<reality_id>/{p.images_name}")
        if p.images_plot:
            print(f"  {'plot':11s} {p.images_dir}/<reality_id>/"
                  f"{p.images_plot_name}")
    return out


def main():
    args = build_parser().parse_args()
    sweeps = [(a, b, c, d) for a in args.w_wall for b in args.w_free
              for c in args.w_scale for d in args.w_aspect]
    for w_wall, w_free, w_scale, w_aspect in sweeps:
        stem = args.stem
        if len(sweeps) > 1:
            stem = (f"{args.stem}_wall{w_wall:g}_free{w_free:g}"
                    f"_scale{w_scale:g}_aspect{w_aspect:g}")
            print(f"\n=== w_wall={w_wall:g}  w_free={w_free:g}  "
                  f"w_scale={w_scale:g}  w_aspect={w_aspect:g} ===", flush=True)
        one_run(params_from(args, w_wall, w_free, w_scale, w_aspect, stem))


if __name__ == "__main__":
    main()
