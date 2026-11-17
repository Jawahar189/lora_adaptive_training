"""Run the coarse search through the reality-major pipeline.

  python scripts/solve.py --reality RLT1788196494610      # one split, verbose
  python scripts/solve.py --csv CURRENT.csv               # a whole set
  python scripts/solve.py --root data --out out/predictions.jsonl
  python scripts/solve.py --reality RLT1788195772679 --viz
  python scripts/solve.py --chain --root data       # capture-major, geo-seeded

Emits the predictions JSONL `scripts/evaluate.py` consumes, plus a per-
hypothesis record for later confidence work. With --reality it also prints each
stage as it completes, which is the shape you want when stepping through
`align/pipeline.py` in a debugger: every stage is a separate call and every
intermediate stays on the object.

Each case is scored as it finishes, so accuracy accumulates while the run goes
rather than arriving at the end of a two-hour sweep. Scoring happens strictly
AFTER the search, reads ground truth only through `align.groundtruth`, and
feeds nothing back: --no-gt turns it off entirely. `scripts/evaluate.py` still
exists for scoring a predictions file on its own.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import chain as chainmod
from align import viz as V
from align.eval.reality import ResultWriter, reality_metrics
from align.io import datasets as datasets_mod
from align.io.cases import CaseIndex
from align.pipeline import RealityPipeline
from align.scales import SheetScales
from align.search import get_backend

VIZ_WIDTH = 1100


def ground_truth(case):
    """The true pose, for OVERLAY ONLY.

    Read here in the script, after the search has finished, and passed to the
    renderer as a comparison layer. It never reaches the pipeline: the solver
    modules cannot import ground truth at all, and a test enforces that.
    """
    from align.frames import decompose_tm, sim3_scale, sim3_yaw, target_to_paper
    from align.groundtruth import gt_trajectory, load_tm
    M = decompose_tm(load_tm(case.rlt_dir), case.structure.design_tm)
    return dict(
        scale=float(sim3_scale(M)),
        yaw_deg=float(np.degrees(sim3_yaw(M)) % 360.0),
        traj_paper=target_to_paper(gt_trajectory(case.rlt_dir).positions,
                                   case.structure.design_tm)[:, :2])


def reality_row(case, p, tm):
    """Per-reality accuracy in metres: RMSE in plan, height beside it, TPR.

    The reporting format, written for every reality of every run alongside the
    transform itself -- see `align.eval.reality`. Runs after the search, like
    every other scorer here, and feeds nothing back.
    """
    from align.groundtruth import gt_trajectory
    gt = gt_trajectory(case.rlt_dir)
    traj = p.trajectory()
    order = {n: k for k, n in enumerate(traj.names)}
    keep = [n for n in gt.names if n in order]
    sfm = traj.positions[[order[n] for n in keep]]
    gtp = np.asarray(gt.positions)[[k for k, n in enumerate(gt.names) if n in order]]
    return reality_metrics(tm, sfm, gtp, p.metres_per_inch,
                           case.structure.georeferenced)


def score_case(case, p, result, tm):
    """Score one finished case against ground truth.

    Runs after the search, never before: nothing here can reach the solver.
    """
    from align.eval.metrics import evaluate
    from align.groundtruth import gt_trajectory, load_tm
    return evaluate(tm, load_tm(case.rlt_dir),
                    gt_trajectory(case.rlt_dir).positions,
                    p.trajectory().positions, case.structure.design_tm,
                    case.structure.geometry(), case.reality_id,
                    case.structure.key, case.structure.status)


def rank_of_truth(result, gt, traj_xy, mpi, tol_m=15.0, tol_deg=15.0):
    """Where the closest hypothesis to truth sits in the ranking.

    The number that matters at this stage. A correct pose ranked 10th means the
    search found it and the score could not tell; no correct pose at all means
    the enumeration missed it. Those call for different fixes.
    """
    best_rank, best_pos, best_yaw = None, float("inf"), float("nan")
    for h in result.hypotheses:
        dp, dy = _deltas(h, gt, traj_xy, mpi)
        if dp < best_pos:
            best_rank, best_pos, best_yaw = h.rank, dp, dy
    within = [h.rank for h in result.hypotheses
              if _deltas(h, gt, traj_xy, mpi)[0] <= tol_m
              and abs(_deltas(h, gt, traj_xy, mpi)[1]) <= tol_deg]
    return dict(closest_rank=best_rank, closest_pos_m=best_pos,
                closest_yaw_deg=best_yaw,
                found_rank=min(within) if within else None,
                n_within=len(within))


def _gt_deltas(h, gt, p, result) -> dict:
    """This hypothesis's distance and yaw from truth, for the record file.

    A transferred pose carries an `up_index` whose gravity hypothesis was never
    searched -- it has no wall band and so no `traj_xy` -- and a chained pose
    can carry one from the split it leaned on. Missing that view is normal;
    crashing the run 19 captures in is not.
    """
    if gt is None:
        return {}
    traj_xy = p.view(getattr(h, "up_index", 0)).get("traj_xy")
    if traj_xy is None:
        traj_xy = p.view(0).get("traj_xy")
    if traj_xy is None:
        return {}
    return dict(zip(("gt_pos_m", "gt_yaw_deg"),
                    _deltas(h, gt, traj_xy, result.metres_per_inch)))


def _deltas(h, gt, traj_xy, mpi):
    """How far one hypothesis is from truth, in metres and degrees."""
    pred = h.sim2.apply(traj_xy)
    d_pos = float(np.linalg.norm(pred.mean(0) - gt["traj_paper"].mean(0)) * mpi)
    d_yaw = float(((h.yaw_deg - gt["yaw_deg"] + 180.0) % 360.0) - 180.0)
    return d_pos, d_yaw


def render(p, result, out_dir, top_n=6, gt=None):
    """Where the search put the capture, and how confident the ranking is.

    `gt`, when given, is drawn as a comparison layer only -- it plays no part
    in the search, which has already finished by the time this runs.
    """
    from PIL import Image, ImageDraw

    out_dir = Path(out_dir) / p.case.reality_id
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = p.sheet()
    geom = p.structure.geometry()
    wall_xy, traj_xy = p.wall_points()
    scale = VIZ_WIDTH / feats.wall.shape[1]
    base = V.mask_rgb(V.downsample_mask(feats.wall, VIZ_WIDTH),
                      fg=(150, 158, 168)).convert("RGB")

    def to_px(xy):
        return geom.paper_to_pixel(np.asarray(xy)[:, :2]) * scale

    panels = []

    # 1. the winning placement
    best = result.best
    img = base.copy()
    d = ImageDraw.Draw(img)
    for x, y in to_px(best.sim2.apply(wall_xy)):
        d.point((x, y), fill=V.RED)
    img = V.draw_path(img, to_px(best.sim2.apply(traj_xy)), width=2)
    # A transferred pose carries viewport -1: it came from shared images, not
    # from a search over any one viewport. Guard the lower bound too, or -1
    # indexes the LAST viewport and draws a box that means nothing.
    vp = (feats.viewports[best.viewport]
          if 0 <= best.viewport < len(feats.viewports) else None)
    if vp is not None:
        d = ImageDraw.Draw(img)
        d.rectangle([vp.x0 * scale, vp.y0 * scale, vp.x1 * scale, vp.y1 * scale],
                    outline=V.BLUE, width=2)
    sub = (f"score {best.score:.3f} · scale {best.scale:.3f} in/unit · "
           f"yaw {best.yaw_deg:.1f}° · viewport {best.viewport}")
    if gt is not None:
        img = V.draw_path(img, to_px(gt["traj_paper"]), color=V.GREEN, width=3)
        dp, dy = _deltas(best, gt, traj_xy, result.metres_per_inch)
        sub += (f"  ||  GROUND TRUTH scale {gt['scale']:.3f} yaw "
                f"{gt['yaw_deg']:.1f}° · off by {dp:.1f} m, {dy:+.1f}°, "
                f"x{best.scale / gt['scale']:.2f} scale")
    panels.append(V.panel(
        img, "Best hypothesis",
        sub + " · red = cloud, blue = predicted walk"
        + (", GREEN = ground truth" if gt is not None else "")
        + ", box = viewport", VIZ_WIDTH))

    # 2. how tightly the top hypotheses agree -- a scattered set is the
    #    honest signal that the score could not decide.
    img = base.copy()
    if gt is not None:
        img = V.draw_path(img, to_px(gt["traj_paper"]), color=V.GREEN, width=4)
    for h in reversed(result.hypotheses[:top_n]):
        img = V.draw_path(img, to_px(h.sim2.apply(traj_xy)),
                          color=V.rank_color(h.rank), width=2 if h.rank else 3)
    spread = float(np.std([[h.sim2.tx, h.sim2.ty]
                           for h in result.hypotheses[:top_n]], axis=0).mean())
    sub = (f"warm = rank 0, cool = worse · translation spread "
           f"{spread * result.metres_per_inch:.1f} m")
    if gt is not None:
        best_i = min(result.hypotheses,
                     key=lambda h: _deltas(h, gt, traj_xy,
                                           result.metres_per_inch)[0])
        dp, dy = _deltas(best_i, gt, traj_xy, result.metres_per_inch)
        sub += (f"  ||  GREEN = ground truth · closest of {len(result.hypotheses)} "
                f"is rank {best_i.rank} at {dp:.1f} m, {dy:+.1f}°")
    panels.append(V.panel(img, f"Top {top_n} walks", sub, VIZ_WIDTH))

    # 3. the score decomposition
    shown = result.hypotheses[:min(12, len(result.hypotheses))]
    # A transferred pose has no score terms at all -- they are nan, and a nan
    # bar length reaches PIL as a nan coordinate and raises. Draw it as zero
    # contribution, which is what "this pose was not scored" looks like.
    def _term(v):
        return float(v) if np.isfinite(v) else 0.0
    rows = [(f"#{h.rank} s={h.scale:.2f} y={h.yaw_deg:.0f}°",
             _term(h.wall), _term(h.free), _term(h.scale_prior), _term(h.aspect))
            for h in shown]
    notes = highlight = None
    sub = "wall NCC, walk penalty, scale prior, local aspect"
    if gt is not None:
        d = [_deltas(h, gt, traj_xy, result.metres_per_inch) for h in shown]
        notes = [f"{dp:6.1f} m {dy:+6.1f}°" for dp, dy in d]
        highlight = int(np.argmin([x[0] for x in d]))
        sub += "  ||  right column = distance from ground truth; marked row is closest"
    panels.append(V.panel(
        V.score_bars(rows, VIZ_WIDTH, notes=notes, highlight=highlight),
        "Score terms", sub, VIZ_WIDTH))

    for name, panel in zip(("H1_best", "H2_topk", "H3_scores"), panels):
        V.save(panel, out_dir / f"{name}.png")
    V.save(V.grid(panels, cols=1), out_dir / "solution.png")
    return out_dir


def solve_one(case, scales, backend, stride, cell_m, verbose=False):  # noqa: D103
    p = RealityPipeline(case, scales=scales, backend=backend, stride=stride,
                        cell_m=cell_m)
    timings = {}

    def stage(name, fn):
        t0 = time.time()
        value = fn()
        timings[name] = time.time() - t0
        if verbose:
            print(f"  {name:<13s} {timings[name]:6.1f}s  {_describe(name, value, p)}",
                  flush=True)
        return value

    stage("cloud", p.cloud)
    stage("sheet", p.sheet)
    stage("gravity", p.gravity)
    stage("floor", p.floor)
    stage("scale_prior", p.scale_prior)
    try:
        stage("wall_points", p.wall_points)
        stage("grid", p.grid)
    except ValueError as exc:
        return p, None, timings, str(exc)
    result = stage("search", p.search)
    return p, result, timings, result.error


def _describe(name, v, p):
    if name == "cloud":
        return f"{len(v):,} points"
    if name == "sheet":
        return (f"ppi {v.ppi:.0f}, wall {v.wall.mean():.3%}, "
                f"{len(v.viewports)} viewports")
    if name == "gravity":
        return (f"{v.method}"
                + ("  SIGN AMBIGUOUS" if v.sign_ambiguous else ""))
    if name == "floor":
        return f"h={v.height:.4f} cloud units, beyond={v.beyond:.4f}"
    if name == "scale_prior":
        s, (lo, hi) = v
        return f"{s:.3f} in/unit, bracket {lo:.3f}..{hi:.3f}"
    if name == "wall_points":
        return f"{len(v[0]):,} wall points, {len(v[1])} cameras"
    if name == "grid":
        return (f"{len(v.scales)} scales x {len(v.yaws_deg)} yaws x "
                f"{len(v.viewports)} viewports = {len(v)}"
                + ("  (yaw fallback sweep)" if v.fallback_yaw else ""))
    if name == "search":
        return f"{len(v.hypotheses)} hypotheses kept"
    return ""


def solve_captures(cases, scales, backend, args):
    """Capture-major: seed each capture from its GPS fix, then chain outward.

    Only captures carrying a real geo anchor are chained. The rest keep the
    split-major behaviour, so a chained result is never silently mixed with an
    independent one.
    """
    import collections
    caps = collections.defaultdict(list)
    for c in cases:
        caps[c.capture_id].append(c)

    out = {}
    for i, (cap, members) in enumerate(sorted(caps.items()), 1):
        mpi = scales[members[0].structure.key]
        pipes, trajs = {}, {}

        def solve_fn(case, prior, _pipes=pipes, _trajs=trajs):
            p = RealityPipeline(case, scales=scales, backend=backend,
                                stride=args.stride, cell_m=args.cell_m,
                                prior=prior)
            _pipes[case.reality_id] = p
            _trajs[case.reality_id] = p.trajectory()
            try:
                return p.search(), p
            except ValueError as exc:
                from align.pipeline import SearchResult
                return SearchResult(reality_id=case.reality_id,
                                    structure=case.structure.key,
                                    metres_per_inch=mpi,
                                    error=str(exc)), p

        # Trajectories are needed up front to order the splits; building the
        # pipelines here also means solve_fn reuses them rather than re-reading.
        for c in members:
            p = RealityPipeline(c, scales=scales, backend=backend,
                                stride=args.stride, cell_m=args.cell_m)
            pipes[c.reality_id] = p
            trajs[c.reality_id] = p.trajectory()

        anchored = any(c.geo is not None for c in members)
        print(f"[{i:3d}/{len(caps)}] {cap} — {len(members)} split(s), "
              f"{'geo-anchored' if anchored else 'NO geo anchor'}", flush=True)
        results = chainmod.solve_capture(members, solve_fn, mpi, trajs)
        for rid, r in results.items():
            out[rid] = (r, pipes.get(rid))
            how = (r.chain or {}).get("how", "?") if r else "?"
            if r is None or not r.hypotheses:
                print(f"      {rid}  {how}  no hypothesis", flush=True)
            else:
                b = r.best
                gate = "gate ok" if (r.chain or {}).get("gate_passed") else "GATE FAILED"
                print(f"      {rid}  {how:34s} scale {b.scale:.3f} "
                      f"yaw {b.yaw_deg:6.1f}  {gate}", flush=True)
    return out


def run_chained(cases, scales, backend, args):
    """Capture-major run, with the same per-case scoring as the split-major one.

    Results are written as each CAPTURE finishes rather than at the end. The
    first version of this accumulated everything and wrote once, which meant a
    four-hour sweep held all of its output in memory and left nothing on disk if
    it died -- and a run that produces nothing until it completes cannot be
    watched while it runs.
    """
    import collections
    caps = collections.defaultdict(list)
    for c in cases:
        caps[c.capture_id].append(c)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored = []
    pf = open(args.out, "w")
    hf = open(args.hypotheses_out, "w")
    mf = None if args.no_gt else open(args.metrics_out, "w")
    try:
        for i, (cap, members) in enumerate(sorted(caps.items()), 1):
            mpi = scales[members[0].structure.key]
            anchored = any(c.geo is not None for c in members)
            print(f"[{i:3d}/{len(caps)}] {cap} — {len(members)} split(s), "
                  f"{'geo-anchored' if anchored else 'NO geo anchor'}", flush=True)
            for rid, (result, p) in solve_one_capture(members, scales, backend,
                                                      args, mpi).items():
                if result is None or not result.hypotheses or p is None:
                    print(f"      {rid}  no hypothesis", flush=True)
                    continue
                b = result.best
                how = (result.chain or {}).get("how", "?")
                gate = "gate ok" if (result.chain or {}).get("gate_passed") else "GATE FAILED"
                tm = p.tm(b)
                pf.write(json.dumps({"reality_id": rid,
                                     "tm": np.asarray(tm).ravel().tolist()}) + "\n")
                line = (f"      {rid}  {how:34s} scale {b.scale:.3f} "
                        f"yaw {b.yaw_deg:6.1f} inl {b.inliers:4d}  {gate}")

                gt = None
                if not args.no_gt:
                    try:
                        gt = ground_truth(p.case)
                        traj_xy = p.view(0).get("traj_xy")
                        if traj_xy is None:
                            _, traj_xy = p.wall_points()
                        m = score_case(p.case, p, result, tm)
                        truth = rank_of_truth(result, gt, traj_xy,
                                              result.metres_per_inch)
                        record = m.as_dict() | truth | {"chain": result.chain}
                        scored.append(record)
                        mf.write(json.dumps(record) + "\n")
                        line += (f" | off {truth['closest_pos_m']:6.1f} m"
                                 f" @rank {truth['closest_rank']}")
                    except Exception as exc:                      # noqa: BLE001
                        line += f"  scoring failed: {exc!r}"[:90]

                hf.write(json.dumps({
                    "reality_id": rid, "structure": p.structure.key,
                    "metres_per_inch": result.metres_per_inch,
                    "scale_trusted": result.scale_trusted,
                    "chain": result.chain,
                    "hypotheses": [
                        h.as_dict() | _gt_deltas(h, gt, p, result)
                        for h in result.hypotheses],
                }) + "\n")

                if args.viz:
                    try:
                        render(p, result, args.viz_dir, gt=gt)
                    except Exception as exc:                      # noqa: BLE001
                        line += f"  viz failed: {exc!r}"[:90]
                print(line, flush=True)
            pf.flush(); hf.flush()
            if mf:
                mf.flush()
    finally:
        pf.close(); hf.close()
        if mf:
            mf.close()

    summarize_chained(scored, args)
    return 0


def solve_one_capture(members, scales, backend, args, mpi):
    """Order, seed and chain a single capture. Returns reality_id -> (result, pipeline)."""
    pipes, trajs = {}, {}

    def solve_fn(case, prior):
        p = RealityPipeline(case, scales=scales, backend=backend,
                            stride=args.stride, cell_m=args.cell_m, prior=prior)
        pipes[case.reality_id] = p
        trajs[case.reality_id] = p.trajectory()
        try:
            return p.search(), p
        except ValueError as exc:
            from align.pipeline import SearchResult
            return SearchResult(reality_id=case.reality_id,
                                structure=case.structure.key,
                                metres_per_inch=mpi, error=str(exc)), p

    for c in members:
        p = RealityPipeline(c, scales=scales, backend=backend,
                            stride=args.stride, cell_m=args.cell_m)
        pipes[c.reality_id] = p
        trajs[c.reality_id] = p.trajectory()

    results = chainmod.solve_capture(members, solve_fn, mpi, trajs)
    return {rid: (r, pipes.get(rid)) for rid, r in results.items()}


def summarize_chained(scored, args):
    if not scored:
        return
    def group(pred):
        return [r for r in scored if pred((r["chain"] or {}).get("how", ""))]
    buckets = (("seeds", group(lambda h: h.startswith("seed"))),
               ("chained", group(lambda h: h.startswith(("chained", "transferred")))),
               ("independent", group(lambda h: h.startswith("independent"))))
    print(f"\n{len(scored)} splits scored")
    for label, g in buckets:
        if not g:
            continue
        # What was ACTUALLY chosen. `closest_*` below describes the candidate
        # set instead -- how good an answer was available -- and the gap
        # between the two is the ranking's failure, not the search's.
        err = np.array([(r.get("utm") or r["paper"])["median"] for r in g])
        unit = "m" if g[0].get("utm") else "in"
        tpr = [r["tpr_1m"] for r in g if r.get("tpr_1m") is not None]
        d = np.array([r["closest_pos_m"] for r in g])
        rk = [r["closest_rank"] for r in g]
        hit = [r["found_rank"] for r in g if r["found_rank"] is not None]
        print(f"  {label:12s} n={len(g):3d}  chosen-pose median error "
              f"{np.median(err):7.1f} {unit}"
              + (f"   tpr@1m median {np.median(tpr):.2f}"
                 f"  ({sum(1 for t in tpr if t >= 0.5)}/{len(tpr)} at 0.5+)"
                 if tpr else "   tpr@1m n/a"))
        print(f"  {'':12s}      closest-to-truth median {np.median(d):6.1f} m"
              f"   median rank {int(np.median(rk))}"
              f"   within 15 m/15deg: {len(hit)}/{len(g)}")
    px = np.array([r["pixel"]["median"] for r in scored])
    print(f"  median pixel error {np.median(px):8.0f} px")
    print(f"  metrics -> {args.metrics_out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--reality", nargs="+", help="one or more reality ids")
    src.add_argument("--csv", nargs="+",
                     help="current-side dataset CSV(s)")
    src.add_argument("--root", help="a data root, e.g. data or data_v; "
                                    "validated against the loaded index")
    src.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="out/predictions.jsonl")
    ap.add_argument("--hypotheses-out", default="out/hypotheses.jsonl")
    ap.add_argument("--backend", choices=["auto", "numpy"], default="auto")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--cell-m", type=float, default=0.20,
                    help="coarse cell size in metres")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--viz", action="store_true",
                    help="render where the search put the capture")
    ap.add_argument("--viz-dir", default="out/solve")
    ap.add_argument("--no-gt", action="store_true",
                    help="do not score against ground truth, and omit it from "
                         "the panels")
    ap.add_argument("--metrics-out", default="out/solve_metrics.jsonl")
    ap.add_argument("--results-stem", default="out/solve_results",
                    help="per-reality RMSE and TPR, plus the transforms; "
                         "written for every run")
    ap.add_argument("--chain", action="store_true",
                    help="capture-major: seed each geo-anchored capture from "
                         "its GPS fix and chain along the walk")
    args = ap.parse_args()

    index = (CaseIndex.load_dataset(datasets_mod.resolve(args.csv))
             if args.csv else CaseIndex.load_all())
    if args.reality:
        wanted = set(args.reality)
        cases = [c for c in index if c.reality_id in wanted]
        missing = wanted - {c.reality_id for c in cases}
        if missing:
            raise SystemExit(f"unknown reality ids: {sorted(missing)}")
    elif args.root:
        cases = [c for c in index if c.structure.root == args.root]
        if not cases:
            raise SystemExit(
                f"no cases under root {args.root!r}; this index has "
                f"{index.roots()}")
    else:
        cases = list(index)
    cases.sort(key=lambda c: (c.structure.key, c.reality_id))
    if args.limit:
        cases = cases[: args.limit]

    scales = SheetScales(CaseIndex.load_all())
    backend = get_backend(args.backend)
    verbose = len(cases) == 1 and not args.chain

    if args.chain:
        return run_chained(cases, scales, backend, args)

    print(f"solving {len(cases)} split(s) on the {backend.name} backend, "
          f"cell {args.cell_m} m\n", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    solved = failed = 0
    scored = []
    mf = None if args.no_gt else open(args.metrics_out, "w")
    # Per-reality RMSE and TPR, and the transform that produced them, for every
    # run. Not conditional on --no-gt only because it needs ground truth to
    # score; with --no-gt the transforms are still written by the writer's
    # predictions file.
    results = ResultWriter(args.results_stem)
    with open(args.out, "w") as pf, open(args.hypotheses_out, "w") as hf:
        for i, case in enumerate(cases, 1):
            if not verbose:
                print(f"[{i:3d}/{len(cases)}] {case.reality_id} "
                      f"{case.structure.key}", end="", flush=True)
            else:
                print(f"{case.reality_id}  {case.structure.key}")
            t0 = time.time()
            try:
                p, result, timings, err = solve_one(
                    case, scales, backend, args.stride, args.cell_m, verbose)
            except Exception as exc:                              # noqa: BLE001
                print(f"   FAILED {exc!r}"[:160], flush=True)
                failed += 1
                continue
            dt = time.time() - t0

            if result is None or not result.hypotheses:
                print(f"   no hypotheses ({err}) [{dt:.0f}s]", flush=True)
                results.write(case, {}, status=err or "no hypotheses")
                failed += 1
                continue

            tm = p.tm(result.best)
            pf.write(json.dumps({"reality_id": case.reality_id,
                                 "tm": np.asarray(tm).ravel().tolist()}) + "\n")

            gt = truth = None
            if not args.no_gt:
                try:
                    gt = ground_truth(case)
                    _, traj_xy = p.wall_points()
                    m = score_case(case, p, result, tm)
                    try:
                        results.write(case, {"predicted":
                                             dict(reality_row(case, p, tm), tm=tm)},
                                      metres_per_inch=p.metres_per_inch,
                                      scale_source=("design_tm" if
                                                    case.structure.georeferenced
                                                    else "estimated"))
                    except Exception as exc:                      # noqa: BLE001
                        print(f"   reality metrics failed: {exc!r}"[:140], flush=True)
                    truth = rank_of_truth(result, gt, traj_xy,
                                          result.metres_per_inch)
                    record = m.as_dict() | truth
                    scored.append(record)
                    mf.write(json.dumps(record) + "\n")
                    mf.flush()
                except Exception as exc:                          # noqa: BLE001
                    print(f"   scoring failed: {exc!r}"[:140], flush=True)

            hf.write(json.dumps({
                "reality_id": case.reality_id,
                "structure": case.structure.key,
                "metres_per_inch": result.metres_per_inch,
                # A structure with no design transform and no usable E1 estimate
                # falls back to a guessed sheet scale, and every threshold below
                # derives from it. Surface that rather than let it pass quietly.
                "scale_trusted": result.scale_trusted,
                "sign_ambiguous": bool(p.gravity().sign_ambiguous),
                "yaw_fallback": bool(p.grid().fallback_yaw),
                "seconds": round(dt, 1),
                "truth": truth,
                "hypotheses": [
                    h.as_dict() | (dict(zip(("gt_pos_m", "gt_yaw_deg"),
                                            _deltas(h, gt, traj_xy,
                                                    result.metres_per_inch)))
                                   if gt is not None else {})
                    for h in result.hypotheses],
            }) + "\n")
            pf.flush(); hf.flush()
            if args.viz:
                try:
                    dest = render(p, result, args.viz_dir, gt=gt)
                    if verbose:
                        print(f"  viz            -> {dest}", flush=True)
                except Exception as exc:                          # noqa: BLE001
                    print(f"   viz failed: {exc!r}"[:140], flush=True)

            solved += 1
            b = result.best
            line = (f"   score {b.score:.3f} scale {b.scale:.3f} "
                    f"yaw {b.yaw_deg:6.1f} [{dt:.0f}s]")
            if truth is not None:
                line += (f" | px {scored[-1]['pixel']['median']:7.0f}"
                         f"  off {truth['closest_pos_m']:6.1f} m"
                         f" at rank {truth['closest_rank']}")
            if not result.scale_trusted:
                line += "  SHEET SCALE GUESSED"
            print(line, flush=True)

    if mf is not None:
        mf.close()
    results.close()

    print(f"\n{solved} solved, {failed} without a hypothesis")
    if results.rows:
        print("\nper-reality accuracy, metres (horizontal; height reported "
              "separately)\n" + results.report())
        for name, path in results.paths.items():
            print(f"  {name:12s} {path}")
    if scored:
        px = np.array([r["pixel"]["median"] for r in scored])
        closest = np.array([r["closest_pos_m"] for r in scored])
        ranks = [r["found_rank"] for r in scored]
        hit = [r for r in ranks if r is not None]
        print(f"\n  median pixel error       {np.median(px):8.0f} px")
        print(f"  closest hypothesis       {np.median(closest):8.1f} m "
              f"(median over splits)")
        print(f"  a pose within 15 m/15deg {len(hit):4d}/{len(scored)} splits"
              + (f", median rank {int(np.median(hit))}" if hit else ""))
        top1 = sum(1 for r in ranks if r == 0)
        print(f"  ...and it ranked FIRST   {top1:4d}/{len(scored)}")
        print(f"  metrics -> {args.metrics_out}")
    print(f"predictions -> {args.out}")
    print(f"hypotheses  -> {args.hypotheses_out}")
    if args.viz:
        print(f"visualisation -> {args.viz_dir}/<reality_id>/")
    if solved:
        print(f"\nscore them with:\n"
              f"  .venv/bin/python scripts/evaluate.py --pred {args.out}"
              + (f" --csv {args.csv}" if args.csv else ""))


if __name__ == "__main__":
    main()
