"""Does the SEARCH find the right scale and yaw? The half e3 handed over for free.

`scripts/e3_prior_oracle.py` placed each split at its true scale and yaw and
asked only where the true POSITION ranked. It ranks in the top 0.04%. That
leaves the question it deliberately did not ask: the search has to discover the
scale and the yaw too, and an early probe put near-truth poses at ranks 39, 6
and 15 of 4050 -- good enough to be found, not good enough to be chosen.

Two failures are possible and they need different fixes, so they are measured
separately:

  ENUMERATION -- the true (scale, yaw) is not among the candidates at all. No
      ranking can recover it; the grid has to change.
  RANKING     -- it is among them and something else scores higher.

Reported for each, plus the split of the blame: a pose can be right in yaw and
wrong in scale, and the two come from different places (the scale bracket from
the rig-height ratio, the yaw from the Manhattan estimate).

The scale prior here is the RATIO of the prior's rig height to this split's --
the same monopod in two frames, see `gravity.relative_scale_prior` -- not the
2.11 m constant divided by a sheet-scale guess.

  python scripts/e6_scale_yaw.py --limit 20
  python scripts/e6_scale_yaw.py --w-free 0.25
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import cloud as cloudmod
from align import reference as R
from align import search as S
from align.eval.oracle import true_placement, wall_band_local
from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex
from align.io.pairing import prior_reference_for
from align.io.ply import read_ply
from align.io.shots import load_shots
from align.sheet import resample_to
from align.gravity import relative_scale_prior

COARSE_CELL_M = 0.20       # align.pipeline's search cell
MAX_SEARCH_PX = 4000       # align.pipeline's cap on the correlated raster
POS_TOL_M = 2.0            # the ceiling is 0.71 m median, so 1 m would be unfair
YAW_TOL_DEG = 10.0         # rerank.ANG_TOL_DEG, for the same reason it is 10 there
SCALE_TOL = 1.15


def search_rasters(ref_map, cell_m=COARSE_CELL_M):
    """The reference resampled to the SEARCH cell, the way the pipeline does it.

    The reference is stored at 0.10 m so the finer stages have something to pool,
    but correlating at that resolution is four times the work for no gain -- and
    the FFT is padded by the template on every side, so the cost is superlinear.
    `resample_to` any-pools, which is what keeps a one-cell-wide wall alive
    through the downsample.
    """
    h, w = ref_map.feats.wall.shape
    bounds = (0.0, 0.0, w * ref_map.res, h * ref_map.res)
    span = max(bounds[2], bounds[3])
    res = max(cell_m / ref_map.metres_per_inch, span / MAX_SEARCH_PX, ref_map.res)
    shape = (max(int(bounds[3] / res), 1), max(int(bounds[2] / res), 1))
    wall = resample_to(ref_map.feats, ref_map.geom, bounds, res, shape)
    free = resample_to(ref_map.free, ref_map.geom, bounds, res, shape)
    return wall.astype(np.float32), free.astype(np.float32), res


def build_grid(ref_map, wall_xy, floor, sheet_wall, cell_m=COARSE_CELL_M):
    """The candidate grid the pipeline would build for this split."""
    s0, bracket = relative_scale_prior(floor, ref_map.camera_height_in)
    if not np.isfinite(s0):
        return None, None, None
    cell = cell_m / ref_map.metres_per_inch
    ras = cloudmod.wall_raster(wall_xy * s0, res=cell)
    cm = cloudmod.manhattan(ras.mask)
    sm = cloudmod.manhattan(sheet_wall)
    yaws, fallback = S.yaw_candidates(cm.angle, sm.angle,
                                      strength=min(cm.strength, sm.strength))
    grid = S.SearchGrid(scales=S.scale_candidates(bracket), yaws_deg=yaws,
                        viewports=[0], fallback_yaw=fallback)
    return grid, s0, bracket


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets_mod.add_arguments(ap)
    ap.add_argument("--stride", type=int, default=3,
                    help="keep every Nth point of the current cloud")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N realities, for a smoke test")
    ap.add_argument("--w-free", type=float, default=0.25,
                    help="weight of the coverage term in the search score")
    ap.add_argument("--out", default="out/e6_scale_yaw.jsonl",
                    help="per-reality grid coverage and rank of the truth")
    args = ap.parse_args()

    ix = PairedIndex.load(datasets_mod.from_args(args))
    cases = list(ix.current)[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bk = S.get_backend()
    rows = []
    t0 = time.time()

    with open(args.out, "w") as fh:
        for i, case in enumerate(cases, 1):
            rec = dict(reality_id=case.reality_id, structure=case.structure.key)
            try:
                ref = prior_reference_for(case, ix)
                if ref is None:
                    rec["status"] = "no prior cloud"
                    raise StopIteration
                ref_map = R.build_cached(ref, case.structure)
                if ref_map is None:
                    rec["status"] = "no usable prior split"
                    raise StopIteration
                pc = read_ply(case.cloud_path, stride=args.stride, min_source_frames=2)
                traj = load_shots(case.rlt_dir)
                best, _grav, gt_local = true_placement(case, ref_map, pc.xyz,
                                                       traj.positions)
                if best is None:
                    rec["status"] = "no planar fit to the truth"
                    raise StopIteration
                _resid, ui, Rm, true_sim2 = best
                wall_xy, traj_xy, floor = wall_band_local(pc.xyz, traj.positions, Rm)
                if wall_xy is None:
                    rec["status"] = "no usable wall band"
                    raise StopIteration

                sheet_map, free_map, sres = search_rasters(ref_map)
                grid, s0, bracket = build_grid(ref_map, wall_xy, floor, sheet_map)
                if grid is None:
                    rec["status"] = "no scale prior"
                    raise StopIteration

                true_yaw = float(np.degrees(true_sim2.theta)) % 360.0
                dy = np.abs((grid.yaws_deg % 360.0 - true_yaw + 180.0) % 360.0 - 180.0)
                ds = np.abs(np.log(grid.scales / true_sim2.s))
                rec.update(
                    n_scales=len(grid.scales), n_yaws=len(grid.yaws_deg),
                    n_candidates=len(grid), yaw_fallback=bool(grid.fallback_yaw),
                    true_scale=true_sim2.s, true_yaw_deg=true_yaw,
                    scale_prior=s0, scale_ratio_prior_over_true=s0 / true_sim2.s,
                    nearest_yaw_deg=float(dy.min()),
                    nearest_scale_logratio=float(ds.min()),
                    scale_in_bracket=bool(bracket[0] <= true_sim2.s <= bracket[1]),
                    yaw_covered=bool(dy.min() <= S.YAW_STEP_DEG),
                    scale_covered=bool(ds.min() <= np.log(1.05)))

                rec["search_cells"] = int(sheet_map.size)
                hyps = S.search(wall_xy, traj_xy, sheet_map, (0.0, 0.0), sres,
                                grid, backend=bk, scale_prior=s0, up_index=ui,
                                w_free=args.w_free, interior=free_map)
                if not hyps:
                    rec["status"] = "no hypotheses"
                    raise StopIteration

                mpi = ref_map.metres_per_inch
                traj_up = (traj.positions @ Rm.T)[:, :2]

                def errs(h):
                    pos = float(np.linalg.norm(
                        h.sim2.apply(traj_up) - gt_local, axis=1).mean()) * mpi
                    yaw = abs(((np.degrees(h.sim2.theta) - true_yaw + 180) % 360) - 180)
                    return pos, float(yaw), float(h.sim2.s / true_sim2.s)

                e = [errs(h) for h in hyps]
                good = [j for j, (p, y, sc) in enumerate(e)
                        if p <= POS_TOL_M and y <= YAW_TOL_DEG
                        and 1 / SCALE_TOL <= sc <= SCALE_TOL]
                rec.update(
                    n_hypotheses=len(hyps),
                    top1_pos_m=e[0][0], top1_yaw_err=e[0][1], top1_scale_ratio=e[0][2],
                    top1_correct=bool(good and good[0] == 0),
                    rank_of_truth=(good[0] if good else None),
                    found_in_top20=bool(good and good[0] < 20),
                    best_pos_m=min(p for p, _, _ in e),
                    # Which half is wrong when the top pose is wrong?
                    top1_yaw_ok=bool(e[0][1] <= YAW_TOL_DEG),
                    top1_scale_ok=bool(1 / SCALE_TOL <= e[0][2] <= SCALE_TOL),
                    status="ok")
            except StopIteration:
                pass
            except Exception as exc:
                rec["status"] = f"error: {exc!r}"[:160]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            rows.append(rec)
            if rec.get("status") == "ok":
                msg = (f"grid {rec['n_scales']}x{rec['n_yaws']}"
                       f"{' FALLBACK' if rec['yaw_fallback'] else ''}"
                       f" | covered s={rec['scale_covered']:d} y={rec['yaw_covered']:d}"
                       f" | top1 {rec['top1_pos_m']:7.2f} m "
                       f"yaw {rec['top1_yaw_err']:5.1f} sc {rec['top1_scale_ratio']:5.2f}"
                       f" | truth rank {rec['rank_of_truth']}")
            else:
                msg = rec["status"]
            print(f"[{i}/{len(cases)}] {case.reality_id}  {msg}", flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(ok)
    print(f"\n{n} of {len(cases)} scored, in {time.time() - t0:.0f}s")
    if not n:
        return

    def pct(f):
        c = sum(1 for r in ok if f(r))
        return f"{c:3d}/{n} ({100 * c / n:3.0f}%)"

    print("\n  ENUMERATION -- is the truth among the candidates at all?")
    print(f"    true scale inside the bracket   {pct(lambda r: r['scale_in_bracket'])}")
    print(f"    a candidate scale within 5%     {pct(lambda r: r['scale_covered'])}")
    print(f"    a candidate yaw within one step {pct(lambda r: r['yaw_covered'])}")
    print(f"    BOTH                            "
          f"{pct(lambda r: r['scale_covered'] and r['yaw_covered'])}")
    print(f"    fell back to a full yaw sweep   {pct(lambda r: r['yaw_fallback'])}")

    print("\n  RANKING -- given the candidates, is the truth chosen?")
    print(f"    top-1 correct (<={POS_TOL_M:g} m, <={YAW_TOL_DEG:g} deg, scale <{SCALE_TOL:g}x)"
          f"  {pct(lambda r: r['top1_correct'])}")
    print(f"    truth anywhere in the returned set  "
          f"{pct(lambda r: r['rank_of_truth'] is not None)}")
    print(f"    truth in the top 20                 {pct(lambda r: r['found_in_top20'])}")
    ranks = [r["rank_of_truth"] for r in ok if r["rank_of_truth"] is not None]
    if ranks:
        print(f"    rank of truth, when present: median {np.median(ranks):.0f}  "
              f"p90 {np.percentile(ranks, 90):.0f}  max {max(ranks)}")

    print("\n  WHEN TOP-1 IS WRONG, WHICH HALF?")
    wrong = [r for r in ok if not r["top1_correct"]]
    if wrong:
        w = len(wrong)
        for lab, f in [("yaw right, scale wrong",
                        lambda r: r["top1_yaw_ok"] and not r["top1_scale_ok"]),
                       ("scale right, yaw wrong",
                        lambda r: r["top1_scale_ok"] and not r["top1_yaw_ok"]),
                       ("both right, position wrong",
                        lambda r: r["top1_scale_ok"] and r["top1_yaw_ok"]),
                       ("both wrong",
                        lambda r: not r["top1_scale_ok"] and not r["top1_yaw_ok"])]:
            c = sum(1 for r in wrong if f(r))
            print(f"    {lab:28s} {c:3d}/{w} ({100 * c / w:3.0f}%)")

    sp = np.array([r["scale_ratio_prior_over_true"] for r in ok])
    print(f"\n  SCALE PRIOR (rig-height ratio) vs truth: median {np.median(sp):.3f}  "
          f"p10 {np.percentile(sp, 10):.3f}  p90 {np.percentile(sp, 90):.3f}")
    p = np.array([r["top1_pos_m"] for r in ok])
    b = np.array([r["best_pos_m"] for r in ok])
    print(f"  top-1 position error: median {np.median(p):.2f} m   "
          f"best in the returned set: median {np.median(b):.2f} m")


if __name__ == "__main__":
    main()
