"""Can the prior BEV be RANKED? The experiment that decides the approach.

Everything measured so far says coverage is not the problem. A current split's
wall band lands on its prior's, the enumeration contains near-truth poses, and
the prior is registered so scale and gravity come free. What has never worked is
choosing between candidates: against a floorplan, `align/rerank.py` records the
true translation sitting at the 94th, 98th and 19.5th percentile of the
masked-NCC surface -- which is why the correlation only ever proposed and a
segment reranker disposed.

So this asks the same question of the new reference, and asks it WITHOUT a
search: place the current split at its true pose, and measure where that pose
sits on the score surface. No enumeration, no ranking heuristics, nothing that
can be tuned. If the true pose is not near the top of the surface when handed to
us for free, no amount of pipeline work will find it.

Pre-registered thresholds, fixed before running:

  * median true-translation percentile must exceed 99.5 on the combined
    surface. The floorplan baseline for the identical measurement is 94 / 98 /
    19.5 (`align/rerank.py`);
  * median oracle inliers must exceed 2x the floorplan's median of 12
    (`docs/METHOD.md` §11, E2).

Failing either is a RESULT, not a setback: it says the reference cannot be
ranked by correlation, and the reference is now a point cloud rather than ink,
so branch-and-bound, FPFH+RANSAC and ICP all become available in a way they
never were against a CAD sheet.

Also reported per split, because the percentile is uninterpretable without them:

  ceiling_m    how far the best LOCAL correction sits from the current capture's
               own registration. The prior's transform is inherited, so this is
               the error a PERFECT matcher would still show.
  overlap      fraction of the current split's cameras within 5 m of a prior one.

Ground truth is read here, after the reference exists, and never reaches it.

  python scripts/e3_prior_oracle.py
  python scripts/e3_prior_oracle.py --limit 10 --w-free 0 0.25 0.5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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

W_FREE_SWEEP = (0.0, 0.25, 0.5, 1.0)
CEILING_M = 3.0
OVERLAP_M = 5.0
INLIER_TOL_M = 0.35


def surface(ref_map, wall_xy, traj_xy, sim2, w_free_values):
    """Score maps at the TRUE scale and yaw, one per free-term weight.

    Mirrors `search.search`'s padding and indexing exactly so the cell the true
    translation lands in is the cell the search would have scored.
    """
    bk = S.get_backend()
    res, origin = ref_map.res, np.zeros(2)
    sheet = np.flipud(ref_map.feats.wall).astype(np.float32)     # back to y-up
    cover = np.flipud(ref_map.free.wall).astype(np.float32)
    H, W = sheet.shape
    yaw = float(np.degrees(sim2.theta))
    tmpl, t_lo = S._stamp(S._rotate(wall_xy, yaw, sim2.s), res)
    if tmpl.shape[0] < 2 or tmpl.shape[1] < 2:
        return None
    traj_r = S._rotate(traj_xy, yaw, sim2.s)
    tj = np.zeros_like(tmpl)
    ti = np.floor((traj_r - t_lo) / res).astype(int)
    ok = ((ti[:, 0] >= 0) & (ti[:, 0] < tmpl.shape[1])
          & (ti[:, 1] >= 0) & (ti[:, 1] < tmpl.shape[0]))
    if ok.sum() >= 2:
        tj[ti[ok, 1], ti[ok, 0]] = 1.0

    pad_y, pad_x = tmpl.shape
    padded = np.zeros((H + 2 * pad_y, W + 2 * pad_x), np.float32)
    padded[pad_y:pad_y + H, pad_x:pad_x + W] = sheet
    inside = np.zeros_like(padded)
    inside[pad_y:pad_y + H, pad_x:pad_x + W] = cover
    PH, PW = padded.shape
    shape = (PH + pad_y, PW + pad_x)

    st = bk.asarray(padded)
    F_sheet, F_sheet2 = bk.rfft2(st, shape), bk.rfft2(st * st, shape)
    F_inside = bk.rfft2(bk.asarray(inside), shape)
    th, tw = tmpl.shape
    n = float(th * tw)
    t_mean = float(tmpl.mean())
    t_zero = bk.asarray(tmpl - t_mean)
    t_energy = max(float(np.sqrt(((tmpl - t_mean) ** 2).sum())), 1e-3)
    fp = bk.asarray(np.ones_like(tmpl))

    num = S._correlate(bk, F_sheet, t_zero, shape)
    s1 = S._correlate(bk, F_sheet, fp, shape)
    s2 = S._correlate(bk, F_sheet2, fp, shape)
    var = s2 - s1 * s1 / n
    ncc = S._valid(bk.to_numpy(num / ((var.clip(min=1e-3) ** 0.5) * t_energy)),
                   (th, tw), (PH, PW))
    mass = float(tj.sum())
    if mass >= 1:
        tj_t = bk.asarray(tj)
        free = (S._valid(bk.to_numpy(S._correlate(bk, F_sheet, tj_t, shape) / mass),
                         (th, tw), (PH, PW))
                - S._valid(bk.to_numpy(S._correlate(bk, F_inside, tj_t, shape) / mass),
                           (th, tw), (PH, PW)))
    else:
        free = np.zeros_like(ncc)

    # Where the true translation lands, inverting search's own mapping.
    px = int(round((sim2.tx + t_lo[0] - origin[0]) / res + pad_x))
    py = int(round((sim2.ty + t_lo[1] - origin[1]) / res + pad_y))
    if not (0 <= py < ncc.shape[0] and 0 <= px < ncc.shape[1]):
        return None
    out = {}
    for wf in w_free_values:
        total = ncc - wf * free
        v = total[py, px]
        out[wf] = dict(pct=float(100.0 * (total < v).mean()),
                       value=float(v), peak=float(total.max()),
                       better=int((total > v).sum()), cells=int(total.size))
    out["_ncc_at_true"] = float(ncc[py, px])
    out["_interior_frac"] = float(cover.mean())
    return out


def ceiling_and_inliers(ref_map, wall_xy, sim2, step_in, span_in):
    """Best local correction, and the occupancy agreement at the true pose."""
    wall = np.flipud(ref_map.feats.wall)
    res = ref_map.res
    h, w = wall.shape
    occ = np.argwhere(wall)
    if not len(occ):
        return float("nan"), 0.0, 0.0
    tree = cKDTree(occ[:, ::-1] * res)             # (x, y) in local inches
    placed = sim2.apply(wall_xy)
    sample = placed[:: max(len(placed) // 20000, 1)]
    tol = INLIER_TOL_M / ref_map.metres_per_inch

    def frac(dx, dy):
        d, _ = tree.query(sample + [dx, dy], distance_upper_bound=tol * 4)
        return float((d < tol).mean())

    base = frac(0.0, 0.0)
    best, off = base, (0.0, 0.0)
    grid = np.arange(-span_in, span_in + 1e-9, step_in)
    for dx in grid:
        for dy in grid:
            f = frac(dx, dy)
            if f > best:
                best, off = f, (dx, dy)
    return float(np.hypot(*off) * ref_map.metres_per_inch), base, best


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets_mod.add_arguments(ap)
    ap.add_argument("--stride", type=int, default=3,
                    help="keep every Nth point of the current cloud")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N realities, for a smoke test")
    ap.add_argument("--w-free", type=float, nargs="+", default=list(W_FREE_SWEEP),
                    help="weights to sweep for the coverage term; 0 disables "
                         "it, 0.25 measured best")
    ap.add_argument("--out", default="out/e3_prior_oracle.jsonl",
                    help="per-reality percentiles, ceiling and overlap")
    args = ap.parse_args()

    ix = PairedIndex.load(datasets_mod.from_args(args))
    cases = list(ix.current)[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    with open(args.out, "w") as fh:
        for i, case in enumerate(cases, 1):
            rec = dict(reality_id=case.reality_id, structure=case.structure.key,
                       georeferenced=case.structure.georeferenced)
            try:
                ref = prior_reference_for(case, ix)
                if ref is None:
                    rec["status"] = "no prior cloud"
                    raise StopIteration
                ref_map = R.build_cached(ref, case.structure)
                if ref_map is None:
                    rec["status"] = "no usable prior split"
                    raise StopIteration
                rec.update(n_prior=len(ref_map.prior_reality_ids),
                           same_day=ref.same_day, thin=ref_map.thin,
                           wall_cells=ref_map.wall_cells,
                           metres_per_inch=ref_map.metres_per_inch,
                           scale_source=ref_map.scale_source)

                pc = read_ply(case.cloud_path, stride=args.stride,
                              min_source_frames=2)
                traj = load_shots(case.rlt_dir)
                best, grav, gt_local = true_placement(case, ref_map, pc.xyz,
                                                      traj.positions)
                if best is None:
                    rec["status"] = "no planar fit to the truth"
                    raise StopIteration
                resid_in, ui, Rm, sim2 = best
                rec.update(up_index=ui, sign_ambiguous=bool(grav.sign_ambiguous),
                           fit_resid_m=resid_in * ref_map.metres_per_inch,
                           true_scale=sim2.s,
                           true_yaw_deg=float(np.degrees(sim2.theta)))

                # camera overlap with the prior, the coverage number
                prior_cams = np.vstack([ref.trajectory(r).positions[:, :2]
                                        for r in ref_map.prior_reality_ids])
                cur_cams = sim2.apply((traj.positions @ Rm.T)[:, :2]) + ref_map.origin
                d, _ = cKDTree(prior_cams).query(cur_cams)
                rec["overlap"] = float(
                    (d * ref_map.metres_per_inch <= OVERLAP_M).mean())

                wall_xy, traj_xy, _ = wall_band_local(pc.xyz, traj.positions, Rm)
                if wall_xy is None:
                    rec["status"] = "no usable wall band"
                    raise StopIteration

                surf = surface(ref_map, wall_xy, traj_xy, sim2, args.w_free)
                if surf is None:
                    rec["status"] = "true pose is off the reference"
                    raise StopIteration
                rec["ncc_at_true"] = surf.pop("_ncc_at_true")
                rec["coverage_frac"] = surf.pop("_interior_frac")
                rec["pct"] = {str(k): v["pct"] for k, v in surf.items()}
                rec["better_cells"] = {str(k): v["better"] for k, v in surf.items()}

                step = 0.25 / ref_map.metres_per_inch
                span = CEILING_M / ref_map.metres_per_inch
                ceil_m, inl_true, inl_best = ceiling_and_inliers(
                    ref_map, wall_xy, sim2, step, span)
                rec.update(ceiling_m=ceil_m, inliers_at_true=inl_true,
                           inliers_at_best=inl_best, status="ok")
            except StopIteration:
                pass
            except Exception as exc:
                rec["status"] = f"error: {exc!r}"[:160]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            rows.append(rec)
            msg = rec.get("status", "?")
            if msg == "ok":
                p = rec["pct"]
                msg = (f"pct " + " ".join(f"{k}:{v:6.2f}" for k, v in p.items())
                       + f" | inl {rec['inliers_at_true']:.2f}"
                       + f" | ceil {rec['ceiling_m']:5.2f} m"
                       + f" | ovl {rec['overlap']:.2f}")
            print(f"[{i}/{len(cases)}] {case.reality_id}  {msg}", flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n{len(ok)} of {len(cases)} scored, in {time.time() - t0:.0f}s")
    if not ok:
        return
    print("\n  true-translation percentile on the score surface "
          "(floorplan baseline: 94.0 / 98.0 / 19.5)")
    for wf in args.w_free:
        v = np.array([r["pct"][str(wf)] for r in ok])
        print(f"    w_free={wf:<5} median {np.median(v):7.3f}  "
              f"p25 {np.percentile(v, 25):7.3f}  "
              f">99.5: {(v > 99.5).sum():3d}/{len(v)}  "
              f">99.9: {(v > 99.9).sum():3d}/{len(v)}")
    inl = np.array([r["inliers_at_true"] for r in ok])
    print(f"\n  occupancy agreement at the true pose: median {np.median(inl):.3f}  "
          f"p10 {np.percentile(inl, 10):.3f}")
    ceil = np.array([r["ceiling_m"] for r in ok])
    print(f"  ceiling (best local correction): median {np.median(ceil):.2f} m  "
          f"p90 {np.percentile(ceil, 90):.2f}  "
          f"under 1 m: {(ceil < 1.0).sum()}/{len(ceil)}")
    ovl = np.array([r["overlap"] for r in ok])
    print(f"  camera overlap with the prior: median {np.median(ovl):.2f}  "
          f"under 0.1: {(ovl < 0.1).sum()}  under 0.5: {(ovl < 0.5).sum()}")
    best_wf = max(args.w_free, key=lambda wf: np.median(
        [r["pct"][str(wf)] for r in ok]))
    med = np.median([r["pct"][str(best_wf)] for r in ok])
    print(f"\n  VERDICT  best w_free={best_wf}, median percentile {med:.3f} "
          f"(threshold 99.5) -> {'PASS' if med > 99.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
