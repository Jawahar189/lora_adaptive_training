"""Does a panorama alone say where a camera is? Retrieval against the prior.

Every prior panorama has a known place on the plan. Describe every panorama
with a frozen DINOv2 (`align.visual`), and for each current camera take the
prior camera whose picture looks most alike. Measured against the truth:

  PLACE     per camera, how far the retrieved prior camera is from where the
            current camera really is;
  SEED      the median retrieved place of a split's first cameras, against
            the GNSS start point it would replace (4.7 m median, 39 m p90);
  POSE      a full planar pose fitted to the retrieved places (RANSAC over the
            walk), judged as `top1_correct` is: 2 m, 10 degrees, 1.15x.

Truth is read here, in a script, and only to score. `align/visual.py` itself
names no ground-truth file.

  python scripts/e8_visual_retrieval.py --limit 5 --extract-only
  python scripts/e8_visual_retrieval.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import reference as refmod
from align import visual
from align.eval.oracle import best_planar_fit, wall_band_local
from align.frames import target_to_paper
from align.gravity import _rotation_to_z, estimate_gravity, relative_scale_prior
from align.groundtruth import gt_trajectory
from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex
from align.io.pairing import prior_reference_for
from align.io.ply import read_ply
from align.io.shots import load_shots

POS_TOL_M, YAW_TOL_DEG, SCALE_TOL = 2.0, 10.0, 1.15


def pose_errors(sim, true_sim2, src_xy, mpi):
    dpos = float(np.linalg.norm(sim.apply(src_xy) - true_sim2.apply(src_xy), axis=1).mean()) * mpi
    dyaw = abs(((np.degrees(sim.theta - true_sim2.theta) + 180) % 360) - 180)
    ratio = sim.s / true_sim2.s
    ok = dpos <= POS_TOL_M and dyaw <= YAW_TOL_DEG and 1 / SCALE_TOL <= ratio <= SCALE_TOL
    return dict(pose_error_m=dpos, yaw_error_deg=float(dyaw), scale_ratio=float(ratio), correct=bool(ok))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets_mod.add_arguments(ap)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reality", nargs="+", default=None)
    ap.add_argument("--model", default=visual.DEFAULT_MODEL)
    ap.add_argument("--rolls", type=int, default=visual.ROLLS)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed-cameras", type=int, default=20)
    ap.add_argument("--radius-m", type=float, default=5.0,
                    help="walk-consistency tolerance, and the SEED bar")
    ap.add_argument("--cache", default="cache/desc")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--out", default="out/e8_visual_retrieval.jsonl")
    args = ap.parse_args()

    ix = PairedIndex.load(datasets_mod.from_args(args))
    cases = list(ix.current)
    if args.reality:
        cases = [c for c in cases if c.reality_id in set(args.reality)]
    cases = cases[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bk = visual.Backbone(args.model)
    rows, t0 = [], time.time()

    with open(args.out, "w") as fh:
        for i, case in enumerate(cases, 1):
            rec = dict(reality_id=case.reality_id, capture_id=case.capture_id,
                       structure=case.structure.structure_id)
            t1 = time.time()
            try:
                ref = prior_reference_for(case, ix)
                if ref is None:
                    rec["status"] = "no prior cloud"; raise StopIteration
                ref_map = refmod.build_cached(ref, case.structure)
                if ref_map is None:
                    rec["status"] = "no usable prior split"; raise StopIteration
                mpi, origin = ref_map.metres_per_inch, ref_map.origin
                traj = load_shots(case.rlt_dir)
                rec["same_day"] = bool(ref.same_day)

                # descriptors: the current split unrolled, the prior pool rolled
                q = visual.describe(case.rlt_dir, traj.names, bk, args.cache,
                                    case.reality_id, rolls=1, workers=args.workers)[:, 0]
                pool, pool_xy, pool_rid = [], [], []
                for rid in ref_map.prior_reality_ids:
                    t = ref.trajectory(rid)
                    split = ref._check(rid)
                    d = visual.describe(split.rlt_dir, t.names, bk, args.cache, rid,
                                        rolls=args.rolls, workers=args.workers)
                    pool.append(d); pool_xy.append(t.positions[:, :2] - origin)
                    pool_rid += [rid] * len(t.names)
                pool, pool_xy = np.concatenate(pool), np.concatenate(pool_xy)
                rec.update(n_query=len(q), n_pool=len(pool))
                if args.extract_only:
                    rec["status"] = "extracted"; raise StopIteration

                # truth, joined by image name, in the local frame
                gt = gt_trajectory(case.rlt_dir)
                g = {n: k for k, n in enumerate(gt.names)}
                keep = [k for k, n in enumerate(traj.names) if n in g]
                if len(keep) < 3:
                    rec["status"] = "no images in common with the truth"; raise StopIteration
                gt_local = target_to_paper(np.asarray(gt.positions)[[g[traj.names[k]] for k in keep]],
                                           case.structure.design_tm)[:, :2] - origin
                pc = read_ply(case.cloud_path, stride=3, min_source_frames=2)
                grav = estimate_gravity(pc.xyz, traj.positions)
                rots = {ui: _rotation_to_z(up) for ui, up in enumerate(grav.alternatives())}
                best = best_planar_fit(gt_local, rots, traj.positions[keep])
                if best is None:
                    rec["status"] = "no planar fit to the truth"; raise StopIteration
                _, ui, Rm, true_sim2 = best
                src = (traj.positions[keep] @ Rm.T)[:, :2]
                _, _, floor = wall_band_local(pc.xyz, traj.positions, Rm)
                s0, bracket = relative_scale_prior(floor, ref_map.camera_height_in)
                rec.update(scale_prior=float(s0) if np.isfinite(s0) else None)

                # PLACE: top-1 retrieved prior camera vs where this camera really is
                idx, sims = visual.retrieve(q[keep], pool, top_k=args.top_k)
                cand = pool_xy[idx]                                   # (n, k, 2) local
                err = np.linalg.norm(cand[:, 0] - gt_local, axis=1) * mpi
                rec.update(place_err_med_m=float(np.median(err)),
                           place_err_p90_m=float(np.percentile(err, 90)),
                           place_within_2m=float((err <= 2).mean()),
                           place_within_5m=float((err <= 5).mean()),
                           place_within_10m=float((err <= 10).mean()),
                           top1_sim_med=float(np.median(sims[:, 0])))

                # SEED: the split's first cameras, as the GNSS start would be used
                order = np.argsort(traj.idx[keep])[: args.seed_cameras]
                seed_err = np.linalg.norm(np.median(cand[order, 0], 0) - np.median(gt_local[order], 0)) * mpi
                rec["seed_err_m"] = float(seed_err)

                # POSE: raw top-1 places, then walk-consistent anchors only
                tol_in = POS_TOL_M / mpi
                fit = visual.ransac_sim2(src, cand[:, 0], tol=tol_in,
                                         bracket=bracket if np.isfinite(s0) else None)
                rec["raw"] = (dict(n_inliers=int(fit[1].sum()), **pose_errors(fit[0], true_sim2, src, mpi))
                              if fit else dict(n_inliers=0, correct=False))
                anchors = visual.consistent_anchors(
                    src, cand, scale=float(s0) if np.isfinite(s0) else float(true_sim2.s),
                    radius=args.radius_m / mpi)
                rec["n_anchors"] = len(anchors)
                if anchors:
                    ai = np.array([a for a, _ in anchors]); aj = np.array([j for _, j in anchors])
                    a_err = np.linalg.norm(cand[ai, aj] - gt_local[ai], axis=1) * mpi
                    rec["anchor_err_med_m"] = float(np.median(a_err))
                    rec["anchor_within_2m"] = float((a_err <= 2).mean())
                    a_order = ai[np.argsort(traj.idx[keep][ai])][: args.seed_cameras]
                    a_j = np.array([dict(anchors)[a] for a in a_order])
                    rec["anchor_seed_err_m"] = float(np.linalg.norm(
                        np.median(cand[a_order, a_j], 0) - np.median(gt_local[a_order], 0)) * mpi)
                    fit = (visual.ransac_sim2(src[ai], cand[ai, aj], tol=tol_in,
                                              bracket=bracket if np.isfinite(s0) else None)
                           if len(ai) >= 3 else None)
                    rec["anchored"] = (dict(n_inliers=int(fit[1].sum()), **pose_errors(fit[0], true_sim2, src, mpi))
                                       if fit else dict(n_inliers=0, correct=False))
                else:
                    rec["anchored"] = dict(n_inliers=0, correct=False)
                rec["status"] = "ok"
            except StopIteration:
                pass
            except Exception as exc:                          # noqa: BLE001
                rec["status"] = f"error: {exc!r}"[:200]
            rec["seconds"] = round(time.time() - t1, 1)
            fh.write(json.dumps(rec) + "\n"); fh.flush(); rows.append(rec)
            if rec.get("status") == "ok":
                msg = (f"pool {rec['n_pool']:5d} | place med {rec['place_err_med_m']:6.1f} m "
                       f"<=5m {rec['place_within_5m']:.2f} | seed {rec['seed_err_m']:6.1f} m | "
                       f"anchors {rec['n_anchors']:3d} seed {rec.get('anchor_seed_err_m', float('nan')):6.1f} m | "
                       f"pose raw {'OK' if rec['raw']['correct'] else '--'} anchored {'OK' if rec['anchored']['correct'] else '--'}")
            else:
                msg = rec["status"]
            print(f"[{i}/{len(cases)}] {case.reality_id} {rec['seconds']:6.1f}s  {msg}", flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n{len(ok)} of {len(cases)} scored in {time.time() - t0:.0f}s")
    if not ok:
        return
    n = len(ok)
    pct = lambda f: f"{sum(1 for r in ok if f(r)):3d}/{n} ({100 * sum(1 for r in ok if f(r)) / n:3.0f}%)"
    for label, sub in (("ALL", ok), ("different-day priors only", [r for r in ok if not r["same_day"]])):
        if not sub:
            continue
        m = len(sub)
        p = lambda f, sub=sub, m=m: f"{sum(1 for r in sub if f(r)):3d}/{m} ({100 * sum(1 for r in sub if f(r)) / m:3.0f}%)"
        print(f"\n  == {label}: {m} realities ==")
        print(f"  PLACE  per-camera top-1 error: median of split medians {np.median([r['place_err_med_m'] for r in sub]):.1f} m;"
              f" cameras within 5 m: mean {np.mean([r['place_within_5m'] for r in sub]):.2f}")
        print(f"  SEED   first-{args.seed_cameras} centre within 5 m   raw {p(lambda r: r['seed_err_m'] <= 5)}"
              f"   anchored {p(lambda r: r.get('anchor_seed_err_m', 1e9) <= 5)}"
              f"   (GNSS: median 4.7 m, p90 39 m)")
        print(f"         raw seed error median {np.median([r['seed_err_m'] for r in sub]):.1f} m  p90 {np.percentile([r['seed_err_m'] for r in sub], 90):.1f} m")
        print(f"  POSE   correct (2 m, 10 deg, 1.15x)   raw {p(lambda r: r['raw']['correct'])}"
              f"   anchored {p(lambda r: r['anchored']['correct'])}")
        print(f"  anchors per split: median {np.median([r['n_anchors'] for r in sub]):.0f}")


if __name__ == "__main__":
    main()
