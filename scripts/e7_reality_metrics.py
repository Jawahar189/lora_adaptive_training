"""Per-reality accuracy: RMSE and TPR in metres, with every transform kept.

The earlier experiments reported where the true pose RANKED. This reports what
the pipeline would actually deliver: take the pose it picks, place every
photograph with it, and measure how far each one lands from where it belongs.
Definitions and file layout live in `align.eval.reality`.

Three poses are scored for every reality, because one number alone cannot be
read:

  predicted  what the search chose. The honest end-to-end result.
  oracle     the best 4-DOF planar fit to the truth -- the parameterisation's
             own limit. A perfect search scores this, not zero, because a
             planar similarity cannot absorb a non-planar discrepancy.
  ceiling    the same pose reached through the PRIOR's registration, inheriting
             the prior's own error. What a flawless match to THIS reference
             would score.

Reading `predicted` without `ceiling` beside it attributes the previous
capture's registration error to this work, so both are always written.

  python scripts/e7_reality_metrics.py
  python scripts/e7_reality_metrics.py --limit 10 --stem out/e7_sample
"""
import argparse
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
from align.eval.reality import (RIG_PLAUSIBLE_M, ResultWriter, compose_pose,
                                reality_metrics, rig_height_m)
from align.frames import target_to_paper
from align.gravity import relative_scale_prior
from align.groundtruth import gt_trajectory
from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex
from align.io.pairing import prior_reference_for
from align.io.ply import read_ply
from align.io.shots import load_shots
from align.sheet import resample_to

COARSE_CELL_M = 0.20
MAX_SEARCH_PX = 4000


def search_rasters(ref_map, cell_m=COARSE_CELL_M):
    """The reference at the SEARCH cell, the way the pipeline resamples it."""
    h, w = ref_map.feats.wall.shape
    bounds = (0.0, 0.0, w * ref_map.res, h * ref_map.res)
    span = max(bounds[2], bounds[3])
    res = max(cell_m / ref_map.metres_per_inch, span / MAX_SEARCH_PX, ref_map.res)
    shape = (max(int(bounds[3] / res), 1), max(int(bounds[2] / res), 1))
    return (resample_to(ref_map.feats, ref_map.geom, bounds, res, shape).astype(np.float32),
            resample_to(ref_map.free, ref_map.geom, bounds, res, shape).astype(np.float32),
            res)


def aligned_positions(case, traj):
    """SfM and ground-truth positions for the same images, in the same order."""
    gt = gt_trajectory(case.rlt_dir)
    order = {n: k for k, n in enumerate(traj.names)}
    shared = [n for n in gt.names if n in order]
    sfm = traj.positions[[order[n] for n in shared]]
    gtp = np.asarray(gt.positions)[[k for k, n in enumerate(gt.names) if n in order]]
    return sfm, gtp


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
    ap.add_argument("--stem", default="out/e7_reality_metrics",
                    help="path prefix; .csv, .jsonl and _predictions.jsonl "
                         "are written from it")
    args = ap.parse_args()

    ix = PairedIndex.load(datasets_mod.from_args(args))
    cases = list(ix.current)[: args.limit]
    bk = S.get_backend()
    t0 = time.time()

    with ResultWriter(args.stem) as out:
        for i, case in enumerate(cases, 1):
            poses, status, ctx = {}, "ok", {}
            try:
                ref = prior_reference_for(case, ix)
                if ref is None:
                    status = "no prior cloud"
                    raise StopIteration
                ref_map = R.build_cached(ref, case.structure)
                if ref_map is None:
                    status = "no usable prior split"
                    raise StopIteration
                mpi = ref_map.metres_per_inch
                geo = case.structure.georeferenced
                ctx.update(metres_per_inch=mpi, scale_source=ref_map.scale_source)

                pc = read_ply(case.cloud_path, stride=args.stride, min_source_frames=2)
                traj = load_shots(case.rlt_dir)
                best, _grav, _ = true_placement(case, ref_map, pc.xyz, traj.positions)
                if best is None:
                    status = "no planar fit to the truth"
                    raise StopIteration
                _resid, ui, Rm, true_sim2 = best
                wall_xy, traj_xy, floor = wall_band_local(pc.xyz, traj.positions, Rm)
                if wall_xy is None:
                    status = "no usable wall band"
                    raise StopIteration

                sfm, gtp = aligned_positions(case, traj)
                cam_up = (traj.positions @ Rm.T)[:, 2]

                # How tall does this capture think its own rig is? It should be
                # about 2.11 m. When it is not, the floor fit found something
                # that is not the floor -- which corrupts the wall band and the
                # scale prior as well as the height, so it is recorded here
                # rather than left to surface as an unexplained bad result.
                rig = rig_height_m(floor.height, true_sim2.s, mpi)
                ctx["rig_height_m"] = rig
                ctx["rig_implausible"] = not (RIG_PLAUSIBLE_M[0] <= rig
                                              <= RIG_PLAUSIBLE_M[1])

                gt_paper = target_to_paper(gtp, case.structure.design_tm)[:, :2]
                prior_cams = np.vstack([ref.trajectory(r).positions[:, :2]
                                        for r in ref_map.prior_reality_ids])
                d, _ = cKDTree(prior_cams).query(gt_paper)
                ctx["overlap_5m"] = float((d * mpi <= 5.0).mean())

                def pose(sim2):
                    tm = compose_pose(sim2, Rm, ref_map, cam_up, floor.height,
                                      case.structure.design_tm)
                    return dict(reality_metrics(tm, sfm, gtp, mpi, geo), tm=tm)

                poses["oracle"] = pose(true_sim2)

                sheet_map, free_map, sres = search_rasters(ref_map)
                s0, bracket = relative_scale_prior(floor, ref_map.camera_height_in)
                if not np.isfinite(s0):
                    status = "no scale prior"
                    raise StopIteration
                cell = COARSE_CELL_M / mpi
                cm = cloudmod.manhattan(
                    cloudmod.wall_raster(wall_xy * s0, res=cell).mask)
                sm = cloudmod.manhattan(sheet_map)
                yaws, fallback = S.yaw_candidates(
                    cm.angle, sm.angle, strength=min(cm.strength, sm.strength))
                ctx["yaw_fallback"] = bool(fallback)
                grid = S.SearchGrid(scales=S.scale_candidates(bracket),
                                    yaws_deg=yaws, viewports=[0],
                                    fallback_yaw=fallback)
                hyps = S.search(wall_xy, traj_xy, sheet_map, (0.0, 0.0), sres,
                                grid, backend=bk, scale_prior=s0, up_index=ui,
                                w_free=args.w_free, interior=free_map)
                if not hyps:
                    status = "no hypotheses"
                    raise StopIteration
                poses["predicted"] = pose(hyps[0].sim2)
            except StopIteration:
                pass
            except Exception as exc:
                status = f"error: {exc!r}"[:160]

            out.write(case, poses, status=status, **ctx)
            p, o = poses.get("predicted"), poses.get("oracle")
            msg = (f"pred xy {p['rmse_xy']:7.2f} z {p['rmse_z']:5.2f} "
                   f"tpr@1 {p['tpr@1.0']:.2f} tpr@2 {p['tpr@2.0']:.2f}"
                   f"  |  oracle xy {o['rmse_xy']:5.2f} z {o['rmse_z']:5.2f}"
                   if p and o else status)
            print(f"[{i}/{len(cases)}] {case.reality_id}  {msg}", flush=True)

        print(f"\n{len(cases)} realities in {time.time() - t0:.0f}s")
        for name, path in out.paths.items():
            print(f"  {name:12s} {path}")
        print("\nPREDICTED\n" + out.report())


if __name__ == "__main__":
    main()
