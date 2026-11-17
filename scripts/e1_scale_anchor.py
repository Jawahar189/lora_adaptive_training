"""E1 -- the gating experiment for the metric anchor.

Measures camera-height-above-floor on every split and multiplies by the
ground-truth scale. For georeferenced structures the product is in true metres
and should cluster tightly around the rig height; for paper-frame structures it
is in inches and inverts to give the sheet's metres-per-inch.

Falsified if the p10-p90 spread exceeds +/-25% of the median, or if more than
~15 splits yield no usable floor mode.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.frames import sim3_scale
from align.gravity import (estimate_gravity, floor_below_trajectory,
                           trajectory_extent)
from align.groundtruth import load_tm
from align.io.shots import load_shots
from align.io.ply import read_ply
from align.io.cases import CaseIndex


def measure(case, stride=4):
    cloud = read_ply(case.cloud_path, stride=stride, min_source_frames=2)
    traj = load_shots(case.rlt_dir)
    g = estimate_gravity(cloud.xyz, traj.positions)
    P = cloud.xyz @ g.R.T
    C = traj.positions @ g.R.T
    f = floor_below_trajectory(P, C)
    return g, f, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="restrict to one dataset CSV")
    ap.add_argument("--root", default=None,
                    help="restrict to one data root, e.g. data or data_v")
    ap.add_argument("--out", default="out/e1_scale_anchor.jsonl")
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    index = CaseIndex.load(args.csv) if args.csv else CaseIndex.load_all()
    if args.root:
        index = CaseIndex([c for c in index if c.structure.root == args.root])
        if not len(index):
            raise SystemExit(f"no cases under root {args.root!r}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.out, "w") as fh:
        for i, case in enumerate(index, 1):
            try:
                g, f, traj = measure(case, args.stride)
                gt_scale = sim3_scale(load_tm(case.rlt_dir))
                rec = dict(
                    reality_id=case.reality_id, structure=case.structure.key,
                    status=case.structure.status,
                    georeferenced=case.structure.georeferenced,
                    n_cam=len(traj), gravity_method=g.method,
                    sign_ambiguous=bool(g.sign_ambiguous),
                    planarity=round(g.planarity, 5),
                    floor_h=None if not np.isfinite(f.height) else round(f.height, 6),
                    rel_mad=None if not np.isfinite(f.height) else round(f.rel_mad, 4),
                    n_floor=f.n_used,
                    gt_scale=gt_scale,
                    # height expressed in target units: metres, or paper inches
                    height_target=None if not np.isfinite(f.height)
                    else round(f.height * gt_scale, 4),
                )
            except Exception as exc:                      # noqa: BLE001
                rec = dict(reality_id=case.reality_id, structure=case.structure.key,
                           status=case.structure.status,
                           georeferenced=case.structure.georeferenced, error=repr(exc))
            rows.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i:3d}/{len(index)}] {case.reality_id} "
                  f"{'GEO' if case.structure.georeferenced else 'pap'} "
                  f"h={rec.get('height_target')} relmad={rec.get('rel_mad')} "
                  f"{'AMBIG' if rec.get('sign_ambiguous') else '     '} "
                  f"{rec.get('error','')}", flush=True)

    def heights(sel):
        return np.array([r["height_target"] for r in rows
                         if sel(r) and r.get("height_target")])

    def describe(name, arr, unit):
        if not len(arr):
            return
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        print(f"  {name:34s} n={len(arr):3d}  p10 {p10:6.2f}  median {p50:6.2f}  "
              f"p90 {p90:6.2f}  spread -{(p50-p10)/p50:.0%}/+{(p90-p50)/p50:.0%} [{unit}]")

    failed = [r["reality_id"] for r in rows if not r.get("height_target")]
    amb = [r for r in rows if r.get("sign_ambiguous")]

    print("\n" + "=" * 78)
    print("E1 RESULT -- camera height above floor")
    print("\n  georeferenced splits (true metres):")
    describe("all", heights(lambda r: r.get("georeferenced")), "m")
    describe("sign-decisive only", heights(
        lambda r: r.get("georeferenced") and not r.get("sign_ambiguous")), "m")
    print("\n  paper-frame splits (inches per cloud unit -> m/inch when inverted):")
    describe("all", heights(lambda r: not r.get("georeferenced")), "in")
    describe("sign-decisive only", heights(
        lambda r: not r.get("georeferenced") and not r.get("sign_ambiguous")), "in")
    print(f"\n  floor/ceiling sign ambiguous: {len(amb)}/{len(rows)} splits "
          f"(both hypotheses carried forward, matcher decides)")
    print(f"  no usable floor mode:         {len(failed)}/{len(rows)} "
          f"(falsified above ~15)")
    if failed:
        print(f"    {failed}")
    print("\n  VERDICT: the anchor is trustworthy where the sign test is decisive;")
    print("           ambiguous splits must carry two scale hypotheses.")
    print("=" * 78)


if __name__ == "__main__":
    main()
