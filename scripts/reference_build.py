"""Build and cache the prior reference for every current split.

One reference per distinct PRIOR SET, not per split: the splits of a current
capture name the same priors, so 131 rows collapse to 58 builds. Run this once
and every later stage reads the cache.

Prints what was kept and what was not, because the drop list is the thing worth
reading: a split dropped for disagreeing with its sibling by 6 m is a data
problem, and a split whose floor fit was overridden is a fit problem, and both
are invisible once the reference is a boolean array.

  python scripts/reference_build.py
  python scripts/reference_build.py --force --stride 6
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import reference as R
from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex
from align.io.pairing import prior_reference_for


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets_mod.add_arguments(ap)
    ap.add_argument("--stride", type=int, default=R.MERGE_STRIDE,
                    help="keep every Nth point of each prior cloud; lower is "
                         "denser and slower")
    ap.add_argument("--cell-m", type=float, default=R.CELL_M,
                    help="reference raster cell, metres. The prior disagrees "
                         "with itself by ~1.4 m, so finer buys little")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even where a cached reference exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N realities, for a smoke test")
    ap.add_argument("--out", default="out/reference_build.jsonl",
                    help="what was kept, dropped and corrected per reference")
    args = ap.parse_args()

    ix = PairedIndex.load(datasets_mod.from_args(args))
    cases = list(ix.current)[: args.limit]
    seen: dict[str, str] = {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    built = skipped = failed = abstain = 0
    t0 = time.time()
    with open(args.out, "w") as fh:
        for i, case in enumerate(cases, 1):
            ref = prior_reference_for(case, ix)
            if ref is None:
                abstain += 1
                fh.write(json.dumps(dict(reality_id=case.reality_id,
                                         status="no prior with a cloud")) + "\n")
                print(f"[{i}/{len(cases)}] {case.reality_id}  ABSTAIN (no prior cloud)")
                continue
            if ref.key in seen and not args.force:
                skipped += 1
                fh.write(json.dumps(dict(reality_id=case.reality_id, key=ref.key,
                                         status="shared", with_=seen[ref.key])) + "\n")
                continue
            t = time.time()
            try:
                r = R.build_cached(ref, case.structure, stride=args.stride,
                                   cell_m=args.cell_m, force=args.force)
            except Exception as exc:
                failed += 1
                fh.write(json.dumps(dict(reality_id=case.reality_id, key=ref.key,
                                         status="error", error=repr(exc)[:200])) + "\n")
                print(f"[{i}/{len(cases)}] {case.reality_id}  FAILED {exc!r}"[:150])
                continue
            dt = time.time() - t
            if r is None:
                failed += 1
                fh.write(json.dumps(dict(reality_id=case.reality_id, key=ref.key,
                                         status="no usable prior split")) + "\n")
                print(f"[{i}/{len(cases)}] {case.reality_id}  no usable prior split")
                continue
            seen[ref.key] = case.reality_id
            built += 1
            h, w = r.feats.wall.shape
            fh.write(json.dumps(dict(
                reality_id=case.reality_id, key=r.key, status="built",
                seconds=round(dt, 1), kept=list(r.prior_reality_ids),
                dropped=[list(d) for d in r.dropped], warnings=r.warnings,
                rows=h, cols=w, res_in=r.res, wall_frac=float(r.feats.wall.mean()),
                cover_frac=float(r.free.wall.mean()), n_points=r.n_points,
                metres_per_inch=r.metres_per_inch, scale_source=r.scale_source,
                rig_m=r.camera_height_in * r.metres_per_inch,
                levels=len(r.levels))) + "\n")
            print(f"[{i}/{len(cases)}] {case.reality_id}  {dt:5.1f}s  "
                  f"{len(r.prior_reality_ids)} kept / {len(r.dropped)} dropped  "
                  f"{h}x{w}  wall {r.feats.wall.mean():.2%}  "
                  f"rig {r.camera_height_in * r.metres_per_inch:.2f} m", flush=True)

    print(f"\n{built} built, {skipped} shared a cached reference, "
          f"{failed} failed, {abstain} abstained, in {time.time() - t0:.0f}s")
    rows = [json.loads(l) for l in open(args.out)]
    good = [r for r in rows if r.get("status") == "built"]
    if good:
        rig = np.array([r["rig_m"] for r in good])
        print(f"  rig height   median {np.median(rig):.2f} m  "
              f"p10 {np.percentile(rig, 10):.2f}  p90 {np.percentile(rig, 90):.2f}")
        wall = np.array([r["wall_frac"] for r in good])
        print(f"  wall density median {np.median(wall):.2%}  "
              f"min {wall.min():.2%}  max {wall.max():.2%}")
        px = np.array([r["rows"] * r["cols"] for r in good])
        print(f"  raster cells median {np.median(px)/1e6:.1f}M  max {px.max()/1e6:.1f}M")
        drops = collections.Counter(
            d[1].split(" by ")[0].split(" gave ")[0] for r in good for d in r["dropped"])
        print(f"  splits dropped: {sum(drops.values())}")
        for why, n in drops.most_common(6):
            print(f"    {n:3d}  {why}")
        corrected = sum(1 for r in good for w in r["warnings"] if "rig height" in w)
        print(f"  floor fits overridden with the known rig height: {corrected}")
        print(f"  references with >1 level: "
              f"{sum(1 for r in good if r['levels'] > 1)}")


if __name__ == "__main__":
    main()
