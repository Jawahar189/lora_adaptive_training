"""Prove that shots.geojson IS the ply-frame trajectory, before anything relies on it.

The solver's camera trajectory used to be synthesized as inv(tm) @ images_full
-- ground truth, un-applied. `shots.geojson` is production's own record of the
same poses and needs no ground truth at all, so switching to it removes the
last inflow of ground truth into solver inputs. That is only true if the two
agree, which is what this measures:

  * the two filename sets must be identical -- a subset would silently shorten
    the trajectory and `scripts/image_errors.py` skips any split whose pose
    count disagrees with its image count, so the split would vanish rather than
    fail;
  * the positions must agree to well under the 0.20 m coarse cell.

Measured at the switch: 273 splits, zero name-set differences, max RMS
1.2e-9 m. Kept as a regression check -- a future dataset drop with a rebuilt or
truncated shots.geojson should fail here, loudly, rather than quietly degrade
the input the whole solve rests on.

  python scripts/check_shots.py --current-csv CUR.csv --prior-csv PRIOR.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.groundtruth import gt_trajectory, load_tm
from align.io import datasets as datasets_mod
from align.io.shots import load_shots

TOL_M = 1e-6


def rlt_dirs(csv_paths):
    """Every split's directory, deduplicated -- 49 realities are in both CSVs."""
    seen = {}
    for name in csv_paths:
        path = Path(name)
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                rid = (row.get("reality_id") or "").strip()
                # tm_path is populated on every row; the cloud path is not.
                if rid and rid not in seen and row.get("tm_path", "").strip():
                    seen[rid] = Path(row["tm_path"]).parent
    return seen


def compare(rlt_dir):
    shots, gt = load_shots(rlt_dir), gt_trajectory(rlt_dir)
    inv = np.linalg.inv(load_tm(rlt_dir))
    expected = {n: p for n, p in zip(gt.names,
                                     gt.positions @ inv[:3, :3].T + inv[:3, 3])}
    missing = sorted(set(expected) - set(shots.names))
    extra = sorted(set(shots.names) - set(expected))
    common = [n for n in shots.names if n in expected]
    if not common:
        return dict(n_shots=len(shots), n_gt=len(gt), n_common=0,
                    missing=len(missing), extra=len(extra),
                    rms_m=float("nan"), max_m=float("nan"))
    have = np.array([shots.positions[shots.names.index(n)] for n in common])
    want = np.array([expected[n] for n in common])
    d = np.linalg.norm(have - want, axis=1)
    return dict(n_shots=len(shots), n_gt=len(gt), n_common=len(common),
                missing=len(missing), extra=len(extra),
                rms_m=float(np.sqrt((d ** 2).mean())), max_m=float(d.max()))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets_mod.add_arguments(ap)
    ap.add_argument("--tol", type=float, default=TOL_M,
                    help=f"metres; a split above this fails the check (default {TOL_M})")
    ap.add_argument("--out", default="out/check_shots.jsonl",
                    help="per-split record of the comparison")
    args = ap.parse_args()

    spec = datasets_mod.from_args(args)
    dirs = rlt_dirs(spec.paths("current") + spec.paths("prior"))
    if not dirs:
        raise SystemExit("no rows found in the given CSVs")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    bad, rms = [], []
    with open(args.out, "w") as fh:
        for i, (rid, rlt) in enumerate(sorted(dirs.items()), 1):
            try:
                rec = compare(rlt)
            except Exception as exc:                  # a missing or broken file
                rec = dict(error=repr(exc)[:160])
                bad.append((rid, rec["error"]))
            rec["reality_id"] = rid
            fh.write(json.dumps(rec) + "\n")
            if "error" not in rec:
                rms.append(rec["rms_m"])
                if rec["missing"] or rec["extra"]:
                    bad.append((rid, f"{rec['missing']} missing, {rec['extra']} extra"))
                elif not (rec["max_m"] <= args.tol):
                    bad.append((rid, f"max {rec['max_m']:.3g} m"))
            if i % 50 == 0:
                print(f"  {i}/{len(dirs)}", flush=True)

    print(f"\n{len(dirs)} splits, {len(rms)} compared")
    if rms:
        print(f"  rms  median {np.median(rms):.3g} m   max {max(rms):.3g} m")
    if bad:
        print(f"  FAILED {len(bad)}:")
        for rid, why in bad[:20]:
            print(f"    {rid}  {why}")
        raise SystemExit(1)
    print(f"  all within {args.tol:g} m -- shots.geojson is the ply trajectory")


if __name__ == "__main__":
    main()
