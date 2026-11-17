"""How well does a prior capture agree with ITSELF? The reference's own quality.

A current split is aligned to the merged bird's-eye view of a prior capture's
splits, each placed by its own registration. If two of those splits disagree
about where the same photograph is, the merged reference is blurred by exactly
that much before the matcher sees it -- and no matcher can recover a pose from a
reference that contradicts itself.

This measures it directly and without touching a point cloud: within a
structure, any two splits that share image filenames should put those images in
the same place. The residual between them is a lower bound on the reference's
sharpness, and it is knowable before any matching code exists.

It reads a prior's registration, which is a legitimate input, so this is an
analysis of the INPUT rather than of the answer -- unlike `scripts/consistency.py`,
which measures the same quantity over current splits and is a ground-truth
self-consistency check.

  python scripts/e5_prior_quality.py
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.frames import apply as apply4
from align.frames import design_scale, target_to_paper
from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex
from align.io.shots import load_shots
from align.priors import PriorReference, PriorSplit


def _paper_positions(split, design_tm):
    """One prior split's cameras in paper inches, keyed by image name.

    Goes through a PriorReference so the read is guarded like any other, even
    though this script is analysis: the guard is cheap and a script that
    bypasses it is a script that teaches people to bypass it.
    """
    ref = PriorReference(
        current_reality_id="__analysis__", current_capture_id="__analysis__",
        current_rlt_dir=Path("/nonexistent"), current_capture_date="",
        structure_key=split.structure_key, splits=[split], design_tm=design_tm)
    traj = ref.trajectory(split.reality_id)
    return dict(zip(traj.names, traj.positions))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets_mod.add_arguments(ap)
    ap.add_argument("--out", default="out/e5_prior_quality.jsonl",
                    help="one record per comparable pair of prior splits")
    ap.add_argument("--min-shared", type=int, default=3,
                    help="images two splits must share to be comparable")
    args = ap.parse_args()

    ix = PairedIndex.load(datasets_mod.from_args(args))
    if not len(ix.prior):
        raise SystemExit("this dataset has no prior side; pass --prior-csv")

    # design_tm and the sheet scale are per structure, and a prior Case carries
    # its own structure, so group there.
    by_structure = collections.defaultdict(list)
    for c in ix.prior:
        by_structure[c.structure.key].append(c)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows, all_rms = [], []
    with open(args.out, "w") as fh:
        for skey, members in sorted(by_structure.items()):
            st = members[0].structure
            mpi = design_scale(st.design_tm) if st.design_tm is not None else float("nan")
            pos = {}
            for c in members:
                split = PriorSplit(reality_id=c.reality_id, capture_id=c.capture_id,
                                   structure_key=skey, capture_date=c.capture_date,
                                   rlt_dir=c.rlt_dir, cloud_path=c.cloud_path,
                                   shots_path=c.shots_path)
                try:
                    pos[c.reality_id] = _paper_positions(split, st.design_tm)
                except Exception as exc:
                    fh.write(json.dumps(dict(structure=skey, reality_id=c.reality_id,
                                             error=repr(exc)[:120])) + "\n")
            cap = {c.reality_id: c.capture_id for c in members}
            ids = sorted(pos)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    shared = sorted(set(pos[a]) & set(pos[b]))
                    if len(shared) < args.min_shared:
                        continue
                    d = np.linalg.norm(
                        np.array([pos[a][n] for n in shared])
                        - np.array([pos[b][n] for n in shared]), axis=1)
                    rms_in = float(np.sqrt((d ** 2).mean()))
                    rec = dict(structure=skey, a=a, b=b,
                               same_capture=cap[a] == cap[b],
                               n_shared=len(shared),
                               rms_in=rms_in, max_in=float(d.max()),
                               metres_per_inch=None if np.isnan(mpi) else mpi,
                               rms_m=None if np.isnan(mpi) else rms_in * mpi)
                    fh.write(json.dumps(rec) + "\n")
                    rows.append(rec)
                    if rec["rms_m"] is not None:
                        all_rms.append((rec["rms_m"], rec["same_capture"], skey, a, b))

    print(f"{len(rows)} comparable split pairs across {len(by_structure)} structures")
    if not all_rms:
        print("  (no georeferenced pairs; residuals are only expressible in inches)")
        return
    same = np.array([r[0] for r in all_rms if r[1]])
    diff = np.array([r[0] for r in all_rms if not r[1]])
    for label, arr in [("within one capture", same), ("across captures", diff)]:
        if not len(arr):
            continue
        print(f"\n  {label}: n={len(arr)}  metres")
        for q in (50, 75, 90, 95):
            print(f"    p{q:<3d} {np.percentile(arr, q):7.2f}")
        print(f"    max  {arr.max():7.2f}")
        for t in (0.5, 1.0, 2.0, 5.0):
            print(f"    over {t:>4.1f} m: {(arr > t).sum():4d} / {len(arr)}")
    print("\n  worst pairs:")
    for m, sc, skey, a, b in sorted(all_rms, reverse=True)[:8]:
        print(f"    {m:8.2f} m  {skey:28s} {a} vs {b}{'  (same capture)' if sc else ''}")


if __name__ == "__main__":
    main()
