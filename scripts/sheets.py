"""Sheet -> wall extraction, on its own.

  python scripts/sheets.py --structure STReMnIXvisa5Lm   # one
  python scripts/sheets.py --root data                   # every structure there
  python scripts/sheets.py --all --force                 # rebuild the whole cache
  python scripts/sheets.py --all --check                 # report, with GT overlap
  python scripts/sheets.py --all --scales                # just the sheet scales

This stage is per STRUCTURE, not per split: a sheet takes up to 42 s and is
shared by as many as 19 splits, so it is extracted once and cached. Everything
downstream reads the cache, which is why this is worth running on its own
before a solve.

`--check` additionally measures how much retained wall sits near each split's
ground-truth walk. That reads ground truth, so it is reporting only and plays
no part in building the cache.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import sheet as sheetmod
from align.io.cases import CaseIndex
from align.scales import SheetScales


def select(index, args):
    structures = {}
    for case in index:
        st = case.structure
        if args.structure and args.structure not in st.key:
            continue
        if args.root and st.root != args.root:
            continue
        structures.setdefault(st.key, (st, []))[1].append(case)
    return structures


def gt_overlap(feats, structure, cases):
    """Fraction of wall ink near the ground-truth walks. Reporting only."""
    from align.frames import target_to_paper
    from align.groundtruth import gt_trajectory
    g = structure.geometry()
    near = []
    for c in cases:
        xy = target_to_paper(gt_trajectory(c.rlt_dir).positions,
                             structure.design_tm)[:, :2]
        px = g.paper_to_pixel(xy).astype(int)
        pad = int(1.5 * feats.ppi)
        x0, y0 = np.clip(px.min(0) - pad, 0, None)
        x1 = min(px.max(0)[0] + pad, feats.wall.shape[1])
        y1 = min(px.max(0)[1] + pad, feats.wall.shape[0])
        if x1 > x0 and y1 > y0:
            near.append(float(feats.wall[y0:y1, x0:x1].mean()))
    return float(np.mean(near)) if near else float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--structure", help="structure id or key substring")
    pick.add_argument("--root", help="a data root; validated against the index")
    pick.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="ignore the cache and re-extract")
    ap.add_argument("--check", action="store_true",
                    help="also report wall ink near the ground-truth walks")
    ap.add_argument("--scales", action="store_true",
                    help="print each structure's metres-per-inch and where it "
                         "came from, then stop")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    index = CaseIndex.load_all()
    scales = SheetScales(index)
    chosen = select(index, args)
    if not chosen:
        raise SystemExit("no structures matched")

    if args.scales:
        # Every threshold in the extraction is derived from this number, so it
        # is worth being able to see it without extracting anything.
        for key in sorted(chosen):
            mpi = scales[key]
            name, _, err = __import__("align.scales", fromlist=["nearest_standard"]) \
                .nearest_standard(mpi)
            flag = "" if scales.is_trusted(key) else "   GUESSED"
            print(f"{key:30s} {mpi:7.2f} m/in  [{scales.source(key):12s}] "
                  f"nearest {name} ({err:.0%}){flag}")
        guessed = [k for k in chosen if not scales.is_trusted(k)]
        print(f"\n{len(chosen) - len(guessed)} measured, {len(guessed)} guessed")
        return

    print(f"{len(chosen)} structure(s)"
          f"{' — rebuilding' if args.force else ''}\n")
    head = (f"{'structure':30s} {'ppi':>5s} {'m/in':>6s} {'src':>9s} "
            f"{'ink%':>6s} {'wall%':>6s} {'vp':>3s} {'photo':>5s} {'s':>5s}")
    if args.check:
        head += f" {'gt-near':>8s}"
    print(head)

    rows = []
    for key, (st, cases) in sorted(chosen.items()):
        mpi = scales[key]
        t0 = time.time()
        try:
            f = sheetmod.prepare(st, mpi, force=args.force, verbose=args.verbose)
        except Exception as exc:                                  # noqa: BLE001
            print(f"{key:30s} FAILED {exc!r}"[:120])
            continue
        dt = time.time() - t0
        line = (f"{key:30s} {f.ppi:5.0f} {mpi:6.2f} {scales.source(key)[:9]:>9s} "
                f"{f.ink_fraction * 100:6.2f} {f.wall.mean() * 100:6.3f} "
                f"{len(f.viewports):3d} {'YES' if f.photo_backed else '':>5s} "
                f"{dt:5.1f}")
        if args.check:
            line += f" {gt_overlap(f, st, cases) * 100:7.3f}%"
        print(line, flush=True)
        rows.append((key, f, scales.source(key)))

    if not rows:
        return
    w = np.array([f.wall.mean() for _, f, _ in rows])
    print(f"\nwall fraction  p10 {np.percentile(w, 10) * 100:.3f}%  "
          f"median {np.median(w) * 100:.3f}%  p90 {np.percentile(w, 90) * 100:.3f}%")
    empty = [k for k, f, _ in rows if f.wall.mean() < 1e-4]
    novp = [k for k, f, _ in rows if not f.viewports]
    guessed = [k for k, _, src in rows if src == "fallback"]
    print(f"near-empty wall maps: {empty or 'none'}")
    print(f"no viewport found   : {novp or 'none'}")
    if guessed:
        print(f"sheet scale GUESSED : {guessed}\n"
              f"  every threshold in the extraction derives from it — treat "
              f"these results as provisional")
    print(f"cache               : {sheetmod.CACHE_DIR}")


if __name__ == "__main__":
    main()
