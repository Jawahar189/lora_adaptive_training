#!/usr/bin/env python
"""Is the chain's start-point anchor actually on the walk?

`geo_first_point.json` identifies a capture moment twice over -- a reality id
and a photograph -- and they disagree. `align.chain.find_anchor` trusts the
photograph and searches every split of the capture for it. This measures
whether the window it picks really does pass near the surveyed point.

Reads ground truth, so it lives in scripts/ rather than align/. The bar: the
distance from the point to the nearest camera in the chosen window is under 2
paper inches for every anchored capture except RLT1788201055417, whose walk
runs off its sheet and which E2 showed carries no wall evidence at all.

    python scripts/check_anchor.py --csv CURRENT.csv [--window 20]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.chain import ANCHOR_WINDOW, capture_geo, find_anchor
from align.frames import target_to_paper
from align.groundtruth import gt_trajectory
from align.io.cases import CaseIndex

BAR_IN = 2.0
KNOWN_BAD = {"RLT1788201055417"}      # walk leaves the sheet; no evidence either


class _Paper:
    """A Trajectory whose positions are paper inches rather than target frame."""

    def __init__(self, traj, positions):
        self.names, self.idx, self.positions = traj.names, traj.idx, positions


def _on_paper(traj, design_tm):
    xy = target_to_paper(np.asarray(traj.positions), design_tm)[:, :2]
    return _Paper(traj, np.c_[xy, np.zeros(len(xy))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--window", type=int, default=ANCHOR_WINDOW)
    args = ap.parse_args()

    cases = []
    seen = set()
    for name in ([args.csv] if isinstance(args.csv, str) else args.csv):
        for c in CaseIndex.load(name).cases:
            if c.reality_id not in seen:
                seen.add(c.reality_id)
                cases.append(c)

    captures = defaultdict(list)
    for c in cases:
        captures[c.capture_id].append(c)

    rows, missing, unanchored = [], [], 0
    for cap, members in sorted(captures.items()):
        geo = capture_geo(members)
        if geo is None:
            unanchored += 1
            continue
        # At solve time `find_anchor` sees SfM-frame positions and returns the
        # window in that frame, which is what the prior needs. To MEASURE the
        # anchor we hand it ground-truth positions mapped to paper instead, so
        # the window comes back in the same units as the start point.
        trajs = {m.reality_id: _on_paper(gt_trajectory(m.rlt_dir),
                                         m.structure.design_tm)
                 for m in members}
        anchor = find_anchor(members, trajs, geo, metres_per_inch=1.0,
                             window=args.window)
        if anchor is None:
            missing.append((cap, geo.image))
            continue
        err = float(np.linalg.norm(anchor.ref - anchor.centre, axis=1).min())
        extent = float(np.linalg.norm(
            anchor.ref - anchor.ref.mean(0), axis=1).max() * 2)
        owner = next((m.reality_id for m in members if m.geo is not None), "")
        rows.append((cap, anchor.reality_id, owner, err, extent, len(anchor.ref)))

    print(f"{len(captures)} captures  {unanchored} with no start point  "
          f"{len(missing)} naming an image in no split  {len(rows)} anchored")
    moved = sum(1 for r in rows if r[1] != r[2])
    print(f"{moved} of {len(rows)} anchors moved to a different split than "
          f"source_reality_id named")

    errs = np.array([r[3] for r in rows])
    ext = np.array([r[4] for r in rows])
    print(f"\nwindow +-{args.window}  n={len(rows)}")
    print(f"  anchor error  median {np.median(errs):.2f} in  "
          f"p90 {np.percentile(errs, 90):.2f}  max {errs.max():.2f}")
    print(f"  window extent median {np.median(ext):.2f} in  "
          f"p90 {np.percentile(ext, 90):.2f}")

    over = [r for r in rows if r[3] >= BAR_IN and r[1] not in KNOWN_BAD]
    print(f"\n{len(rows) - len(over)} of {len(rows)} within {BAR_IN} in"
          + (f"; {len(over)} over:" if over else ""))
    for cap, rid, owner, err, extent, n in sorted(over, key=lambda r: -r[3]):
        print(f"  {rid}  {err:7.2f} in   (point filed under {owner or '-'})")
    for cap, image in missing:
        print(f"  {cap}: start point names {image}, in no split")
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
