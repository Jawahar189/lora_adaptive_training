#!/usr/bin/env python
"""Which way of finding the cloud's wall segments ranks the true pose better?

`align.rerank` can obtain segments two ways (see `SEGMENT_MODE`):

  "pose"   LSD on the cloud rasterized at each candidate's yaw and scale, which
           is what `count_inliers` defines -- and one LSD run per distinct
           (yaw, scale), 252 of them on a typical grid at ~157 ms each.
  "cloud"  one LSD for the whole split, endpoints carried to each pose.

"pose" is exact by construction, but exactness is against a measure that is
itself pose-dependent: the segment COUNT swings with raster aliasing, so two
candidates are not scored against the same denominator. "cloud" is 252x cheaper
and scores every candidate against one segment set.

Cost is not the argument -- ranking is. This reports, for each mode: the error
of the top-ranked pose, where the closest candidate sits in the ranking, and
how the TRUE pose itself scores against the best candidate.

    python scripts/compare_segment_modes.py RLT... [RLT...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import align.search as searchmod
from align import rerank as rerankmod
from align.frames import Sim2, decompose_tm, sim3_scale, sim3_yaw
from align.groundtruth import load_tm
from align.io.cases import CaseIndex
from align.pipeline import RealityPipeline

DEFAULT = ["RLT1788196494610", "RLT1788223477293",
           "RLT1788195295881", "RLT1788194386776"]


def find_case(rid, csvs):
    for name in csvs:
        for c in CaseIndex.load(name).cases:
            if c.reality_id == rid:
                return c
    return None


def truth(case) -> Sim2:
    M = decompose_tm(load_tm(case.rlt_dir), case.structure.design_tm)
    return Sim2(s=sim3_scale(M), theta=sim3_yaw(M),
                tx=float(M[0, 3]), ty=float(M[1, 3]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("realities", nargs="*", default=DEFAULT)
    ap.add_argument("--csv", nargs="+",
                    required=True)
    args = ap.parse_args()

    print(f"{'reality':<20}{'mode':>7}{'cands':>7}{'keys':>6}{'s':>7}"
          f"{'top err m':>11}{'closest m':>11}{'@rank':>7}"
          f"{'true inl':>10}{'best inl':>10}")
    for rid in args.realities:
        case = find_case(rid, args.csv)
        if case is None:
            print(f"{rid:<20} not in any CSV")
            continue
        p = RealityPipeline(case, rerank=False)
        keep = searchmod.TOP_K_TOTAL
        searchmod.TOP_K_TOTAL = 10 ** 9          # judge every proposal
        try:
            res = p.search()
        finally:
            searchmod.TOP_K_TOTAL = keep
        if res.error or not res.hypotheses:
            print(f"{rid:<20} {res.error}")
            continue

        gt, mpi = truth(case), res.metres_per_inch
        geom, feats = case.structure.geometry(), p.sheet()

        def err_m(sim):
            return float(np.hypot(sim.tx - gt.tx, sim.ty - gt.ty) * mpi)

        for mode in ("pose", "cloud"):
            up = getattr(res.hypotheses[0], "up_index", 0)
            poses = [h for h in res.hypotheses if getattr(h, "up_index", 0) == up]
            wall = p.view(up)["wall_xy"]
            t0 = time.perf_counter()
            sc = rerankmod.build_scorer(poses, wall, feats, geom, mpi,
                                        segment_mode=mode)
            sc.score_all(poses)
            dt = time.perf_counter() - t0
            scored = sorted(poses, key=lambda h: (-h.inliers, -h.score))
            d = [err_m(h.sim2) for h in scored]
            k = int(np.argmin(d))
            # The true pose is not in the candidate set; score it directly to
            # see whether the measure would have preferred it if offered.
            true_inl = sc.score(gt)[0]
            print(f"{rid:<20}{mode:>7}{len(poses):>7}{len(sc._cache):>6}{dt:7.1f}"
                  f"{d[0]:11.1f}{d[k]:11.1f}{k:>7}"
                  f"{true_inl:>10}{scored[0].inliers:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
