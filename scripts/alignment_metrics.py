#!/usr/bin/env python
"""Per-reality alignment metrics, flattened to a CSV.

Reads the predictions and metrics `scripts/solve.py` writes. The summary
statistics come straight from the metrics file; the three `tpr` columns are
recomputed from the per-image distances, which the summary does not keep -- and
recomputing them is the point, because they are the same fraction reached
through three different unit conversions and must agree.

The point of the CSV is `tpr@1m` -- the fraction of a reality's photographs
placed within a metre of where they belong. That threshold is only directly
meaningful where the target frame is metres, which is 61 of 132 splits. For the
other 71 the frame is paper inches and a metre has to come through the sheet
scale, so every row carries the threshold in all three units and says where the
scale came from:

    one_m_in_inches = 1 / metres_per_inch
    one_m_in_px     = ppi / metres_per_inch

and `scale_source` is `design_tm` (exact), `e1` (pooled from the camera-height
anchor) or `fallback` (a guess -- treat `tpr_1m` on those rows as a guess too).

`tpr_1m`, `tpr_1m_inches` and `tpr_1m_px` are the SAME fraction reached through
three different unit conversions. They should agree to rounding on every row;
where they do not, the frame algebra has a bug, not the split.

    python scripts/alignment_metrics.py --metrics out/solve_metrics.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.eval.metrics import image_errors
from align.groundtruth import gt_trajectory, load_tm
from align.io.cases import CaseIndex
from align.pipeline import RealityPipeline
from align.scales import SheetScales

COLUMNS = [
    "reality_id", "structure_id", "capture_id", "root",
    "chain_how", "seeded_from", "gate_passed",
    "metres_per_inch", "scale_source", "ppi",
    "one_m_in_inches", "one_m_in_px",
    "tpr_1m", "tpr_1m_inches", "tpr_1m_px",
    "median_err_m", "p90_err_m",
    "median_err_in", "p90_err_in",
    "median_err_px", "p90_err_px",
    "yaw_err_deg", "yaw_err_mod90_deg", "scale_ratio",
    "n_images", "closest_rank", "closest_pos_m", "georeferenced", "status",
]


def three_unit_tpr(case, pred_tm, one_m_in_inches, one_m_in_px):
    """tpr@1m in metres, inches and pixels, from the same per-image distances.

    `ErrorStats.within` is keyed by the thresholds it was built with, which are
    not the 1 m equivalents, so this goes back to the distances. Doing so is
    what makes the three columns a check rather than three restatements of one
    number: they are computed independently and must land on the same fraction.
    """
    p = RealityPipeline(case)
    d_in, d_px, d_m = image_errors(
        pred_tm, load_tm(case.rlt_dir), gt_trajectory(case.rlt_dir).positions,
        p.trajectory().positions, case.structure.design_tm,
        case.structure.geometry())[:3]
    return (float((d_m <= 1.0).mean()) if d_m is not None else None,
            float((d_in <= one_m_in_inches).mean()),
            float((d_px <= one_m_in_px).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="out/solve_metrics.jsonl")
    ap.add_argument("--predictions", default=None,
                    help="defaults to predictions.jsonl beside --metrics")
    ap.add_argument("--out", default="out/alignment_metrics.csv")
    ap.add_argument("--csv", nargs="+",
                    required=True)
    args = ap.parse_args()

    cases, seen = [], set()
    for name in args.csv:
        if not Path(name).exists():
            continue
        for c in CaseIndex.load(name).cases:
            if c.reality_id not in seen:
                seen.add(c.reality_id)
                cases.append(c)
    by_id = {c.reality_id: c for c in cases}
    scales = SheetScales(cases)      # SheetScales iterates cases, not structures

    pred_path = Path(args.predictions or
                     Path(args.metrics).with_name("predictions.jsonl"))
    preds = {}
    if pred_path.exists():
        with open(pred_path) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    preds[r["reality_id"]] = np.asarray(r["tm"]).reshape(4, 4)
    else:
        print(f"no predictions at {pred_path}; tpr columns will come from the "
              f"metrics file alone and cannot be cross-checked")

    rows, disagree = [], []
    with open(args.metrics) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            rid = m["reality_id"]
            case = by_id.get(rid)
            chain = m.get("chain") or {}
            skey = m.get("structure", "")
            mpi = m.get("metres_per_inch")
            if mpi is None or not np.isfinite(mpi or float("nan")):
                mpi = scales.get(skey, float("nan"))
            ppi = m.get("ppi", float("nan"))

            one_m_in = 1.0 / mpi if mpi and np.isfinite(mpi) and mpi > 0 else float("nan")
            one_m_px = one_m_in * ppi if np.isfinite(one_m_in) else float("nan")

            utm, paper, pixel = m.get("utm"), m.get("paper"), m.get("pixel")
            tpr_m, tpr_in, tpr_px = m.get("tpr_1m"), None, None
            pred = preds.get(rid)
            if case is not None and pred is not None and np.isfinite(one_m_in):
                try:
                    tpr_m, tpr_in, tpr_px = three_unit_tpr(
                        case, pred, one_m_in, one_m_px)
                except Exception as exc:                          # noqa: BLE001
                    print(f"  {rid}: could not recompute tpr ({exc!r})"[:110])
            got = [t for t in (tpr_m, tpr_in, tpr_px) if t is not None]
            if len(got) > 1 and max(got) - min(got) > 0.02:
                disagree.append((rid, tpr_m, tpr_in, tpr_px))

            med_m = (utm or {}).get("median")
            med_in = (paper or {}).get("median")
            med_px = (pixel or {}).get("median")

            rows.append({
                "reality_id": rid,
                "structure_id": skey,
                "capture_id": getattr(case, "capture_id", ""),
                "root": skey.split("/")[0],   # "data" / "data2"
                "chain_how": chain.get("how", ""),
                "seeded_from": chain.get("seeded_from") or "",
                "gate_passed": chain.get("gate_passed", ""),
                "metres_per_inch": mpi,
                "scale_source": scales.source(skey),
                "ppi": ppi,
                "one_m_in_inches": one_m_in,
                "one_m_in_px": one_m_px,
                "tpr_1m": tpr_m,
                "tpr_1m_inches": tpr_in,
                "tpr_1m_px": tpr_px,
                "median_err_m": med_m,
                "p90_err_m": (utm or {}).get("p90"),
                "median_err_in": med_in,
                "p90_err_in": (paper or {}).get("p90"),
                "median_err_px": med_px,
                "p90_err_px": (pixel or {}).get("p90"),
                "yaw_err_deg": m.get("yaw_err_deg"),
                "yaw_err_mod90_deg": m.get("yaw_err_mod90_deg"),
                "scale_ratio": (float(np.exp(m["scale_log_err"]))
                                if m.get("scale_log_err") is not None else None),
                "n_images": m.get("n_images"),
                "closest_rank": m.get("closest_rank"),
                "closest_pos_m": m.get("closest_pos_m"),
                "georeferenced": m.get("georeferenced"),
                "status": m.get("status", ""),
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} realities -> {args.out}")
    src = {}
    for r in rows:
        src[r["scale_source"]] = src.get(r["scale_source"], 0) + 1
    print("scale source: " + "  ".join(f"{k}={v}" for k, v in sorted(src.items())))
    got = [r["tpr_1m"] for r in rows if r["tpr_1m"] is not None]
    if got:
        print(f"tpr@1m over {len(got)} georeferenced: "
              f"median {np.median(got):.3f}  mean {np.mean(got):.3f}  "
              f"{sum(1 for t in got if t >= 0.5)} at or above 0.5")
    if disagree:
        print(f"\n{len(disagree)} rows where tpr@1m differs between units by "
              f"more than 0.02 -- a frame bug, not a property of the split:")
        for rid, a, b, c in disagree[:10]:
            print(f"  {rid}  m={a}  in={b}  px={c}")
    else:
        print("tpr@1m agrees across metres, inches and pixels on every row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
