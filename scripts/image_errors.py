#!/usr/bin/env python
"""Per-image alignment error: how far each photograph landed from where it belongs.

One row per IMAGE, not per reality. This is the metric -- everything else is a
summary of it.

For each image the solved `tm` places its camera, ground truth says where it
really was, and the distance between them is reported in all three frames:

  err_m    metres. Only where the structure carries a real design transform;
           blank for the 71 splits whose target frame is paper inches, because
           a metre there would be a number in invented units.
  err_in   paper inches -- the frame the solver actually works in.
  err_px   floormap pixels -- what a reviewer sees when the capture is plotted.
           ppi runs 87.9..600 across the structures, so this is not derivable
           from the inch error without the per-sheet scale.

`within_1m` uses metres where they exist and otherwise the sheet scale
(1 m = 1/metres_per_inch inches); `scale_source` says whether that scale is
exact (`design_tm`), pooled from the camera-height anchor (`e1`), or a guess
(`fallback`), in which case read the flag as a guess too.

    python scripts/image_errors.py --predictions out/selected/predictions.jsonl \
        --csv SELECTED.csv --out out/image_errors.csv

Writes a companion per-reality roll-up beside it unless --no-summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.frames import is_identity, target_to_paper
from align.groundtruth import gt_trajectory, load_tm
from align.io.cases import CaseIndex
from align.pipeline import RealityPipeline
from align.scales import SheetScales

COLUMNS = [
    "reality_id", "structure_id", "capture_id", "root",
    "image", "image_index",
    "err_m", "err_in", "err_px", "within_1m",
    "pred_paper_x", "pred_paper_y", "gt_paper_x", "gt_paper_y",
    "pred_px_x", "pred_px_y", "gt_px_x", "gt_px_y",
    "metres_per_inch", "scale_source", "ppi", "one_m_in_inches", "one_m_in_px",
    "chain_how", "gate_passed",
]

SUMMARY_COLUMNS = [
    "reality_id", "structure_id", "capture_id", "root",
    "chain_how", "gate_passed",
    "n_images", "tpr_1m", "tpr_2m", "tpr_5m",
    "median_err_m", "p90_err_m", "max_err_m",
    "median_err_in", "p90_err_in",
    "median_err_px", "p90_err_px",
    "metres_per_inch", "scale_source", "ppi", "one_m_in_inches", "one_m_in_px",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="out/selected/predictions.jsonl")
    ap.add_argument("--metrics", default=None,
                    help="solve_metrics.jsonl, for the chain columns; "
                         "defaults to one beside --predictions")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="out/image_errors.csv")
    ap.add_argument("--summary-out", default=None,
                    help="per-reality roll-up; defaults to <out>_by_reality.csv")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    cases, seen = [], []
    for name in args.csv:
        for c in CaseIndex.load(name).cases:
            if c.reality_id not in seen:
                seen.append(c.reality_id)
                cases.append(c)
    by_id = {c.reality_id: c for c in cases}
    scales = SheetScales(cases)

    preds = {}
    with open(args.predictions) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                preds[r["reality_id"]] = np.asarray(r["tm"], float).reshape(4, 4)

    chain = {}
    mpath = Path(args.metrics or
                 Path(args.predictions).with_name("solve_metrics.jsonl"))
    if mpath.exists():
        with open(mpath) as fh:
            for line in fh:
                if line.strip():
                    m = json.loads(line)
                    chain[m["reality_id"]] = m.get("chain") or {}

    rows, summary, skipped = [], [], []
    for rid, tm in preds.items():
        case = by_id.get(rid)
        if case is None:
            skipped.append((rid, "not in the CSV"))
            continue
        try:
            traj = gt_trajectory(case.rlt_dir)
            sfm = np.asarray(RealityPipeline(case).trajectory().positions, float)
        except Exception as exc:                                  # noqa: BLE001
            skipped.append((rid, repr(exc)[:60]))
            continue
        if len(sfm) != len(traj.names):
            skipped.append((rid, f"{len(sfm)} sfm poses vs {len(traj.names)} images"))
            continue

        design_tm = case.structure.design_tm
        geom = case.structure.geometry()
        georef = not is_identity(design_tm)
        mpi = scales.get(case.structure.key, float("nan"))
        one_m_in = 1.0 / mpi if mpi and np.isfinite(mpi) and mpi > 0 else float("nan")
        one_m_px = one_m_in * geom.ppi if np.isfinite(one_m_in) else float("nan")

        # Where the solved transform puts each camera, against where it was.
        pred_pos = np.asarray(sfm) @ tm[:3, :3].T + tm[:3, 3]
        gt_pos = np.asarray(traj.positions, float)

        pred_paper = target_to_paper(pred_pos, design_tm)[:, :2]
        gt_paper = target_to_paper(gt_pos, design_tm)[:, :2]
        pred_px = geom.paper_to_pixel(pred_paper)
        gt_px = geom.paper_to_pixel(gt_paper)

        d_in = np.linalg.norm(pred_paper - gt_paper, axis=1)
        d_px = np.linalg.norm(pred_px - gt_px, axis=1)
        d_m = np.linalg.norm(pred_pos - gt_pos, axis=1) if georef else None
        # A metre is a metre where the frame is metres; elsewhere it has to come
        # through the sheet scale, which is what `scale_source` qualifies.
        within = (d_m <= 1.0) if d_m is not None else (d_in <= one_m_in)

        ch = chain.get(rid, {})
        for i, name in enumerate(traj.names):
            rows.append({
                "reality_id": rid,
                "structure_id": case.structure.key,
                "capture_id": case.capture_id,
                "root": case.structure.key.split("/")[0],
                "image": name,
                "image_index": int(traj.idx[i]),
                "err_m": float(d_m[i]) if d_m is not None else "",
                "err_in": float(d_in[i]),
                "err_px": float(d_px[i]),
                "within_1m": int(bool(within[i])),
                "pred_paper_x": float(pred_paper[i, 0]),
                "pred_paper_y": float(pred_paper[i, 1]),
                "gt_paper_x": float(gt_paper[i, 0]),
                "gt_paper_y": float(gt_paper[i, 1]),
                "pred_px_x": float(pred_px[i, 0]),
                "pred_px_y": float(pred_px[i, 1]),
                "gt_px_x": float(gt_px[i, 0]),
                "gt_px_y": float(gt_px[i, 1]),
                "metres_per_inch": mpi,
                "scale_source": scales.source(case.structure.key),
                "ppi": geom.ppi,
                "one_m_in_inches": one_m_in,
                "one_m_in_px": one_m_px,
                "chain_how": ch.get("how", ""),
                "gate_passed": ch.get("gate_passed", ""),
            })

        d_thresh = d_m if d_m is not None else d_in * mpi
        summary.append({
            "reality_id": rid,
            "structure_id": case.structure.key,
            "capture_id": case.capture_id,
            "root": case.structure.key.split("/")[0],
            "chain_how": ch.get("how", ""),
            "gate_passed": ch.get("gate_passed", ""),
            "n_images": len(traj.names),
            "tpr_1m": float((d_thresh <= 1.0).mean()),
            "tpr_2m": float((d_thresh <= 2.0).mean()),
            "tpr_5m": float((d_thresh <= 5.0).mean()),
            "median_err_m": float(np.median(d_m)) if d_m is not None else "",
            "p90_err_m": float(np.percentile(d_m, 90)) if d_m is not None else "",
            "max_err_m": float(d_m.max()) if d_m is not None else "",
            "median_err_in": float(np.median(d_in)),
            "p90_err_in": float(np.percentile(d_in, 90)),
            "median_err_px": float(np.median(d_px)),
            "p90_err_px": float(np.percentile(d_px, 90)),
            "metres_per_inch": mpi,
            "scale_source": scales.source(case.structure.key),
            "ppi": geom.ppi,
            "one_m_in_inches": one_m_in,
            "one_m_in_px": one_m_px,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images across {len(summary)} realities -> {args.out}")

    if not args.no_summary and summary:
        spath = args.summary_out or str(
            Path(args.out).with_name(Path(args.out).stem + "_by_reality.csv"))
        with open(spath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(summary)
        print(f"{len(summary)} realities -> {spath}")

    if rows:
        tpr = np.array([s["tpr_1m"] for s in summary])
        med_m = [s["median_err_m"] for s in summary if s["median_err_m"] != ""]
        print(f"\ntpr@1m per reality: median {np.median(tpr):.3f}  "
              f"{int((tpr >= 0.5).sum())}/{len(tpr)} at 0.5 or better  "
              f"{int((tpr >= 0.9).sum())} at 0.9 or better")
        if med_m:
            print(f"median per-image error, over {len(med_m)} georeferenced "
                  f"realities: median {np.median(med_m):.1f} m  "
                  f"best {min(med_m):.2f} m  worst {max(med_m):.0f} m")
        allpx = np.array([r["err_px"] for r in rows])
        print(f"all {len(allpx)} images: median {np.median(allpx):.0f} px  "
              f"p90 {np.percentile(allpx, 90):.0f} px")
    for rid, why in skipped:
        print(f"  skipped {rid}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
