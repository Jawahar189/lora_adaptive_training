"""Gather the eval runs of one model into a single results CSV.

`bevaligner.evaluate` writes one `<stem>.csv` per split through
`align.eval.reality.ResultWriter`, in the same columns every other method in
this repo writes. This puts them side by side, one row per capture with the
split it came from, and prints the per-split summary that goes in the write-up.

Two things it refuses to hide.

Rows that did not solve are KEPT, with `status` saying why, because a recall
quoted over the rows that happened to solve is not a recall. `solved_fraction`
is reported separately for the same reason.

Metres are split on `scale_source`. Where it is `design_tm` the scale is that
transform's own and the metre figures are exact; on the rest it rests on a
rig-height estimate that measured about 3x off on one probe pair. Pooling the
two and quoting one median would be quoting a number that means two things.

    python scripts/collect_results.py \\
        --eval out/bevaligner/eval/prior_bev_train.csv=train \\
               out/bevaligner/eval/prior_bev_val.csv=val \\
               out/bevaligner/eval/prior_bev_test.csv=test \\
        --out out/bevaligner/eval/prior_bev_results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# the recalls this repo reports; tpr@N is the share of a capture's cameras
# placed within N metres, averaged over captures
THRESHOLDS = ("tpr@0.5", "tpr@1.0", "tpr@2.0")


def summarise(frame: pd.DataFrame, label: str) -> dict:
    solved = frame[frame.status == "ok"] if "status" in frame else frame
    row = {"split": label, "captures": len(frame), "solved": len(solved),
           "solved_fraction": len(solved) / max(len(frame), 1)}
    for column in THRESHOLDS:
        if column in frame:
            # a capture that did not solve scores 0, not missing
            scores = frame[column].fillna(0.0) if "status" in frame else frame[column]
            row[column + "_mean"] = float(scores.mean())
    if "rmse_xy" in solved and len(solved):
        row["rmse_xy_median_m"] = float(solved.rmse_xy.median())
    if "scale_source" in solved and len(solved):
        exact = solved[solved.scale_source == "design_tm"]
        row["exact_scale_fraction"] = len(exact) / len(solved)
        if len(exact):
            row["rmse_xy_median_m_design_tm"] = float(exact.rmse_xy.median())
            for column in THRESHOLDS:
                if column in exact:
                    row[column + "_mean_design_tm"] = float(exact[column].fillna(0.0).mean())
    for column in ("n_inliers", "overlap", "prior_splits"):
        if column in solved and len(solved):
            row[column + "_median"] = float(pd.to_numeric(
                solved[column], errors="coerce").median())
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", nargs="+", required=True, metavar="CSV=SPLIT",
                    help="an evaluate.py output and the split it scored")
    ap.add_argument("--out", required=True, help="the combined per-capture CSV")
    ap.add_argument("--summary", default=None,
                    help="the per-split table; default: <out> with _summary")
    args = ap.parse_args(argv)

    frames, summaries = [], []
    for item in args.eval:
        path, _, label = item.partition("=")
        path = Path(path)
        if not path.exists():
            print(f"  missing, skipped: {path}")
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "split", label or path.stem)
        frames.append(frame)
        summaries.append(summarise(frame, label or path.stem))
    if not frames:
        raise SystemExit("no eval CSVs found")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out, index=False)

    summary = pd.DataFrame(summaries)
    summary_path = Path(args.summary) if args.summary else \
        out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n{len(combined)} captures -> {out}")
    print(f"per-split summary   -> {summary_path}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.round(3).to_string(index=False))
    if "exact_scale_fraction" in summary:
        print("\nRead the *_design_tm columns as the metre figures. On the rest the"
              "\nscale is a rig-height estimate, not a measurement.")


if __name__ == "__main__":
    main()
