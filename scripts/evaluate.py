"""Image-to-image error for a set of predicted transforms.

  python scripts/evaluate.py --pred out/predictions.jsonl

`--pred` is JSONL, one object per split: {"reality_id": "RLT...", "tm": [16 floats]}.
Splits absent from the file are reported as abstained.

Error is reported in floormap PIXELS for every split, and additionally in UTM
METRES for structures carrying a real design transform. Where design_tm is
absent or identity the target frame is the paper itself, so there are no metres
to report and the pixel column stands alone.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.eval.metrics import evaluate
from align.frames import is_identity
from align.groundtruth import gt_trajectory, load_tm
from align.io.shots import load_shots
from align.io.cases import CaseIndex


def load_predictions(path):
    preds = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("tm") is None:
            continue
        preds[r["reality_id"]] = np.asarray(r["tm"], dtype=np.float64).reshape(4, 4)
    return preds


def run(index, preds):
    rows, abstained = [], []
    for case in index:
        tm = preds.get(case.reality_id)
        if tm is None:
            abstained.append(case)
            continue
        st = case.structure
        rows.append(evaluate(tm, load_tm(case.rlt_dir),
                             gt_trajectory(case.rlt_dir).positions,
                             load_shots(case.rlt_dir).positions,
                             st.design_tm, st.geometry(),
                             case.reality_id, st.key, st.status))
    return rows, abstained


def report(rows, abstained, out_path=None):
    print(f"{'reality_id':22s} {'structure':26s} {'ppi':>5s} "
          f"{'px med':>8s} {'px p90':>8s} {'<=25px':>7s} "
          f"{'m med':>7s} {'m p90':>7s} {'tpr@1m':>7s} {'yaw':>6s}")
    for m in rows:
        u = m.utm
        print(f"{m.reality_id:22s} {m.structure:26s} {m.ppi:5.0f} "
              f"{m.pixel.median:8.1f} {m.pixel.p90:8.1f} "
              f"{m.pixel.within['25.0']:7.2f} "
              f"{u.median if u else float('nan'):7.2f} "
              f"{u.p90 if u else float('nan'):7.2f} "
              f"{m.tpr_1m if m.tpr_1m is not None else float('nan'):7.2f} "
              f"{m.yaw_err_deg:6.1f}"
              .replace("nan", "  — "))

    geo = [m for m in rows if m.georeferenced]
    pap = [m for m in rows if not m.georeferenced]
    print("\n" + "=" * 78)
    print(f"IMAGE-TO-IMAGE ERROR   ({len(rows)} evaluated, {len(abstained)} abstained)")
    if rows:
        px = np.array([m.pixel.median for m in rows])
        print(f"\n  pixels (all {len(rows)} splits, each at its own sheet ppi)")
        print(f"    median-of-medians {np.median(px):8.1f} px   "
              f"p90 {np.percentile(px, 90):8.1f} px")
        for t in ("5.0", "10.0", "25.0", "50.0", "100.0"):
            frac = np.mean([m.pixel.within[t] for m in rows])
            print(f"    images within {float(t):6.0f} px : {frac:.3f}")
    if geo:
        md = np.array([m.utm.median for m in geo])
        print(f"\n  UTM metres ({len(geo)} georeferenced splits)")
        print(f"    median-of-medians {np.median(md):8.2f} m    "
              f"p90 {np.percentile(md, 90):8.2f} m")
        print(f"    macro tpr@1m      {np.mean([m.tpr_1m for m in geo]):8.3f}")
    if pap:
        print(f"\n  {len(pap)} splits have no georeference (design_tm absent or "
              f"identity) — pixel error only, by design")
    print("=" * 78)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            for m in rows:
                fh.write(json.dumps(m.as_dict()) + "\n")
        print(f"per-split metrics -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="JSONL of predicted transforms")
    ap.add_argument("--csv", default=None, help="restrict to one dataset CSV")
    ap.add_argument("--root", default=None,
                    help="restrict to one data root, e.g. data or data_v")
    ap.add_argument("--out", default="out/metrics.jsonl")
    args = ap.parse_args()

    index = CaseIndex.load(args.csv) if args.csv else CaseIndex.load_all()
    if args.root:
        index = CaseIndex([c for c in index if c.structure.root == args.root])
        if not len(index):
            raise SystemExit(f"no cases under root {args.root!r}")
    rows, abstained = run(index, load_predictions(args.pred))
    report(rows, abstained, args.out)


if __name__ == "__main__":
    main()
