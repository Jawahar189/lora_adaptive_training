"""Select splits whose surveyed start point can be trusted, and write the index.

`geo_first_point.json` names a photograph. The distance from the point to the
nearest camera in a window around that photograph is how good the anchor is,
and that is what this selects on -- the same quantity the solver's prior relies
on, so the subset is defined by what actually matters rather than a proxy.

Two files:

  --out                       the selected rows, source row VERBATIM, written
                              beside the other dataset indexes so CaseIndex and
                              every --csv script work on it unchanged
  out/start_point_quality.csv the audit record for ALL splits, so the excluded
                              ones stay visible rather than vanishing

Both measure against ground truth, which is why this lives in scripts/ and not
in align/.
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.frames import target_to_paper
from align.groundtruth import gt_trajectory
from align.io.cases import CaseIndex
from align.scales import SheetScales

REPO = Path(__file__).resolve().parents[1]
DEFAULT_WINDOW = 20


def source_rows(sources):
    """Every dataset row, keyed by reality id, plus the superset header.

    The two CSVs do NOT share a column set -- the 105 has `localisation_status`
    and `localisation tpr@1m`, the 28 does not -- so the output uses the union
    and leaves those blank where the source had none. One column is named
    `" residue images, images_full"` and its name CONTAINS A COMMA, so this has
    to go through csv.DictReader; splitting on commas silently corrupts it.
    """
    rows, header = {}, []
    # Order follows the LAST source that defines a column, so the output header
    # matches the LAST source given rather than interleaving the others.
    for name in reversed(sources):
        path = REPO / name
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for col in reader.fieldnames or []:
                if col not in header:
                    header.append(col)
            if name == sources[-1]:
                header = list(reader.fieldnames or header)
            for row in reader:
                rid = (row.get("reality_id") or "").strip()
                if not rid:
                    continue          # the 105 carries one blank row
                rows[rid] = row
    return rows, header


def anchor_error(case, members, window):
    """Distance from the start point to the nearest camera in the window.

    The start point belongs to a CAPTURE. It names a photograph, and that
    photograph is often in a sibling split rather than the one
    `source_reality_id` claims -- 21 of 59 -- so every member is searched.
    Returns (host_reality_id, error_in_inches) or (None, nan).
    """
    geo = next((c.geo for c in members if c.geo), None)
    if geo is None:
        return None, float("nan")
    for split in members:
        traj = gt_trajectory(split.rlt_dir)
        if geo.image not in traj.names:
            continue
        order = np.argsort(traj.idx)
        paper = target_to_paper(traj.positions,
                                split.structure.design_tm)[:, :2][order]
        k = int(np.where(order == traj.names.index(geo.image))[0][0])
        lo = max(0, k - window)
        sub = paper[lo:k + window + 1]
        point = target_to_paper(np.array([[geo.x, geo.y, geo.z]]),
                                split.structure.design_tm)[0, :2]
        return split.reality_id, float(np.linalg.norm(sub - point, axis=1).min())
    return None, float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help="half-width, in images, of the window around the named "
                         "photograph (default 20)")
    ap.add_argument("--threshold-in", type=float, default=None,
                    help="keep splits under this anchor error in paper inches; "
                         "default is the median under the chosen window, "
                         "recomputed rather than hardcoded")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="the dataset CSVs to select from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-out", default="out/start_point_quality.csv")
    args = ap.parse_args()

    index = CaseIndex.load_dataset(args.csv)
    scales = SheetScales(index)
    rows, header = source_rows(args.csv)

    captures = collections.defaultdict(list)
    for case in index:
        captures[case.capture_id].append(case)

    measured = {}
    for cap, members in captures.items():
        measured[cap] = anchor_error(members[0], members, args.window)

    errs = np.array([e for _, e in measured.values() if np.isfinite(e)])
    threshold = args.threshold_in
    if threshold is None:
        threshold = float(np.median(errs))
    print(f"window +-{args.window} images   n={len(errs)} captures with a "
          f"locatable start point")
    print(f"anchor error: median {np.median(errs):.3f} in  "
          f"p90 {np.percentile(errs, 90):.3f}  max {errs.max():.3f}")
    print(f"threshold: {threshold:.3f} in\n")

    audit, selected = [], []
    for cap, members in sorted(captures.items()):
        host, err = measured[cap]
        keep = bool(np.isfinite(err) and err < threshold)
        for case in sorted(members, key=lambda c: c.reality_id):
            st = case.structure
            audit.append(dict(
                reality_id=case.reality_id, capture_id=cap,
                structure_id=st.structure_id, root=st.root,
                n_splits_in_capture=len(members),
                georeferenced=int(st.design_tm is not None),
                metres_per_inch=round(scales[st.key], 4),
                scale_source=scales.source(st.key),
                has_start_point=int(host is not None or any(c.geo for c in members)),
                anchor_split=host or "",
                is_anchor_split=int(host == case.reality_id),
                anchor_window=args.window,
                anchor_err_in=("" if not np.isfinite(err) else round(err, 4)),
                anchor_err_m=("" if not np.isfinite(err)
                              else round(err * scales[st.key], 3)),
                threshold_in=round(threshold, 4),
                selected=int(keep)))
            if keep and case.reality_id in rows:
                selected.append(rows[case.reality_id])

    Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.audit_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit[0]))
        w.writeheader(); w.writerows(audit)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, restval="")
        w.writeheader()
        for row in selected:
            w.writerow({k: row.get(k, "") for k in header})

    caps = len({r["capture_id"] for r in audit if r["selected"]})
    multi = len({r["capture_id"] for r in audit
                 if r["selected"] and r["n_splits_in_capture"] > 1})
    print(f"selected {len(selected)} splits across {caps} captures "
          f"({multi} multi-split)")
    print(f"  -> {args.out}")
    print(f"  -> {args.audit_out}   (all {len(audit)} splits, for audit)")


if __name__ == "__main__":
    main()
