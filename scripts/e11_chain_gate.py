"""Fit the chain's confidence gate against an eval that has already been scored.

    python scripts/e11_chain_gate.py --eval out/bevaligner/eval/test_viz

Reads `<stem>.jsonl` after the fact -- no model, no GPU, seconds to run -- and
asks the only two questions that decide whether chaining helps:

  * of the captures holding a correctly-placed split, how many does a given
    threshold FIND one in? (seed recall)
  * of the seeds it finds, how many are actually correct? (seed precision)

Precision is the one to optimise. A missing seed costs the recovery ladder, one
extra unconstrained search; a WRONG seed is propagated into every split after
it, so it costs the capture. `scripts/e9_window_oracle.py` is the model for
reading an eval JSONL post hoc like this.

A caveat that matters for reading the output. The margin is computed over
materially different candidate poses, which needs the `centre_x`/`centre_y` that
`inference/propose.py` now records on every candidate. An eval written before
that reports `margin = inf` everywhere, and the margin column is then not a
threshold but a constant -- the script says so rather than quietly fitting it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevaligner.inference.gate import GateThresholds       # noqa: E402

GOOD = 0.5          # a reality is placed when more than half its images land <1 m


def load(stem: Path) -> list[dict]:
    """Scored rows, with the metrics lifted out of `poses.predicted`.

    `ResultWriter` nests them there and flattens them only into the CSV, which
    does not carry `candidates`. The JSONL has both, so it is read instead and
    flattened here -- the same shape `ResultWriter.rows` has in memory.
    """
    path = stem if stem.suffix == ".jsonl" else Path(f"{stem}.jsonl")
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        predicted = (row.get("poses") or {}).get("predicted")
        if row.get("status") != "ok" or not predicted:
            continue
        metrics = {k: v for k, v in predicted.items() if k != "tm"}
        if metrics.get("tpr@1.0") is None:
            continue
        out.append(dict({k: v for k, v in row.items() if k != "poses"}, **metrics))
    return out


def margin_of(row: dict, thresholds: GateThresholds, metres_per_inch: float) -> float:
    """The winner's fractional lead over the best materially different candidate."""
    candidates = [c for c in row.get("candidates") or []
                  if c.get("centre_x") is not None and not c.get("refine")]
    winner = next((c for c in candidates
                   if c.get("up_index") == row.get("up_index")
                   and c.get("window_index") == row.get("window")), None)
    if winner is None:
        return float("inf")
    centre = np.array([winner["centre_x"], winner["centre_y"]], float)
    yaw = float(winner.get("yaw_deg", 0.0))
    best = float(winner.get("num_inliers", row.get("n_inliers", 0)))
    rival = 0.0
    for candidate in candidates:
        if candidate is winner:
            continue
        other = np.array([candidate["centre_x"], candidate["centre_y"]], float)
        far = np.linalg.norm(other - centre) * metres_per_inch > thresholds.distinct_m
        turned = abs(((float(candidate.get("yaw_deg", 0.0)) - yaw + 180) % 360)
                     - 180) > thresholds.distinct_deg
        if far or turned:
            rival = max(rival, float(candidate["num_inliers"]))
    if rival == 0.0:
        return float("inf")
    return (best - rival) / max(best, 1.0)


def features(rows: list[dict], thresholds: GateThresholds) -> list[dict]:
    out = []
    for row in rows:
        mpi = float(row.get("metres_per_inch") or 1.0)
        out.append(dict(
            capture=row.get("capture_id") or "", reality=row.get("reality_id"),
            good=bool(row["tpr@1.0"] > GOOD),
            inlier_frac=float(row.get("inlier_frac") or 0.0),
            n_inliers=int(row.get("n_inliers") or 0),
            refined=bool(row.get("refined")),
            rig=float(row.get("rig_height_m") or float("nan")),
            margin=margin_of(row, thresholds, mpi)))
    return out


def keeps(item: dict, thresholds: GateThresholds, require_refined: bool) -> bool:
    low, high = thresholds.rig_plausible
    return (item["inlier_frac"] >= thresholds.min_inlier_fraction
            and item["n_inliers"] >= thresholds.min_inliers
            and item["margin"] >= thresholds.min_margin
            and np.isfinite(item["rig"]) and low <= item["rig"] <= high
            and (item["refined"] or not require_refined))


def report(items: list[dict], thresholds: GateThresholds,
           require_refined: bool) -> dict:
    """Row-level precision/recall, and the capture-level numbers that matter."""
    kept = [i for i in items if keeps(i, thresholds, require_refined)]
    by_capture = defaultdict(list)
    for item in items:
        by_capture[item["capture"]].append(item)
    multi = {c: v for c, v in by_capture.items() if len(v) > 1}
    placeable = {c for c, v in multi.items() if any(i["good"] for i in v)}
    seeded, correct = set(), 0
    for capture, members in multi.items():
        gated = [i for i in members if keeps(i, thresholds, require_refined)]
        if not gated:
            continue
        seeded.add(capture)
        # The seed is the strongest-supported gated split, which is what the
        # chain's own seeding picks.
        if max(gated, key=lambda i: i["n_inliers"])["good"]:
            correct += 1
    return dict(
        kept=len(kept),
        precision=(sum(i["good"] for i in kept) / len(kept)) if kept else 0.0,
        recall=(sum(i["good"] for i in kept)
                / max(sum(i["good"] for i in items), 1)),
        captures_seeded=len(seeded), captures_placeable=len(placeable),
        seed_precision=correct / max(len(seeded), 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, type=Path,
                        help="an eval stem or its .jsonl")
    parser.add_argument("--top", type=int, default=15,
                        help="how many of the best thresholds to print")
    args = parser.parse_args(argv)

    rows = load(args.eval)
    base = GateThresholds()
    items = features(rows, base)
    good = sum(i["good"] for i in items)
    print(f"{len(items)} scored rows, {good} good (tpr@1.0 > {GOOD}), "
          f"base rate {good / max(len(items), 1):.3f}")
    # Two different reasons the margin can be inf everywhere, and they call for
    # opposite responses: an eval written before `centre_x` existed has to be
    # re-run, whereas an eval where no candidate was materially different is
    # telling you the search only ever found one answer -- which is information,
    # not a gap.
    with_pose = sum(1 for row in rows
                    if any(c.get("centre_x") is not None
                           for c in row.get("candidates") or []))
    finite = [i["margin"] for i in items if np.isfinite(i["margin"])]
    if with_pose == 0:
        print("  NOTE: no candidate carries a pose, so every margin is inf -- this "
              "eval predates `centre_x` on the candidates. Re-run an eval before "
              "reading the margin column.")
    elif not finite:
        print(f"  NOTE: {with_pose}/{len(rows)} rows carry candidate poses, but no "
              "candidate anywhere was materially different from its winner, so "
              f"every margin is inf. Nothing to fit at this sample size.")
    else:
        print(f"  margin is real on {len(finite)}/{len(items)} rows "
              f"(median {np.median(finite):.3f})")

    results = []
    for fraction in (0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        for inliers in (0, 30, 60, 120, 200):
            for margin in (0.0, 0.02, 0.05, 0.1, 0.2):
                for refined in (False, True):
                    thresholds = GateThresholds(min_inlier_fraction=fraction,
                                                min_inliers=inliers,
                                                min_margin=margin)
                    summary = report(items, thresholds, refined)
                    if summary["captures_seeded"] < 5:
                        continue
                    results.append(((fraction, inliers, margin, refined), summary))
    results.sort(key=lambda r: (-r[1]["seed_precision"], -r[1]["captures_seeded"]))
    print(f"\n  seeding over the {report(items, base, True)['captures_placeable']} "
          f"multi-split captures that contain a correct split:")
    print("   frac  inl  margin  refined |  seeded  seed prec |  rows  prec  recall")
    for (fraction, inliers, margin, refined), summary in results[:args.top]:
        print(f"   {fraction:<5.2f} {inliers:<4d} {margin:<7.2f} {str(refined):<7} |"
              f"  {summary['captures_seeded']:<6d}  {summary['seed_precision']:<9.3f} |"
              f"  {summary['kept']:<5d} {summary['precision']:.3f} "
              f"{summary['recall']:.3f}")

    current = report(items, base, base.require_refined)
    print(f"\n  the defaults in gate.py ({base.min_inlier_fraction}, "
          f"{base.min_inliers}, {base.min_margin}, refined): "
          f"{current['captures_seeded']} captures seeded at "
          f"{current['seed_precision']:.3f}, {current['kept']} rows at "
          f"precision {current['precision']:.3f} recall {current['recall']:.3f}")


if __name__ == "__main__":
    main()
