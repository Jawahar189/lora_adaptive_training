#!/usr/bin/env python
"""Does COLMAP's geometric verification make the matcher's pose better?

`bevaligner.inference.verify` puts COLMAP's LO-RANSAC in front of the Sim2
fit, so correspondences no single planar motion can explain never reach it. That
is a plausible story, not a result. The flag ships off; this is what decides
whether it should.

The argument is NOT that verification throws matches away -- it always will.
It is whether the pose moves closer to truth, and in particular whether the
`(up_index, window)` the inlier count picks changes. A wrong window winning on a
crowd of mutually-inconsistent matches is the failure mode verification is aimed
at, so "the winner changed, and the new one is better" is the finding worth
having; "fewer matches, same pose" means it bought nothing.

Each mode re-runs the whole evaluation on the same cases with the same seed, so
the only difference between the three columns is the verifier.

    python scripts/e10_colmap_verification.py --ckpt out/.../best.ckpt --split val
    python scripts/e10_colmap_verification.py --ckpt ... --limit 40 --modes none affine

Reads `<stem>.jsonl`, not `<stem>.csv`: `align.eval.reality.COLUMNS` is a fixed
tuple written with `extrasaction="ignore"`, so `n_matches`, `n_verified`,
`n_inliers`, `up_index` and `window` reach the JSONL record only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODE_HELP = "verification modes to compare, in the order the table shows them"


def load_rows(stem: Path) -> dict:
    """reality_id -> the record, for the captures that produced a pose."""
    rows = {}
    with open(f"{stem}.jsonl") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("status") == "ok":
                rows[record["reality_id"]] = record
    return rows


def _metric(record, name):
    pose = (record.get("poses") or {}).get("predicted")
    value = None if pose is None else pose.get(name)
    return None if value is None else float(value)


def _median(values):
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else float("nan")


def _mean(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else float("nan")


def summarise(rows: dict, baseline: dict | None, common: set | None) -> dict:
    """One table column.

    Accuracy is reported over `common` -- the captures EVERY mode placed --
    because a mode that declines the hard ones would otherwise look accurate
    for having answered only the easy ones. The captures it declined are
    counted separately, scored by what the baseline made of them, which is the
    only way to see whether refusing to place them was right.
    """
    records = list(rows.values())
    shared = [r for rid, r in rows.items() if common is None or rid in common]
    moved = better = worse = 0
    declined, declined_good = 0, 0
    if baseline is not None:
        for rid, record in rows.items():
            other = baseline.get(rid)
            if other is None:
                continue
            if (record.get("up_index"), record.get("window")) != (
                    other.get("up_index"), other.get("window")):
                moved += 1
                mine, theirs = _metric(record, "rmse_xy"), _metric(other, "rmse_xy")
                if mine is not None and theirs is not None:
                    better += mine < theirs
                    worse += mine > theirs
        for rid, other in baseline.items():
            if rid in rows:
                continue
            declined += 1
            # A pose the baseline got within a metre of is one this mode gave up
            # on for nothing. These are the cost of verification, in captures.
            baseline_rmse = _metric(other, "rmse_xy")
            if baseline_rmse is not None and baseline_rmse <= 1.0:
                declined_good += 1
    return dict(
        placed=len(records), common=len(shared),
        tpr1=_mean([_metric(r, "tpr@1.0") for r in shared]),
        tpr15=_mean([_metric(r, "tpr@1.5") for r in shared]),
        rmse=_median([_metric(r, "rmse_xy") for r in shared]),
        matches=_median([r.get("n_matches") for r in records]),
        verified=_median([r.get("n_verified") for r in records]),
        inliers=_median([r.get("n_inliers") for r in records]),
        seconds=_median([r.get("seconds") for r in records]),
        moved=moved, moved_better=better, moved_worse=worse,
        declined=declined, declined_good=declined_good)


def table(summaries: dict[str, dict]) -> str:
    names = list(summaries)
    rows = [
        ("captures placed", "placed", "{:.0f}"),
        ("declined vs baseline", "declined", "{:.0f}"),
        ("  ... that were GOOD", "declined_good", "{:.0f}"),
        ("-- on the common set --", None, None),
        ("common captures", "common", "{:.0f}"),
        ("tpr@1.0m  (mean)", "tpr1", "{:.3f}"),
        ("tpr@1.5m  (mean)", "tpr15", "{:.3f}"),
        ("rmse_xy m (median)", "rmse", "{:.2f}"),
        ("-- over all placed --", None, None),
        ("matches   (median)", "matches", "{:.0f}"),
        ("verified  (median)", "verified", "{:.0f}"),
        ("inliers   (median)", "inliers", "{:.0f}"),
        ("seconds   (median)", "seconds", "{:.1f}"),
        ("winner moved", "moved", "{:.0f}"),
        ("  ... and got closer", "moved_better", "{:.0f}"),
        ("  ... and got further", "moved_worse", "{:.0f}"),
    ]
    width = max(len(label) for label, _, _ in rows) + 2
    out = [" " * width + "".join(f"{n:>14s}" for n in names),
           "-" * (width + 14 * len(names))]
    for label, key, fmt in rows:
        if key is None:
            out.append(label)
            continue
        cells = []
        for name in names:
            value = summaries[name].get(key)
            try:
                cells.append(f"{fmt.format(value):>14s}")
            except (TypeError, ValueError):
                cells.append(f"{'-':>14s}")
        out.append(f"{label:<{width}s}" + "".join(cells))
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="val",
                        help="a split name in the cache's splits.json, or 'all'")
    parser.add_argument("--modes", nargs="+", default=["none", "affine", "homography"],
                        help=MODE_HELP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--tolerance-m", type=float, default=0.5)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--whole-prior", action="store_true")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--out", default=None,
                        help="directory for the three runs; default <run>/eval/e10")
    parser.add_argument("--csv", nargs="+", default=None)
    parser.add_argument("--prior-csv", nargs="+", default=None)
    parser.add_argument("--reuse", action="store_true",
                        help="table the runs already in --out instead of re-running")
    args = parser.parse_args(argv)

    import os
    import torch
    from bevaligner.config import REPO_ROOT
    from bevaligner.data.datamodule import build_capture_store, load_paired_index
    from bevaligner.data.splits import load_splits
    from bevaligner.evaluate import evaluate
    from bevaligner.inference import HAVE_PYCOLMAP, MODES, PoseProposalOptions
    from bevaligner.lightning.module import BEVAlignerModule

    for mode in args.modes:
        if mode not in MODES:
            parser.error(f"--modes must come from {MODES}, not {mode!r}")
    if not HAVE_PYCOLMAP and set(args.modes) - {"none"}:
        parser.error("pycolmap is not installed: pip install 'pycolmap>=4.2,<5'")

    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "cache" / "hf"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = BEVAlignerModule.from_checkpoint(args.ckpt)
    config = module.config
    if args.csv:
        config.data.csv_files = list(args.csv)
    if args.prior_csv:
        config.data.prior_csv_files = list(args.prior_csv)
    index = load_paired_index(config)
    cases = list(index.current)
    store = build_capture_store(config)
    store.index = index
    if args.split != "all":
        splits = load_splits(store.paths.splits)
        named = sorted(k for k, v in splits.items() if isinstance(v, list))
        if args.split not in splits:
            parser.error(f"no split {args.split!r} in {store.paths.splits}; "
                         f"it has {named}")
        structures = set(splits[args.split])
        cases = [c for c in cases if c.structure.key in structures]
    if not cases:
        parser.error(f"no captures in split {args.split!r}")
    if args.limit:
        cases = cases[:args.limit]

    out_dir = Path(args.out) if args.out else config.run_dir / "eval" / "e10"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} captures x {len(args.modes)} modes -> {out_dir}")
    print(f"  checkpoint: {args.ckpt}   split: {args.split}")

    collected = {}
    for mode in args.modes:
        stem = out_dir / f"{Path(args.ckpt).stem}_{args.split}_{mode}"
        if args.reuse and Path(f"{stem}.jsonl").exists():
            print(f"\n== {mode}: reusing {stem}.jsonl")
        else:
            print(f"\n== {mode} ==")
            started = time.time()
            options = PoseProposalOptions(
                resolution=args.resolution or config.data.eval_resolution,
                num_queries=args.queries, tolerance_m=args.tolerance_m,
                min_inliers=args.min_inliers, refine_on_footprint=not args.no_refine,
                refine_footprint_margin=config.data.refine_footprint_margin,
                geometric_verification=mode)
            evaluate(module, cases, stem, store, index, options, device,
                     stride=args.stride, whole_prior=args.whole_prior)
            print(f"   {time.time() - started:.0f}s")
        collected[mode] = load_rows(stem)

    common = set.intersection(*(set(r) for r in collected.values())) if collected else set()
    baseline = collected[args.modes[0]]
    summaries = {m: summarise(r, baseline, common) for m, r in collected.items()}

    print("\n" + table(summaries))
    print(f"\nbaseline: {args.modes[0]};  common set: {len(common)} captures")
    print("A mode earns the default by holding tpr@1.0 on the COMMON set while")
    print("declining few GOOD captures. Cutting matches is not the result -- it")
    print("always cuts matches -- and accuracy over a mode's own survivors is not")
    print("the result either, since declining the hard ones flatters it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
