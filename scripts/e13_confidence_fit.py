"""Re-derive the confidence coefficients from an eval that has already been scored.

    python scripts/e13_confidence_fit.py --eval out/bevaligner/chain/a_base \
        --check out/bevaligner/chain/b_chain

Reads `<stem>.jsonl` after the fact -- no model, no GPU, seconds to run -- and
answers the only question that makes the number in
`bevaligner/inference/confidence.py` worth printing: does a stated confidence
of 0.8 mean eight realities in ten within 2 m? `scripts/e11_chain_gate.py` is
the same exercise for the gate's thresholds, and its row reader, margin and
feature extraction are imported rather than copied -- a second definition of the
margin is how `confidence` and `gate_margin` would come to disagree inside one
row.

The fit is scored by five-fold cross-validation GROUPED BY CAPTURE. A per-row
split leaks: the splits of one capture walk the same floor against the same
prior, so a row's siblings predict it even when the signals do not. The grouped
number is two to three points lower than the in-sample one and is the one worth
quoting.

`--check` applies the fitted coefficients to a second run untouched by the fit.
That is the only test that matters for a score meant to travel: a model carrying
the ABSOLUTE inlier count scores as well as this one in-family and then predicts
a mean of 0.03 on a run whose matcher proposes fewer matches, against a base
rate of 0.39. Scale-free signals are not a stylistic preference.

A caveat inherited from e11: the margin needs the `centre_x`/`centre_y` that
`inference/propose.py` records on every candidate, and an eval written before
that reports `margin = inf` everywhere -- a constant rather than a signal. The
script says so rather than quietly fitting it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e11_chain_gate import features, load                     # noqa: E402

from bevaligner.inference.confidence import (CONFIDENCE,     # noqa: E402
                                               CONFIDENCE_NO_REFINE,
                                               confidence_from)
from bevaligner.inference.gate import GateThresholds         # noqa: E402

# Within this, in metres, the placement is right. The score is fitted against
# this event and means nothing without it stated.
WITHIN_M = 2.0
BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 1.01)


def table(rows, thresholds: GateThresholds, within_m: float):
    """`(X, y, captures)` -- the four signals, the answer, and the fold key."""
    items = features(rows, thresholds)
    x, y, captures = [], [], []
    for row, item in zip(rows, items):
        rmse = row.get("rmse_m", row.get("rmse_xy"))
        if rmse is None:
            continue
        margin = 1.0 if not np.isfinite(item["margin"]) else min(item["margin"], 1.0)
        low, high = thresholds.rig_plausible
        x.append([item["inlier_frac"], float(item["refined"]), margin,
                  float(np.isfinite(item["rig"]) and low <= item["rig"] <= high)])
        y.append(float(float(rmse) < within_m))
        captures.append(item["capture"])
    return np.array(x, float), np.array(y, float), np.array(captures)


def fit(x, y, l2: float = 1.0, iterations: int = 60):
    """Logistic regression by IRLS, returning coefficients on the RAW features.

    Fitted in standardised space because the four columns differ by an order of
    magnitude in scale and the Hessian is better conditioned there, then mapped
    back so the constants can be read beside the quantities they weight. L2 of 1
    on the standardised coefficients, which is what keeps a column that is
    constant in some fold (`refined`, in a run where the zoomed pass never
    agreed) from running away.

    Plain numpy on purpose: four coefficients are not worth a scikit-learn
    dependency, and this box does not have one.
    """
    mean, deviation = x.mean(0), x.std(0) + 1e-9
    z = np.c_[np.ones(len(x)), (x - mean) / deviation]
    w = np.zeros(z.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-z @ w))
        weights = np.clip(p * (1 - p), 1e-6, None)
        gradient = z.T @ (y - p) - l2 * w
        hessian = z.T @ (z * weights[:, None]) + l2 * np.eye(z.shape[1])
        w = w + np.linalg.solve(hessian, gradient)
    raw = w[1:] / deviation
    return float(w[0] - (w[1:] * mean / deviation).sum()), raw


def apply_raw(x, bias, raw):
    return 1.0 / (1.0 + np.exp(-(bias + x @ raw)))


def auc(y, p) -> float:
    """Rank-based, with ties averaged. `nan` when one class is missing."""
    positive, negative = y.sum(), (1 - y).sum()
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    sorted_p, index = p[order], 0
    while index < len(sorted_p):                      # average the tied ranks
        stop = index
        while stop + 1 < len(sorted_p) and sorted_p[stop + 1] == sorted_p[index]:
            stop += 1
        if stop > index:
            ranks[order[index:stop + 1]] = (index + 1 + stop + 1) / 2
        index = stop + 1
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def folds(captures, k: int = 5, seed: int = 0):
    """Fold index per row, assigned by capture so no capture spans a split."""
    unique = np.unique(captures)
    np.random.default_rng(seed).shuffle(unique)
    assignment = {capture: i % k for i, capture in enumerate(unique)}
    return np.array([assignment[capture] for capture in captures])


def reliability(p, y, label: str, rmse=None) -> str:
    base = y.mean()
    lines = [f"  {label}: n={len(y)} base={base:.3f} AUC={auc(y, p):.3f} "
             f"Brier={((p - y) ** 2).mean():.4f} (base {base * (1 - base):.4f}) "
             f"mean p={p.mean():.3f}",
             "    confidence       n    mean p   actual   median rmse"]
    for low, high in zip(BINS, BINS[1:]):
        mask = (p >= low) & (p < high)
        if not mask.sum():
            continue
        median = (f"{np.median(rmse[mask]):8.2f} m" if rmse is not None
                  else "         -")
        lines.append(f"    [{low:.2f},{high:.2f})  {mask.sum():4d}   {p[mask].mean():.3f}"
                     f"    {y[mask].mean():.3f}   {median}")
    return "\n".join(lines)


def rmse_of(rows, within_m):
    values = [row.get("rmse_m", row.get("rmse_xy")) for row in rows]
    return np.array([float(v) for v in values if v is not None])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, type=Path,
                        help="stem of the scored run to FIT on (<stem>.jsonl)")
    parser.add_argument("--check", type=Path, action="append", default=[],
                        help="stem of a run to apply the fitted coefficients to; "
                             "repeatable")
    parser.add_argument("--within-m", type=float, default=WITHIN_M,
                        help=f"what counts as placed, in metres (default {WITHIN_M})")
    parser.add_argument("--no-refine", action="store_true",
                        help="fit the variant that drops the `refined` term")
    args = parser.parse_args()

    thresholds = GateThresholds()
    rows = load(args.eval)
    x, y, captures = table(rows, thresholds, args.within_m)
    if not len(y):
        raise SystemExit(f"{args.eval}: no scored rows with an rmse")
    refined_term = not args.no_refine
    if not refined_term:
        x = np.delete(x, 1, axis=1)

    infinite = int((x[:, 2 if refined_term else 1] >= 1.0).sum())
    print(f"fit on {args.eval}: {len(y)} scored rows, "
          f"{int(y.sum())} within {args.within_m:g} m (base {y.mean():.3f})")
    if infinite == len(y):
        print("  WARNING: every row has margin = inf -- this eval predates the "
              "candidate centres, so the margin is a constant, not a signal")
    else:
        print(f"  margin = inf (no materially different rival) on {infinite} rows; "
              f"{y[x[:, 2 if refined_term else 1] >= 1.0].mean():.3f} of those are "
              f"within {args.within_m:g} m")

    # Grouped cross-validation first: it is the number worth quoting, and
    # printing it above the constants keeps the two from being read apart.
    fold = folds(captures)
    out_of_fold = np.zeros(len(y))
    for k in range(fold.max() + 1):
        train, test = fold != k, fold == k
        bias, raw = fit(x[train], y[train])
        out_of_fold[test] = apply_raw(x[test], bias, raw)
    rmse = rmse_of(rows, args.within_m)
    print(reliability(out_of_fold, y, "grouped-CV (out of fold)",
                      rmse if len(rmse) == len(y) else None))

    bias, raw = fit(x, y)
    names = (["inlier_fraction", "refined", "margin", "rig_plausible"] if refined_term
             else ["inlier_fraction", "margin", "rig_plausible"])
    constant = "CONFIDENCE" if refined_term else "CONFIDENCE_NO_REFINE"
    print(f"\n  {constant} = ({bias:.4f}, "
          + ", ".join(f"{value:.4f}" for value in raw) + ")")
    print("    " + "  ".join(f"{name} {value:+.4f}" for name, value in zip(names, raw)))
    shipped = CONFIDENCE if refined_term else CONFIDENCE_NO_REFINE
    drift = max(abs(a - b) for a, b in zip((bias, *raw), shipped))
    print(f"    largest drift from the coefficients in confidence.py: {drift:.4f}")

    # And the shipped constants, scored as they actually run -- through
    # `confidence_from`, not through a second copy of the arithmetic here.
    def shipped_scores(matrix):
        if refined_term:
            return np.array([confidence_from(f, m, bool(r), bool(g))
                             for f, r, m, g in matrix])
        return np.array([confidence_from(f, m, False, bool(g), refined_term=False)
                         for f, m, g in matrix])

    print("\n" + reliability(shipped_scores(x), y, "the shipped constants, in sample",
                             rmse if len(rmse) == len(y) else None))

    for stem in args.check:
        held_rows = load(stem)
        held_x, held_y, _ = table(held_rows, thresholds, args.within_m)
        if not len(held_y):
            print(f"\n  {stem}: no scored rows with an rmse")
            continue
        if not refined_term:
            held_x = np.delete(held_x, 1, axis=1)
        held_rmse = rmse_of(held_rows, args.within_m)
        print("\n" + reliability(shipped_scores(held_x), held_y, f"{stem} (held out)",
                                 held_rmse if len(held_rmse) == len(held_y) else None))


if __name__ == "__main__":
    main()
