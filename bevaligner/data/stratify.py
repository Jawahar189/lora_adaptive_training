"""Split a dataset CSV into train/val/test by STRUCTURE, stratified by size.

Captures of one building share its floor plan, so the split is structure-
disjoint for the same reason `bevaligner/data/splits.py` is: a split by
capture would validate on sheets the network has memorised.

Two strata axes, chosen because they are the two independent things that make
one structure unlike another here:

  #realities   how much walk the building has, binned into terciles
  sheet area   paper_width x paper_height, constant per structure, a proxy for
               floor area

Realities-per-capture was the obvious second axis and is not used: 45 of 48
structures have exactly two captures, so it is 0.99 rank-correlated with
#realities and carries no signal that the first axis does not already have.
Sheet area is 0.36 correlated, which is the point of a second axis.

Allocation is largest-remainder within each stratum, then a global repair pass
that moves the least-costly structure between splits until the structure counts
hit the requested ratio exactly. Deterministic in `--seed`.

Rows are written VERBATIM, so `CaseIndex` and every `--csv` script -- and
`data.csv_files` / `data.val_csv_files` in the BEVAligner config -- read them
unchanged.

`datasets/` is gitignored, so neither the input nor the splits are in the repo.

Stratify the CURRENT index only. A prior row is not a training example on this
branch -- it is part of a target -- so concatenating the two indexes, as earlier
runs did, would both inflate the counts a stratum is balanced on and put rows in
a split that nothing ever draws. The prior side follows its current row through
`data.prior_csv_files`, and because the split is by structure both land on the
same side of it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.75, "val": 0.15, "test": 0.10}


# ---------------------------------------------------------------- strata

def structure_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per structure: the quantities the strata are cut from."""
    width = pd.to_numeric(rows["paper_width"], errors="coerce")
    height = pd.to_numeric(rows["paper_height"], errors="coerce")
    per_structure = rows.assign(_sheet_area=width * height).groupby("structure_id")
    table = per_structure.agg(
        project_id=("project_id", "first"),
        n_realities=("reality_id", "nunique"),
        n_captures=("capture_id", "nunique"),
        sheet_area=("_sheet_area", "first"))
    table["realities_per_capture"] = table.n_realities / table.n_captures
    return table.sort_index()


def tercile_bin(values: pd.Series, labels) -> pd.Series:
    """Quantile bins that survive heavy ties: cut on rank, not on value."""
    ranks = values.rank(method="average", pct=True)
    edges = np.linspace(0, 1, len(labels) + 1)
    return pd.cut(ranks, bins=edges, labels=labels, include_lowest=True).astype(str)


def sheet_area_bin(area: pd.Series) -> pd.Series:
    """small / standard / large around the modal sheet size.

    21 of 48 structures sit on exactly one sheet area, so a quantile cut
    collapses; the mode is the natural middle bin.
    """
    modal_area = area.mode().iloc[0]
    return pd.Series(np.where(area < modal_area, "area:small",
                     np.where(area > modal_area, "area:large", "area:standard")),
                     index=area.index)


def assign_strata(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["realities_bin"] = tercile_bin(
        table.n_realities, ["rlt:small", "rlt:medium", "rlt:large"])
    table["area_bin"] = sheet_area_bin(table.sheet_area)
    table["stratum"] = table.realities_bin + " | " + table.area_bin
    return table


# ------------------------------------------------------------ allocation

def largest_remainder(total: int, fractions: dict[str, float]) -> dict[str, int]:
    """Whole counts summing to `total`, as close to `fractions` as integers get."""
    exact = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: int(math.floor(value)) for name, value in exact.items()}
    short = total - sum(counts.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - counts[name]),
                                            -fractions[name], name))
    for name in order[:short]:
        counts[name] += 1
    return counts


def allocate(table: pd.DataFrame, fractions: dict[str, float],
             seed: int) -> pd.Series:
    """A split label per structure: proportional inside each stratum, then
    repaired so the global structure counts match `fractions` exactly."""
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for stratum in sorted(table.stratum.unique()):
        members = table.index[table.stratum == stratum].to_numpy()
        members = members[rng.permutation(len(members))]
        counts = largest_remainder(len(members), fractions)
        cursor = 0
        # train first, so a one-member stratum lands in train rather than test
        for name in SPLIT_NAMES:
            for structure in members[cursor:cursor + counts[name]]:
                assignment[structure] = name
            cursor += counts[name]
    labels = pd.Series(assignment, name="split").reindex(table.index)
    labels = repair(labels, table, fractions, rng)
    return balance_realities(labels, table, fractions)


def repair(labels: pd.Series, table: pd.DataFrame, fractions: dict[str, float],
           rng: np.random.Generator) -> pd.Series:
    """Move structures between splits until the totals are exactly right.

    Each move is taken from the stratum that is currently most over-represented
    in the donor split, so stratification degrades as little as the arithmetic
    allows.
    """
    labels = labels.copy()
    targets = largest_remainder(len(table), fractions)
    for _ in range(4 * len(table)):
        counts = {name: int((labels == name).sum()) for name in SPLIT_NAMES}
        surplus = [n for n in SPLIT_NAMES if counts[n] > targets[n]]
        deficit = [n for n in SPLIT_NAMES if counts[n] < targets[n]]
        if not deficit:
            break
        donor, receiver = surplus[0], deficit[0]
        candidates = labels.index[labels == donor]
        # the donor's most crowded stratum gives up a member
        crowding = table.loc[candidates, "stratum"].value_counts()
        pool = candidates[table.loc[candidates, "stratum"] == crowding.index[0]]
        labels[pool[rng.integers(len(pool))]] = receiver
    return labels


def balance_realities(labels: pd.Series, table: pd.DataFrame,
                      fractions: dict[str, float]) -> pd.Series:
    """Swap structures WITHIN a stratum to even out the reality counts.

    Equal structure shares do not give equal reality shares: a stratum holds
    buildings of merely similar size, and which one goes to val decides several
    percent of the walk. A swap inside one stratum leaves both the structure
    counts and the strata crosstab untouched, so this is free accuracy on the
    quantity that actually matters -- how much data each split holds.
    """
    labels = labels.copy()
    realities = table.n_realities
    targets = {name: realities.sum() * fraction for name, fraction in fractions.items()}

    def error(current: pd.Series) -> float:
        return sum((realities[current == name].sum() - targets[name]) ** 2
                   for name in SPLIT_NAMES)

    for _ in range(len(table)):
        best, best_error = None, error(labels)
        for stratum in table.stratum.unique():
            members = table.index[table.stratum == stratum]
            for left in members:
                for right in members:
                    if labels[left] >= labels[right]:
                        continue
                    trial = labels.copy()
                    trial[left], trial[right] = labels[right], labels[left]
                    trial_error = error(trial)
                    if trial_error < best_error - 1e-9:
                        best, best_error = (left, right), trial_error
        if best is None:
            break
        left, right = best
        labels[left], labels[right] = labels[right], labels[left]
    return labels


# ---------------------------------------------------------------- report

def shown(path: Path) -> str:
    """Repo-relative when it can be, absolute when `--out-dir` points elsewhere."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    grouped = table.groupby("split")
    summary = grouped.agg(structures=("n_realities", "size"),
                          realities=("n_realities", "sum"),
                          median_realities=("n_realities", "median"),
                          median_sheet_area=("sheet_area", "median"))
    summary["structure_pct"] = (100 * summary.structures / summary.structures.sum()).round(1)
    summary["reality_pct"] = (100 * summary.realities / summary.realities.sum()).round(1)
    return summary.reindex([n for n in SPLIT_NAMES if n in summary.index])




def write_splits(source_csv, out_dir, prefix: str = "100_prod",
                 fractions: dict[str, float] | None = None,
                 seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Write `<prefix>_{train,val,test}.csv` and the strata audit; return the
    per-structure table with its `split` column.

    Rows are copied VERBATIM from `source_csv`, so `CaseIndex` and every
    `--csv` script read the splits unchanged.
    """
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"fractions sum to {total:g}, not 1")

    source_csv, out_dir = _under_repo(source_csv), _under_repo(out_dir)
    rows = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    if rows.reality_id.duplicated().any():
        raise ValueError(f"{source_csv} has duplicate reality_id values")

    table = assign_strata(structure_table(rows))
    table["split"] = allocate(table, fractions, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    split_of = table.split.to_dict()
    for name in SPLIT_NAMES:
        keep = rows[rows.structure_id.map(split_of) == name]
        path = out_dir / f"{prefix}_{name}.csv"
        keep.to_csv(path, index=False)
        if verbose:
            print(f"{shown(path)}: {len(keep)} realities, "
                  f"{keep.structure_id.nunique()} structures")
    audit = out_dir / f"{prefix}_structure_strata.csv"
    table.to_csv(audit)
    if verbose:
        print(f"{shown(audit)}: per-structure strata and assignment\n")
        print(summarise(table).to_string())
        print("\nstructures per stratum per split")
        print(pd.crosstab(table.stratum, table.split).reindex(
            columns=list(SPLIT_NAMES), fill_value=0).to_string())
    return table


def _under_repo(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
