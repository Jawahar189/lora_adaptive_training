"""Train / validation by structure, never by capture.

A capture is aligned against earlier captures of the same building, so a split
by capture would validate against a reference the network has already been
trained on from the other side. Splitting by structure is also what keeps a
PAIR together: a current row and every prior split it is matched against share
a structure, so neither can straddle the two sides.

Validation structures are drawn from those with at least `min_val_captures`
usable captures so the pose metric has some variety per building. Given an
explicit validation CSV, its rows are the validation set and any structure they
name is removed from training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def split_by_structure(cases, capture_results, num_val_structures: int = 10, seed: int = 0,
                       min_val_captures: int = 3):
    """(train_cases, val_cases) over the usable rows, structure-disjoint."""
    usable = [c for c in cases if capture_results[c.reality_id].ok]
    captures_per_structure: dict[str, int] = {}
    for case in usable:
        key = case.structure.key
        captures_per_structure[key] = captures_per_structure.get(key, 0) + 1
    eligible = [key for key in sorted(captures_per_structure)
                if captures_per_structure[key] >= min_val_captures]
    rng = np.random.default_rng(seed)
    num_val = min(num_val_structures, max(len(eligible) - 1, 0))
    val_structures = (set(rng.choice(eligible, size=num_val, replace=False).tolist())
                      if num_val else set())
    return ([c for c in usable if c.structure.key not in val_structures],
            [c for c in usable if c.structure.key in val_structures])


def split_by_csv(train_cases, val_cases, capture_results):
    """Validation is the second CSV; its structures leave the training set."""
    val = [c for c in val_cases if capture_results[c.reality_id].ok]
    val_structures = {c.structure.key for c in val}
    train = [c for c in train_cases
             if capture_results[c.reality_id].ok and c.structure.key not in val_structures]
    return train, val


def save_splits(path: Path, train_cases, val_cases, seed: int | None = None) -> dict:
    splits = dict(seed=seed,
                  train=sorted({c.structure.key for c in train_cases}),
                  val=sorted({c.structure.key for c in val_cases}),
                  train_realities=[c.reality_id for c in train_cases],
                  val_realities=[c.reality_id for c in val_cases],
                  num_train_captures=len(train_cases),
                  num_val_captures=len(val_cases))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(splits, indent=1))
    print(f"splits: {len(splits['train'])} train structures ({len(train_cases)} captures), "
          f"{len(splits['val'])} val ({len(val_cases)} captures) -> {path}")
    return splits


def load_splits(path: Path) -> dict:
    return json.loads(Path(path).read_text())
