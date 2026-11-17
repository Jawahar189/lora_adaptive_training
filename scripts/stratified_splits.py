#!/usr/bin/env python3
"""Split a dataset CSV into train/val/test by STRUCTURE, stratified by size.

Thin wrapper; see `bevaligner/data/stratify.py` for what it does and why
those strata. `bevaligner.pipeline` calls `write_splits` directly.

    .venv/bin/python scripts/stratified_splits.py
    .venv/bin/python scripts/stratified_splits.py --train 0.8 --val 0.15 --test 0.05
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevaligner.data.stratify import DEFAULT_FRACTIONS, write_splits  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="datasets/100_prod_merged.csv")
    parser.add_argument("--out-dir", default="datasets/splits")
    parser.add_argument("--prefix", default="100_prod")
    parser.add_argument("--train", type=float, default=DEFAULT_FRACTIONS["train"])
    parser.add_argument("--val", type=float, default=DEFAULT_FRACTIONS["val"])
    parser.add_argument("--test", type=float, default=DEFAULT_FRACTIONS["test"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        write_splits(args.csv, args.out_dir, args.prefix,
                     {"train": args.train, "val": args.val, "test": args.test},
                     seed=args.seed)
    except ValueError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
