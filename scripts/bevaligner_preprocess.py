#!/usr/bin/env python3
"""Build the BEVAligner training cache. Thin wrapper; see
`bevaligner/data/preprocess.py` for what it does and every flag.

    .venv/bin/python scripts/bevaligner_preprocess.py --workers 16
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevaligner.data.preprocess import main  # noqa: E402

if __name__ == "__main__":
    main()
