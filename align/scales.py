"""Metres per paper inch, per structure.

Exact where the structure carries a design transform. Where it does not — 71 of
132 splits — the sheet's scale is otherwise unavailable, and the camera-height
anchor supplies it: a 360 rig rides at a near-constant height, so measuring that
height in paper inches and dividing the real height by it gives metres per inch.

Measured: this lands within 15% of a standard architectural scale on 14 of 15
un-georeferenced structures, and the five 48x36 sheets of one project agree with
each other to about 2%.

The scale is a property of the SHEET, not of any one walk, so per-split
estimates are pooled across the structure before use.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from align.frames import design_scale
from align.gravity import CAMERA_HEIGHT_M

REPO_ROOT = Path(__file__).resolve().parents[1]
# E1 writes out/e1_<tag>.jsonl, and the tag depends on how the run was scoped,
# so discover the outputs rather than naming one. Every split that appears in
# any of them contributes once, keyed by reality id.
E1_GLOB = "out*/e1_*.jsonl"

# Standard architectural scales, metres per inch. Snapping to these is a
# VALIDATOR, not a constraint: the georeferenced structures include 3.733,
# 4.492 and 4.608, which sit on no standard rung at all.
LADDER = {
    '1/4"=1\'': 1.2192, '3/16"=1\'': 1.6256, '1:100': 2.54, '1/8"=1\'': 2.4384,
    '1:120': 3.048, '3/32"=1\'': 3.2512, '1:150': 3.81, '1/16"=1\'': 4.8768,
    '1:200': 5.08, '1:250': 6.35, '1/32"=1\'': 9.7536, '1:480': 12.192,
}

# Used only when a structure has no design transform AND no usable E1 estimate.
FALLBACK_METRES_PER_INCH = 3.0


def nearest_standard(mpi: float) -> tuple[str, float, float]:
    """Closest standard scale, and the relative error to it."""
    name, val = min(LADDER.items(), key=lambda kv: abs(np.log(kv[1] / mpi)))
    return name, val, float(abs(mpi - val) / val)


def from_camera_height(height_inches: float,
                       camera_height_m: float = CAMERA_HEIGHT_M) -> float:
    """Invert the camera-height anchor: rig height in metres / height in inches."""
    return float(camera_height_m / height_inches)


def e1_paths(root=REPO_ROOT) -> list[Path]:
    """Every E1 output on disk, newest last."""
    return sorted(Path(root).glob(E1_GLOB), key=lambda p: p.stat().st_mtime)


def load_e1_estimates(paths=None, root=REPO_ROOT) -> dict[str, list[float]]:
    """Per-structure metres-per-inch estimates from the E1 runs on disk.

    Only sign-decisive splits contribute: where the floor/ceiling test tied, the
    height could be either surface and the implied scale could be either value.
    A split seen in several runs counts once, the most recent winning.
    """
    if paths is None:
        paths = e1_paths(root)
    elif isinstance(paths, (str, Path)):
        paths = [Path(paths)]

    by_split: dict[str, tuple[str, float]] = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("height_target") and not r.get("georeferenced")
                    and not r.get("sign_ambiguous")):
                by_split[r["reality_id"]] = (
                    r["structure"], from_camera_height(r["height_target"]))

    pooled: dict[str, list[float]] = {}
    for structure, mpi in by_split.values():
        pooled.setdefault(structure, []).append(mpi)
    return pooled


class SheetScales:
    """Resolves metres-per-inch for every structure in an index."""

    def __init__(self, index, e1_path=None):
        self._pooled = load_e1_estimates(e1_path)
        self._scales: dict[str, float] = {}
        self._source: dict[str, str] = {}
        for case in index:
            st = case.structure
            if st.key in self._scales:
                continue
            if st.design_tm is not None:
                self._scales[st.key] = design_scale(st.design_tm)
                self._source[st.key] = "design_tm"
            elif self._pooled.get(st.key):
                self._scales[st.key] = float(np.median(self._pooled[st.key]))
                self._source[st.key] = f"e1 (n={len(self._pooled[st.key])})"
            else:
                self._scales[st.key] = FALLBACK_METRES_PER_INCH
                self._source[st.key] = "fallback"

    def __getitem__(self, key: str) -> float:
        return self._scales.get(key, FALLBACK_METRES_PER_INCH)

    def get(self, key: str, default=None):
        return self._scales.get(key, default)

    def source(self, key: str) -> str:
        """Where this structure's scale came from. `fallback` means it is a
        guess and every threshold derived from it is suspect -- surface it."""
        return self._source.get(key, "fallback")

    def is_trusted(self, key: str) -> bool:
        return self.source(key) != "fallback"

    def report(self) -> str:
        lines = []
        for k in sorted(self._scales):
            mpi = self._scales[k]
            name, _, err = nearest_standard(mpi)
            lines.append(f"{k:30s} {mpi:7.2f} m/in  [{self.source(k):12s}] "
                         f"nearest {name} ({err:.0%})")
        return "\n".join(lines)
