"""The camera trajectory: what every stage downstream actually consumes.

A `Trajectory` is positions plus the image index parsed out of the filename,
and nothing else. Two things depend on that shape:

  * gravity, scale, the walk term and the Manhattan estimate all work from
    POSITIONS alone -- the euler convention in the pose files is undocumented
    and a wrong guess fails silently, so rotations are deliberately not
    surfaced;
  * `align.chain` orders a capture's splits, classifies their junctions and
    transfers poses across them entirely by IMAGE INDEX, so the index has to
    survive the load rather than being recomputed from list position.

`frame` says which space the positions live in -- "ply" for the arbitrary
reconstruction frame the solver works in, "target" for UTM metres or paper
inches. Mixing the two silently produces a plausible answer in the wrong place,
so every producer sets it and consumers that care assert on it.

This module names no file. It exists so the solver can hold a trajectory
without importing the module that reads ground truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_IMG_IDX = re.compile(r"IMG_(\d+)")


@dataclass
class Trajectory:
    names: list[str]
    idx: np.ndarray          # image index parsed from the filename
    positions: np.ndarray    # (N, 3)
    frame: str               # "ply" | "target"

    def __len__(self) -> int:
        return len(self.names)


def image_index(name: str) -> int:
    """The IMG_ number in a filename, or -1.

    -1 rather than an exception: a split whose names do not carry an index
    still has a usable trajectory for gravity and scale, it just cannot be
    chained, and that is a better failure than losing the split entirely.
    """
    m = _IMG_IDX.search(name)
    return int(m.group(1)) if m else -1
