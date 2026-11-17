"""Where a split's files sit on disk, without hardcoding the directory shape.

Two layouts are in play and there will be a third:

    <root>/<structure_id>/<capture_id>/<reality_id>/     the original mirror
    <root>/PRJ<project>__<structure_id>/<reality_id>/    the current export

The second has no capture level at all, so code that walked a fixed number of
`.parent` hops off the point cloud silently resolved `root` to a directory four
levels too high, found no design transform, and keyed every structure the same.
Rather than branch on a layout flag, find the anchor: `floormap.png` sits in the
structure's directory in both, so walk up until it appears. Its parent is the
root and anything in between is the capture level.

Not derived from `panovggt_point_cloud_path`: 9 of the 322 rows leave that
column blank. `tm_path` is populated on every row, so `rlt_dir` comes from
there and a missing cloud is a property of the case rather than a crash.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SHEET_NAME = "floormap.png"
MAX_DEPTH = 3


@dataclass(frozen=True)
class SplitLayout:
    rlt_dir: Path
    capture_dir: Path | None     # absent in the flattened layout
    structure_dir: Path          # the directory holding floormap.png
    root_dir: Path

    @property
    def root(self) -> str:
        return self.root_dir.name


def detect(rlt_dir, sheet_name: str = SHEET_NAME) -> SplitLayout:
    """Resolve one split's directories by finding the sheet above it."""
    rlt_dir = Path(rlt_dir)
    up = rlt_dir
    for depth in range(1, MAX_DEPTH + 1):
        up = up.parent
        if (up / sheet_name).exists():
            return SplitLayout(
                rlt_dir=rlt_dir,
                capture_dir=rlt_dir.parent if depth > 1 else None,
                structure_dir=up,
                root_dir=up.parent,
            )
    raise FileNotFoundError(
        f"no {sheet_name} within {MAX_DEPTH} levels above {rlt_dir}")


def detect_from_sheet(rlt_dir, sheet_path) -> SplitLayout:
    """The same, but told where the sheet is -- the CSV always names it.

    Avoids touching the disk, which matters when resolving 322 rows, and works
    for a structure whose sheet is missing.
    """
    rlt_dir, structure_dir = Path(rlt_dir), Path(sheet_path).parent
    try:
        depth = len(rlt_dir.relative_to(structure_dir).parts)
    except ValueError:
        raise ValueError(f"{rlt_dir} does not sit under {structure_dir}") from None
    return SplitLayout(
        rlt_dir=rlt_dir,
        capture_dir=rlt_dir.parent if depth > 1 else None,
        structure_dir=structure_dir,
        root_dir=structure_dir.parent,
    )
