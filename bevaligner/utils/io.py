"""Reading captures and sheets, through `align.io` and `align.sheet`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from align import cloud as align_cloud
from align import sheet as align_sheet
from align.io.cases import Case, CaseIndex, Structure  # noqa: F401
from align.io.floormap import every_nth as every_nth_image  # noqa: F401
from align.io.floormap import poses_to_images, shot_poses  # noqa: F401
from align.io.shots import load_shots
from align.io.traj import Trajectory  # noqa: F401
from align.frames import target_to_paper
from align.reference import metres_per_inch_for  # noqa: F401

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_cases(csv_paths: Iterable[str | Path], role: str = "current") -> list[Case]:
    """Every case in the CSVs, in order, de-duplicated by reality id."""
    seen, cases = set(), []
    for path in csv_paths:
        for case in CaseIndex.load(Path(path), repo_root=REPO_ROOT, role=role):
            if case.reality_id not in seen:
                seen.add(case.reality_id)
                cases.append(case)
    return cases


def load_point_cloud_xyz(case: Case, stride: int = 1, min_source_frames: int = 2) -> np.ndarray:
    if case.cloud_path is None:
        raise FileNotFoundError(f"{case.reality_id}: no point cloud")
    return align_cloud.load(case.cloud_path, stride=stride,
                            min_source_frames=min_source_frames).xyz


def load_camera_trajectory(case: Case) -> Trajectory:
    """Camera positions in the point-cloud frame (not ground truth)."""
    return load_shots(case.rlt_dir)


# ---------------------------------------------------------------------------
# the GNSS fix

# A capture's `geo.txt` is a per-image fix, one line per photograph, written
# beside the cloud. `align.io.cases` deliberately does not load it -- the older
# generation's `geo_first_point.json` is a surveyed point, this is a track, and
# the provenance question that stalled it is still open. It is read HERE, for
# this pipeline only, because what it is wanted for is a search prior rather
# than a measurement: an error of a few paper inches is useless as truth and
# ample as a place to start looking.
#
# Measured against ground truth over 160 realities of
# `datasets/manifest_all_19_aug_23_sep_current.csv`:
#
#   georeferenced structures   median 2.8 paper inches, p75 7.5, p90 12.1
#   everything else            median ~38 inches -- a different frame entirely
#
# Hence the two conditions below. The declared CRS cannot carry the test on its
# own: the unusable files name `EPSG:326xx` just as the usable ones do, while
# their coordinates are single digits. The magnitude is the honest signal.
GEO_UTM_MIN_ABS = 1e4
# Only the walk's START in geo.txt is a real fix; the per-photo fixes after it are
# not (over the 84 geo.txt files of datasets/bev_projects100: frames 1-5 median
# 1.8 m off, later frames median 24 m). So only the split holding the walk's start
# -- its first frame within this window, the one production switches the chain
# on with -- is seeded from geo.txt; every other split is placed by the chain.
GEO_START_MAX_FRAME = 50


@dataclass
class GeoTrack:
    """Where each photograph was fixed, in PAPER INCHES."""
    names: list[str]
    paper_xy: np.ndarray            # (N, 2)
    crs: str

    def __len__(self) -> int:
        return len(self.names)

    def first_of(self, trajectory: Trajectory) -> np.ndarray | None:
        """The fix for the earliest photograph of `trajectory` that carries one.

        Walk order, not file order: `Trajectory.idx` is the image index, and a
        split whose first camera has no fix should fall through to its second
        rather than losing its seed.
        """
        index = {name: row for row, name in enumerate(self.names)}
        for position in np.argsort(trajectory.idx):
            row = index.get(trajectory.names[int(position)])
            if row is not None:
                return self.paper_xy[row]
        return None


def _parse_geo_txt(path: Path) -> tuple[str, list[str], np.ndarray] | None:
    """`(crs, names, xyz)` from a geo.txt, or None when it is not one."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None
    names, rows = [], []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
        names.append(parts[0])
    if not rows:
        return None
    return lines[0].strip(), names, np.asarray(rows, dtype=np.float64)


def load_geo_track(case: Case) -> GeoTrack | None:
    """The per-image GNSS fix in paper inches, or None when it is not usable.

    None rather than an exception on every failure -- a missing or unreadable
    fix is a property of the capture, the way an absent cloud is, and the seed
    policy that calls this already has a path for not having one. A split whose
    first frame is past GEO_START_MAX_FRAME is not the walk's start: None.
    """
    if not case.structure.georeferenced:
        return None
    if int(np.min(load_camera_trajectory(case).idx)) > GEO_START_MAX_FRAME:
        return None
    parsed = _parse_geo_txt(Path(case.rlt_dir) / "geo.txt")
    if parsed is None:
        return None
    crs, names, xyz = parsed
    if np.abs(xyz[:, :2]).max() < GEO_UTM_MIN_ABS:
        return None                 # a local frame wearing a UTM label
    paper = target_to_paper(xyz, case.structure.design_tm)[:, :2]
    if not np.isfinite(paper).all():
        return None
    return GeoTrack(names=names, paper_xy=paper, crs=crs)


def extract_sheet_features(structure: Structure, metres_per_inch: float | None):
    """Wall extraction for the sheet, cached under `cache/sheet/`."""
    return align_sheet.prepare(structure, metres_per_inch)


def load_sheet_image(structure: Structure) -> Image.Image:
    return Image.open(structure.sheet_path).convert("RGB")


def sheet_size_px(structure: Structure) -> tuple[int, int]:
    with Image.open(structure.sheet_path) as image:
        return image.size   # (width, height)


def render_floormap(case: Case, images: dict, out_path, gt_target_xyz=None,
                    width: int = 1800, subtitle: str = ""):
    """One reality's per-photograph poses drawn on its sheet.

    `align.viz_floorplan` is imported inside the call because it pulls in the
    plotting stack, which a data worker has no use for.
    """
    from align import viz_floorplan
    return viz_floorplan.render(case, images, out_path, gt_target_xyz=gt_target_xyz,
                                width=width, subtitle=subtitle)
