"""Everything a training sample needs that is slow, computed once per capture.

The dataset is driven by a CSV; `store.CaptureStore` calls the functions here
on demand for rows the cache lacks. This module is also a CLI to do that work
ahead of time. Two stages, each skipping work that is already on disk:

  captures    cloud -> gravity -> ground-truth planar fit (which also settles
              the up sign) -> wide wall band. One .npz + one .json per reality.
              Run over the CURRENT rows and the PRIOR rows alike: a prior split
              is the same object, and its cached `gt_sim2` is its registration,
              which `data/prior_bev.py` pushes into paper inches.
  pairing     each current row's prior pool -- the named prior captures' splits
              whose camera box meets the row's -- then manifest.csv and splits.
              Cheap, and redone every time.

Ground truth is read here, through `..utils.groundtruth`, and nowhere else in
the data path. The manifest records why any capture was skipped, so the
denominator of every later number is visible.

    python -m bevaligner.data.preprocess --csv datasets/100_prod_current.csv --workers 16
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils import geometry, gravity, groundtruth, io
from ..utils.metrics import RIG_PLAUSIBLE_M, rig_height_m

MANIFEST_COLUMNS = ("reality_id", "capture_id", "structure_key", "role", "status",
                    "num_band_points", "num_cameras", "residual_inches", "residual_m",
                    "up_index", "sign_ambiguous", "pair_status",
                    "prior_splits", "prior_dropped", "prior_levels", "overlap",
                    "same_day", "later_dated",
                    "metres_per_inch", "metres_per_inch_source",
                    "camera_rig_height_m", "rig_height_in",
                    "rig_height_implausible", "height_corrected", "seconds")


@dataclass
class CaptureResult:
    """One row of the manifest: did this capture make it into the dataset, and why not."""

    reality_id: str
    capture_id: str
    structure_key: str
    role: str
    # Did the slow per-row work succeed? `ok` means there is an .npz: a gravity
    # frame, a wall band and a ground-truth fit. Nothing about priors.
    status: str = "ok"
    num_band_points: int = 0
    num_cameras: int = 0
    residual_inches: float = math.nan     # ground-truth planar fit residual
    residual_m: float = math.nan
    up_index: int = -1
    sign_ambiguous: bool = False
    # Did this row get a target? A SEPARATE verdict from `status`, and the
    # separation is load-bearing. A reality is a current row and, for a later
    # capture, one of its prior splits -- 14,635 realities across the two sides
    # of these manifests against 13,485 current rows, so most are both. Writing
    # "this row has no prior of its own" into `status` would then disqualify it
    # as somebody else's TARGET, which is a different question entirely and one
    # `align/io/cases.py:287` warns is not even well formed. Doing exactly that
    # collapsed a full run from ~9,000 usable rows to 30: each gated row emptied
    # the next row's pool, which gated that row in turn.
    #
    # "" = not a current row in this run. Otherwise ok / no_prior /
    # no_usable_prior_split / low_overlap.
    pair_status: str = ""
    # the prior pool this row was paired with, and how well it covers the walk
    prior_splits: int = 0
    prior_dropped: int = 0
    prior_levels: int = 0
    overlap: float = math.nan
    # recorded, never filtered on: the two date sources in the manifests
    # disagree on a third of the edges, so they cannot carry a rule
    same_day: bool = False
    later_dated: bool = False
    metres_per_inch: float = math.nan
    metres_per_inch_source: str = ""
    camera_rig_height_m: float = math.nan
    # the rig height in PAPER INCHES -- the unit every prior-side threshold is
    # measured in, because the monopod is one physical object present in both
    # captures whatever the frame's scale turns out to be
    rig_height_in: float = math.nan
    rig_height_implausible: bool = False
    height_corrected: bool = False
    seconds: float = 0.0

    @property
    def prepared(self) -> bool:
        """The slow work succeeded, so there is a band on disk. This, and only
        this, is what makes a reality usable as somebody's prior split."""
        return self.status == "ok"

    @property
    def ok(self) -> bool:
        """Usable as a TRAINING ROW: prepared, and paired with a prior."""
        return self.prepared and self.pair_status == "ok"


class CachePaths:
    """Where each piece of the on-disk memo lives."""

    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)
        self.captures = self.root / "captures"
        self.manifest = self.root / "manifest.csv"
        self.splits = self.root / "splits.json"

    def capture_npz(self, reality_id: str) -> Path:
        return self.captures / f"{reality_id}.npz"

    def capture_json(self, reality_id: str) -> Path:
        return self.captures / f"{reality_id}.json"


# ----------------------------------------------------------------------------
# stage 1: captures

def _prior_true_up(case):
    """PRIOR_TRUE_UP=1: a prior's up from its own tm.json (it is registered), so its BEV is built in the
    true frame whatever the estimate or ASSUME_Z_UP say; None otherwise (the gravity step decides)."""
    if case.role != "prior" or os.environ.get("PRIOR_TRUE_UP", "") in ("", "0"):
        return None
    tm_path = Path(case.rlt_dir) / "tm.json"
    if not tm_path.is_file():
        return None
    data = json.loads(tm_path.read_text())
    tm = np.asarray(data["tm"] if isinstance(data, dict) else data, dtype=np.float64).reshape(4, 4)
    return tm[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def prepare_capture(case, paths: CachePaths, stride: int, min_source_frames: int,
                    max_residual_m: float,
                    height_override_in: float | None = None) -> CaptureResult:
    """One reality's slow work, cached.

    `height_override_in` re-cuts the wall band at a known rig height instead of
    the fitted one, in PAPER INCHES. The floor histogram latches onto stacked
    material or a scaffold deck often enough to matter -- 23 of 98 prior splits
    measured in `align/reference.py`, every one too LOW -- and a wrong height
    does not merely mis-score a split, it cuts the band to a slab a few
    centimetres thick. The correction cannot be applied to the cached band
    afterwards, because the wide band was already cut at the wrong height, so it
    has to come back through here. `store.correct_prior_heights` decides who
    needs it, once the pool's own median height is known.
    """
    started = time.time()
    result = CaptureResult(case.reality_id, case.capture_id, case.structure.key, case.role)
    try:
        trajectory = io.load_camera_trajectory(case)
        cloud_xyz = io.load_point_cloud_xyz(case, stride, min_source_frames)
    except Exception as error:                                    # noqa: BLE001
        result.status = f"load_failed: {type(error).__name__}"
        return _finish(result, started)
    result.num_cameras = len(trajectory)
    if len(trajectory) < 3 or len(cloud_xyz) < 1000:
        result.status = "too_small"
        return _finish(result, started)

    gravity_fit, views = gravity.gravity_views(cloud_xyz, trajectory.positions, up=_prior_true_up(case))
    result.sign_ambiguous = bool(gravity_fit.sign_ambiguous)
    planar_fit = groundtruth.ground_truth_planar_fit(
        case, trajectory, gravity.rotations_per_hypothesis(gravity_fit))
    if planar_fit is None:
        result.status = "no_gt_fit"
        return _finish(result, started)
    residual_inches, up_index, rotation_to_gravity_up, gt_sim2 = planar_fit
    result.residual_inches, result.up_index = float(residual_inches), int(up_index)
    view = views[up_index]
    override = (height_override_in is not None and np.isfinite(height_override_in)
                and height_override_in > 0 and gt_sim2.s > 0)
    if isinstance(view, str) and not override:
        result.status = view
        return _finish(result, started)
    if override:
        # `view` may still be the reason string here: a split whose floor
        # histogram found nothing is precisely the one a known height rescues.
        view = gravity.rebanded_view(
            cloud_xyz, trajectory.positions, up_index, rotation_to_gravity_up,
            height_override_in / gt_sim2.s,
            0.0 if isinstance(view, str) else view.floor_relative_mad)
        if isinstance(view, str):
            result.status = view
            return _finish(result, started)
        result.height_corrected = True

    if case.structure.georeferenced:
        metres_per_inch = geometry.design_scale(case.structure.design_tm)
        scale_source = "design_tm"
    else:
        metres_per_inch, scale_source = io.metres_per_inch_for(
            case.structure, view.camera_height_above_floor * gt_sim2.s)
    result.metres_per_inch = float(metres_per_inch)
    result.metres_per_inch_source = scale_source
    result.residual_m = float(residual_inches * metres_per_inch)
    result.rig_height_in = float(view.camera_height_above_floor * gt_sim2.s)
    result.camera_rig_height_m = float(
        rig_height_m(view.camera_height_above_floor, gt_sim2.s, metres_per_inch))
    result.rig_height_implausible = not (
        RIG_PLAUSIBLE_M[0] <= result.camera_rig_height_m <= RIG_PLAUSIBLE_M[1])
    if result.residual_m > max_residual_m:
        result.status = "bad_fit"
        return _finish(result, started)

    result.num_band_points = int(len(view.band_xy))
    paths.captures.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths.capture_npz(case.reality_id),
        band_xy=view.band_xy, band_relative_height=view.band_relative_height,
        band_camera_index=view.band_camera_index,
        camera_xy=view.camera_xy, camera_z=view.camera_z,
        camera_paper_xy=gt_sim2.apply(view.camera_xy),
        # named, so two splits of a prior capture can be checked against each
        # other on the images they share (`prior_bev._consistency_drops`)
        camera_names=np.array(list(trajectory.names), dtype=np.str_),
        # a split of a parent carries the parent's whole cloud (`prior_bev.cut_split_priors`)
        parent=np.str_(case.parent),
        # the only paper-frame z in the cache: `gt_sim2` is planar, and without
        # a z the splits of a prior capture cannot be clustered into storeys
        camera_paper_z_median=np.float64(
            groundtruth.camera_paper_z_median(case, trajectory)),
        up_index=np.int32(up_index), sign_ambiguous=np.bool_(gravity_fit.sign_ambiguous),
        rotation_to_gravity_up=rotation_to_gravity_up,
        gt_sim2_params=groundtruth.sim2_to_params(gt_sim2),
        residual_inches=np.float64(residual_inches),
        camera_height_above_floor=np.float64(view.camera_height_above_floor),
        height_corrected=np.bool_(result.height_corrected),
        floor_relative_mad=np.float64(view.floor_relative_mad),
        metres_per_inch=np.float64(metres_per_inch),
        structure_key=np.str_(case.structure.key), role=np.str_(case.role),
        reality_id=np.str_(case.reality_id))
    return _finish(result, started)


def _finish(result: CaptureResult, started: float) -> CaptureResult:
    result.seconds = round(time.time() - started, 2)
    return result


# ----------------------------------------------------------------------------
# stage 2: pairing + manifest

def write_manifest(path: Path, results: list[CaptureResult]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({column: getattr(result, column) for column in MANIFEST_COLUMNS})


def read_manifest(path: Path) -> list[CaptureResult]:
    results = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            values = {}
            for field in dataclasses.fields(CaptureResult):
                value = row[field.name]
                if field.type in ("int", int):
                    value = int(value)
                elif field.type in ("float", float):
                    value = float(value)
                elif field.type in ("bool", bool):
                    value = value == "True"
                values[field.name] = value
            results.append(CaptureResult(**values))
    return results


# ----------------------------------------------------------------------------
# driver: fill the store for a CSV ahead of training

def prepare_from_csv(csv_paths, prior_csv_paths, cache_dir, workers: int = 16,
                     limit: int | None = None, force: bool = False, stride: int = 1,
                     min_source_frames: int = 2, max_residual_m: float = 0.5,
                     reality_ids=None, num_val_structures: int = 10,
                     split_seed: int = 0, data_config=None):
    """The same work the DataModule does on demand, for every row of the CSVs.

    Both sides are prepared: a prior split needs exactly the same .npz a current
    row does, and its cached `gt_sim2` is the registration `data/prior_bev.py`
    reads. Limiting or naming realities limits the CURRENT side only -- the
    prior pool of whatever survives is prepared whole, or the target has holes.
    """
    from ..config import DataConfig
    from .splits import save_splits, split_by_structure
    from .store import CaptureStore, summarise_capture_results
    from align.io import datasets as datasets_mod
    from align.io.cases import PairedIndex

    spec = datasets_mod.resolve([str(p) for p in csv_paths],
                                [str(p) for p in prior_csv_paths] or None,
                                name="preprocess")
    index = PairedIndex.load(spec)
    cases = list(index.current)
    if reality_ids:
        wanted = set(reality_ids)
        cases = [c for c in cases if c.reality_id in wanted]
    if limit:
        cases = cases[:limit]
    print(f"{len(cases)} current rows over "
          f"{len({c.structure.key for c in cases})} structures; "
          f"prior side {len(index.prior)} rows. {index.coverage()}")
    store = CaptureStore(cache_dir, stride, min_source_frames, max_residual_m,
                         data_config or DataConfig())
    results = store.ensure_prepared(cases, index, workers=workers, force=force)
    usable = [c for c in cases if results[c.reality_id].ok]
    train_cases, val_cases = split_by_structure(cases, results, num_val_structures,
                                                split_seed)
    save_splits(store.paths.splits, train_cases, val_cases, split_seed)
    print(summarise_capture_results(results, [c.reality_id for c in cases], "current"))
    print()
    prior_only = [c.reality_id for c in index.prior
                  if c.reality_id not in {x.reality_id for x in cases}]
    print(summarise_capture_results(results, [c.reality_id for c in index.prior],
                                    "prior-side"))
    print(f"  ({len(prior_only)} of them appear ONLY as prior splits; the rest are "
          f"current rows too, and are counted in both tables)")
    print(f"\n{len(usable)} current rows usable; manifest at {store.paths.manifest}")
    return results


def main(argv=None):
    from ..config import DataConfig, REPO_ROOT
    defaults = DataConfig()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", default=defaults.csv_files,
                        help="current-side capture index CSV(s); these rows ARE the dataset")
    parser.add_argument("--prior-csv", nargs="+", default=defaults.prior_csv_files,
                        help="prior-side CSV(s); the targets are built from these rows")
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--realities", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stride", type=int, default=defaults.cloud_stride)
    parser.add_argument("--min-source-frames", type=int, default=defaults.min_source_frames)
    parser.add_argument("--max-residual-m", type=float,
                        default=defaults.max_gt_fit_residual_m)
    parser.add_argument("--num-val-structures", type=int,
                        default=defaults.num_val_structures)
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    args = parser.parse_args(argv)
    config = DataConfig()
    prepare_from_csv([REPO_ROOT / p for p in args.csv],
                     [REPO_ROOT / p for p in (args.prior_csv or [])],
                     REPO_ROOT / args.cache_dir,
                     workers=args.workers, limit=args.limit, force=args.force,
                     stride=args.stride, min_source_frames=args.min_source_frames,
                     max_residual_m=args.max_residual_m, reality_ids=args.realities,
                     num_val_structures=args.num_val_structures,
                     split_seed=args.split_seed, data_config=config)


if __name__ == "__main__":
    main()
