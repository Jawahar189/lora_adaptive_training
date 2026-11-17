"""The target: the previous capture of the same building, seen from above.

A current reality is placed against the bird's-eye view of the capture that came
before it, not against the architect's drawing. `docs/PRIOR_CAPTURE.md` is why;
this module is that target for the learned matcher, as `align/reference.py` is
for the classical solver.

    prior capture ─┬─ reality A ──[its own gt_sim2]──┐
                   ├─ reality B ──[its own gt_sim2]──┼──▶ PAPER INCHES ──▶ merge
                   └─ reality C ──[its own gt_sim2]──┘
    current reality ────────────[its own gt_sim2]──────▶ the same paper inches

Both sides land in one frame, so the ground-truth chain the floor-plan path
already used survives untouched except for its last link. `target_to_paper` is
the identity where a structure carries no `design_tm`, which is why this target
needs none: it was the *sheet* that had no geometry to land on, not the frame.

Three things are not free and are done here, following `align/reference.py`:

  * a prior capture is 1-79 separately reconstructed splits, and two of them can
    disagree about where the same photograph is. Splits disagreeing with a
    majority by more than `CONSISTENCY_TOL` are dropped -- a single
    mis-registered split blurs the whole reference and nothing downstream can
    tell that from a hard pair;
  * the rig height is often mis-fitted, always too low, and a wrong height cuts
    the wall band to a slab. That correction needs the cloud, so it happens in
    `preprocess.py` via `height_override_in`; here it is only detected.

Everything is rendered from POINTS at the crop's own resolution. There is no
parent raster: a site is a median 352 m across and a quarter of them exceed a
kilometre, so a cached raster fine enough to match against would be tens of
megapixels for a window a few tens of metres wide.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ..utils import affine
from ..utils.gravity import BAND_HI, BAND_LO
from . import density

# Both thresholds below are measured in RIG HEIGHTS, not metres and not a
# fraction of the site. The monopod is one physical object that appears in every
# split of every capture, so its height in paper inches is a length this code
# can trust even where the metre scale is unknown -- the same cancellation
# `docs/PRIOR_CAPTURE.md` §5 uses for the pose scale. The constants are
# `align/reference.py`'s own, divided by CAMERA_HEIGHT_M = 2.11.
#
# 5.0 m: catches a GROSSLY mis-registered split, not a slightly disagreeing one.
# Splits of one capture disagree by a median of 1.41 m, p90 3.71, max 10.78
# (`scripts/e5_prior_quality.py`), so a 1 m rule discarded 13 of 22 splits on one
# capture while 5 m isolates the two genuine outliers.
CONSISTENCY_TOL_RIG_HEIGHTS = 5.0 / 2.11
MIN_SHARED_IMAGES = 3
# Of the expected rig height. `align/reference.py:HEIGHT_BAND`, same numbers.
HEIGHT_BAND = (0.55, 1.65)
MIN_BAND_POINTS = 200
# 3.0 m: the prior walked here, so the model may be asked about here. p25 of
# camera-to-prior-camera distance is 0.43 m, so 1 m would exclude the truth.
COVERAGE_DILATE_RIG_HEIGHTS = 3.0 / 2.11
# Content counts as valid a little beyond the cells it actually fills, so a
# sparse wall is one region rather than a dotted line. About one DINO patch.
CONTENT_DILATE_PX = 16
# The multi-date mosaic (`mosaic_latest`): the fewest wall points a date needs
# in a cell to own it. The cell side is a setting, in metres, carried into rig
# heights through the same 2.11 m as the constants above.
MOSAIC_MIN_POINTS = 20
CAMERA_HEIGHT_M = 2.11
# The split cut's nearest-camera queries (`cut_split_priors`): every core but two.
SPLIT_CUT_WORKERS = max(1, (os.cpu_count() or 4) - 2)


# ----------------------------------------------------------------------------
# one split

@dataclass
class PriorSplit:
    """One reality of a prior capture, already in paper inches."""

    reality_id: str
    capture_id: str
    capture_date: str
    band_paper_xy: np.ndarray          # (N, 2) paper inches
    band_relative_height: np.ndarray   # (N,) height above the nearest camera, normalised
    camera_paper_xy: np.ndarray        # (M, 2)
    camera_paper_z_median: float       # ground-truth paper-frame z of the cameras
    camera_names: tuple[str, ...]
    rig_height_in: float               # camera height above its own floor, paper inches
    parent: str = ""                   # the reality this one was split from, if any


def load_split(npz_path: Path, capture_id: str = "", capture_date: str = "") -> PriorSplit:
    """A cached capture .npz read as a prior split.

    The transform is the split's own `gt_sim2`, which is its registration. Using
    it is not leakage -- `docs/PRIOR_CAPTURE.md` §7 -- but it is only ever
    legitimate for a split the caller has already established is a prior OF the
    current case. `PriorPoolStore.pool_for` is the only thing that decides that.
    """
    stored = np.load(npz_path, allow_pickle=False)
    scale, theta, tx, ty = (float(v) for v in stored["gt_sim2_params"])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    gt = np.array([[scale * cos_t, -scale * sin_t, tx],
                   [scale * sin_t, scale * cos_t, ty]])
    names = tuple(str(n) for n in stored["camera_names"]) \
        if "camera_names" in stored.files else ()
    camera_z = (float(stored["camera_paper_z_median"])
                if "camera_paper_z_median" in stored.files else 0.0)
    return PriorSplit(
        reality_id=str(stored["reality_id"]),
        capture_id=capture_id,
        capture_date=capture_date,
        band_paper_xy=affine.apply(gt, stored["band_xy"]).astype(np.float32),
        band_relative_height=stored["band_relative_height"],
        camera_paper_xy=stored["camera_paper_xy"].astype(np.float64),
        camera_paper_z_median=camera_z,
        camera_names=names,
        rig_height_in=float(stored["camera_height_above_floor"]) * scale,
        parent=str(stored["parent"]) if "parent" in stored.files else "")


def load_split_cameras(npz_path: Path) -> np.ndarray:
    """Just the camera positions in paper inches. `np.load` is lazy per array,
    so the overlap gate over every pool costs a few megabytes, not the cache."""
    stored = np.load(npz_path, allow_pickle=False)
    return stored["camera_paper_xy"].astype(np.float64)


# ----------------------------------------------------------------------------
# the pool: every split of every prior capture this case may see

@dataclass
class PriorPool:
    """The merged prior, in paper inches, for one current case."""

    key: str
    splits: tuple[PriorSplit, ...]
    dropped: tuple[tuple[str, str], ...] = ()
    metres_per_inch: float = float("nan")
    scale_source: str = ""
    num_levels: int = 1
    _merged: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- merged
    def _merge(self) -> dict:
        # Keyed rather than "is the dict empty": `_merged` also holds the
        # derived caches below, so an empty-dict test would see one of those
        # and skip the merge itself.
        if "band_xy" not in self._merged:
            self._merged = dict(
                band_xy=np.vstack([s.band_paper_xy for s in self.splits]),
                band_relative_height=np.concatenate(
                    [s.band_relative_height for s in self.splits]),
                camera_xy=np.vstack([s.camera_paper_xy for s in self.splits]))
        return self._merged

    @property
    def band_paper_xy(self) -> np.ndarray:
        return self._merge()["band_xy"]

    @property
    def band_relative_height(self) -> np.ndarray:
        return self._merge()["band_relative_height"]

    @property
    def camera_paper_xy(self) -> np.ndarray:
        return self._merge()["camera_xy"]

    @property
    def reality_ids(self) -> tuple[str, ...]:
        return tuple(s.reality_id for s in self.splits)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """A square extent over the merged band, trimmed like the query's.

        Memoised: it percentiles the whole merged band, and the search asks for
        it once per gravity hypothesis and again for every window it tiles.
        """
        if "extent" not in self._merged:
            self._merged["extent"] = density.square_extent(self.band_paper_xy)
        return self._merged["extent"]

    @property
    def inference_band_points(self) -> np.ndarray:
        """The band `render_target` renders when it is not drawing a sample.

        With no `rng` the band is the fixed `BAND_LO`/`BAND_HI` one and nothing
        is thinned, so every window of every capture this pool serves renders
        the SAME points -- a few hundred thousand of them, re-masked and
        re-copied per window until this cached them. Held as float64 because
        that is what `render_density_map` casts to anyway.
        """
        if "inference_band" not in self._merged:
            heights = self.band_relative_height
            kept = (heights > BAND_LO) & (heights < BAND_HI)
            self._merged["inference_band"] = np.asarray(
                self.band_paper_xy[kept], dtype=np.float64)
        return self._merged["inference_band"]

    @property
    def span(self) -> float:
        lo_x, _, hi_x, _ = self.extent
        return float(hi_x - lo_x)

    @property
    def rig_height_in(self) -> float:
        """The pool's own rig height, the unit every threshold here is in."""
        return float(np.median([s.rig_height_in for s in self.splits]))

    @property
    def bbox(self) -> np.ndarray:
        """The ground this pool covers, for anchoring the window on it."""
        return trimmed_bbox(self.camera_paper_xy)

    def band_mask(self, rng: np.random.Generator | None,
                  band_low_range, band_high_range) -> np.ndarray:
        """The wide band narrowed, exactly as the query side narrows its own.

        Drawn independently of the query's: these are two different clouds a day
        apart, and asking the model to match a band to the SAME band would teach
        it a correspondence the real pair does not have.
        """
        if rng is None:
            low, high = BAND_LO, BAND_HI
        else:
            low, high = rng.uniform(*band_low_range), rng.uniform(*band_high_range)
        heights = self.band_relative_height
        return (heights > low) & (heights < high)


def pool_key(reality_ids) -> str:
    return hashlib.sha1("|".join(sorted(reality_ids)).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# selection: which splits of the pool survive

def _consistency_drops(splits, tol_in: float) -> dict[str, str]:
    """Splits of one capture that share images must agree about where they are.

    Compares each split's own cameras against another's for the images they
    share; blames whichever disagrees with fewer others, so a majority survives.
    """
    positions = {s.reality_id: dict(zip(s.camera_names, s.camera_paper_xy))
                 for s in splits if s.camera_names}
    drops: dict[str, str] = {}
    ids = list(positions)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = set(positions[a]) & set(positions[b])
            if len(shared) < MIN_SHARED_IMAGES:
                continue
            delta = np.array([positions[a][n] for n in shared]) \
                - np.array([positions[b][n] for n in shared])
            rms = float(np.sqrt((delta ** 2).sum(axis=1).mean()))
            if rms > tol_in:
                drops.setdefault(b, f"disagrees with {a} by {rms:.1f} in")
    return drops


def select_splits(splits) -> tuple[list[PriorSplit], list, int]:
    """(kept, dropped, num_levels), with every threshold in rig heights.

    No level clustering. A prior pool never leaves one structure -- 0 of 11,160
    rows have a pool that crosses one -- and a structure carries exactly one
    floor plan (598 of 598), so its splits are on one floor by construction.
    Measured, the camera-z spread within a structure has a median of 0.18 m.
    `align/reference.py` clusters storeys because the classical solver's
    reference is assembled before that guarantee is known to it; here it is
    known, and splitting on height only threw away splits that had drifted.
    `num_levels` stays in the signature, always 1, because the manifest column
    is worth keeping as a place to notice if that ever stops being true.
    """
    dropped = []
    usable = []
    for split in splits:
        if len(split.band_paper_xy) < MIN_BAND_POINTS:
            dropped.append((split.reality_id, f"{len(split.band_paper_xy)} band points"))
        elif not np.isfinite(split.rig_height_in) or split.rig_height_in <= 0:
            dropped.append((split.reality_id, "no floor mode"))
        else:
            usable.append(split)
    if not usable:
        return [], dropped, 0

    rig_in = float(np.median([s.rig_height_in for s in usable]))
    bad = _consistency_drops(usable, CONSISTENCY_TOL_RIG_HEIGHTS * rig_in)
    if bad and len(bad) < len(usable):
        dropped += sorted(bad.items())
        usable = [s for s in usable if s.reality_id not in bad]
    return usable, dropped, 1


def cut_split_priors(splits) -> list[PriorSplit]:
    """A split of a parent reality, cut back to the walls it walked itself.

    The platform splits a reality but leaves each part the parent's whole cloud, with
    only its own images and its own tm -- fitted on those images, so right near them and
    extrapolated across the rest. The siblings carry the same cloud under their own tms,
    so the pool stacks one set of walls several times, a few degrees apart.

    Such a split keeps only the wall nearer its own cameras than those of any other split
    of its capture: the shared cloud is shared out, each part placed by the split whose
    images stand on it. Splits without a parent, and every split's cameras, are left as
    they are; a split left with no wall of its own adds nothing and is dropped.
    """
    out = []
    for split in splits:
        others = [s.camera_paper_xy for s in splits
                  if s is not split and s.capture_id == split.capture_id]
        if not split.parent or not others:
            out.append(split)
            continue
        band = split.band_paper_xy
        own = cKDTree(split.camera_paper_xy).query(band, workers=SPLIT_CUT_WORKERS)[0]
        keep = own <= cKDTree(np.concatenate(others)).query(band, workers=SPLIT_CUT_WORKERS)[0]
        if keep.any():
            out.append(replace(split, band_paper_xy=band[keep],
                               band_relative_height=split.band_relative_height[keep]))
    return out


def mosaic_latest(splits, cell_in: float) -> list[PriorSplit]:
    """Each cell's walls from the newest capture date that has wall there.

    A prior drawn from several visits disagrees with itself: the same wall
    registered twice a little apart, and a site that changed in between.
    Stacking them doubles walls. Instead, per cell of side `cell_in`, only the
    points of the most recent date with MOSAIC_MIN_POINTS there survive, and
    every split of that date keeps all of its points there -- one capture is
    never cut against itself. Cameras are left whole; only the wall band moves.
    """
    dates = sorted({s.capture_date for s in splits}, reverse=True)
    if len(dates) < 2:
        return list(splits)
    origin = np.min([s.band_paper_xy.min(0) for s in splits], axis=0)
    span = np.max([s.band_paper_xy.max(0) for s in splits], axis=0) - origin
    stride = int(span.max() / cell_in) + 2

    def keys(split):
        ij = np.floor((split.band_paper_xy - origin) / cell_in).astype(np.int64)
        return ij[:, 0] * stride + ij[:, 1]

    split_keys = [keys(s) for s in splits]
    claimed: set[int] = set()
    owned: dict[str, np.ndarray] = {}
    for date in dates:
        cells, counts = np.unique(np.concatenate(
            [k for s, k in zip(splits, split_keys) if s.capture_date == date]),
            return_counts=True)
        mine = [c for c in cells[counts >= MOSAIC_MIN_POINTS] if c not in claimed]
        owned[date] = np.asarray(mine, dtype=np.int64)
        claimed.update(mine)
    out = []
    for split, k in zip(splits, split_keys):
        keep = np.isin(k, owned[split.capture_date])
        if keep.any():
            out.append(replace(split, band_paper_xy=split.band_paper_xy[keep],
                               band_relative_height=split.band_relative_height[keep]))
    return out


# ----------------------------------------------------------------------------
# overlap: does the prior actually cover this walk?

def trimmed_bbox(xy: np.ndarray, trim: float = 2.5) -> np.ndarray:
    """(lo_x, lo_y, hi_x, hi_y) over the 2.5-97.5 percentiles.

    Trimmed rather than min/max because a single stray camera -- and there are
    plenty, one walk's positions span 397 m -- would otherwise stretch a box
    across a site and make it overlap everything.
    """
    p = np.asarray(xy, dtype=np.float64)[:, :2]
    return np.r_[np.percentile(p, trim, axis=0), np.percentile(p, 100 - trim, axis=0)]


def boxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Do two trimmed boxes enclose any common ground at all?

    This one predicate decides both which prior splits form a row's target and
    whether the row is usable, and it takes no threshold. Measured over 500
    rows: filtering a pool to the splits that pass keeps 17% of them and costs
    0.2pp of the rows that pass the gate, because the other 83% are parts of the
    site the walk never went near. Tightening it only loses rows -- a 5%
    containment threshold costs 1.2pp and a 20% one costs 6.6pp.

    Area, not intersection: boxes that merely touch along an edge share no
    ground. IoU was measured and rejected, because it punishes a large prior
    that CONTAINS a small walk, which is the best case rather than a bad one --
    best-candidate median IoU 0.16 against 0.55 for containment.
    """
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return bool(width > 0 and height > 0)


def _cell_keys(xy: np.ndarray, cell: float, origin: np.ndarray, stride: int) -> np.ndarray:
    """Integer cell ids for (N,2) points, as one int per cell. Exact, no hashing."""
    c = np.floor((np.asarray(xy, dtype=np.float64)[:, :2] - origin) / cell).astype(np.int64)
    return c[:, 0] * stride + c[:, 1]


def band_occupancy(current_band_xy: np.ndarray, prior_band_xy: np.ndarray,
                   cell: float) -> tuple[np.ndarray, float]:
    """How much of this walk's wall the prior also has. (per-cell mask, share).

    The criterion. It measures the thing the matcher actually sees -- wall --
    rather than where the surveyor stood or how big a bounding box is, and that
    distinction is not academic. Measured on six rows read off rendered pairs:

        reality              bbox  camera prox  occupancy   the picture
        RLT1787766560895     1.00         0.00       0.00   disjoint
        RLT1787606942928     1.00         0.08       1.00   walls line up
        RLT1787786690443     0.55         0.00       0.40   partial

    A box around a diagonal walk is mostly empty, so a current walk sitting in
    that emptiness scores 1.00 on the box and shares nothing. A prior that
    walked the corridors while the current walked inside a room puts their
    cameras far apart and their walls exactly together. Occupancy gets both.

    The prior is dilated by one cell so a half-cell offset is not a miss, and
    the cell is one rig height (about 2.11 m), which needs no paper scale and no
    design transform. It is not a sensitive number: 1 m and 2 m cells agreed on
    every row above.
    """
    cur = np.asarray(current_band_xy, dtype=np.float64)[:, :2]
    pri = np.asarray(prior_band_xy, dtype=np.float64)[:, :2]
    if len(cur) == 0 or len(pri) == 0 or not np.isfinite(cell) or cell <= 0:
        return np.zeros(len(cur), dtype=bool), 0.0
    origin = np.minimum(cur.min(0), pri.min(0))
    span = max(float((np.maximum(cur.max(0), pri.max(0)) - origin).max()), 1e-9)
    stride = int(span / cell) + 8                 # +8 leaves room for the dilation
    prior_cells = np.unique(_cell_keys(pri, cell, origin, stride))
    dilated = np.unique(np.concatenate(
        [prior_cells + dx * stride + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)]))
    mask = np.isin(_cell_keys(cur, cell, origin, stride), dilated)
    return mask, float(mask.mean())


def cameras_in_bbox(camera_paper_xy: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Which of this walk's cameras stand on ground the prior's box covers.

    What the target window is anchored on, and what biases the query's camera
    window, so a walk that leaves its prior's area part-way through still trains
    on the part that did not. Measured, a median of 0.97 of a walk's cameras
    fall inside its kept pool's box, and 94% of rows have at least 0.2 there.
    """
    cameras = np.asarray(camera_paper_xy, dtype=np.float64)
    if len(cameras) == 0:
        return np.zeros(0, dtype=bool)
    return ((cameras[:, 0] >= bbox[0]) & (cameras[:, 0] <= bbox[2])
            & (cameras[:, 1] >= bbox[1]) & (cameras[:, 1] <= bbox[3]))


# ----------------------------------------------------------------------------
# the target crop

@dataclass
class TargetCrop:
    """A square window of the prior, rendered, with the affine that put it there."""

    image: np.ndarray            # (S, S) float32 in [0, 1], y-down
    paper_to_crop: np.ndarray    # 2x3: paper inches -> crop pixel
    valid_mask: np.ndarray       # (S, S) bool: where the prior actually walked
    window: tuple[float, float, float, float]

    @property
    def size(self) -> int:
        return int(self.image.shape[0])


def _square(centre_x: float, centre_y: float, side: float):
    half = side / 2
    return (centre_x - half, centre_y - half, centre_x + half, centre_y + half)


def choose_window(pool: PriorPool, anchor_paper_xy: np.ndarray, rng, data_config,
                  size: int, patch_size: int):
    """The square of paper the target crop covers.

    Log-uniform in side between `zoom_footprint_margin` x the anchors' span and
    the pool's own extent, capped so the capture never spans fewer than
    `min_footprint_patches` patches -- the same policy, and the same cap, that
    `transforms.random_zoom_box` applied to a sheet viewport. The cap is what
    keeps a multi-kilometre site from rendering the walk into eight pixels, so
    no separate limit in metres is needed.

    `rng=None` is the deterministic evaluation window: the tightest the cap
    allows, centred on the anchors. That mirrors the old validation crop, which
    was the viewport `assign_capture_viewport` had picked using the same truth.
    """
    anchors = np.asarray(anchor_paper_xy, dtype=np.float64)
    low, high = anchors.min(0), anchors.max(0)
    centre_x, centre_y = (low + high) / 2
    anchor_side = float((high - low).max())
    max_zoom_out = size / max(data_config.min_footprint_patches * patch_size, 1e-3)
    smallest = max(anchor_side * data_config.zoom_footprint_margin, 1e-6)
    largest = min(max(pool.span, smallest), max(anchor_side * max_zoom_out, smallest))
    if rng is None or not data_config.zoom_crops or largest <= smallest:
        return _square(centre_x, centre_y, largest if rng is None else smallest)
    side = float(np.exp(rng.uniform(np.log(smallest), np.log(largest))))
    # placed at random, but always still holding the anchors
    slack_x = max(side - (high[0] - low[0]), 0.0) / 2
    slack_y = max(side - (high[1] - low[1]), 0.0) / 2
    return _square(centre_x + rng.uniform(-slack_x, slack_x),
                   centre_y + rng.uniform(-slack_y, slack_y), side)


def _dilate(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    kernel = max(int(kernel_px) | 1, 3)
    return cv2.dilate(mask.astype(np.uint8),
                      np.ones((kernel, kernel), np.uint8)).astype(bool)


def coverage_mask(image: np.ndarray, camera_paper_xy: np.ndarray, window, size: int,
                  rig_height_in: float) -> np.ndarray:
    """Where the prior has something to say, in the crop's pixels.

    This is the target's validity, and it decides where the matcher's softmax is
    allowed to put mass. It is the union of two things:

      * where the prior has WALL, dilated by about a patch. This is the bulk of
        it, and getting it wrong is expensive: an earlier version marked only the
        camera track, which left 3-5% of patches valid on a real pair -- the
        prior's walls stand several metres off the walk, so almost every true
        correspondence was being masked out and most samples failed outright;
      * where the prior WALKED, dilated by ~3 m. A corridor whose walls fell
        below the density threshold was still observed, and the truth can sit
        there: p25 of camera-to-prior-camera distance is 0.43 m.

    Outside the union the prior genuinely has no opinion. It replaces the sheet
    crop's white padding, and unlike padding it says something true.
    """
    lo_x, lo_y, hi_x, hi_y = window
    span = max(hi_x - lo_x, 1e-12)
    units_per_pixel = span / size
    points = np.asarray(camera_paper_xy, dtype=np.float64)
    col = np.floor((points[:, 0] - lo_x) / units_per_pixel).astype(np.int64)
    row = size - 1 - np.floor((points[:, 1] - lo_y) / units_per_pixel).astype(np.int64)
    inside = (col >= 0) & (col < size) & (row >= 0) & (row < size)
    walked = np.zeros((size, size), dtype=bool)
    if inside.any():
        walked[row[inside], col[inside]] = True
    walked = _dilate(walked, round(COVERAGE_DILATE_RIG_HEIGHTS * rig_height_in
                                   / units_per_pixel))
    return _dilate(image > 0, CONTENT_DILATE_PX) | walked


def draw_target_rotation(rng, data_config, anchor_paper_xy) -> np.ndarray | None:
    """This target's continuous yaw, as paper -> rotated paper, or None.

    About the anchor box's CENTRE, which is the point `choose_window` centres
    on, so the window it then picks sits on the same ground it would have sat
    on unrotated -- simply a rotated square of paper rather than an
    axis-aligned one. Rotate the anchors with this BEFORE handing them to
    `choose_window` and both of that function's guarantees survive untouched:
    a rotated window has to be |cos| + |sin| larger to hold the same box, and
    asking for the box in the rotated frame asks for exactly that and no more.
    Without it, anchors in the corners fall out of frame at large angles.

    Drawn independently of the query's angle, for a stronger reason than the
    band jitter's: the matcher only ever sees the DIFFERENCE of the two, so a
    shared angle is an exact no-op that costs the full compute.
    """
    degrees = float(data_config.target_rotation_degrees)
    if rng is None or degrees <= 0:
        return None
    anchors = np.asarray(anchor_paper_xy, dtype=np.float64)[:, :2]
    centre = (anchors.min(0) + anchors.max(0)) / 2
    return affine.rotation_about(np.deg2rad(rng.uniform(-degrees, degrees)),
                                 float(centre[0]), float(centre[1]))


def render_target(pool: PriorPool, window, size: int, rng, data_config,
                  min_points: int = MIN_BAND_POINTS,
                  paper_rotation: np.ndarray | None = None) -> TargetCrop | None:
    """The prior, counted into the window at the crop's own resolution.

    `paper_rotation` is the continuous yaw from `draw_target_rotation`, and
    `window` is then a square of the frame it lands in, not of paper. Nothing
    is resampled: the points turn on their way into the count, so the crop
    covers a rotated square of real paper rather than an interpolation of an
    axis-aligned one.
    """
    if rng is None:
        points = pool.inference_band_points          # the same points every window
    else:
        points = pool.band_paper_xy[pool.band_mask(
            rng, data_config.band_low_range, data_config.band_high_range)]
    if rng is not None:
        keep_low, keep_high = data_config.target_point_keep_fraction
        if keep_low < 1.0 or keep_high < 1.0:
            # the prior's own point dropout: it is as partial a capture as the
            # current one, and only the query side has ever been thinned
            keep = rng.uniform(keep_low, keep_high)
            points = points[rng.random(len(points)) < keep]
    if len(points) < min_points:
        return None
    cameras = pool.camera_paper_xy
    if paper_rotation is not None:
        points = affine.apply(paper_rotation, points)
        # The cameras turn WITH the points, or `coverage_mask` dilates the track
        # where the prior no longer is: `valid_mask` and the image would then
        # disagree and `sample_ground_truth_pairs` would filter the truth
        # against a stale mask, silently.
        cameras = affine.apply(paper_rotation, cameras)
    if rng is None:
        gamma, min_count = 0.5, 1
    else:
        gamma = float(rng.choice(data_config.density_gammas))
        min_count = int(rng.choice(data_config.density_min_counts))
    rendered = density.render_density_map(points, size, extent=window,
                                          gamma=gamma, min_count=min_count)
    if rendered.image.max() <= 0:
        return None
    valid = coverage_mask(rendered.image, cameras, rendered.extent,
                          size, pool.rig_height_in)
    paper_to_crop = affine.invert(rendered.density_px_to_world_xy)
    if paper_rotation is not None:
        # again before the chain is built: paper -> rotated paper -> crop px
        paper_to_crop = affine.compose(paper_to_crop, paper_rotation)
    crop = TargetCrop(image=rendered.image, paper_to_crop=paper_to_crop,
                      valid_mask=valid, window=rendered.extent)
    if rng is not None:
        crop = rotate_target(crop, int(rng.integers(0, 4)))
    return crop


def patch_valid_mask(valid_mask: np.ndarray, patch_size: int,
                     min_fraction: float = 0.5) -> np.ndarray:
    """(S,S) pixel validity -> (S/p, S/p) patch validity: a patch counts when at
    least `min_fraction` of its pixels are inside the prior's coverage."""
    side = valid_mask.shape[0]
    patches_per_side = side // patch_size
    trimmed = valid_mask[:patches_per_side * patch_size, :patches_per_side * patch_size]
    coverage = trimmed.reshape(patches_per_side, patch_size,
                               patches_per_side, patch_size).mean(axis=(1, 3))
    return coverage >= min_fraction


def rotate_target(crop: TargetCrop, turns: int) -> TargetCrop:
    if turns % 4 == 0:
        return crop
    size = crop.size
    rotation = affine.rotation_quarter_turns(turns, size, size)
    return TargetCrop(image=np.ascontiguousarray(np.rot90(crop.image, turns)),
                      paper_to_crop=affine.compose(rotation, crop.paper_to_crop),
                      valid_mask=np.ascontiguousarray(np.rot90(crop.valid_mask, turns)),
                      window=crop.window)


# ----------------------------------------------------------------------------
# the store side: resolve a case's pool, and keep a few in memory

class PriorPoolStore:
    """Builds pools from the capture cache, memoized per worker.

    The pools are the big objects in the data path -- a median pool is 4 splits
    and ~170k points, a p90 one 14 splits -- and a pool serves a median of 4
    current realities, so a small cache pays for itself without holding the
    dataset in RAM twelve times over.
    """

    def __init__(self, paths, cache_size: int = 24, mosaic: bool = False,
                 mosaic_cell_m: float = 5.0):
        self.paths = paths
        self.mosaic = mosaic
        self.mosaic_cell_rig_heights = mosaic_cell_m / CAMERA_HEIGHT_M
        self._bboxes: dict[str, np.ndarray | None] = {}
        self._build = lru_cache(maxsize=cache_size)(self._build_uncached)

    def _build_uncached(self, ids_dates: tuple) -> PriorPool | None:
        splits = []
        dropped = []
        for reality_id, capture_id, capture_date in ids_dates:
            path = self.paths.capture_npz(reality_id)
            if not path.exists():
                dropped.append((reality_id, "not in the capture cache"))
                continue
            try:
                splits.append(load_split(path, capture_id, capture_date))
            except Exception as error:                              # noqa: BLE001
                dropped.append((reality_id, f"{type(error).__name__}"))
        if not splits:
            return None
        keep, more_dropped, num_levels = select_splits(splits)
        keep = cut_split_priors(keep)
        if keep and self.mosaic:
            rig_in = float(np.median([s.rig_height_in for s in keep]))
            keep = mosaic_latest(keep, self.mosaic_cell_rig_heights * rig_in)
        if not keep:
            return None
        return PriorPool(key=pool_key(s.reality_id for s in keep),
                         splits=tuple(keep),
                         dropped=tuple(dropped + more_dropped),
                         num_levels=num_levels)

    def pool(self, ids_dates) -> PriorPool | None:
        return self._build(tuple(sorted(ids_dates)))

    def bbox_of(self, reality_id: str) -> np.ndarray | None:
        """One reality's camera box, memoised.

        `load_split_cameras` reads a single small array out of the .npz, so a
        pool's membership is decided without ever touching a wall band -- which
        is the whole point, since the bands of the 83% that do not overlap are
        exactly what the filter exists to avoid loading.
        """
        if reality_id not in self._bboxes:
            path = self.paths.capture_npz(reality_id)
            box = None
            if path.exists():
                try:
                    cameras = load_split_cameras(path)
                    if len(cameras) >= 3:
                        box = trimmed_bbox(cameras)
                except Exception:                                   # noqa: BLE001
                    box = None
            self._bboxes[reality_id] = box
        return self._bboxes[reality_id]

    def overlapping(self, reality_ids, current_bbox) -> tuple:
        """Those of `reality_ids` whose box meets the current walk's."""
        if current_bbox is None:
            return ()
        return tuple(r for r in reality_ids
                     if (box := self.bbox_of(r)) is not None
                     and boxes_overlap(current_bbox, box))
