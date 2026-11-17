"""The prior capture's bird's-eye view, shaped like a sheet so the search can eat it.

`align.search` and `align.rerank` reach the floorplan through exactly four
surfaces: a boolean wall mask at full resolution, a list of viewports, a
paper<->pixel geometry, and `sheet.resample_to`. None of them care that the
mask came from a PNG. So the prior capture's merged wall band is packaged as a
`SheetFeatures` over its own raster, and every one of those consumers works
unchanged.

Everything here is in PAPER INCHES. The prior's transform lands in the target
frame, which is UTM metres for some structures and paper inches for others -- a
reference built "in the target frame" would make a 0.20 m cell mean two
different things on one dataset, and would do it silently rather than raising.
`align.priors` converts on the way out; this module never sees the target frame.

Two conventions that will cost a day if you get them wrong:

  * `feats.wall` is stored Y-DOWN, flipped from the rasteriser's y-up output,
    because `sheet.resample_to` crops through `paper_to_pixel` and flips back.
    Get it backwards and NOTHING raises: the search returns a pose mirrored
    about the horizontal axis and the reranker agrees with it, because both go
    through the same resample. Only the final transform is wrong.
  * `manhattan` must therefore run on `feats.wall`, not on the y-up raster --
    `search.yaw_candidates` negates the sheet angle on exactly that assumption.

The resolution is set by the FINEST consumer, not the search: the reranker
works at `rerank.TOL_M / rerank.TOL_PX`, about 0.05 m, while the search asks for
0.20 m and `resample_to` any-pools down correctly. Coarser here would silently
degrade the term that actually decides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from align import cloud as cloudmod
from align import segments
from align.frames import SheetGeometry, design_scale
from align.gravity import CAMERA_HEIGHT_M, MAX_REL_MAD, FloorFit, floor_below_trajectory
from align.scales import FALLBACK_METRES_PER_INCH, from_camera_height
from align.sheet import SheetFeatures, Viewport

MERGE_STRIDE = 6
# 0.10 m. Not the reranker's 0.05 m tolerance (TOL_M / TOL_PX), which was the
# first choice: storing the whole reference that fine put four rasters over 20M
# cells and one at 44M, for no gain. The reference's own splits disagree with
# each other by a median of 1.41 m (`scripts/e5_prior_quality.py`), so 0.10 m is
# already fourteen times finer than the data it represents, and still twice as
# fine as the search's COARSE_CELL_M so `resample_to` has something to pool.
CELL_M = 0.10
MAX_REF_PX = 6000
MIN_WALL_CELLS = 2000            # below this the reference is too thin to match
PAD_M = 2.0
COVERAGE_DILATE_M = 3.0          # p25 of camera-to-prior-camera distance at 2 m
                                 # is only 0.43, so 1 m would exclude the truth
LEVEL_GAP_M = 2.0
# Catches a GROSSLY mis-registered prior, not a slightly disagreeing one.
# `scripts/e5_prior_quality.py` measured 43 comparable pairs within a capture:
# median 1.41 m, p75 2.22, p90 3.71, max 10.78. Splits of one capture simply do
# not agree to a metre, so a 1 m filter discarded normal data -- 13 of 22 splits
# on one capture -- while 5 m isolates the two genuine outliers (10.78 and
# 10.13 m, the same pair docs/METHOD.md §11 already records at 10.8 m).
# That median is also the reference's intrinsic blur, and a floor under any
# accuracy this approach can reach.
CONSISTENCY_TOL_M = 5.0
HEIGHT_BAND = (0.55, 1.65)       # of the expected rig height, before correcting
# MAX_REL_MAD lives in align.gravity now, shared with the current-side floor-fit
# gate (align.gravity.usable_floor); re-imported above so this stays the name
# every caller here already uses.
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "reference"


@dataclass
class PriorLevel:
    """One storey. A superimposed two-storey reference is unsolvable, and looks
    exactly like a ranking failure, so levels are separated rather than merged."""
    z_in: float
    reality_ids: tuple[str, ...]
    wall: np.ndarray             # y-up, level-only


@dataclass
class Reference:
    key: str
    feats: SheetFeatures         # wall = merged prior wall, Y-DOWN
    free: SheetFeatures          # wall = where the prior walked, same grid
    geom: SheetGeometry
    origin: np.ndarray           # (2,) paper inches of local (0, 0)
    res: float                   # paper inches per cell
    camera_height_in: float      # the prior rig height -- the scale anchor
    camera_height_mad: float
    metres_per_inch: float
    scale_source: str            # "design_tm" | "prior_camera_height" | "fallback"
    floor_z_in: float            # paper-frame z of the prior floor
    # The prior's camera heights, in paper inches. The current capture is
    # levelled against these rather than against the prior's floor: the same rig
    # walked both, and matching cameras uses one floor fit where landing floors
    # compounds two. Measured at the oracle pose, median rmse_z 0.077 m against
    # 0.150 m for floor-landing.
    prior_camera_z: np.ndarray | None = None
    # Their plan positions too: scoring asks how much of the current walk the
    # prior actually covered, and re-reading every prior split's shots to answer
    # that would undo the point of building the reference once.
    prior_camera_xy: np.ndarray | None = None
    levels: list[PriorLevel] = field(default_factory=list)
    prior_reality_ids: tuple[str, ...] = ()
    dropped: list[tuple[str, str]] = field(default_factory=list)
    n_points: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def wall_cells(self) -> int:
        return int(self.feats.wall.sum())

    @property
    def thin(self) -> bool:
        """Too little line-work to place anything against."""
        return self.wall_cells < MIN_WALL_CELLS

    @property
    def scale_trusted(self) -> bool:
        return self.scale_source != "fallback"

    def to_paper(self, local_xy: np.ndarray) -> np.ndarray:
        """Local solve frame -> paper inches."""
        return np.asarray(local_xy)[:, :2] + self.origin


@dataclass
class SplitFloor:
    reality_id: str
    pts: np.ndarray              # paper inches
    cams: np.ndarray
    floor: FloorFit
    fitted_in: float             # what the histogram found


@dataclass
class SplitBEV:
    reality_id: str
    wall_xy: np.ndarray          # paper inches, y-up
    cam_xy: np.ndarray
    cam_z: np.ndarray
    floor: FloorFit
    height_in: float             # rig height above the local floor, inches
    floor_z_in: float
    corrected: bool = False


def split_floor(ref, reality_id: str, stride: int = MERGE_STRIDE) -> SplitFloor:
    """One prior split's cloud, cameras and fitted floor, in paper inches.

    No gravity estimate: the prior is already registered, so paper z is up.
    """
    pts = ref.points(reality_id, stride=stride)
    cams = ref.trajectory(reality_id).positions
    floor = floor_below_trajectory(pts, cams)
    return SplitFloor(reality_id, pts, cams, floor, float(floor.height))


def band_at(sf: SplitFloor, height_in: float, corrected: bool) -> SplitBEV:
    """Cut the wall band at a given rig height.

    `cloud.wall_band` takes only the scalar -- its locality comes from banding
    each point against its NEAREST CAMERA's z, so substituting a corrected
    height keeps the per-camera behaviour that stops a stair smearing the band.
    """
    floor = FloorFit(height=height_in, mad=sf.floor.mad, n_used=sf.floor.n_used,
                     per_camera=sf.floor.per_camera, beyond=sf.floor.beyond)
    band = cloudmod.wall_band(sf.pts, sf.cams, floor)
    return SplitBEV(
        reality_id=sf.reality_id,
        wall_xy=sf.pts[band][:, :2],
        cam_xy=sf.cams[:, :2],
        cam_z=sf.cams[:, 2],
        floor=floor,
        height_in=height_in,
        floor_z_in=float(np.median(sf.cams[:, 2]) - height_in),
        corrected=corrected,
    )


def _usable(bev: SplitBEV, n_band: int,
            max_rel_mad: float = MAX_REL_MAD) -> str | None:
    if not np.isfinite(bev.height_in) or bev.height_in <= 0:
        return "no floor mode"
    if bev.floor.rel_mad > max_rel_mad:
        return f"floor rel_mad {bev.floor.rel_mad:.2f}"
    # NOT gated on FloorFit.beyond. That test exists to tell a floor from a
    # ceiling when the up-vector's SIGN is unknown, and its <=0.008 threshold was
    # measured in the ply frame for exactly that job. A prior is registered, so
    # up is known and the surface under the cameras is the floor by
    # construction; `beyond` here just counts clutter below floor level, which
    # on a construction site is openings, penetrations and stairwells. Measured
    # over 105 prior splits: median 0.006 but p90 0.165, so gating at 0.05 would
    # discard 38 of them -- including the split that placed best in the oracle
    # experiment (0.16 m from truth).
    if n_band < cloudmod.MIN_BAND_POINTS:
        return f"{n_band} wall points"
    return None


def _consistency_drops(ref, bevs, tol_in):
    """Prior splits of one capture that share images must agree in the target frame.

    A single mis-registered prior poisons the whole reference, and nothing
    downstream can tell that from a hard split. This is the only in-solver guard
    against it, and it is legitimate: it compares shots pushed through each
    split's own transform, never a pose file.
    """
    trajs = {b.reality_id: ref.trajectory(b.reality_id) for b in bevs}
    pos = {rid: dict(zip(t.names, t.positions)) for rid, t in trajs.items()}
    drops = {}
    ids = [b.reality_id for b in bevs]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = set(pos[a]) & set(pos[b])
            if len(shared) < 3:
                continue
            d = np.linalg.norm(
                np.array([pos[a][n] for n in shared])
                - np.array([pos[b][n] for n in shared]), axis=1)
            rms = float(np.sqrt((d ** 2).mean()))
            if rms > tol_in:
                # Blame the one that agrees with fewer others: keep the majority.
                drops.setdefault(b, f"disagrees with {a} by {rms:.1f} in")
    return drops


def _levels(bevs, gap_in):
    """Single-linkage clustering of per-split floor height."""
    order = sorted(bevs, key=lambda b: b.floor_z_in)
    groups, cur = [], [order[0]]
    for b in order[1:]:
        if b.floor_z_in - cur[-1].floor_z_in > gap_in:
            groups.append(cur)
            cur = [b]
        else:
            cur.append(b)
    groups.append(cur)
    return groups


def metres_per_inch_for(structure, camera_height_in: float) -> tuple[float, str]:
    """Metres per paper inch, preferring the sheet's own transform.

    Where the structure carries no design transform the scale comes from the
    PRIOR's cameras rather than from the pooled estimate over current clouds:
    the prior is registered, so its rig height is measured on data whose frame
    is already known.
    """
    if structure is not None and structure.design_tm is not None:
        return design_scale(structure.design_tm), "design_tm"
    if np.isfinite(camera_height_in) and camera_height_in > 0:
        mpi = from_camera_height(camera_height_in)
        if 0.5 <= mpi <= 80.0:
            return mpi, "prior_camera_height"
    return FALLBACK_METRES_PER_INCH, "fallback"


def build(ref, structure=None, stride: int = MERGE_STRIDE,
          cell_m: float = CELL_M, pad_m: float = PAD_M,
          coverage_dilate_m: float = COVERAGE_DILATE_M,
          level_gap_m: float = LEVEL_GAP_M,
          consistency_tol_m: float = CONSISTENCY_TOL_M,
          max_rel_mad: float = MAX_REL_MAD,
          height_band: tuple[float, float] = HEIGHT_BAND,
          max_ref_px: int = MAX_REF_PX,
          min_wall_cells: int = MIN_WALL_CELLS) -> Reference | None:
    """Merge one prior pool into a sheet-shaped reference. None when unusable.

    Every threshold is a parameter rather than a constant read from the module,
    so a caller can sweep one without editing the file and without the swept
    value disagreeing with the cached rasters -- `build_cached` keys the cache on
    the ones that change the raster.
    """
    dropped, warnings, floors = [], [], []
    for split in ref.with_clouds():
        try:
            floors.append(split_floor(ref, split.reality_id, stride=stride))
        except Exception as exc:                       # a broken or empty ply
            dropped.append((split.reality_id, repr(exc)[:80]))
    if not floors:
        return None

    # The rig height is the one thing about a prior we already know: a 360
    # camera on a monopod, CAMERA_HEIGHT_M = 2.11 (p10 1.88 / p90 2.66). The
    # floor histogram sometimes latches onto clutter just under the lens
    # instead -- measured over 98 georeferenced prior splits, the implied rig
    # height has median 1.94 m but p10 0.58 m, and 23 of them land outside
    # 1.2-3.5 m, every one of them too LOW. A wrong height does not merely
    # mis-score the split, it cuts the wall band to a slab a few centimetres
    # thick, so it cannot be left alone.
    #
    # Dropping those 23 would throw away nearly a quarter of the reference. The
    # height is known, so substitute it instead and say so.
    fitted = np.array([f.fitted_in for f in floors])
    good = fitted[np.isfinite(fitted) & (fitted > 0)]
    pool_median = float(np.median(good)) if len(good) else float("nan")
    mpi, scale_source = metres_per_inch_for(structure, pool_median)
    # Where the sheet scale is known the expected height is exact; where it is
    # not, the pool's own median is the only reference available -- which makes
    # this a consistency test between splits rather than against 2.11 m.
    expected_in = (CAMERA_HEIGHT_M / mpi if scale_source == "design_tm"
                   else pool_median)

    bevs = []
    for f in floors:
        h, corrected = f.fitted_in, False
        plausible = (np.isfinite(h) and h > 0 and np.isfinite(expected_in)
                     and height_band[0] * expected_in <= h <= height_band[1] * expected_in)
        if not plausible:
            if not np.isfinite(expected_in) or expected_in <= 0:
                dropped.append((f.reality_id, "no floor mode and no expected height"))
                continue
            h, corrected = expected_in, True
        try:
            bev = band_at(f, h, corrected)
        except ValueError as exc:
            dropped.append((f.reality_id, str(exc)[:80]))
            continue
        why = _usable(bev, len(bev.wall_xy), max_rel_mad)
        if why:
            dropped.append((f.reality_id, why))
            continue
        if corrected:
            warnings.append(
                f"{f.reality_id}: floor fit gave {f.fitted_in * mpi:.2f} m, "
                f"used the known rig height {h * mpi:.2f} m")
        bevs.append(bev)
    if not bevs:
        return None

    heights = np.array([b.height_in for b in bevs])
    camera_height_in = float(np.median(heights))
    camera_height_mad = float(np.median(np.abs(heights - camera_height_in)))

    bad = _consistency_drops(ref, bevs, consistency_tol_m / mpi)
    if bad and len(bad) < len(bevs):
        dropped += sorted(bad.items())
        bevs = [b for b in bevs if b.reality_id not in bad]
    elif bad:
        warnings.append("every split disagrees with another; kept them all")

    groups = _levels(bevs, level_gap_m / mpi)
    if len(groups) > 1:
        warnings.append(
            f"{len(groups)} floor levels "
            f"{[round(g[0].floor_z_in, 1) for g in groups]}; kept the largest")
    keep = max(groups, key=lambda g: sum(len(b.wall_xy) for b in g))

    wall_xy = np.vstack([b.wall_xy for b in keep])
    cam_xy = np.vstack([b.cam_xy for b in keep])
    pad = pad_m / mpi
    lo = np.minimum(wall_xy.min(0), cam_xy.min(0)) - pad
    hi = np.maximum(wall_xy.max(0), cam_xy.max(0)) + pad
    res = max(cell_m / mpi, float((hi - lo).max()) / max_ref_px)
    bounds = (lo[0], lo[1], hi[0], hi[1])

    # min_count 2: one stray point is speckle, and a thinner reference is a
    # more discriminative one.
    wall_up = segments.rasterize(wall_xy, res, bounds=bounds, min_count=2).mask
    cov_up = segments.rasterize(cam_xy, res, bounds=bounds, min_count=1).mask
    k = max(int(round(coverage_dilate_m / mpi / res)) | 1, 3)
    cov_up = cv2.dilate(cov_up.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    cov_up &= ~wall_up

    h, w = wall_up.shape
    geom = SheetGeometry(paper_w=w * res, paper_h=h * res, img_w=w, img_h=h)
    geom.check()
    wall = np.flipud(wall_up)                 # pixel convention: row 0 at max y
    cover = np.flipud(cov_up)

    def _feats(mask):
        return SheetFeatures(
            ink=mask, wall=mask, ppi=1.0 / res,
            paper_w=geom.paper_w, paper_h=geom.paper_h,
            viewports=[Viewport(0, 0, w, h, float(mask.sum()))],
            photo_backed=False, ink_fraction=float(mask.mean()))

    out = Reference(
        key=ref.key, feats=_feats(wall), free=_feats(cover), geom=geom,
        origin=lo, res=res,
        camera_height_in=camera_height_in, camera_height_mad=camera_height_mad,
        metres_per_inch=mpi, scale_source=scale_source,
        floor_z_in=float(np.median([b.floor_z_in for b in keep])),
        prior_camera_z=np.concatenate([b.cam_z for b in keep]),
        prior_camera_xy=np.vstack([b.cam_xy for b in keep]),
        levels=[PriorLevel(z_in=float(np.median([b.floor_z_in for b in g])),
                           reality_ids=tuple(b.reality_id for b in g),
                           wall=wall_up if g is keep else np.zeros_like(wall_up))
                for g in groups],
        prior_reality_ids=tuple(b.reality_id for b in keep),
        dropped=dropped, n_points=int(len(wall_xy)), warnings=warnings)
    if out.wall_cells < min_wall_cells:
        out.warnings.append(
            f"only {out.wall_cells} wall cells; too thin to match against")
    return out


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
# A prior capture set is shared by every split of the current capture that names
# it -- 58 distinct sets across the 131 current rows, one referenced by 10 of
# them -- so building per split would repeat the same merge up to ten times.
def cache_path(key: str, stride: int, cell_m: float,
               cache_dir=CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{key}_s{stride}_c{cell_m:.4f}.npz"


def _save(path: Path, r: Reference) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = r.feats.wall.shape
    np.savez_compressed(
        path,
        wall=np.packbits(r.feats.wall), free=np.packbits(r.free.wall),
        shape=np.array([h, w]), origin=r.origin,
        scalars=np.array([r.res, r.camera_height_in, r.camera_height_mad,
                          r.metres_per_inch, r.floor_z_in, r.n_points]),
        prior_camera_z=(r.prior_camera_z if r.prior_camera_z is not None
                        else np.zeros(0)),
        prior_camera_xy=(r.prior_camera_xy if r.prior_camera_xy is not None
                         else np.zeros((0, 2))),
        scale_source=np.array(r.scale_source),
        prior_reality_ids=np.array(r.prior_reality_ids),
        dropped=np.array([f"{a}\t{b}" for a, b in r.dropped]),
        warnings=np.array(r.warnings),
    )


def _load(path: Path, key: str) -> Reference:
    z = np.load(path, allow_pickle=False)
    h, w = (int(x) for x in z["shape"])
    n = h * w

    def unpack(name):
        return np.unpackbits(z[name])[:n].astype(bool).reshape(h, w)

    res, cam_h, cam_mad, mpi, floor_z, n_pts = (float(x) for x in z["scalars"])
    geom = SheetGeometry(paper_w=w * res, paper_h=h * res, img_w=w, img_h=h)

    def feats(mask):
        return SheetFeatures(
            ink=mask, wall=mask, ppi=1.0 / res,
            paper_w=geom.paper_w, paper_h=geom.paper_h,
            viewports=[Viewport(0, 0, w, h, float(mask.sum()))],
            photo_backed=False, ink_fraction=float(mask.mean()))

    return Reference(
        key=key, feats=feats(unpack("wall")), free=feats(unpack("free")),
        geom=geom, origin=z["origin"], res=res,
        camera_height_in=cam_h, camera_height_mad=cam_mad,
        metres_per_inch=mpi, scale_source=str(z["scale_source"]),
        floor_z_in=floor_z,
        prior_camera_z=(z["prior_camera_z"] if "prior_camera_z" in z.files
                        and len(z["prior_camera_z"]) else None),
        prior_camera_xy=(z["prior_camera_xy"] if "prior_camera_xy" in z.files
                         and len(z["prior_camera_xy"]) else None),
        prior_reality_ids=tuple(str(x) for x in z["prior_reality_ids"]),
        dropped=[tuple(str(x).split("\t")) for x in z["dropped"]],
        n_points=int(n_pts),
        warnings=[str(x) for x in z["warnings"]])


def build_cached(ref, structure=None, stride: int = MERGE_STRIDE,
                 cell_m: float = CELL_M, cache_dir=CACHE_DIR,
                 force: bool = False, **kw) -> Reference | None:
    """`build`, memoized on disk by the prior set. None is NOT cached.

    The cache key carries stride and cell size, the two that change the raster's
    shape. Sweeping any OTHER parameter needs `force=True`, or the run silently
    reads back a reference built with the previous value.
    """
    path = cache_path(ref.key, stride, cell_m, cache_dir)
    if path.exists() and not force:
        try:
            return _load(path, ref.key)
        except Exception:
            pass                      # a stale or truncated cache rebuilds
    r = build(ref, structure, stride=stride, cell_m=cell_m, **kw)
    if r is not None:
        _save(path, r)
    return r
