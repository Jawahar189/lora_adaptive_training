"""Rank proposed poses by how much wall geometry they actually explain.

The correlation score cannot rank. Measured with scale and yaw held at ground
truth, the TRUE translation sits at the 94th, 98th and 19.5th percentile of the
masked-NCC surface on three test splits -- on one of them the term prefers 80%
of all other placements. Softening the sheet map and changing the footprint
were both tried and neither repaired it. Correlating ink overlap simply does
not distinguish a correct partial match from a coincidental dense one.

Segment matching does. E2 established this before any matcher existed: at the
true pose, cloud wall segments find a median of 12 counterparts on the sheet
within 30 cm and 10 degrees, and 18 splits with none at all are the ones with
no evidence to find. That measurement is what this module turns into a ranking.

So the search proposes and this disposes -- the split `docs/METHOD.md` calls
for. Inlier counting is far more discriminative than correlation but cannot be
evaluated over every translation at once, which is exactly why correlation
generates the candidates.

`count_inliers` below is the definition: rasterize the cloud AT THIS POSE, run
LSD on it, resample the sheet, distance-transform it, match. It is also 344 ms,
and nothing in it depends on the candidate except through a similarity applied
to segment endpoints. `InlierScorer` hoists all of it out -- 0.06 ms per
candidate, about 6000x -- which is what makes it affordable to rank EVERY
candidate instead of a pool. That matters more than the time saved: the pose
closest to truth was measured sitting at correlation score-rank 671 of 12,600,
far outside any pool worth paying for, so a pool was capping accuracy.
"""
from __future__ import annotations

import cv2
import numpy as np

from align import sheet as sheetmod
from align.frames import Sim2
from align.segments import (detect_segments, match_inliers, orientation_map,
                            rasterize)

TOL_M = 0.30          # how close a cloud wall must sit to plan line-work
ANG_TOL_DEG = 10.0    # the ground truth's own rotation is out by up to 14.5,
                      # so a tighter gate would reject correct matches
TOL_PX = 6.0          # cells across the tolerance; at 2 neither a wall's
                      # distance nor its orientation is measurable
MAX_RASTER_PX = 2500
PREPARED_MAX_PX = 3500  # the prepared grid covers every placement rather than
                        # one, so it needs more cells to hold the same
                        # resolution; it is built once, so it can afford them
MIN_SEG_LEN_PX = 8.0
N_SAMPLES = 12        # points sampled along each segment
MIN_FRAC = 0.7        # of those, the share that must agree

# How the cloud's wall segments are obtained. Both are exact in their own
# terms; they differ in what they treat as a property of the pose.
#
#   "pose"   reproduces `count_inliers` exactly -- LSD on the cloud rasterized
#            AT this yaw and scale. One run per distinct (yaw, scale).
#   "cloud"  one LSD for the whole split, on the unrotated cloud, with the
#            endpoints carried to each pose.
#
# "pose" is the definition and what the equivalence tests check. "cloud" is
# 253x fewer LSD runs and arguably the better measure -- a wall is a property of
# the building, not of the guess being tested, and re-detecting per pose makes
# the segment COUNT swing (333..362 on one split at a single scale) purely from
# raster aliasing, so two candidates are not scored against the same
# denominator.
#
# `scripts/compare_segment_modes.py` put the two side by side over every
# candidate of the four test splits. Reranking cost 3.1-55.1 s in "pose" and
# 0.5-1.7 s in "cloud", 12x to 42x, and the top-ranked pose landed within 5 m
# of the same place either way (67.6/67.6, 14.6/13.0, 31.1/31.9, 40.9/37.5 m
# from truth). No accuracy signal separates them, and one of them fits in the
# time budget, so: "cloud".
#
# Read that as weak evidence deliberately: the ranking is poor in BOTH modes --
# on three of the four splits the true pose scores no better than a candidate
# tens of metres away -- so "no measurable difference" partly means there was
# not much to measure. Switch back to "pose" if the ranking ever becomes good
# enough for the difference to show.
SEGMENT_MODE = "cloud"
CLOUD_KEY = (float("nan"), float("nan"))   # cache slot for the split-wide set


def _placed_bounds(xy, pad=0.5):
    lo = np.asarray(xy).min(0) - pad
    hi = np.asarray(xy).max(0) + pad
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def count_inliers(wall_xy, sim2, feats, geom, metres_per_inch,
                  tol_m: float = TOL_M, ang_tol_deg: float = ANG_TOL_DEG):
    """Wall segments of one placed cloud that the sheet actually accounts for.

    Rasterizes the cloud at this pose, extracts its segments, and matches them
    against the sheet's wall line-work by distance AND orientation. Returns
    (inliers, n_segments).

    This is the reference implementation and the definition of the measure;
    `InlierScorer` is checked against it. Prefer the scorer for anything
    scoring more than a handful of poses.
    """
    placed = sim2.apply(np.asarray(wall_xy))
    tol_in = tol_m / max(metres_per_inch, 1e-9)
    res = tol_in / TOL_PX
    bounds = _placed_bounds(placed)
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    if span / res > MAX_RASTER_PX:
        res = span / MAX_RASTER_PX
    ras = rasterize(placed, res=res, bounds=bounds, min_count=1)
    if ras.mask.shape[0] < 4 or ras.mask.shape[1] < 4:
        return 0, 0
    segs = detect_segments(ras.mask, min_len_px=MIN_SEG_LEN_PX)
    if len(segs) == 0:
        return 0, 0
    plan = sheetmod.resample_to(feats, geom, bounds, res, ras.mask.shape)
    if not plan.any():
        return 0, int(len(segs))
    inl, _ = match_inliers(segs, plan, tol_px=max(tol_in / res, 1.5),
                           ang_tol_deg=ang_tol_deg)
    return int(inl.sum()), int(len(segs))


class InlierScorer:
    """The same measure, with everything a pose does not change prepared once.

    Two observations make this exact rather than an approximation.

    The sheet's distance and orientation maps do not depend on the pose at all,
    so they are built once over a region covering every placement. The
    reference builds them over each placement's own bounding box, which clips
    line-work just outside it; the prepared maps see slightly more, and the
    only other difference is a resolution the union region may coarsen by a few
    percent -- far below the 30 cm tolerance either way.

    The cloud's segments depend on the pose only through YAW AND SCALE. The
    reference rasterizes over `_placed_bounds(placed)`, which tracks the cloud,
    so translating a pose shifts the raster's origin by exactly as much as the
    points and leaves the mask -- and therefore LSD's output -- unchanged. So
    segments are computed once per distinct (yaw, scale) and cached, with the
    reference's own resolution rule, and scoring adds a translation. A search
    emits a few hundred candidates per (yaw, scale); this turns hundreds of
    LSD runs into one.

    What remains per candidate is sampling two arrays, and `score_all` does
    even that for a whole (yaw, scale) group at once.
    """

    def __init__(self, wall_xy, feats, geom, metres_per_inch, bounds,
                 tol_m: float = TOL_M, ang_tol_deg: float = ANG_TOL_DEG,
                 segment_mode: str = SEGMENT_MODE,
                 reference_scale: float | None = None):
        self.wall = np.asarray(wall_xy, dtype=np.float64)[:, :2]
        self.segment_mode = segment_mode
        self.reference_scale = reference_scale
        self.ang_tol = float(np.radians(ang_tol_deg))
        self.tol_in = float(tol_m / max(metres_per_inch, 1e-9))

        x0, y0, x1, y1 = bounds
        span = max(x1 - x0, y1 - y0)
        self.res = max(self.tol_in / TOL_PX, span / PREPARED_MAX_PX)
        self.origin = np.array([x0, y0], dtype=np.float64)
        self.tol_px = max(self.tol_in / self.res, 1.5)
        shape = (max(int(np.ceil((y1 - y0) / self.res)), 1),
                 max(int(np.ceil((x1 - x0) / self.res)), 1))

        plan = sheetmod.resample_to(feats, geom, bounds, self.res, shape)
        self.shape = plan.shape
        self.empty = not plan.any()
        if self.empty:
            self.dist = np.full(plan.shape, 1e6, np.float32)
            self.ang = np.zeros(plan.shape, np.float32)
        else:
            solid = plan.astype(np.uint8)
            self.dist = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 3)
            self.ang = orientation_map(plan)

        self._cache: dict[tuple[float, float], np.ndarray] = {}
        self._t = np.linspace(0.0, 1.0, N_SAMPLES)

    # -- the cloud side, once per (yaw, scale) ----------------------------

    def segments_for(self, sim2) -> np.ndarray:
        """Wall segments of the cloud rotated and scaled by `sim2`, in inches.

        Not translated: translation is what varies within a cache entry.
        """
        if self.segment_mode == "cloud":
            return self._placed_once(sim2)
        key = (round(float(sim2.theta), 9), round(float(sim2.s), 9))
        segs = self._cache.get(key)
        if segs is None:
            self._cache[key] = segs = self._detect(sim2)
        return segs

    def _placed_once(self, sim2) -> np.ndarray:
        """One segment set for the whole split, carried to this pose."""
        base = self._cache.get(CLOUD_KEY)
        if base is None:
            s = self.reference_scale or float(sim2.s)
            flat = Sim2(s=s, theta=0.0, tx=0.0, ty=0.0)
            base = self._detect(flat) / s       # -> cloud units
            self._cache[CLOUD_KEY] = base
        c, sn = np.cos(sim2.theta), np.sin(sim2.theta)
        M = np.array([[c, sn], [-sn, c]]) * sim2.s
        return np.hstack([base[:, :2] @ M, base[:, 2:] @ M])

    def _detect(self, sim2) -> np.ndarray:
        c, s = np.cos(sim2.theta), np.sin(sim2.theta)
        placed = self.wall @ (np.array([[c, s], [-s, c]]) * sim2.s)
        bounds = _placed_bounds(placed)
        span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        res = max(self.tol_in / TOL_PX, span / MAX_RASTER_PX)
        ras = rasterize(placed, res=res, bounds=bounds, min_count=1)
        if min(ras.mask.shape) < 4:
            return np.zeros((0, 4))
        segs = detect_segments(ras.mask, min_len_px=MIN_SEG_LEN_PX)
        if len(segs) == 0:
            return np.zeros((0, 4))
        out = segs.astype(np.float64) * res
        out[:, [0, 2]] += ras.origin[0]
        out[:, [1, 3]] += ras.origin[1]
        return out

    # -- the per-candidate side -------------------------------------------

    def _count(self, segs, offsets) -> np.ndarray:
        """Inliers for one segment set under (M, 2) translations."""
        a = segs[None, :, None, :2] + offsets[:, None, None, :]
        b = segs[None, :, None, 2:] + offsets[:, None, None, :]
        pts = a + (b - a) * self._t[None, None, :, None]      # (M, S, n, 2)

        grid = (pts - self.origin) / self.res
        h, w = self.shape
        px = np.clip(grid[..., 0], 0, w - 1).astype(np.int64)
        py = np.clip(grid[..., 1], 0, h - 1).astype(np.int64)
        inside = ((grid[..., 0] >= 0) & (grid[..., 0] < w)
                  & (grid[..., 1] >= 0) & (grid[..., 1] < h))

        seg_ang = np.arctan2(segs[:, 3] - segs[:, 1],
                             segs[:, 2] - segs[:, 0]) % np.pi
        da = np.abs(self.ang[py, px] - seg_ang[None, :, None])
        da = np.minimum(da, np.pi - da)       # a wall has direction, not sense
        ok = inside & (self.dist[py, px] <= self.tol_px) & (da <= self.ang_tol)
        return (ok.mean(axis=2) >= MIN_FRAC).sum(axis=1)

    def score(self, sim2) -> tuple[int, int]:
        """(inliers, n_segments) for one pose."""
        segs = self.segments_for(sim2)
        if len(segs) == 0 or self.empty:
            return 0, int(len(segs))
        t = np.array([[sim2.tx, sim2.ty]], dtype=np.float64)
        return int(self._count(segs, t)[0]), int(len(segs))

    def score_all(self, hypotheses, chunk: int = 256) -> None:
        """Set `inliers` and `n_segments` on every hypothesis.

        Grouped by (yaw, scale) so each group runs one LSD and one vectorised
        sampling pass over its translations.
        """
        groups: dict[tuple[float, float], list] = {}
        for h in hypotheses:
            key = (round(float(h.sim2.theta), 9), round(float(h.sim2.s), 9))
            groups.setdefault(key, []).append(h)
        for poses in groups.values():
            segs = self.segments_for(poses[0].sim2)
            n = int(len(segs))
            if n == 0 or self.empty:
                for h in poses:
                    h.inliers, h.n_segments = 0, n
                continue
            # Chunked: the sample tensor is (M, S, 12, 2) and S runs to a few
            # hundred, so a whole group at once can be a gigabyte.
            for i in range(0, len(poses), chunk):
                part = poses[i:i + chunk]
                offs = np.array([[p.sim2.tx, p.sim2.ty] for p in part], dtype=np.float64)
                counts = self._count(segs, offs)
                for h, c in zip(part, counts):
                    h.inliers, h.n_segments = int(c), n


def build_scorer(hypotheses, wall_xy, feats, geom, metres_per_inch,
                 tol_m: float = TOL_M, ang_tol_deg: float = ANG_TOL_DEG,
                 pad: float = 0.5, segment_mode: str = SEGMENT_MODE
                 ) -> InlierScorer:
    """One scorer whose region covers every placement these poses produce."""
    step = max(len(wall_xy) // 400, 1)
    sample = np.asarray(wall_xy)[::step]
    corners = np.vstack([h.sim2.apply(sample) for h in hypotheses])
    return InlierScorer(wall_xy, feats, geom, metres_per_inch,
                        _placed_bounds(corners, pad), tol_m=tol_m,
                        ang_tol_deg=ang_tol_deg, segment_mode=segment_mode,
                        reference_scale=float(np.median(
                            [h.sim2.s for h in hypotheses])))


def rerank(hypotheses, wall_xy_by_up, feats, geom, metres_per_inch,
           limit: int | None = None, tol_m: float = TOL_M,
           ang_tol_deg: float = ANG_TOL_DEG,
           segment_mode: str = SEGMENT_MODE):
    """Score the proposals by inlier count and re-sort.

    `wall_xy_by_up` maps a hypothesis's `up_index` to that gravity hypothesis's
    wall points, since a split with an ambiguous floor carries two. `limit`
    exists for callers that still want a pool; leaving it None scores
    everything, which is the point of `InlierScorer` -- the pose closest to
    truth has been measured at correlation score-rank 671, outside any pool.

    Ties on inliers fall back to the correlation score, which is a reasonable
    tie-break even though it is a poor ranking on its own.
    """
    considered = hypotheses if limit is None else hypotheses[:limit]
    by_up: dict[int, list] = {}
    for h in considered:
        by_up.setdefault(getattr(h, "up_index", 0), []).append(h)

    for up, poses in by_up.items():
        wall = wall_xy_by_up.get(up)
        if wall is None:
            continue
        build_scorer(poses, wall, feats, geom, metres_per_inch,
                     tol_m=tol_m, ang_tol_deg=ang_tol_deg,
                     segment_mode=segment_mode).score_all(poses)

    ranked = sorted(hypotheses, key=lambda h: (-h.inliers, -h.score))
    for i, h in enumerate(ranked):
        h.rank = i
    return ranked
