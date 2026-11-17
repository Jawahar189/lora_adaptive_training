"""Run every stage for ONE reality, in one pass, from Python.

The stage scripts are stage-major: each loops over all 132 splits doing a
single stage, so one reality's cloud gets read four times and its floor fitted
four times across E1, E2 and the visualiser -- and nothing can be stepped
through a split at a time.

This is the other axis. Construct a pipeline for one split and call stages
individually or all at once:

    p = RealityPipeline(case)
    p.gravity()        # or jump straight to
    p.search()         # which resolves everything it needs first
    p.run()

Every stage is memoized, so calling one twice is free and calling a late stage
from cold works. Every intermediate stays on the object, which is the point:
break anywhere and inspect anything.

Sheet work is per STRUCTURE and already cached on disk, so it is resolved once
and shared rather than repeated per split.

Leakage boundary: nothing here touches ground truth at all. The trajectory is
still a constructor argument, but its default is now `align.io.shots.load_shots`
-- production's own per-image SfM poses, read from the file the reconstruction
stage writes. It used to be synthesized by un-applying the ground-truth
transform, which was sanctioned but was the last path from ground truth into
solver inputs; the two agree to 1.2e-9 m over all 273 splits
(`scripts/check_shots.py`), so the swap costs nothing and closes that path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from align import cloud as cloudmod
from align import rerank as rerankmod
from align import search as searchmod
from align import sheet as sheetmod
from align.frames import compose_tm, design_scale
from align.io.shots import load_shots
from align.gravity import (CAMERA_HEIGHT_M, GravityFit, _rotation_to_z,
                           estimate_gravity, floor_below_trajectory,
                           scale_prior)
from align.scales import FALLBACK_METRES_PER_INCH, SheetScales
from align.search import Hypothesis, SearchGrid

# Metres. The coarse cell: a tolerance must span several cells or neither a
# wall's distance nor its orientation is measurable -- at 2 px across the
# tolerance, two test splits scored 0 and 29 oracle inliers; at 6 px the same
# two scored 73 and 111.
COARSE_CELL_M = 0.20
TOL_M = 0.30
MAX_RASTER_PX = 4000

# How many correlation proposals the segment reranker gets to look at.
# None means all of them, which is the right answer and used to be unaffordable:
# reranking cost 344 ms a candidate, so the pool swept 60 -> 16.0 m, 300 ->
# 12.6 m, 1200 -> 12.8 m from truth while the cost went 99 s -> 173 s -> 473 s.
# But the pose closest to truth on that split sat at correlation score-rank 671
# of 12,600 -- the plateau was the pool running out of candidates, not the
# ranking converging. `align.rerank.InlierScorer` prepares per (yaw, scale)
# instead of per candidate, so the whole set now costs a few seconds and there
# is no reason to throw any of it away.
RERANK_POOL = None

# Search the union of the sheet's viewports rather than each one.
# The resolution is max(cell, span / MAX_RASTER_PX), which is the cell either
# way on every sheet measured, so the union loses no resolution -- and it
# already contains each viewport, so it loses no coverage. Measured: 29 s
# against 78.6 s for five viewports on the same split. Set False for a sheet
# whose plans sit far enough apart that the union is mostly blank paper.
UNION_VIEWPORT = True


def _stage(fn):
    """Memoize a stage on the instance under `_<name>`."""
    key = f"_{fn.__name__}"

    def wrapper(self, *a, **kw):
        if getattr(self, key, None) is None or kw.get("force"):
            kw.pop("force", None)
            setattr(self, key, fn(self, *a, **kw))
        return getattr(self, key)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@dataclass
class SearchResult:
    reality_id: str
    structure: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    metres_per_inch: float = float("nan")
    scale_trusted: bool = True
    up_indices: list = field(default_factory=list)
    error: str | None = None
    # Set by align.chain when the pose came from a chain rather than a search.
    chain: dict | None = None

    @property
    def best(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None


class RealityPipeline:
    """All stages for one split. Each method is a stage; each is memoized."""

    def __init__(self, case, trajectory=None, scales: SheetScales | None = None,
                 stride: int = 3, cell_m: float = COARSE_CELL_M,
                 backend=None, trajectory_fn: Callable = load_shots,
                 prior=None, rerank: bool = True,
                 rerank_pool: int | None = RERANK_POOL,
                 union_viewport: bool = UNION_VIEWPORT):
        self.case = case
        self.structure = case.structure
        self.stride = stride
        self.cell_m = cell_m
        self._scales = scales
        self._backend = backend
        self._explicit_traj = trajectory
        # A TranslationPrior from the chain, saying where a known camera should
        # land. Restricts the translation search to a small disc, which is
        # where the unaided score fails.
        self.prior = prior
        # Segment matching is far more discriminative than the correlation but
        # cannot be evaluated over every translation, so the correlation
        # proposes and this judges -- now every proposal, not a pool.
        self.rerank = rerank
        self.rerank_pool = rerank_pool
        self.union_viewport = union_viewport
        self._trajectory_fn = trajectory_fn

        self._cloud = None
        self._trajectory = None
        self._sheet = None
        self._gravity = None
        self._floor = None
        self._scale_prior = None
        self._wall_points = None
        self._grid = None
        self._search = None
        self._views = None

    # -- inputs ------------------------------------------------------------
    @property
    def metres_per_inch(self) -> float:
        if self.structure.design_tm is not None:
            return design_scale(self.structure.design_tm)
        if self._scales is not None:
            return self._scales[self.structure.key]
        return FALLBACK_METRES_PER_INCH

    @property
    def scale_trusted(self) -> bool:
        if self.structure.design_tm is not None:
            return True
        return self._scales.is_trusted(self.structure.key) if self._scales else False

    @_stage
    def trajectory(self):
        """Camera positions in the arbitrary ply frame."""
        if self._explicit_traj is not None:
            return self._explicit_traj
        return self._trajectory_fn(self.case.rlt_dir)

    @_stage
    def cloud(self):
        return cloudmod.load(self.case.cloud_path, stride=self.stride)

    @_stage
    def sheet(self):
        st = self.structure
        return sheetmod.extract_cached(st.sheet_path, st.paper_w, st.paper_h,
                                       st.key, self.metres_per_inch)

    # -- geometry ----------------------------------------------------------
    @_stage
    def gravity(self) -> GravityFit:
        return estimate_gravity(self.cloud().xyz, self.trajectory().positions)

    # -- gravity hypotheses ------------------------------------------------
    # Where the floor/ceiling test ties, BOTH up-vectors are carried through to
    # scoring. This is not a nicety: an inverted up-vector mirrors the top-down
    # projection, and a mirrored building cannot match any floor plan at any
    # pose. 49 of 132 splits are ambiguous, and picking one silently is wrong
    # about half of those by construction -- measured, 2 of 6 sampled splits
    # had the sign the ground truth disagrees with.
    def ups(self) -> list[np.ndarray]:
        return self.gravity().alternatives()

    def view(self, up_index: int = 0) -> dict:
        """Everything downstream of one choice of up-vector, memoized."""
        if self._views is None:
            self._views = {}
        if up_index in self._views:
            return self._views[up_index]
        up = self.ups()[up_index]
        R = _rotation_to_z(up)
        P = self.cloud().xyz @ R.T
        C = self.trajectory().positions @ R.T
        floor = floor_below_trajectory(P, C)
        v = dict(up=up, R=R, floor=floor)
        if np.isfinite(floor.height) and floor.height > 0:
            s, (lo, hi) = scale_prior(floor, CAMERA_HEIGHT_M)
            mpi = self.metres_per_inch
            v["scale_prior"] = (s / mpi, (lo / mpi, hi / mpi))
            band = cloudmod.wall_band(P, C, floor)
            if band.sum() >= cloudmod.MIN_BAND_POINTS:
                v["wall_xy"] = P[band][:, :2]
                v["traj_xy"] = C[:, :2]
        self._views[up_index] = v
        return v

    @_stage
    def floor(self):
        return self.view(0)["floor"]

    @_stage
    def scale_prior(self):
        """Paper inches per cloud unit, and the bracket to search."""
        v = self.view(0)
        if "scale_prior" not in v:
            raise ValueError("no usable floor; cannot derive a scale")
        return v["scale_prior"]

    @_stage
    def wall_points(self):
        """Banded wall points and the walk, in the primary gravity frame."""
        v = self.view(0)
        if "wall_xy" not in v:
            raise ValueError("wall band is empty")
        return v["wall_xy"], v["traj_xy"]

    # -- search ------------------------------------------------------------
    def grid_for(self, up_index: int = 0) -> SearchGrid:
        v = self.view(up_index)
        if "wall_xy" not in v:
            raise ValueError("wall band is empty")
        s0, bracket = v["scale_prior"]
        cell = self.cell_m / self.metres_per_inch          # inches per cell
        ras = cloudmod.wall_raster(v["wall_xy"] * s0, res=cell)
        cm = cloudmod.manhattan(ras.mask)
        sm = cloudmod.manhattan(self.sheet().wall[::4, ::4])
        yaws, fallback = searchmod.yaw_candidates(
            cm.angle, sm.angle, strength=min(cm.strength, sm.strength))
        return SearchGrid(scales=searchmod.scale_candidates(bracket),
                          yaws_deg=yaws, viewports=self.viewports(),
                          fallback_yaw=fallback)

    def viewports(self) -> list:
        """The regions to correlate over."""
        vps = self.sheet().viewports
        if not self.union_viewport or len(vps) <= 1:
            return vps
        return [sheetmod.Viewport(
            x0=min(v.x0 for v in vps), y0=min(v.y0 for v in vps),
            x1=max(v.x1 for v in vps), y1=max(v.y1 for v in vps),
            wall_mass=float(sum(v.wall_mass for v in vps)))]

    @_stage
    def grid(self) -> SearchGrid:
        return self.grid_for(0)

    @_stage
    def search(self) -> SearchResult:
        res = SearchResult(reality_id=self.case.reality_id,
                           structure=self.structure.key,
                           metres_per_inch=self.metres_per_inch,
                           scale_trusted=self.scale_trusted)
        geom = self.structure.geometry()
        cell = self.cell_m / self.metres_per_inch
        bk = self._backend or searchmod.get_backend()
        usable = [i for i in range(len(self.ups())) if "wall_xy" in self.view(i)]
        if not usable:
            res.error = "no usable floor or wall band for any gravity hypothesis"
            return res
        res.up_indices = usable

        for ui in usable:
            v = self.view(ui)
            wall_xy, traj_xy = v["wall_xy"], v["traj_xy"]
            s_prior, _ = v["scale_prior"]
            grid = self.grid_for(ui)
            self._search_one(res, wall_xy, traj_xy, grid, s_prior, geom, cell,
                             bk, ui)

        # Correlation proposed; segment matching decides -- over EVERY
        # proposal, because the correlation ranking is weak enough that the
        # right pose is routinely hundreds deep in its list.
        res.hypotheses.sort(key=lambda h: -h.score)
        if self.rerank_pool is not None:
            res.hypotheses = res.hypotheses[: self.rerank_pool]
        if self.rerank:
            res.hypotheses = rerankmod.rerank(
                res.hypotheses,
                {i: self.view(i)["wall_xy"] for i in usable},
                self.sheet(), geom, self.metres_per_inch)
        res.hypotheses = res.hypotheses[: searchmod.TOP_K_TOTAL]
        for i, h in enumerate(res.hypotheses):
            h.rank = i
        return res

    def _search_one(self, res, wall_xy, traj_xy, grid, s_prior, geom, cell,
                    bk, up_index):
        for vi, vp in enumerate(grid.viewports):
            lo = geom.pixel_to_paper(np.array([[vp.x0, vp.y1]]))[0]
            hi = geom.pixel_to_paper(np.array([[vp.x1, vp.y0]]))[0]
            bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
            span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            vres = max(cell, span / MAX_RASTER_PX)
            shape = (max(int((bounds[3] - bounds[1]) / vres), 1),
                     max(int((bounds[2] - bounds[0]) / vres), 1))
            smap = sheetmod.resample_to(self.sheet(), geom, bounds, vres,
                                        shape).astype(np.float32)
            if not smap.any():
                continue
            res.hypotheses += searchmod.search(
                wall_xy, traj_xy, smap, (bounds[0], bounds[1]), vres, grid,
                viewport_index=vi, backend=bk, scale_prior=s_prior,
                prior=self.prior, up_index=up_index)

    # -- convenience -------------------------------------------------------
    def tm(self, hypothesis: Hypothesis | None = None) -> np.ndarray | None:
        """Compose a hypothesis into the final 4x4, ply -> target frame."""
        h = hypothesis or self.search().best
        if h is None:
            return None
        R = self.view(getattr(h, "up_index", 0))["R"]
        ply_to_paper = h.sim2.to_sim3(tilt_R=R, tz=0.0)
        return compose_tm(ply_to_paper, self.structure.design_tm)

    def run(self) -> SearchResult:
        """Every stage, in order."""
        self.search()
        return self._search
