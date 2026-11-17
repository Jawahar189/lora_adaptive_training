"""Gravity, floor height, and the metric scale anchor.

The point cloud's scale is arbitrary (ground-truth tm scales span 0.168..122.3,
2.9 decades), which makes a blind scale search hopeless. But a 360 camera on a
monopod sits at a near-constant height above the floor, so measuring
camera-height-above-floor in cloud units and dividing by that constant pins the
scale to within ~15%.

The floor must be measured LOCALLY under the trajectory. A global z-histogram
picks up outdoor ground, mezzanines and ceilings and is useless.

Everything here is scale-free: distances are expressed as fractions of the
trajectory's own horizontal extent until a scale is applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# Metres. Measured by scripts/e1_scale_anchor.py over all 132 splits, on the
# georeferenced subset where the target frame is true metres, restricted to
# splits whose floor/ceiling sign test was decisive and excluding
# STR1756226440646 (whose own ground truth contradicts itself by 14.6 m):
#   n=27, p10 1.88, median 2.11, p90 2.66  -- spread -11%/+26%
CAMERA_HEIGHT_M = 2.11

# A fit this dispersed is not measuring a surface. Shared with align.reference,
# which is where this threshold was originally measured and validated
# (`align.reference.MAX_REL_MAD` re-exports this).
MAX_REL_MAD = 0.50


@dataclass
class GravityFit:
    up: np.ndarray          # unit vector, cloud frame, pointing away from the floor
    R: np.ndarray           # 3x3 rotating the cloud so `up` -> +z
    planarity: float        # trajectory plane-fit residual / extent; small is good
    method: str
    sign_ambiguous: bool = False   # floor and ceiling are indistinguishable
    floor: "FloorFit | None" = None
    # Both candidates' fits, as ((up, FloorFit), ...). `resolve_up_sign` has to
    # fit BOTH signs to choose between them, and whoever goes on to build a
    # view for one of those signs would otherwise fit it a second time -- over
    # the same cloud, for the same answer. Fitting a floor means a kd-tree over
    # the whole cloud, so that repeat is the most expensive thing a capture does.
    sign_fits: tuple = ()

    def floor_for(self, up: np.ndarray) -> "FloorFit | None":
        """The fit already computed for this up vector, or None if there is none."""
        for candidate, fit in self.sign_fits:
            if np.array_equal(candidate, up):
                return fit
        return None

    def alternatives(self) -> list[np.ndarray]:
        """Up-vector hypotheses to carry forward. Normally one; two when the
        floor/ceiling test ties, in which case the matcher decides rather than
        this module guessing."""
        return [self.up, -self.up] if self.sign_ambiguous else [self.up]


@dataclass
class FloorFit:
    height: float           # camera height above floor, in cloud units
    mad: float              # median absolute deviation across cameras, same units
    n_used: int
    per_camera: np.ndarray
    beyond: float = np.nan      # fraction of nearby points lying PAST the surface

    @property
    def rel_mad(self) -> float:
        return float(self.mad / self.height) if self.height > 0 else np.inf

    @property
    def quality(self) -> float:
        """Low `beyond` means this really is a floor: nothing lies beneath a
        slab, whereas roof structure sits above a ceiling. Measured on the 27
        dataset_28 splits, the true floor scores <=0.008 and the ceiling
        typically >=0.03."""
        if not np.isfinite(self.height) or self.height <= 0:
            return -np.inf
        return float(-self.beyond)


def _rotation_to_z(up: np.ndarray) -> np.ndarray:
    """Rotation taking `up` to +z, via the shortest arc."""
    u = up / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    c = float(np.dot(u, z))
    # Near-antiparallel too, not only exactly: a cloud stored already z-up hands
    # in -z with a cross product of ~1e-9, where 1 + c rounds to exactly 0.
    if np.linalg.norm(v) < 1e-12 or 1.0 + c < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def trajectory_extent(traj_xy: np.ndarray) -> float:
    """Horizontal size of the walk, used as the cloud's own length unit."""
    lo, hi = np.percentile(traj_xy[:, :2], [2, 98], axis=0)
    return float(max(np.linalg.norm(hi - lo), 1e-9))


def estimate_gravity(cloud_xyz: np.ndarray, traj_xyz: np.ndarray) -> GravityFit:
    """Up vector from the trajectory plane, with a point-normal fallback.

    A person walking a floor traces a plane; its normal is gravity. When the
    walk is not planar (ramps, stairs, multi-storey) the residual flags it and
    we fall back to the consensus of near-vertical surface normals.
    """
    C = np.asarray(traj_xyz, dtype=np.float64)
    ctr = C.mean(0)
    _, sv, vt = np.linalg.svd(C - ctr, full_matrices=False)
    extent = trajectory_extent(C)
    planarity = float(sv[2] / np.sqrt(max(len(C), 1)) / extent)
    up = vt[2]
    method = "trajectory-plane"

    if planarity > 0.02:
        up_alt = _gravity_from_normals(cloud_xyz)
        if up_alt is not None:
            if np.dot(up_alt, up) < 0:
                up_alt = -up_alt
            up, method = up_alt, "surface-normals"

    up = up / np.linalg.norm(up)
    up, ambiguous, floor, sign_fits = resolve_up_sign(cloud_xyz, traj_xyz, up)
    return GravityFit(up=up, R=_rotation_to_z(up), planarity=planarity,
                      method=method, sign_ambiguous=ambiguous, floor=floor,
                      sign_fits=sign_fits)


def assumed_z_up(cloud_xyz: np.ndarray, traj_xyz: np.ndarray) -> GravityFit:
    """The cloud's own +z taken as up, no estimate: for clouds stored z-up already (v9/v10, whose
    verified tm.json is level), where the estimate can only lose -- it turns some upside down or
    on their side. Only the floor under the cameras is fitted, for the camera height."""
    up = np.array([0.0, 0.0, 1.0])
    floor = floor_below_trajectory(np.asarray(cloud_xyz), np.asarray(traj_xyz))
    return GravityFit(up=up, R=np.eye(3), planarity=float("nan"), method="assumed z-up",
                      sign_ambiguous=False, floor=floor, sign_fits=((up, floor),))


def known_up(cloud_xyz: np.ndarray, traj_xyz: np.ndarray, up: np.ndarray) -> GravityFit:
    """A known up vector, cloud frame (a registered capture's, from its tm.json): no estimate, the floor
    fitted under it for the camera height."""
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    R = _rotation_to_z(up)
    floor = floor_below_trajectory(np.asarray(cloud_xyz) @ R.T, np.asarray(traj_xyz) @ R.T)
    return GravityFit(up=up, R=R, planarity=float("nan"), method="known up",
                      sign_ambiguous=False, floor=floor, sign_fits=((up, floor),))


# Below this gap in `beyond` the two candidate surfaces are indistinguishable.
SIGN_TIE_TOL = 0.004

# Once both up-vector hypotheses reach search, `beyond` has no residual power
# to break the tie -- checked over 91 sign-ambiguous realities in deploy2,
# the lower-`beyond` view matches ground truth only 57% of the time, barely
# above chance. The search score margin between the two views' best
# hypotheses is a real, if overlapping, signal instead: median 0.059 when the
# wrong up_index wins, 0.116 when the right one does. At this threshold it
# catches 61/75 wrong picks, at a cost of flagging 4/16 correct ones too.
SIGN_MARGIN_MIN = 0.10

# `beyond` ties on that same population because a real floor and its ceiling
# mirror can both come back clean in a flat-roofed building -- but the FIT
# itself is not equally good: a true floor gives every sampled camera nearly
# the same height, while "floor" fitted against roof structure (trusses,
# ducts, sloped decking) does not. `FloorFit.rel_mad` measures exactly that
# and, checked over the same 91 realities, matches ground truth 77/91 (85%)
# just by preferring the lower one -- vs. 57% for `beyond` and worse than
# chance for Manhattan wall-rectilinearity or raw wall-point count, both
# tried and rejected. At this gap it resolves 64 of the 91 outright (skipping
# the dual search these otherwise fall through to, where the wrong up_index
# wins 61/75 of the time): 58 correct, 6 wrong (91%).
REL_MAD_TIE_TOL = 0.01


def resolve_up_sign(cloud_xyz, traj_xyz, up) -> tuple[np.ndarray, bool, "FloorFit", tuple]:
    """Decide which end of the gravity axis is up.

    Mass-below-the-trajectory fails badly in tall industrial and airport spaces,
    where ceilings and roof structure carry more points than the slab. The
    reliable fact is that nothing lies beneath a floor, so fit a surface for
    both signs and keep the one with less mass past it.

    No distance rule works here: in a hangar the floor is much nearer than the
    ceiling, while in a 3.6 m room a camera at 2.1 m is nearer the ceiling. When
    `beyond` genuinely ties -- a flat-roofed building can look clean from
    either side -- `rel_mad` gets a second vote: a true floor gives every
    sampled camera nearly the same height, roof structure does not. Only when
    both tie is it a real ambiguity, said so, and left for the floorplan
    match to decide.
    """
    fits = []
    for cand in (up, -up):
        R = _rotation_to_z(cand)
        fits.append((cand, floor_below_trajectory(np.asarray(cloud_xyz) @ R.T,
                                                  np.asarray(traj_xyz) @ R.T)))
    (u0, f0), (u1, f1) = fits
    both = tuple(fits)            # kept so neither sign is ever fitted twice
    if not np.isfinite(f0.height):
        return u1, False, f1, both
    if not np.isfinite(f1.height):
        return u0, False, f0, both
    if abs(f0.beyond - f1.beyond) >= SIGN_TIE_TOL:
        return (u0, False, f0, both) if f0.beyond <= f1.beyond else (u1, False, f1, both)
    if abs(f0.rel_mad - f1.rel_mad) >= REL_MAD_TIE_TOL:
        return (u0, False, f0, both) if f0.rel_mad <= f1.rel_mad else (u1, False, f1, both)
    return u0, True, f0, both


def _gravity_from_normals(cloud_xyz: np.ndarray, n_sample: int = 30000):
    """Gravity as the direction least represented among local surface normals:
    walls dominate, and every wall normal is perpendicular to up."""
    P = np.asarray(cloud_xyz)
    if len(P) < 500:
        return None
    idx = np.random.default_rng(0).choice(len(P), min(n_sample, len(P)), replace=False)
    S = P[idx]
    tree = cKDTree(S)
    _, nbr = tree.query(S, k=min(20, len(S)))
    d = S[nbr] - S[nbr].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", d, d)
    w, v = np.linalg.eigh(cov)
    normals = v[:, :, 0]                       # smallest-eigenvalue direction
    planar = w[:, 1] > 8.0 * w[:, 0]           # keep well-defined surfaces
    if planar.sum() < 100:
        return None
    N = normals[planar]
    M = np.einsum("ni,nj->ij", N, N)           # sum of outer products
    wv, vv = np.linalg.eigh(M)
    return vv[:, 0]                            # direction most orthogonal to all normals


def floor_below_trajectory(cloud_xyz_up: np.ndarray, traj_xyz_up: np.ndarray,
                           radius_frac: float = 0.06) -> FloorFit:
    """Camera height above the local floor, in cloud units.

    For every camera, histogram the drop to points beneath it within a local
    radius and take the nearest strong mode -- that is the floor under the
    walker's feet, not the terrain outside or the slab two storeys down.
    """
    P = np.asarray(cloud_xyz_up, dtype=np.float64)
    C = np.asarray(traj_xyz_up, dtype=np.float64)
    L = trajectory_extent(C)
    radius = max(radius_frac * L, 1e-6)

    # Half this function used to be the tree BUILD, and the default build pays
    # for query speed this never uses: it is thrown away after ~120 ball
    # queries. An unbalanced, non-compact tree builds twice as fast, answers
    # those queries just as quickly (checked on clustered clouds too) and
    # returns the same neighbour sets -- and every use below (a histogram, a
    # percentile, a mean of 0/1) reads a set, not an order.
    tree = cKDTree(P[:, :2], balanced_tree=False, compact_nodes=False)
    step = max(1, len(C) // 120)          # 120 cameras is plenty and keeps this fast
    sampled = range(0, len(C), step)
    # The neighbourhood of a camera depends on the radius, not on the window,
    # and the window is the only thing the second pass changes -- so the ball
    # queries and the drops they imply are gathered once and both passes
    # histogram the same arrays. Asked for every camera at once, scipy walks
    # the tree in C instead of once per camera from Python.
    neighbourhoods = tree.query_ball_point(C[list(sampled), :2], radius)
    neigh: list[np.ndarray] = []
    for camera_index, idx in zip(sampled, neighbourhoods):
        if len(idx) < 40:
            continue
        neigh.append(C[camera_index, 2] - P[idx, 2])

    heights: list[float] = []
    # Pass 1: a generous window to get the scale, then pass 2 refines around it.
    window = (0.0, 0.35 * L)
    for _ in range(2):
        heights = []
        for all_drop in neigh:
            drop = all_drop[(all_drop > window[0]) & (all_drop < window[1])]
            if len(drop) < 25:
                continue
            hist, edges = np.histogram(drop, bins=60)
            k = int(np.argmax(hist))
            if hist[k] < 10:
                continue
            heights.append(0.5 * (edges[k] + edges[k + 1]))
        if not heights:
            return FloorFit(float("nan"), float("nan"), 0, np.array([]))
        h0 = float(np.median(heights))
        window = (0.45 * h0, 1.7 * h0)

    arr = np.asarray(heights)
    h = float(np.median(arr))
    mad = float(np.median(np.abs(arr - h)))

    # Mass lying past the detected surface, with the margin scaled by the local
    # vertical extent so it means the same thing in a corridor and in a hangar.
    frac = []
    for d in neigh:
        span = np.percentile(d, 98) - np.percentile(d, 2)
        frac.append(float((d > h + 0.08 * span).mean()))
    beyond = float(np.median(frac)) if frac else 1.0
    return FloorFit(height=h, mad=mad, n_used=len(arr), per_camera=arr, beyond=beyond)


def usable_floor(floor: FloorFit, max_rel_mad: float = MAX_REL_MAD) -> str | None:
    """None when a floor fit is trustworthy, else why it isn't.

    Scale-free by design: `rel_mad` compares the per-camera spread to the fitted
    height itself, so this catches an internally-inconsistent (likely
    multi-modal, wrong-surface) fit before it ever reaches a metric estimate.
    Mirrors the first two checks of `align.reference._usable`; deliberately
    excludes `floor.beyond` for the same reason that function does not gate on
    it -- a coherent fit on a real but wrong surface looks identical to a
    coherent fit on the true floor by that measure.
    """
    if not np.isfinite(floor.height) or floor.height <= 0:
        return "no floor mode"
    if floor.rel_mad > max_rel_mad:
        return f"floor rel_mad {floor.rel_mad:.2f}"
    return None


def corrected_floor_height(floor: FloorFit, scale: float, metres_per_inch: float,
                           camera_height_m: float = CAMERA_HEIGHT_M
                           ) -> "FloorFit | None":
    """The floor height that would make this rig plausible, given a solved scale.

    The current side cannot check a floor fit against the known monopod height
    directly -- cloud units are arbitrary until a scale exists, and the scale
    itself is derived from the (possibly wrong) fit, which is exactly the
    chicken-and-egg the prior side does not have (see align.reference.build).
    Once a scale has been searched, though, the same physical fact the prior
    side substitutes with can be inverted: this is the height that would put the
    camera at `camera_height_m` given that scale, everything else about the
    fit (dispersion, sample count, per-camera detail) carried through unchanged.
    """
    if not np.isfinite(scale) or scale <= 0:
        return None
    if not np.isfinite(metres_per_inch) or metres_per_inch <= 0:
        return None
    h = camera_height_m / (scale * metres_per_inch)
    return FloorFit(height=h, mad=floor.mad, n_used=floor.n_used,
                    per_camera=floor.per_camera, beyond=floor.beyond)


def scale_prior(floor: FloorFit, camera_height: float = CAMERA_HEIGHT_M
                ) -> tuple[float, tuple[float, float]]:
    """Target-frame units per cloud unit, plus the bracket to search.

    For georeferenced structures the target is metres and this is the scale
    directly. For paper-frame structures the same number is inches-per-cloud-unit.
    """
    if not np.isfinite(floor.height) or floor.height <= 0:
        return float("nan"), (float("nan"), float("nan"))
    s = camera_height / floor.height
    lo, hi = (0.65, 1.60) if floor.rel_mad > 0.15 else (0.85, 1.30)
    return s, (s * lo, s * hi)


def relative_scale_prior(floor: FloorFit, reference_height: float,
                         reference_rel_mad: float = 0.0
                         ) -> tuple[float, tuple[float, float]]:
    """Paper inches per cloud unit, from the SAME rig measured in two frames.

    `scale_prior` divides the 2.11 m monopod by the fitted cloud height, and the
    caller then divides by metres-per-inch to reach paper -- so the result is
    only as good as the sheet-scale estimate, which on a paper-frame structure
    is itself a guess (measured spread within one structure: up to 6.95x).

    When the reference is a registered prior capture, its rig height is already
    known IN PAPER INCHES, and the constant cancels:

        s = (H / h_cloud) / (H / h_prior) = h_prior / h_cloud

    What is left is a ratio of two measurements of one monopod, with no scale
    estimate and no 2.11 in it at all. The bracket widens on the worse of the
    two floor fits, since either one being loose loosens the ratio.
    """
    if (not np.isfinite(floor.height) or floor.height <= 0
            or not np.isfinite(reference_height) or reference_height <= 0):
        return float("nan"), (float("nan"), float("nan"))
    s = reference_height / floor.height
    worst = max(floor.rel_mad, reference_rel_mad)
    lo, hi = (0.65, 1.60) if worst > 0.15 else (0.85, 1.30)
    return s, (s * lo, s * hi)
