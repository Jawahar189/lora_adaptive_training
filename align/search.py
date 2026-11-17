"""Coarse pose search: a scale bracket and a yaw estimate in, ranked poses out.

The score has three terms, and the middle one is the reason this works:

    score(s, theta, viewport, t) =
          w_wall   * maskedNCC(cloud_wall, sheet_wall)[t]
        - w_free   * (sheet wall mass under the translated walk)[t]
        + w_aspect * oriented-bbox aspect agreement          (per hypothesis)

Measured, in that order:

  * Plain binary correlation puts the peak on the title block and lands ~25
    inches from truth on 5 of 6 test cases. Normalising by the local energy
    under the template footprint -- the masked NCC below -- drops 4 of those to
    1.8-7.8 inches. Normalisation is not a refinement here, it is the term.

  * A blind search over scale and yaw still got only 1 of 3 right; the other
    two locked onto a 90-degree-rotated, 1.36x-rescaled pose, because filled
    occupancy overlap is nearly invariant to exactly that transformation on a
    rectangular building. Scoring where the WALK lands breaks it: a person does
    not walk through walls, and that is information the cloud alone does not
    carry.

    The walk term has two halves, and both are needed. Penalising ink under
    the walk alone is worse than useless: blank paper carries no ink, so the
    lowest-penalty placement is off the drawing entirely. Measured on a real
    split, the top-ranked pose scored the BEST walk penalty of any candidate
    (0.006) precisely because it sat on empty sheet, 114 m from truth. So the
    term is "inside the building AND not through its walls", not just the
    second half.


This module proposes; it does not decide. `segments.match_inliers` is far more
discriminative but cannot be evaluated over every translation at once, so the
correlation generates candidates and inlier counting ranks them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from align.frames import Sim2

# Weights are NOT fitted -- 27 splits is far too few to fit three numbers
# without memorising them. They come from the measured separations above, and
# scripts/search.py sweeps each over a decade to show the ranking is stable.
W_WALL = 1.0
W_FREE = 0.5
# Not a small prior: this is what cancels masked NCC's scale bias (see above).
W_SCALE = 1.0
# Off by default: unmeasured (see `search`'s aspect-crop docstring for why this
# is a LOCAL term, not the global one aspect_agreement's own callers avoided).
W_ASPECT = 0.0
# Log-scale spread of the E1 camera-height prior: -11%/+26% is about +-0.18 in
# log terms, so a hypothesis a full bracket-width from the prior is penalised
# but not excluded.
SCALE_SIGMA_LOG = 0.18

SOFT_PRIOR_WEIGHT = 0.5       # how much a translation is favoured near the prior
TOP_K_PER_HYPOTHESIS = 5      # peaks kept from one score map
TOP_K_TOTAL = 20              # hypotheses carried forward overall
NMS_RADIUS = 3                # cells, when picking peaks out of a score map


@dataclass
class TranslationPrior:
    """Where a known part of the walk is expected to land on the sheet.

    The constraint is on POINTS, not on the translation itself: the chain says
    "the camera that was standing here is somewhere near this spot", and what
    that permits of the translation depends on the scale and yaw being tried.
    `ref` holds those cloud points in the cloud's own arbitrary frame, one row
    each; `centre` and `radius` are in paper units.

    `ref` is a WINDOW rather than a single camera because a surveyed start
    point describes a moment, not a frame. Measured over the 57 captures whose
    named photograph could be located, in paper inches:

        reference set        err med   err p90   extent med   extent p90
        the named image         0.44      1.65         0.00         0.00
        +-20 images             0.36      1.47         1.88         3.27
        +-50 images             0.27      1.23         4.23         7.91
        the whole split         0.27      1.21         8.86        20.71

    The window beats the exact camera on error -- it absorbs the slop between
    the surveyed moment and the exact frame -- while keeping the admissible
    region tight. Widening further buys almost nothing (p90 1.47 -> 1.21) for a
    five-fold wider region, and a p90 extent of 20.7 inches on a 36-inch sheet
    constrains nothing at all. Admitting a pose where ANY of them lands in the
    disc makes the admissible set the union of their discs.
    """
    centre: np.ndarray
    radius: float
    ref: np.ndarray
    # How much the score is pulled toward the disc's centre. SOFT_PRIOR_WEIGHT
    # suits a surveyed anchor; zero makes the disc a pure constraint, which
    # is what a candidate injected for comparison needs -- its score must be
    # the same wall/free/scale sum every other candidate got.
    soft_weight: float = SOFT_PRIOR_WEIGHT

    def __post_init__(self):
        self.ref = np.asarray(self.ref, dtype=np.float64).reshape(-1, 2)

    def allowed_translations(self, scale: float, yaw_deg: float) -> np.ndarray:
        """(M, 2) disc centres the translation may lie in, for one hypothesis."""
        return (np.asarray(self.centre, dtype=np.float64)[:2]
                - _rotate(self.ref, yaw_deg, scale))

    def allowed_translation(self, scale: float, yaw_deg: float) -> np.ndarray:
        """The first admissible centre. Kept for callers wanting a single point."""
        return self.allowed_translations(scale, yaw_deg)[0]


@dataclass
class Hypothesis:
    """One candidate pose, with the score decomposed so failures are readable."""
    sim2: Sim2
    score: float
    wall: float
    free: float
    scale_prior: float
    scale: float
    yaw_deg: float
    viewport: int
    up_index: int = 0          # which gravity hypothesis, for sign-ambiguous splits
    rank: int = -1
    prior_dist: float = float("nan")   # paper units from the prior centre
    outside_prior: bool = False        # True when the disc held nothing
    # Filled by align.rerank. The correlation proposes; these decide.
    inliers: int = -1
    n_segments: int = -1
    # Oriented-bbox agreement against the LOCAL sheet crop under this placement
    # (see `search`'s peak loop) -- nan wherever a hypothesis was never scored
    # against a sheet at all (e.g. align.chain's transferred poses).
    aspect: float = float("nan")

    def as_dict(self) -> dict:
        return dict(score=self.score, wall=self.wall, free=self.free,
                    scale_prior=self.scale_prior, aspect=self.aspect,
                    scale=self.scale,
                    yaw_deg=self.yaw_deg, prior_dist=self.prior_dist,
                    outside_prior=self.outside_prior, inliers=self.inliers,
                    n_segments=self.n_segments,
                    viewport=self.viewport, up_index=self.up_index,
                    rank=self.rank, tx=self.sim2.tx, ty=self.sim2.ty,
                    theta=self.sim2.theta, s=self.sim2.s)


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class _Numpy:
    """Reference backend. Correct, portable, and slow enough that the GPU path
    exists; kept so results can be cross-checked between the two."""
    name = "numpy"

    @staticmethod
    def rfft2(a, shape):
        return np.fft.rfft2(a, shape)

    @staticmethod
    def irfft2(a, shape):
        return np.fft.irfft2(a, shape)

    @staticmethod
    def asarray(a):
        return np.asarray(a, dtype=np.float32)

    @staticmethod
    def to_numpy(a):
        return np.asarray(a)

    @staticmethod
    def flip2(a):
        return a[::-1, ::-1]


class _Torch:
    """GPU backend. Same maths; the win is batching many templates against one
    pre-transformed sheet."""
    name = "torch"

    def __init__(self, device=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def rfft2(self, a, shape):
        return self.torch.fft.rfft2(a, s=shape)

    def irfft2(self, a, shape):
        return self.torch.fft.irfft2(a, s=shape)

    def asarray(self, a):
        return self.torch.as_tensor(np.asarray(a, dtype=np.float32),
                                    device=self.device)

    def to_numpy(self, a):
        return a.detach().cpu().numpy()

    def flip2(self, a):
        # torch rejects negative strides, so reversal is an explicit op
        return self.torch.flip(a, dims=(-2, -1))


def get_backend(prefer: str = "auto"):
    if prefer == "numpy" or (prefer == "auto" and not _have_torch()):
        return _Numpy()
    return _Torch()


# --------------------------------------------------------------------------
# the score maps
# --------------------------------------------------------------------------
def _correlate(bk, F_image, template, shape):
    """Cross-correlation of `template` over `image`, image already transformed.

    Implemented as a convolution with the reversed template, so the peak at
    index n corresponds to the template ORIGIN sitting at n - (t - 1). Callers
    must slice that offset away; `_valid` does it.
    """
    return bk.irfft2(F_image * bk.rfft2(bk.flip2(template), shape), shape)


def _valid(corr, tshape, ishape):
    """Crop a correlation to placements where the template lies fully inside.

    Index n of the raw convolution means template origin at n - (t - 1), so the
    fully-contained placements start at t - 1 and the cell (0, 0) of the result
    is the template sitting flush with the image origin.
    """
    th, tw = tshape
    H, W = ishape
    return corr[th - 1:H, tw - 1:W]


def masked_ncc(bk, sheet, template, footprint, shape, eps: float = 1e-3):
    """Normalised cross-correlation of a masked template over the sheet map.

    The template is mean-subtracted within the footprint and the result divided
    by the sheet's local standard deviation under it, so a dense block of ink
    cannot win by being dense -- that is the title-block failure, worth ~25
    inches of error.

    The footprint is the region the comparison covers, NOT the template's
    occupied cells. Using the occupied cells makes the mean exactly 1 for a
    binary template, the mean-subtracted template identically zero, and the
    whole correlation zero. Pass the covered area; `search` passes the
    template's full rectangle.
    """
    n = float(bk.to_numpy(footprint).sum())
    if n < 1:
        return None
    F_sheet = bk.rfft2(sheet, shape)
    F_sheet2 = bk.rfft2(sheet * sheet, shape)

    t_mean = float(bk.to_numpy((template * footprint).sum()) / n)
    t_zero = (template - t_mean) * footprint
    t_energy = float(bk.to_numpy((t_zero * t_zero).sum())) ** 0.5

    num = _correlate(bk, F_sheet, t_zero, shape)
    s1 = _correlate(bk, F_sheet, footprint, shape)
    s2 = _correlate(bk, F_sheet2, footprint, shape)
    var = s2 - s1 * s1 / n
    denom = (var.clip(min=eps) ** 0.5) * max(t_energy, eps)
    return num / denom


def interior_mask(sheet_map: np.ndarray, close_cells: int = 3) -> np.ndarray:
    """Enclosed space on the sheet: inside the drawn outline, off the ink.

    Closes small gaps in the wall map, floods in from the border, and keeps what
    the flood could not reach. Blank margin outside the building is reachable
    from the border and is therefore NOT interior -- which is the whole point,
    since it stops an empty region scoring as perfect free space.
    """
    import cv2
    wall = (np.asarray(sheet_map) > 0).astype(np.uint8)
    if not wall.any():
        return np.zeros(wall.shape, bool)
    k = np.ones((close_cells, close_cells), np.uint8)
    closed = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, k)
    h, w = closed.shape
    flood = closed.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 2)
    for seed in ((w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(flood, mask, seed, 2)
    return (flood == 0)


def free_space_map(bk, sheet, traj_template, shape):
    """Fraction of the walk landing on drawn line-work, over all translations.

    One half of the walk term; see `inside_map` for the other. Used alone only
    in tests, where the walk is guaranteed to be over the drawing.
    """
    mass = float(bk.to_numpy(traj_template).sum())
    if mass < 1:
        return None
    return _correlate(bk, bk.rfft2(sheet, shape), traj_template, shape) / mass


def inside_map(bk, interior, traj_template, shape):
    """Fraction of the walk falling inside enclosed space, over all translations."""
    mass = float(bk.to_numpy(traj_template).sum())
    if mass < 1:
        return None
    return _correlate(bk, bk.rfft2(interior, shape), traj_template, shape) / mass


def aspect_agreement(cloud_xy: np.ndarray, sheet_wall: np.ndarray,
                     res: float) -> float:
    """How well the ROTATED cloud's extent matches the drawn content's.

    Compares width to width and height to height as an ordered pair, not as a
    ratio. A ratio of max to min extent is invariant under a 90-degree
    rotation, which makes it blind to exactly the degeneracy this term exists
    to penalise -- the first version of this function had that bug, and a
    synthetic test caught it.

    `cloud_xy` must already be rotated and scaled, so both extents are measured
    along the same paper axes. 1.0 is a perfect match.
    """
    if len(cloud_xy) < 3 or not sheet_wall.any():
        return 0.0
    cw, ch = (np.asarray(cloud_xy).max(0) - np.asarray(cloud_xy).min(0))[:2]
    ys, xs = np.nonzero(sheet_wall)
    sw = float(xs.max() - xs.min() + 1) * res
    sh = float(ys.max() - ys.min() + 1) * res
    if min(cw, ch, sw, sh) <= 0:
        return 0.0
    return float(np.exp(-abs(np.log(cw / sw)) - abs(np.log(ch / sh))))


def scale_plausibility(scale: float, prior: float,
                       sigma_log: float = SCALE_SIGMA_LOG) -> float:
    """How well a candidate scale agrees with the camera-height prior.

    1.0 at the prior, falling off as a Gaussian in log scale. This is the
    counterweight to masked NCC's bias toward small templates.
    """
    if not (np.isfinite(scale) and np.isfinite(prior)) or scale <= 0 or prior <= 0:
        return 0.0
    return float(np.exp(-0.5 * (np.log(scale / prior) / sigma_log) ** 2))


def peaks(score: np.ndarray, k: int, radius: int = NMS_RADIUS) -> list[tuple]:
    """Top-k local maxima with simple non-maximum suppression."""
    s = np.array(score, dtype=np.float64, copy=True)
    out = []
    h, w = s.shape
    for _ in range(k):
        idx = int(np.argmax(s))
        y, x = divmod(idx, w)
        v = s[y, x]
        if not np.isfinite(v) or v == -np.inf:
            break
        out.append((float(v), int(y), int(x)))
        s[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = -np.inf
    return out


# --------------------------------------------------------------------------
# hypothesis enumeration and the search itself
# --------------------------------------------------------------------------
@dataclass
class SearchGrid:
    """The hypotheses to evaluate. Kept separate so it can be inspected."""
    scales: np.ndarray
    yaws_deg: np.ndarray
    viewports: list
    fallback_yaw: bool = False

    def __len__(self) -> int:
        return len(self.scales) * len(self.yaws_deg) * max(len(self.viewports), 1)


YAW_STEP_DEG = 3.0            # see yaw_candidates
YAW_FALLBACK_STEP_DEG = 4.0   # ...when the Manhattan estimate is unusable


def yaw_candidates(cloud_angle: float, sheet_angle: float,
                   window_deg: float = 10.0, step_deg: float = YAW_STEP_DEG,
                   strength: float = 1.0, min_strength: float = 0.08
                   ) -> tuple[np.ndarray, bool]:
    """Yaw samples, in degrees, for rotating the cloud into the paper frame.

    Both angles are known only modulo 90 degrees, which leaves four rotations.
    A window around each absorbs the residual: the estimator was within 5
    degrees on 16 of 27 splits and within 10 on 19, and the ground truth's own
    rotation is off by as much as 14.5 degrees, so a window narrower than that
    would be chasing noise.

    The step is 3 degrees for the same reason the window is 10: the ground
    truth's own rotation has a median error of 1.57 degrees and a p90 of 9.7,
    so sampling finer than that resolves noise. It was 1.5, which doubled the
    grid to measure the same thing twice.

    Paper y runs up while sheet rows run down, so the sheet's angle enters
    negated.
    """
    if strength < min_strength:
        return np.arange(0.0, 360.0, 4.0), True
    base = (np.degrees(-sheet_angle - cloud_angle)) % 90.0
    offsets = np.arange(-window_deg, window_deg + 1e-9, step_deg)
    return np.unique(np.concatenate(
        [(base + 90.0 * k + offsets) % 360.0 for k in range(4)])), False


def scale_candidates(bracket: tuple[float, float], steps: int = 9) -> np.ndarray:
    """Log-spaced scale ladder across the camera-height bracket."""
    lo, hi = bracket
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0:
        raise ValueError(f"unusable scale bracket {bracket}")
    return np.exp(np.linspace(np.log(lo), np.log(hi), steps))


def _distance_to_nearest(cx: np.ndarray, cy: np.ndarray,
                         shape: tuple[int, int], pad_cells: float) -> np.ndarray:
    """Per-cell distance, in cells, to the nearest of several points.

    A `hypot` per point per cell is the obvious way and costs M times the grid
    on every hypothesis. Stamping the points and distance-transforming costs
    one pass whatever M is. The canvas is padded by the prior's radius so a
    point sitting just off the grid still constrains the cells it reaches.
    """
    H, W = shape
    cx = np.atleast_1d(np.asarray(cx, dtype=np.float64))
    cy = np.atleast_1d(np.asarray(cy, dtype=np.float64))
    if len(cx) == 1:
        yy, xx = np.ogrid[:H, :W]
        return np.hypot(xx - cx[0], yy - cy[0])

    p = int(np.ceil(pad_cells)) + 1
    canvas = np.ones((H + 2 * p, W + 2 * p), np.uint8)
    ix = np.round(cx).astype(int) + p
    iy = np.round(cy).astype(int) + p
    ok = (ix >= 0) & (ix < canvas.shape[1]) & (iy >= 0) & (iy < canvas.shape[0])
    if not ok.any():
        return np.full(shape, np.inf)
    canvas[iy[ok], ix[ok]] = 0
    d = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    return d[p:p + H, p:p + W].astype(np.float64)


def _rotate(xy: np.ndarray, deg: float, scale: float) -> np.ndarray:
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a), np.cos(a)]])
    return (np.asarray(xy)[:, :2] @ R.T) * scale


def _stamp(xy: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Binary template from points, plus the paper-frame origin of cell (0,0)."""
    lo = xy.min(0)
    idx = np.floor((xy - lo) / res).astype(int)
    h = int(idx[:, 1].max()) + 1
    w = int(idx[:, 0].max()) + 1
    t = np.zeros((h, w), np.float32)
    t[idx[:, 1], idx[:, 0]] = 1.0
    return t, lo


def search(cloud_xy: np.ndarray, traj_xy: np.ndarray, sheet_map: np.ndarray,
           sheet_origin: tuple[float, float], res: float, grid: SearchGrid,
           viewport_index: int = 0, up_index: int = 0, backend=None,
           w_wall: float = W_WALL, w_free: float = W_FREE,
           w_scale: float = W_SCALE, w_aspect: float = W_ASPECT,
           scale_prior: float | None = None,
           prior: TranslationPrior | None = None,
           interior: np.ndarray | None = None,
           top_k: int = TOP_K_PER_HYPOTHESIS) -> list[Hypothesis]:
    """Score every (scale, yaw) over all translations; return the best poses.

    `sheet_map` is the wall map on a paper-frame grid whose cell (0, 0) sits at
    `sheet_origin`, row 0 at MINIMUM paper y -- the convention `rasterize` and
    `sheet.resample_to` both use, so indices line up.

    `interior` overrides the "inside the building" half of the walk term. The
    default, `interior_mask`, floods from the border and so only works on a
    CLOSED outline: against an open point set it marks 0.1-13% of the map and the
    walk term collapses into a pure penalty that drives the answer off into blank
    space. Callers matching against a point cloud pass their own coverage mask --
    where the reference was actually walked -- on the same grid as `sheet_map`.

    The map is zero-padded by the largest template before correlating. Without
    that, only placements lying wholly inside the viewport are representable,
    and a capture that runs past the drawn area -- which is common, since
    viewports are cut to the ink and walks are not -- can have NO expressible
    translation at its true pose. Padding also lets every hypothesis share one
    forward transform of the sheet instead of recomputing it per template.

    `w_aspect` weights `aspect_agreement` against a LOCAL crop of the sheet --
    the padded region the template would actually occupy at each surviving
    peak, not the whole sheet. The whole-sheet comparison is what
    `align.search.aspect_agreement`'s own tests found unsound (the merged
    reference commonly spans far more than one capture's footprint, so a
    global aspect ratio does not discriminate a locally-wrong rotation); the
    local crop is sized and placed exactly like the correlation template
    itself, so it is a translation-aware readout, not a whole-sheet one.
    """
    bk = backend or get_backend()
    H, W = sheet_map.shape

    # Build every template first: the padding and the shared FFT plan both
    # depend on the largest one.
    built = []
    for s in grid.scales:
        for yaw in grid.yaws_deg:
            cloud_r = _rotate(cloud_xy, yaw, s)
            tmpl, t_lo = _stamp(cloud_r, res)
            if tmpl.shape[0] < 2 or tmpl.shape[1] < 2:
                continue
            traj_r = _rotate(traj_xy, yaw, s)
            tj = np.zeros_like(tmpl)
            ti = np.floor((traj_r - t_lo) / res).astype(int)
            ok = ((ti[:, 0] >= 0) & (ti[:, 0] < tmpl.shape[1])
                  & (ti[:, 1] >= 0) & (ti[:, 1] < tmpl.shape[0]))
            if ok.sum() >= 2:
                tj[ti[ok, 1], ti[ok, 0]] = 1.0
            built.append((float(s), float(yaw), tmpl, t_lo, tj, cloud_r))
    if not built:
        return []

    pad_y = max(t[2].shape[0] for t in built)
    pad_x = max(t[2].shape[1] for t in built)
    padded = np.zeros((H + 2 * pad_y, W + 2 * pad_x), np.float32)
    padded[pad_y:pad_y + H, pad_x:pad_x + W] = sheet_map
    PH, PW = padded.shape
    shape = (PH + pad_y, PW + pad_x)

    inside = np.zeros_like(padded)
    inside[pad_y:pad_y + H, pad_x:pad_x + W] = (
        interior_mask(sheet_map) if interior is None
        else np.asarray(interior, dtype=np.float32))

    sheet_t = bk.asarray(padded)
    F_sheet = bk.rfft2(sheet_t, shape)
    F_sheet2 = bk.rfft2(sheet_t * sheet_t, shape)
    F_inside = bk.rfft2(bk.asarray(inside), shape)

    out: list[Hypothesis] = []
    for s, yaw, tmpl, t_lo, tj, cloud_r in built:
        th, tw = tmpl.shape
        t = bk.asarray(tmpl)
        # Footprint is the covered rectangle, not the occupied cells: inside the
        # area the capture covers, "no wall here" is evidence too.
        fp = bk.asarray(np.ones_like(tmpl))
        n = float(th * tw)

        t_mean = float(tmpl.mean())
        t_zero = bk.asarray(tmpl - t_mean)
        t_energy = max(float(np.sqrt(((tmpl - t_mean) ** 2).sum())), 1e-3)

        num = _correlate(bk, F_sheet, t_zero, shape)
        s1 = _correlate(bk, F_sheet, fp, shape)
        s2 = _correlate(bk, F_sheet2, fp, shape)
        var = s2 - s1 * s1 / n
        ncc = num / ((var.clip(min=1e-3) ** 0.5) * t_energy)

        ncc_np = _valid(bk.to_numpy(ncc), (th, tw), (PH, PW))
        if ncc_np.size == 0:
            continue

        mass = float(tj.sum())
        if mass >= 1:
            tj_t = bk.asarray(tj)
            on_wall = _correlate(bk, F_sheet, tj_t, shape) / mass
            in_room = _correlate(bk, F_inside, tj_t, shape) / mass
            # One quantity: the walk should be inside the building and not
            # through its walls. Without the first half, blank paper wins.
            free_np = (_valid(bk.to_numpy(on_wall), (th, tw), (PH, PW))
                       - _valid(bk.to_numpy(in_room), (th, tw), (PH, PW)))
        else:
            free_np = np.zeros_like(ncc_np)

        sp = (scale_plausibility(s, scale_prior)
              if scale_prior is not None else 1.0)
        total = w_wall * ncc_np - w_free * free_np + w_scale * sp

        # A translation prior turns a whole-sheet search into a small disc,
        # which is exactly where the unaided score fails. Distances are computed
        # per cell, then used both to exclude and to weight.
        dist = None
        outside = False
        if prior is not None:
            H_v, W_v = total.shape
            cs = prior.allowed_translations(s, yaw)
            cx = (cs[:, 0] + t_lo[0] - sheet_origin[0]) / res + pad_x
            cy = (cs[:, 1] + t_lo[1] - sheet_origin[1]) / res + pad_y
            dist = _distance_to_nearest(cx, cy, (H_v, W_v),
                                        prior.radius / res) * res
            inside = dist <= prior.radius
            if inside.any():
                # Favour the centre of the disc, then exclude everything beyond.
                total = total + prior.soft_weight * np.exp(
                    -0.5 * (dist / max(prior.radius, 1e-9)) ** 2)
                total = np.where(inside, total, -np.inf)
            else:
                # Nothing representable inside the disc. Fall back to the
                # unrestricted map rather than returning nothing, and say so.
                outside = True

        for value, py, px in peaks(total, top_k):
            if not np.isfinite(value):
                break
            # The sheet exactly under this placement, at the template's own
            # footprint -- not the whole sheet. `py, px` are already the
            # template's origin in `padded`'s frame (see `_valid`'s docstring).
            local_sheet = padded[py:py + th, px:px + tw]
            asp = aspect_agreement(cloud_r, local_sheet, res)
            value = value + w_aspect * asp
            # Padded-grid offset back to paper: undo the pad, then the stamp.
            tx = sheet_origin[0] + (px - pad_x) * res - t_lo[0]
            ty = sheet_origin[1] + (py - pad_y) * res - t_lo[1]
            out.append(Hypothesis(
                sim2=Sim2(s=s, theta=float(np.radians(yaw)),
                          tx=float(tx), ty=float(ty)),
                score=float(value), wall=float(ncc_np[py, px]),
                free=float(free_np[py, px]), scale_prior=float(sp),
                aspect=float(asp),
                scale=s, yaw_deg=yaw,
                viewport=viewport_index, up_index=up_index,
                prior_dist=(float(dist[py, px]) if dist is not None
                            else float("nan")),
                outside_prior=outside))

    out.sort(key=lambda h: -h.score)
    for i, h in enumerate(out):
        h.rank = i
    return out
