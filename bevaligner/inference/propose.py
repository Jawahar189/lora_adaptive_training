"""One pose per capture, chosen over every gravity hypothesis and prior window.

Stage one: the prior is cut into overlapping square windows, and for each
(up alternative, window) the matcher proposes correspondences and RANSAC fits a
similarity; the most inliers win. Stage two, optional: the prior is re-rendered
around the winning pose's footprint, where the capture fills the frame the way
the training zoom windows did, and the pose is re-fitted. The result carries
what `align.runner.visual_local_search` expects, so it can be dropped in as a
proposer stage later.

The window side is chosen WITHOUT the answer. The monopod appears in both
captures, so the walk's size in paper inches is its size in cloud units times
`prior rig height / this walk's camera height` -- the scale cancellation of
`docs/PRIOR_CAPTURE.md` §5. Nothing here reads the current capture's transform.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data import density as density_module
from ..data import prior_bev
from ..utils import affine, geometry
from ..utils.gravity import GravityView
from .matching import (MATCHERS, Matches, match_density_keypoints_to_crop,
                       match_density_to_crop)
from .ransac import fit_planar_pose
from .verify import verify_matches


# A window holding fewer prior wall points than this is not worth a crop. The
# same floor `align.cloud` puts on a capture's own band, for the same reason.
MIN_WINDOW_BAND_POINTS = 200


@dataclass
class PoseProposal:
    sim2: geometry.Sim2                  # gravity-up cloud xy -> paper inches
    up_index: int
    rotation_to_gravity_up: np.ndarray
    window_index: int
    num_inliers: int
    inlier_fraction: float
    num_matches: int
    # How many matches geometric verification kept, or None when it did not run
    # -- which is not the same as "it ran and kept everything".
    num_verified: int | None
    camera_height_above_floor: float
    camera_z: np.ndarray
    refined: bool = False
    # Whether the zoomed pass RAN, which is a property of the run rather than of
    # the pose. `confidence_for` needs it to pick its coefficients and
    # `gate_proposal` takes the same thing as `refine_was_on`; carried here so
    # every caller reads one answer instead of re-deriving it from its own
    # options.
    refine_was_on: bool = True
    method: str = "bevaligner"
    candidates: list[dict] = field(default_factory=list)
    # What the winning fit was made of, kept only when the caller asks
    # (`PoseProposalOptions.keep_match_artifacts`), for drawing.
    density_map: object | None = None
    crop: object | None = None
    matches: Matches | None = None
    inlier_mask: np.ndarray | None = None
    # Which matches COLMAP's LO-RANSAC could explain with one planar motion.
    # None when nothing was verified; `matches` is never sliced, so this and
    # `inlier_mask` index the same array and the drawing can tell the three
    # states apart.
    verified_mask: np.ndarray | None = None


def pose_transform_matrix_for(proposal: PoseProposal, design_tm) -> np.ndarray:
    return geometry.pose_transform_matrix(proposal.sim2, proposal.rotation_to_gravity_up,
                                          proposal.camera_z,
                                          proposal.camera_height_above_floor, design_tm)


def as_solver_fit_dict(proposal: PoseProposal, scale_prior=None) -> dict:
    """The dictionary `align.runner.visual_local_search` consumes."""
    return dict(sim2=proposal.sim2, up_index=proposal.up_index,
                n_inliers=proposal.num_inliers, inlier_frac=proposal.inlier_fraction,
                how=proposal.method, scale_prior=scale_prior)


@dataclass
class PoseProposalOptions:
    resolution: int = 1024
    num_queries: int = 1024
    keep_fraction: float = 0.5
    mutual_check: bool = True
    tolerance_m: float = 0.5
    ransac_iterations: int = 2000
    min_inliers: int = 12
    seed: int = 0
    refine_on_footprint: bool = True
    refine_footprint_margin: float = 1.6
    # The matcher resolves about one patch; on a coarse window a patch can be
    # several metres, so the inlier tolerance is at least this many patches.
    tolerance_patches: float = 0.5
    # Stage one sees the capture a few patches wide, so a patch-level mutual
    # check there would keep one match per patch; it is applied on the zoomed pass.
    mutual_check_on_viewport: bool = False
    # Stage one's window side, as a multiple of the walk's estimated paper size.
    search_zoom_out: float = 4.0
    # Windows step this fraction of their side, so a walk near a seam still
    # falls whole inside a neighbour.
    window_stride_fraction: float = 0.5
    # A kilometre-wide site tiled for a 30 m walk is a thousand windows. The
    # richest by prior wall content are kept: that ranking reads only the
    # reference, never the answer.
    max_windows: int = 32
    # A refined pose must agree with the coarse one to within this many coarse patches.
    refine_agreement_patches: float = 2.0
    # COLMAP's LO-RANSAC over the matches before the Sim2 fit: "none",
    # "affine" or "homography". It only ever REMOVES matches -- the pose is
    # still the Sim2 `align.visual.ransac_sim2` fits, so the up-sign mechanism
    # that rejects a mirrored match set is untouched. See `verify.py`.
    geometric_verification: str = "none"
    # The verifier's tolerance as a multiple of the fit's own. Above 1 it only
    # throws away what no planar motion could explain at any tolerance; at 1 it
    # is as strict as the fit that follows it.
    verification_tolerance_scale: float = 1.0
    # Which matcher proposes the correspondences (`matching.MATCHERS`): "dense",
    # each query's soft-argmax over the prior's patches, or "keypoints", points
    # drawn on both maps and matched one to one. The keypoint matcher always
    # applies its own exact mutual check, so the two `mutual_check` options above
    # are the dense matcher's alone.
    matcher: str = "dense"
    # keep the winning density map, crop, matches and inlier mask on the proposal
    keep_match_artifacts: bool = False
    # ---------------------------------------------------------------- the prior
    # Where a caller already believes the walk starts, in PAPER INCHES, and how
    # far it is willing to be wrong. `None` is the whole of this file's history:
    # `candidate_windows` tiles the prior and ranks by wall content, and nothing
    # below changes. Set, and the tiling is replaced by a handful of windows
    # around the centre -- see `_windows_near` for why the side is NOT shrunk.
    prior_centre: tuple[float, float] | None = None
    prior_radius_in: float = 0.0
    # Windows drawn around the prior. Five is the centre plus the four axis
    # offsets; the diagonals come in after that.
    max_windows_with_prior: int = 5
    # ... plus this many of the wall-richest windows, kept whatever the prior
    # says. A prior that is wrong is then a window that WINS from outside the
    # disc, which the row records, rather than a search that quietly had no
    # chance. Zero makes the prior a hard filter.
    insurance_windows: int = 2
    # (lo, hi) on the Sim2 scale, passed to `fit_planar_pose`. A capture is one
    # rig, so a sibling split's solved scale brackets this one's.
    scale_bracket: tuple[float, float] | None = None


def estimated_paper_span(view: GravityView, pool: prior_bev.PriorPool) -> float:
    """How big this walk is in paper inches, without knowing where it is.

    One rig measured in two frames: the prior's camera height in paper inches
    over this walk's camera height in cloud units is the scale, so the walk's
    own extent converts. `docs/PRIOR_CAPTURE.md` §5 -- no paper-scale estimate
    and no constant.
    """
    points = view.select_band_points()
    if len(points) < 2 or view.camera_height_above_floor <= 0:
        return float(pool.span)
    low = np.percentile(points, 2.5, axis=0)
    high = np.percentile(points, 97.5, axis=0)
    scale = pool.rig_height_in / view.camera_height_above_floor
    return float((high - low).max() * scale)


def candidate_windows(pool: prior_bev.PriorPool, side: float, options) -> list[tuple]:
    """The windows stage one searches, richest in wall first.

    With no prior this is the tiling it has always been. With one it is a few
    squares around the prior plus `insurance_windows` of the richest, which is
    what `_windows_near` builds.
    """
    return [window for window, _from_prior in windows_with_origin(pool, side, options)]


def windows_with_origin(pool: prior_bev.PriorPool, side: float,
                        options) -> list[tuple[tuple, bool]]:
    """Each window, and whether the prior rather than wall content put it there.

    The flag reaches the row. A run whose winners keep coming from outside the
    disc is a run whose priors are wrong, and that is worth seeing directly
    rather than inferring from the error.
    """
    if options.prior_centre is None:
        return [(window, False) for window in _tiled_windows(pool, side, options)]
    return _windows_near(pool, side, options)


def _tiled_windows(pool: prior_bev.PriorPool, side: float, options) -> list[tuple]:
    """Overlapping squares of `side` tiling the prior, richest in wall first."""
    lo_x, lo_y, hi_x, hi_y = pool.extent
    side = float(min(max(side, 1e-6), max(hi_x - lo_x, 1e-6)))
    step = max(side * options.window_stride_fraction, 1e-9)
    xs = np.arange(lo_x, max(hi_x - side, lo_x) + step, step)
    ys = np.arange(lo_y, max(hi_y - side, lo_y) + step, step)
    points = pool.band_paper_xy

    # Every window edge, as the exact floats the comparisons use. A point with
    # x <= px < x + side lands in a cell between x's edge and (x + side)'s edge,
    # so counting cells reproduces the half-open test exactly, for any stride.
    edges_x, inv_x = np.unique(np.concatenate([xs, xs + side]), return_inverse=True)
    edges_y, inv_y = np.unique(np.concatenate([ys, ys + side]), return_inverse=True)
    x0, x1 = inv_x[:len(xs)], inv_x[len(xs):]
    y0, y1 = inv_y[:len(ys)], inv_y[len(ys):]

    # Cell of each point = last edge <= p. Points outside every window drop out.
    bx = np.searchsorted(edges_x, points[:, 0], side="right") - 1
    by = np.searchsorted(edges_y, points[:, 1], side="right") - 1
    nx, ny = len(edges_x) - 1, len(edges_y) - 1
    inside = (bx >= 0) & (bx < nx) & (by >= 0) & (by < ny)
    hist = np.bincount(bx[inside] * ny + by[inside],
                       minlength=nx * ny).reshape(nx, ny)

    # Summed-area table with a zero row and column in front: any window's
    # count is four lookups, and every window is looked up at once.
    sat = np.zeros((nx + 1, ny + 1), dtype=np.int64)
    sat[1:, 1:] = hist.cumsum(0).cumsum(1)
    counts = (sat[x1[:, None], y1[None, :]] - sat[x0[:, None], y1[None, :]]
              - sat[x1[:, None], y0[None, :]] + sat[x0[:, None], y0[None, :]])

    flat = counts.ravel()                      # x-major, the old loop order
    nonzero = np.flatnonzero(flat)
    if nonzero.size == 0:
        return [pool.extent]
    k = min(int(options.max_windows), nonzero.size)
    if k <= 0:
        return []
    # Stable sort on -count keeps ties in scan order, as the old sort did.
    best = nonzero[np.argsort(-flat[nonzero], kind="stable")[:k]]
    ix, iy = np.divmod(best, len(ys))
    return [(float(xs[i]), float(ys[j]), float(xs[i] + side), float(ys[j] + side))
            for i, j in zip(ix, iy)]


def _square(centre_x: float, centre_y: float, side: float) -> tuple:
    half = side / 2
    return (centre_x - half, centre_y - half, centre_x + half, centre_y + half)


def _windows_near(pool: prior_bev.PriorPool, side: float,
                  options) -> list[tuple[tuple, bool]]:
    """A handful of windows around `options.prior_centre`, then the insurance.

    `side` is NOT shrunk to the disc. The prior says where the walk STARTS, not
    how big it is: at `search_zoom_out` x the estimated span, a window centred
    on the start point already contains any walk that begins there, and a
    smaller one would cut the walk's tail out of the crop -- which costs the
    matches that place it, not just the margin.

    The ring exists because the walk may begin at the disc's RIM rather than at
    its centre, in which case a single centred window has it against an edge.
    Offsets go out by `min(prior_radius_in, side / 4)`: a quarter of the side is
    as far as a window can move while still containing the centre's own start
    point, so beyond that the ring stops buying coverage and starts buying
    crops.
    """
    centre_x, centre_y = (float(value) for value in options.prior_centre)
    reach = min(max(float(options.prior_radius_in), 0.0), side / 4)
    diagonal = reach / np.sqrt(2.0)
    offsets = [(0.0, 0.0),
               (reach, 0.0), (-reach, 0.0), (0.0, reach), (0.0, -reach),
               (diagonal, diagonal), (diagonal, -diagonal),
               (-diagonal, diagonal), (-diagonal, -diagonal)]

    points = pool.band_paper_xy
    seen, near = set(), []
    for dx, dy in offsets:
        if len(near) >= max(int(options.max_windows_with_prior), 0):
            break
        window = _square(centre_x + dx, centre_y + dy, side)
        key = tuple(round(value, 6) for value in window)
        if key in seen:
            continue
        seen.add(key)
        inside = ((points[:, 0] >= window[0]) & (points[:, 0] < window[2])
                  & (points[:, 1] >= window[1]) & (points[:, 1] < window[3]))
        # Fewer than this and `render_target` has nothing to match against; the
        # window is dropped here so `window_index` still means something.
        if int(inside.sum()) < MIN_WINDOW_BAND_POINTS:
            continue
        near.append((window, True))

    # The insurance. Kept even when the prior found nothing, which is the case
    # where a disc has landed off the prior entirely -- the search then looks
    # exactly like an unconstrained one restricted to its best few windows, and
    # the row says every winner came from outside the prior.
    insurance = []
    if options.insurance_windows > 0:
        for window in _tiled_windows(pool, side, options):
            if len(insurance) >= int(options.insurance_windows):
                break
            key = tuple(round(value, 6) for value in window)
            if key in seen:
                continue
            seen.add(key)
            insurance.append((window, False))
    return near + insurance


def _verified_count(verified):
    """How many matches verification kept, or None when it did not run."""
    return None if verified is None else int(verified.sum())


def _match_and_fit(model, density_map, crop, metres_per_inch, options,
                   rng, device, patch_size, mutual_check):
    if options.matcher == "keypoints":
        matches: Matches = match_density_keypoints_to_crop(
            model, density_map, crop, num_queries=options.num_queries,
            keep_fraction=options.keep_fraction, rng=rng, device=device)
    elif options.matcher == "dense":
        matches = match_density_to_crop(
            model, density_map, crop, num_queries=options.num_queries,
            keep_fraction=options.keep_fraction, mutual_check=mutual_check,
            rng=rng, device=device)
    else:
        raise ValueError(f"unknown matcher {options.matcher!r}; choose from {MATCHERS}")
    inches_per_crop_px = 1.0 / max(affine.isotropic_scale(crop.paper_to_crop), 1e-12)
    tolerance_inches = max(options.tolerance_m / metres_per_inch,
                           options.tolerance_patches * patch_size * inches_per_crop_px)
    # Geometric verification first, so the Sim2 fit never sees a correspondence
    # no planar motion could explain. `matches` is NOT sliced: the drawing wants
    # to show what was thrown away, and the inlier fraction stays over every
    # match proposed, so a window where verification killed most of its matches
    # cannot out-score one where it killed none.
    verified = verify_matches(
        matches.source_world_xy, matches.target_paper_xy,
        tolerance_inches * options.verification_tolerance_scale,
        mode=options.geometric_verification, seed=options.seed)
    fit = _fit_on(matches, verified, metres_per_inch, options, tolerance_inches)
    return matches, fit, tolerance_inches, verified


def _fit_on(matches: Matches, verified, metres_per_inch, options, tolerance_inches):
    """`fit_planar_pose` over the verified subset, with the mask grown back.

    RANSAC runs on the survivors, but the mask it returns has to index the full
    match set so every caller -- inlier counting, the fraction, the drawing --
    keeps working on one set of indices.
    """
    source, target = matches.source_world_xy, matches.target_paper_xy
    index = None if verified is None else np.flatnonzero(verified)
    if index is not None:
        source, target = source[index], target[index]
    fit = fit_planar_pose(source, target, metres_per_inch,
                          options.tolerance_m, options.ransac_iterations, options.seed,
                          options.min_inliers, scale_bracket=options.scale_bracket,
                          tolerance_inches=tolerance_inches)
    if fit is None or index is None:
        return fit
    sim2, subset_mask = fit
    full = np.zeros(len(matches.source_world_xy), dtype=bool)
    full[index[subset_mask]] = True
    return sim2, full


def propose_pose(model, views: dict[int, GravityView],
                 pool: prior_bev.PriorPool, data_config,
                 metres_per_inch: float, options: PoseProposalOptions = PoseProposalOptions(),
                 device=None, patch_size: int | None = None) -> PoseProposal | None:
    """`views`: usable gravity hypotheses; `pool`: the merged prior capture.

    The windows searched are derived from the prior and from this walk's own
    size, never from its transform.
    """
    device = device or next(model.parameters()).device
    patch_size = patch_size or model.patch_size

    best, best_density_map, candidates = None, None, []
    best_tolerance_inches = best_window = None
    for up_index, view in views.items():
        band_points = view.select_band_points()
        if len(band_points) < 200:
            continue
        density_map = density_module.render_density_map(band_points, options.resolution)
        side = options.search_zoom_out * estimated_paper_span(view, pool)
        windows = windows_with_origin(pool, side, options)
        rng = np.random.default_rng(options.seed)
        for window_index, (window, from_prior) in enumerate(windows):
            crop = prior_bev.render_target(pool, window, options.resolution, None,
                                           data_config)
            if crop is None:
                continue
            matches, fit, tolerance_inches, verified = _match_and_fit(
                model, density_map, crop, metres_per_inch, options, rng,
                device, patch_size,
                options.mutual_check and options.mutual_check_on_viewport)
            num_inliers = int(fit[1].sum()) if fit else 0
            candidate = dict(
                up_index=up_index, window_index=window_index,
                num_matches=len(matches.source_world_xy),
                num_verified=_verified_count(verified), num_inliers=num_inliers,
                mean_confidence=(float(matches.confidence.mean())
                                 if len(matches.confidence) else 0.0),
                from_prior=bool(from_prior))
            candidates.append(candidate)
            if fit is None:
                continue
            sim2, _ = fit
            # Where this candidate put the walk, as four plain floats. The row
            # is json.dumps'd, so a Sim2 here would take every row down with it;
            # and a centroid in paper inches is what the gate's margin test
            # actually compares, so nothing is reconstructed later.
            #
            # Recorded on EVERY run, not just a chained one: it is what lets the
            # gate be calibrated from an ordinary independent eval, and it has
            # no say in which pose wins.
            centre = sim2.apply(view.camera_xy).mean(0)
            candidate.update(centre_x=float(centre[0]), centre_y=float(centre[1]),
                             yaw_deg=float(np.degrees(sim2.theta)),
                             scale=float(sim2.s))
            inlier_fraction = num_inliers / max(len(matches.source_world_xy), 1)
            if best is None or (num_inliers, inlier_fraction) > (best.num_inliers,
                                                                 best.inlier_fraction):
                keep = options.keep_match_artifacts
                best = PoseProposal(
                    sim2=sim2, up_index=up_index,
                    rotation_to_gravity_up=view.rotation_to_gravity_up,
                    window_index=window_index, num_inliers=num_inliers,
                    inlier_fraction=inlier_fraction,
                    num_matches=len(matches.source_world_xy),
                    num_verified=_verified_count(verified),
                    camera_height_above_floor=view.camera_height_above_floor,
                    camera_z=view.camera_z,
                    refine_was_on=options.refine_on_footprint,
                    density_map=density_map if keep else None,
                    crop=crop if keep else None,
                    matches=matches if keep else None,
                    inlier_mask=fit[1] if keep else None,
                    verified_mask=verified if keep else None)
                best_density_map, best_tolerance_inches = density_map, tolerance_inches
                best_window = window
    if best is None:
        return None
    best.candidates = candidates

    if options.refine_on_footprint and best_window is not None:
        # where the coarse pose says the walk sits, in paper inches
        footprint = best.sim2.apply(views[best.up_index].select_band_points())
        low, high = footprint.min(0), footprint.max(0)
        centre = (low + high) / 2
        side = max(float((high - low).max()) * options.refine_footprint_margin, 1e-6)
        half = side / 2
        crop = prior_bev.render_target(
            pool, (centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half),
            options.resolution, None, data_config)
        rng = np.random.default_rng(options.seed + 1)
        matches, fit, _, verified = (_match_and_fit(model, best_density_map, crop,
                                                   metres_per_inch, options, rng, device,
                                                   patch_size, options.mutual_check)
                                     if crop is not None else (None, None, None, None))
        if fit is not None:
            sim2, inlier_mask = fit
            num_inliers = int(inlier_mask.sum())
            cameras = views[best.up_index].camera_xy
            shift_inches = float(np.median(np.linalg.norm(
                sim2.apply(cameras) - best.sim2.apply(cameras), axis=1)))
            agrees = shift_inches <= (options.refine_agreement_patches
                                      * best_tolerance_inches / options.tolerance_patches)
            candidates.append(dict(
                up_index=best.up_index, window_index=best.window_index, refine=True,
                num_matches=len(matches.source_world_xy),
                num_verified=_verified_count(verified), num_inliers=num_inliers,
                agrees=bool(agrees), shift_m=shift_inches * metres_per_inch,
                mean_confidence=(float(matches.confidence.mean())
                                 if len(matches.confidence) else 0.0)))
            # the zoomed crop matches at finer scale; trust it when it agrees with the
            # coarse pose and is supported at least as well
            if agrees and num_inliers >= max(options.min_inliers, int(0.7 * best.num_inliers)):
                keep = options.keep_match_artifacts
                best = PoseProposal(
                    sim2=sim2, up_index=best.up_index,
                    rotation_to_gravity_up=best.rotation_to_gravity_up,
                    window_index=best.window_index, num_inliers=num_inliers,
                    inlier_fraction=num_inliers / max(len(matches.source_world_xy), 1),
                    num_matches=len(matches.source_world_xy),
                    num_verified=_verified_count(verified),
                    camera_height_above_floor=best.camera_height_above_floor,
                    camera_z=best.camera_z, refined=True,
                    refine_was_on=options.refine_on_footprint, candidates=candidates,
                    density_map=best_density_map if keep else None,
                    crop=crop if keep else None,
                    matches=matches if keep else None,
                    inlier_mask=inlier_mask if keep else None,
                    verified_mask=verified if keep else None)
    return best
