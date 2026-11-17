"""Place a capture's splits in walk order, each leaning on the one before.

A capture is one continuous walk cut into pieces, so the pieces are ordered and
where one ends the next begins. Measured over 1,742 adjacent pairs in 250
multi-split captures of `datasets/manifest_all_19_aug_23_sep_current.csv`, the
distance from the end of split A to the start of split B, in PAPER INCHES:

    contiguous (consecutive frame indices)   428 pairs  median 0.15  p90  0.50
    shared (the same photographs in both)    388 pairs  median 0.95  p90  2.84
    gap (recording stopped and restarted)    926 pairs  median 6.57  p90 18.01

Against a search that today tiles the whole prior into 32 windows ranked by wall
content -- so in the units the SEARCH works in, a placed neighbour pins its
successor to a fraction of one window, and translation is exactly the dimension
the matcher cannot resolve on its own.

Read those inches as inches, though, and not as physical closeness. The same
junctions in METRES, where `metres_per_inch` runs 2.36 to 30.4 across these
structures:

    contiguous   median  0.8 m   p90    1.8 m   p99   14 m
    shared       median  5.1 m   p90   15.3 m   p99   41 m
    gap          median 45.1 m   p90  122.6 m   p99  252 m

Contiguous and shared junctions really are tight. A GAP is not, and a gap is 92%
of all junctions: the recording stopped and the person walked somewhere else.
The prior still helps there -- 21 paper inches is a small part of a sheet -- but
nothing about a gap junction is physically constrained, which is why
`JUNCTION_MAX_M` can police the other two kinds and not this one.

The pure functions that order splits and classify their junctions come from
`align.chain`, which is the same idea for the classical solver. What does NOT
come from there:

  * `solve_capture`, whose seed is `find_anchor` over `geo_first_point.json` --
    absent from every current dataset, so it would take its no-anchor branch and
    run everything independently;
  * `gate_result`, which reads a ranked list of correlation scores this pipeline
    does not produce (see `gate.py`);
  * the radii. `Junction.radius_m` is in METRES, and `metres_per_inch` runs
    2.36 to 30.4 across the validation structures -- 15 m is 6.4 paper inches on
    one sheet and 0.49 on another, so a metre radius is a disc that excludes the
    truth on coarse sheets. The radii here are in paper inches, which is the
    frame the error was measured in.

`expected_start` is not imported either, for the same reason: it returns
`kind.radius_m / metres_per_inch`. The centre it computes is one line
(`sim2_a.apply(A's meeting camera)`) and is written out below so that nobody
later "fixes" it back onto the metre radius.

WHAT THE CHAIN DOES NOT CONSTRAIN is yaw. A single-point prior is satisfied just
as well by a pose rotated about that point, and heading continuity cannot be
substituted: the turn at a contiguous junction has a median of 52.8 degrees.
People turn where the recording breaks. Yaw has to come from the matcher.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from align.chain import (Junction, capture_order,  # noqa: F401
                         classify_junction, transfer_pose, walk_outward)

from ..utils.geometry import Sim2
from ..utils.io import Trajectory
from ..utils.metrics import rig_height_m
from .gate import Gate, GateThresholds, gate_proposal

# Search radii, in PAPER INCHES, from the measured table above with headroom:
# roughly the worst case seen, not the median.
RADIUS_SHARED_IN = 6.0          # 2x the measured p90 of 2.84
RADIUS_CONTIGUOUS_IN = 2.0      # 4x the measured p90 of 0.50
RADIUS_GAP_IN = 24.0            # the measured p90 of 18.01
# The GNSS fix, whose p90 against ground truth is 12.1 inches. Four inches is
# `align.chain.RADIUS_GEO_IN`, and is for a SURVEYED start point -- which this
# is not. A radius that excludes the truth is worse than no radius at all.
RADIUS_GEO_IN = 16.0
# The oracle seed, for the diagnostic run that asks whether the CHAIN works
# rather than whether the seed did. A couple of inches because the fix is exact;
# a radius any tighter would start testing the window geometry instead.
RADIUS_ORACLE_IN = 2.0
# Past this the disc is most of a sheet and the prior has stopped meaning
# anything; the ladder skips straight to an unconstrained search instead of
# pretending to be guided.
MAX_PRIOR_RADIUS_IN = 40.0

# REACHING BACK. A split whose immediate neighbour could not be placed is not
# stranded: anything already placed in the same capture is an anchor, and a
# capture stays on one floor, so a distant sibling is barely further away than
# an adjacent one. Measured end-to-start separation by walk-position gap, in
# paper inches, over `datasets/splits/prior_bev_test.csv`:
#
#     gap    1   p90 15.3"      gap  6-10   p90 22.5"
#     gap  2-3   p90 17.0"      gap 11-20   p90 22.8"
#     gap  4-5   p90 20.1"      gap 21-40   p90 25.4"
#
# About an inch of p90 per position over the first ten, then flat. p90 never
# exceeds 25.4" at ANY gap, which is what makes reaching back nearly free: the
# radius grows by inches, and `_windows_near` clips the ring offset to side/4
# and caps the count, so a wider disc costs no extra window and no extra solve.
REACH_RADIUS_GROWTH_IN = 1.0
# Where that growth saturates. 33" is 1.3x the worst p90 seen at any gap, and
# deliberately short of MAX_PRIOR_RADIUS_IN: unbounded growth would cross it and
# silently route every deep reach-back to an unconstrained search -- the
# opposite of what reaching back is for.
REACH_RADIUS_MAX_IN = 33.0
# The post-hoc junction ceiling, in METRES ON THE GROUND and per junction kind.
#
# It was briefly in paper inches, which is wrong and was caught by a placement
# that passed at 29.84" while being 100 m out: `metres_per_inch` runs 2.36 to
# 30.4 across these structures, so one inch-valued ceiling is an order of
# magnitude looser on some sheets than others. The RADIUS stays in paper inches,
# because the search windows are -- but whether two splits of one walk could
# really have met is a physical question, and it gets a physical answer.
#
# Measured end-to-start separation by kind, metres (n = 8,471 pairs):
#
#     contiguous  median  0.8   p90    1.8   p99   14
#     shared      median  5.1   p90   15.3   p99   41
#     gap         median 45.1   p90  122.6   p99  252
#
# Taken at the p99 of each. Note what this means for GAP, which is 92% of all
# junctions: there is NO tight bound. The recording stopped and the person
# walked somewhere else, and a hundred metres is an ordinary distance for that.
# This check polices contiguous and shared junctions, where continuity is real;
# it cannot police a gap, and saying so is better than pretending otherwise.
JUNCTION_MAX_M = {Junction.CONTIGUOUS: 15.0,
                  Junction.SHARED: 45.0,
                  Junction.GAP: 300.0}
JUNCTION_MAX_DEFAULT_M = 300.0

# A transferred pose whose shared photographs do not agree this well is not a
# transfer, it is two splits that happen to name some of the same files.
TRANSFER_MAX_RESIDUAL_IN = 2.0

# After a split is placed, the junction it was placed through is measured again
# under its OWN solved pose. More than this many radii of daylight and one of
# the two trajectories is broken -- the measured junctions run to 107 inches on
# contiguous pairs and far worse on gaps -- so the pose is not propagated even
# though it scored well. Propagating through a broken junction carries the break
# into every split after it.
JUNCTION_SANITY_FACTOR = 3.0

# How wide a sibling's scale brackets this one. B's floor fit is the noisy term;
# correct placements show about this much spread in implied rig height.
SCALE_BRACKET_SLACK = 0.20


# ---------------------------------------------------------------------------
# what travels along the chain

@dataclass
class PosePrior:
    """Where a split is believed to be, in paper inches, and how wrong that may be."""
    centre: np.ndarray
    radius_in: float
    source: str                       # geo | junction | transfer | oracle


@dataclass
class Placed:
    """A split that passed the gate, in the form the next split needs."""
    reality_id: str
    sim2: Sim2
    up_index: int
    trajectory: Trajectory            # gravity-rotated, the frame sim2 maps from
    rig_height_m: float


@dataclass
class ChainResult:
    """One split's outcome, in the shape the caller turns into a row.

    `metres_per_inch` rather than the `Prepared` it came from: a capture runs to
    70 splits, and holding every split's wall band until the capture finishes
    would cost more memory than the whole search.
    """
    status: str
    proposal: object | None = None
    metres_per_inch: float = float("nan")
    trajectory: Trajectory | None = None
    context: dict = field(default_factory=dict)
    # Only when `ChainPolicy.keep_prepared` asks: the wall band and the prior
    # pool a drawing needs. Off by default because a 70-split capture holding
    # all of them at once costs more memory than the search does.
    prepared: object | None = None


@dataclass
class ChainPolicy:
    """The knobs the CLI exposes. Defaults are the plan's, and need calibration."""
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    radius_scale: float = 1.0
    max_windows_with_prior: int = 5
    insurance_windows: int = 2
    # 0 = the prior attempt only, 1 = also the widened one, 2 = also the full
    # unconstrained search that today's flow does.
    retries: int = 2
    max_seed_attempts: int = 3
    # How many walk positions back a split may look for a sibling to lean on.
    # 1 is "the immediate neighbour, or nothing", which is what this replaced.
    # 20 covers 98% of the splits that have an anchor at all, and the radius
    # saturates long before that, so reaching further is free.
    max_reach: int = 20
    reach_radius_growth_in: float = REACH_RADIUS_GROWTH_IN
    # Walks over the capture. A split that failed when almost nothing was placed
    # deserves another look once its neighbours are down; 1 is a single walk.
    passes: int = 2
    # Fix-bearing splits offered as the capture's seed. The first gets the full
    # exhaustive ladder, the rest are screened at one rung.
    max_geo_seed_attempts: int = 3
    # The one cost bound on seeding, in SOLVES, across both seed loops.
    max_seed_solves: int = 8
    use_scale_bracket: bool = True
    seconds_per_capture: float = 0.0  # 0 = unbounded
    seed_source: str = "geo"          # geo | oracle | search | none
    # Hold each split's `Prepared` on its result, for callers that draw. See
    # `ChainResult.prepared` for why this is not the default.
    keep_prepared: bool = False
    # ... and never more than this many at once. A capture runs to 70 splits and
    # the drawing happens only after the capture finishes, so "keep them all"
    # is unbounded in the one direction that matters. A caller drawing the first
    # N placements needs a handful, not a hundred.
    max_kept_prepared: int = 12


# ---------------------------------------------------------------------------
# frames

def gravity_trajectory(trajectory: Trajectory,
                       rotation_to_gravity_up: np.ndarray) -> Trajectory:
    """`trajectory` in the gravity-up frame a `PoseProposal.sim2` maps FROM.

    The trap this exists to close: `align.chain` was written against the
    classical solver, whose Sim2 applies to raw trajectory positions. This
    pipeline's Sim2 applies to GRAVITY-ROTATED ones. Feeding `transfer_pose` the
    raw positions produces a transform that is wrong by the gravity rotation and
    looks entirely plausible.
    """
    rotated = np.asarray(trajectory.positions, dtype=np.float64) \
        @ np.asarray(rotation_to_gravity_up, dtype=np.float64).T
    return Trajectory(list(trajectory.names), np.asarray(trajectory.idx), rotated,
                      trajectory.frame)


def orderable(trajectories: dict) -> bool:
    """Whether these splits can be put in walk order at all.

    `align.io.traj.image_index` returns -1 for a filename carrying no IMG_
    number, and `load_shots` then sorts by name. A capture of such splits would
    get an order that looks fine and means nothing, so it is refused outright
    rather than chained on a fiction.
    """
    if len(trajectories) < 2:
        return False
    for trajectory in trajectories.values():
        idx = np.asarray(trajectory.idx)
        if len(idx) == 0 or (idx < 0).mean() > 0.2 or len(np.unique(idx)) < 2:
            return False
    return True


def radius_for(kind: Junction, scale: float = 1.0) -> float:
    """The search radius for a junction, in paper inches."""
    base = {Junction.SHARED: RADIUS_SHARED_IN,
            Junction.CONTIGUOUS: RADIUS_CONTIGUOUS_IN,
            Junction.GAP: RADIUS_GAP_IN}.get(kind, RADIUS_GAP_IN)
    return base * max(float(scale), 0.0)


def meeting_camera(trajectory: Trajectory, last: bool) -> np.ndarray:
    """The camera at the end of a split that meets its neighbour.

    Splits are ordered by image index, so the LAST camera of the earlier split
    meets the FIRST of the later one. Walking backward reverses which end is
    wanted, which is the whole of the `last` flag.
    """
    idx = np.asarray(trajectory.idx)
    position = int(np.argmax(idx)) if last else int(np.argmin(idx))
    return np.asarray(trajectory.positions)[position][:2]


def reach_anchor(reality_id: str, neighbour: str | None, order_pos: dict,
                 placed: dict, *, max_reach: int) -> tuple[str, int, bool] | None:
    """The nearest already-placed split, measured in WALK POSITIONS.

    Returns `(anchor, reach, anchor_is_earlier)`, or None when nothing placed is
    within `max_reach`. `reach` is 1 for the immediate neighbour, so
    `max_reach=1` reproduces the behaviour this replaced: lean on the split next
    door or give up.

    `placed` only ever receives splits that passed the gate -- `run` adds to it
    under `gate.passed` alone -- so "nearest placed" is already "nearest
    trusted", and because it grows as the walk proceeds, the right answer is
    simply whatever is in it at the moment of asking.

    Ties go to the side `neighbour` is on, which is the direction the walk is
    coming from. That is what keeps the adjacent case exactly what it was: when
    the immediate neighbour is placed, it is the unique winner at reach 1.
    """
    here = order_pos.get(reality_id)
    if here is None:
        return None
    toward = None
    if neighbour in order_pos:
        toward = order_pos[neighbour] < here
    best = None
    for rid in placed:
        there = order_pos.get(rid)
        if there is None or there == here:
            continue
        reach = abs(there - here)
        if reach > max_reach:
            continue
        earlier = there < here
        key = (reach, 0 if (toward is None or earlier == toward) else 1)
        if best is None or key < best[0]:
            best = (key, rid, reach, earlier)
    return None if best is None else (best[1], best[2], best[3])


def reach_radius_in(kind: Junction, reach: int, scale: float = 1.0,
                    growth_in: float = REACH_RADIUS_GROWTH_IN) -> float:
    """`radius_for`, widened by how far back the anchor sits.

    Saturates at `REACH_RADIUS_MAX_IN`. Deliberately NOT clamped to
    `MAX_PRIOR_RADIUS_IN`: that test lives once, in `run_ladder`, and repeating
    it here would quietly make a `--chain-radius-scale 2` prior usable that the
    ladder refuses today.
    """
    extra = max(int(reach) - 1, 0) * max(float(growth_in), 0.0)
    base = min(radius_for(kind, 1.0) + extra, REACH_RADIUS_MAX_IN)
    return base * max(float(scale), 0.0)


def junction_prior(placed: Placed, kind: Junction, forward: bool,
                   scale: float = 1.0, reach: int = 1,
                   growth_in: float = REACH_RADIUS_GROWTH_IN) -> PosePrior:
    """Where the next split starts, from the pose of a split already placed.

    `forward` means THE ANCHOR SITS EARLIER IN WALK ORDER -- not "the walk is
    going forwards". With a reach-back anchor the two can differ, and getting it
    backwards centres the prior on the far end of the anchor's walk: a
    plausible-looking point tens of inches from the truth.

    This is `align.chain.expected_start`'s centre with its metre radius left
    behind -- see the module docstring. Do not replace it with a call there.
    """
    camera = meeting_camera(placed.trajectory, last=forward)
    centre = placed.sim2.apply(np.asarray([camera]))[0]
    return PosePrior(centre=np.asarray(centre, dtype=np.float64),
                     radius_in=reach_radius_in(kind, reach, scale, growth_in),
                     source="junction")


def scale_bracket_from(rig_height_a_m: float, views: dict, metres_per_inch: float,
                       slack: float = SCALE_BRACKET_SLACK) -> tuple[float, float] | None:
    """A bracket on this split's Sim2 scale, from a sibling's solved rig height.

    One capture is one rig on one monopod, so the implied height is the same
    number in both splits. `rig_height_m = camera_height_above_floor * s * mpi`
    inverts to the scale this split must have had.

    The two gravity hypotheses put the camera at different heights above their
    respective floors, and `propose_pose` searches both under one bracket, so
    the bracket is the UNION over the usable views. A bracket that fits only the
    hypothesis that happens to be right would reject the right pose whenever the
    other hypothesis is the one being scored.
    """
    if not np.isfinite(rig_height_a_m) or rig_height_a_m <= 0 or metres_per_inch <= 0:
        return None
    los, his = [], []
    for view in views.values():
        height = float(view.camera_height_above_floor)
        if height <= 0:
            continue
        centre = rig_height_a_m / (height * metres_per_inch)
        los.append(centre * (1 - slack))
        his.append(centre * (1 + slack))
    if not los:
        return None
    return float(min(los)), float(max(his))


# ---------------------------------------------------------------------------
# the transfer, for splits that share photographs

@dataclass
class Transfer:
    sim2: Sim2
    up_index: int
    residual_in: float
    n_shared: int


def transfer_from(placed: Placed, prepared, trajectory: Trajectory) -> Transfer | None:
    """This split's pose from a solved sibling they share photographs with.

    The same photograph is at the same place in the world in both splits, so
    fitting the sibling's PAPER positions against this split's arbitrary-frame
    ones gives the pose outright, with no search.

    Fitted under BOTH gravity hypotheses and the closer one kept. That is not
    tidiness: the hypothesis whose fit is worse is the one whose floor normal
    points the wrong way, so a junction that shares photographs resolves this
    split's gravity sign for free -- which `sign_ambiguous` otherwise costs a
    whole extra branch of the search.

    It resolves the sign only where the shared stretch BENDS. A reflected
    straight line is still a straight line, and a similarity maps one onto the
    other exactly, so a shared run down a single corridor fits both hypotheses
    equally well and the tie is broken arbitrarily. That is not a failure -- the
    pose is right either way, because the fit that produced it is right either
    way -- but it does mean the sign is not always decided here.
    """
    best = None
    for up_index, view in prepared.views.items():
        rotated = gravity_trajectory(trajectory, view.rotation_to_gravity_up)
        fitted = transfer_pose(placed.trajectory, rotated, placed.sim2)
        if fitted is None:
            continue
        sim2, residual = fitted
        if best is None or residual < best.residual_in:
            shared = len(set(placed.trajectory.names) & set(trajectory.names))
            best = Transfer(sim2=sim2, up_index=up_index,
                            residual_in=float(residual), n_shared=shared)
    if best is None or best.residual_in > TRANSFER_MAX_RESIDUAL_IN:
        return None
    return best


def proposal_from_transfer(transfer: Transfer, prepared):
    """A `PoseProposal` carrying a transferred pose and no matcher evidence.

    Built only when the verifying search came back with nothing. The transfer is
    the most reliable signal in the data -- Umeyama on photographs the two
    splits genuinely share, median 0.95 paper inches -- so discarding it in
    favour of no pose at all would be throwing away the best junction we have.
    What it does NOT have is support of its own, so `num_inliers` is zero, the
    row says `transferred_unverified`, and the gate is never consulted: it would
    be scoring evidence that was never collected.
    """
    from .propose import PoseProposal
    view = prepared.views[transfer.up_index]
    return PoseProposal(
        sim2=transfer.sim2, up_index=transfer.up_index,
        rotation_to_gravity_up=view.rotation_to_gravity_up, window_index=-1,
        num_inliers=0, inlier_fraction=0.0, num_matches=0, num_verified=None,
        camera_height_above_floor=view.camera_height_above_floor,
        camera_z=view.camera_z, method="transfer")


def transfer_prior(transfer: Transfer, prepared, scale: float = 1.0) -> PosePrior:
    """A disc around where the transfer says this split's walk sits.

    Centred on the walk's centroid rather than on its start: the transfer places
    the whole split, not just its first camera, so there is no reason to throw
    the rest away.
    """
    centre = transfer.sim2.apply(prepared.views[transfer.up_index].camera_xy).mean(0)
    return PosePrior(centre=np.asarray(centre, dtype=np.float64),
                     radius_in=radius_for(Junction.SHARED, scale), source="transfer")


def geo_candidates(order, by_id, trajectories, geo_track_for, limit: int):
    """Fix-bearing splits offered as a capture's seed, best first.

    Candidate ONE is the earliest in the walk carrying both a track and a
    photograph the trajectory names -- exactly the split the single-attempt
    version took, so a limit of 1 is that behaviour and a wider run is a strict
    superset of it. Keeping it fixed is what makes a wider run readable: re-rank
    everything and a worse result could mean either "the extra candidates hurt"
    or "we moved the first one", with no way to tell which.

    The rest go by photograph count, the heuristic the unconstrained seed loop
    already uses: more images is more of the walk, and the inlier count the gate
    turns on tracks it.

    Worth knowing why more than one is worth trying. Of the 59 captures whose
    geo seed failed to gate, EVERY ONE has at least one split whose fix is
    within `RADIUS_GEO_IN` of the truth, and 55 of 59 have two or more; their
    median fix error, 3.2 inches, is better than that of the captures that did
    gate. The fixes are not the problem -- the one split that was tried is.
    """
    found = []
    for rid in order:
        track = geo_track_for(by_id[rid])
        if track is None:
            continue
        centre = track.first_of(trajectories[rid])
        if centre is not None:
            found.append((rid, centre))
    if not found:
        return []
    rest = sorted(found[1:], key=lambda item: -len(trajectories[item[0]]))
    return ([found[0]] + rest)[:max(int(limit), 1)]


# ---------------------------------------------------------------------------
# the recovery ladder

@dataclass
class Attempt:
    label: str
    proposal: object | None
    gate: Gate
    prior: PosePrior | None


def _with_prior(options, prior: PosePrior | None, bracket, policy: ChainPolicy,
                window_factor: int = 1):
    """`options` aimed at a prior. Without one this returns `options` itself."""
    if prior is None:
        return options
    return replace(options,
                   prior_centre=(float(prior.centre[0]), float(prior.centre[1])),
                   prior_radius_in=float(prior.radius_in),
                   max_windows_with_prior=policy.max_windows_with_prior * window_factor,
                   insurance_windows=policy.insurance_windows,
                   scale_bracket=bracket)


def run_ladder(solve, prepared, options, prior: PosePrior | None, bracket,
               policy: ChainPolicy, *, deadline: float | None = None,
               exhaustive: bool = False, screen: bool = False,
               widen: bool = True) -> list[Attempt]:
    """Search this split, widening and then giving up on the prior as it fails.

        L1  the prior, at its junction's radius, with the sibling's scale bracket
        L2  the prior widened threefold, twice the windows, bracket dropped
        L3  no prior at all -- byte for byte the search today's flow runs

    Every attempt is returned, failures included, so the caller can write the
    best of them when none passed and can say in the row how many it took.

    L3 is what makes this safe to turn on: the worst a chained split can do is
    the search it would have had anyway, plus the time spent finding that out.

    `screen` is for seed CANDIDATES after the first: one rung, then give up, so
    offering three of them costs barely more than offering one.

    `widen=False` drops L2. It is for REACH-BACK splits, measured: of 36 splits
    that a reach-back gave a prior to, 26 still failed the gate, and tripling a
    disc that was already the right place is not what they needed. Dropping the
    rung takes them from 1.59 full-search equivalents back to 1.22, against the
    1.00 they cost before they had a prior at all.

    `exhaustive` runs every rung even after one has passed, and is for SEEDS.
    A seed is the one split in a capture with nothing to lean on, so the only
    prior available to it is `geo.txt` -- whose p90 against ground truth is 12
    paper inches, against junctions of 0.15 to 6.6. A prior that coarse can
    place a split well enough to pass the gate while an unconstrained search
    would have placed it BETTER, and stopping at the first pass takes the worse
    answer: measured, on RLT1790009494426, 1.14 m from a geo-seeded rung where
    the unconstrained search reaches 0.65 m. One extra search per capture is a
    cheap price for not propagating the worse of the two, and `--chain-retries`
    still bounds it. Every other split stops at the first pass, because its
    prior is a junction rather than a GNSS fix and is the better information.
    """
    attempts: list[Attempt] = []

    def expired() -> bool:
        return deadline is not None and time.time() > deadline

    def attempt(label, use_prior, bracket_, window_factor=1):
        proposal = solve(prepared, _with_prior(options, use_prior, bracket_, policy,
                                               window_factor))
        gate = gate_proposal(proposal, prepared.metres_per_inch, policy.thresholds,
                             refine_was_on=options.refine_on_footprint)
        attempts.append(Attempt(label, proposal, gate, use_prior))
        return gate.passed

    def settled() -> bool:
        return any(a.gate.passed for a in attempts) and not exhaustive

    usable = prior is not None and prior.radius_in <= MAX_PRIOR_RADIUS_IN
    if usable:
        passed = attempt("chained", prior, bracket)
        if screen and not passed:
            # A seed CANDIDATE the fix could not place at its own radius. The
            # rungs below it are an unconstrained search by another name, and
            # the seed loop has somewhere cheaper to spend them: the next
            # candidate's prior. Stop here and let it. A candidate that DID
            # pass falls through and climbs the rest of the ladder in this same
            # call, so its first rung is never solved twice.
            return attempts
        if widen and not settled() and policy.retries >= 1 and not expired():
            widened = replace(prior, radius_in=prior.radius_in * 3)
            attempt("widened", widened, None, window_factor=2)
    # No prior, or a prior that got us nowhere. An unconstrained search is both
    # the recovery and the thing we would have done without any of this.
    if not usable or (not settled() and policy.retries >= 2 and not expired()):
        # "primary" when there was never a prior -- the caller's own label for
        # the attempt then stands, so a seed stays a seed in the row.
        attempt("recovered" if usable else "primary", None, None)
    return attempts


def _support(attempt: "Attempt") -> tuple[float, int]:
    return (attempt.proposal.inlier_fraction, attempt.proposal.num_inliers)


def best_attempt(attempts: list[Attempt]) -> Attempt | None:
    """The best-supported attempt that passed, else the best that found a pose.

    Picking the best rather than the first matters only under
    `run_ladder(exhaustive=True)`; everywhere else the ladder stops at its first
    pass, so there is only one to choose from.
    """
    passed = [a for a in attempts if a.gate.passed and a.proposal is not None]
    if passed:
        return max(passed, key=_support)
    scored = [a for a in attempts if a.proposal is not None]
    if not scored:
        return attempts[-1] if attempts else None
    return max(scored, key=_support)


# ---------------------------------------------------------------------------
# the capture

def junction_distance_in(placed: Placed, proposal, prepared,
                         trajectory: Trajectory, forward: bool) -> float:
    """How far apart the two splits' meeting cameras actually ended up.

    Measured after the fact, under each split's OWN solved pose, which is what
    makes it a check rather than a restatement of the prior.
    """
    rotated = gravity_trajectory(
        trajectory, prepared.views[proposal.up_index].rotation_to_gravity_up)
    here = proposal.sim2.apply(np.asarray([meeting_camera(rotated, last=not forward)]))[0]
    there = placed.sim2.apply(np.asarray([meeting_camera(placed.trajectory,
                                                         last=forward)]))[0]
    return float(np.linalg.norm(np.asarray(here) - np.asarray(there)))


def _placed_from(reality_id, proposal, prepared, trajectory) -> Placed:
    rotated = gravity_trajectory(
        trajectory, prepared.views[proposal.up_index].rotation_to_gravity_up)
    return Placed(reality_id=reality_id, sim2=proposal.sim2,
                  up_index=proposal.up_index, trajectory=rotated,
                  rig_height_m=rig_height_m(proposal.camera_height_above_floor,
                                            proposal.sim2.s,
                                            prepared.metres_per_inch))


def place_capture_chain(members, prepare, solve, options, policy: ChainPolicy,
                        *, geo_track_for=None, trajectory_for=None,
                        log=print) -> dict:
    """Place every split of one capture in walk order. Returns reality_id -> ChainResult.

    `prepare(case, context) -> (Prepared | None, status | None)` and
    `solve(prepared, options) -> PoseProposal | None` are the caller's; the
    sequencing is here. `geo_track_for(case) -> GeoTrack | None` supplies the
    bootstrap fix.

    The chain only ever leans on a split that PASSED the gate. Where one fails,
    it keeps its own best-effort pose and the chain re-seeds from the next split
    that passes, rather than carrying a bad pose through the rest of the capture.
    """
    by_id = {case.reality_id: case for case in members}
    trajectory_for = trajectory_for or _trajectory_of
    trajectories, results = {}, {}
    for case in members:
        try:
            trajectories[case.reality_id] = trajectory_for(case)
        except Exception:                                         # noqa: BLE001
            trajectories[case.reality_id] = None
    usable_trajectories = {rid: t for rid, t in trajectories.items() if t is not None}

    prepared_cache: dict = {}
    placed: dict[str, Placed] = {}
    searched: set = set()
    spent = {"solves": 0}
    kept = {"n": 0}
    started = time.time()

    def deadline():
        if not policy.seconds_per_capture:
            return None
        return started + float(policy.seconds_per_capture)

    def prepare_once(rid):
        if rid not in prepared_cache:
            context: dict = {}
            prepared, status = prepare(by_id[rid], context)
            prepared_cache[rid] = (prepared, status, context)
        return prepared_cache[rid]

    def _release(rid):
        """Drop a finished split's wall band. `Placed` keeps what the chain needs."""
        prepared_cache.pop(rid, None)

    def _keep(prepared):
        """`prepared`, if a caller asked to draw and the quota is not yet spent.

        The cap has to be applied HERE rather than on the cache, because what
        keeps a wall band alive is the reference the `ChainResult` carries, not
        the cache entry -- dropping the cache while every result still holds one
        frees nothing at all.
        """
        if not policy.keep_prepared or kept["n"] >= policy.max_kept_prepared:
            return None
        kept["n"] += 1
        return prepared

    def run(rid, prior, how, *, seeded_from=None, kind=None, bracket=None,
            transfer=None, forward=True, exhaustive=False, screen=False,
            reach=None, pass_no=1, widen=True):
        """One split, through the ladder, gated, recorded."""
        prepared, status, context = prepare_once(rid)
        if prepared is None:
            results[rid] = ChainResult(status, context=dict(
                context, chain_mode="chain", chain_how=f"{how} ({status})"),
                trajectory=trajectories.get(rid))
            return None
        attempts = run_ladder(solve, prepared, options, prior, bracket, policy,
                              deadline=deadline(), exhaustive=exhaustive,
                              screen=screen, widen=widen)
        if any(a.label in ("primary", "recovered") for a in attempts):
            # This split has now had its unconstrained search. Recorded rather
            # than inferred from `policy.retries`, which is wrong for a screened
            # seed candidate that only ever ran one rung.
            searched.add(rid)
        best = best_attempt(attempts)
        chain = dict(chain_mode="chain", chain_seeded_from=seeded_from,
                     chain_junction=kind.value if kind else None,
                     chain_attempt=len(attempts),
                     chain_prior_radius_in=(round(prior.radius_in, 2) if prior else None),
                     chain_prior_centre_x=(round(float(prior.centre[0]), 3)
                                           if prior else None),
                     chain_prior_centre_y=(round(float(prior.centre[1]), 3)
                                           if prior else None),
                     chain_reach=reach, chain_pass=pass_no,
                     chain_n_shared=(transfer.n_shared if transfer else None),
                     chain_fit_residual_in=(round(transfer.residual_in, 3)
                                            if transfer else None))
        scale = prepared.metres_per_inch

        proposal = None if best is None else best.proposal
        gate = Gate(False, reason="no pose") if best is None else best.gate
        # The ladder names the RUNG; the caller names the ROLE the split was
        # playing. They are different facts and both go in the row.
        #
        # The role wins outright when there was no neighbour to lean on -- a
        # seed that needed its disc widened is still the capture's seed, and
        # labelling it "widened" loses the only thing about it that matters to
        # anyone reading the chain. Where there WAS a neighbour, a fallback rung
        # is the more informative label: "recovered" says the prior was abandoned
        # and the chain re-seeded here, which "chained" would hide.
        rung = None if best is None else best.label
        if best is None or seeded_from is None or best.label in ("chained", "primary"):
            label = how
        else:
            label = best.label
        chain["chain_rung"] = rung

        if proposal is None and transfer is not None:
            # The search found nothing to verify the transfer with. Keep the
            # transfer: it is a pose, it is a good one, and the row says plainly
            # that nothing independent backed it up.
            proposal = proposal_from_transfer(transfer, prepared)
            gate = Gate(True, reason="", rig_height_m=rig_height_m(
                proposal.camera_height_above_floor, proposal.sim2.s, scale))
            label = "transferred_unverified"
        if proposal is None:
            results[rid] = ChainResult("no_pose", metres_per_inch=scale, context=dict(
                context, **chain, chain_how=f"{label} (no pose)", **gate.as_row()),
                trajectory=usable_trajectories.get(rid),
                prepared=_keep(prepared))
            _release(rid)
            return None

        chain["chain_winner_from_prior"] = _winner_from_prior(proposal)

        # The junction the pose was reached through, re-measured under the pose
        # itself. A split can score beautifully and still sit somewhere its own
        # trajectory says it cannot be.
        if gate.passed and seeded_from in placed and prior is not None:
            realised = junction_distance_in(placed[seeded_from], proposal, prepared,
                                            usable_trajectories[rid], forward)
            chain["chain_junction_realised_in"] = round(realised, 2)
            # Two ceilings, in the two units that mean something. The first is
            # the prior's own: a pose that landed far outside the disc it was
            # searched in. The second is physical, and is the one that catches a
            # walk that could not have gone where the pose says it did.
            realised_m = realised * scale
            ceiling_m = JUNCTION_MAX_M.get(kind, JUNCTION_MAX_DEFAULT_M)
            chain["chain_junction_realised_m"] = round(realised_m, 1)
            if realised > JUNCTION_SANITY_FACTOR * max(prior.radius_in, 1e-6):
                gate = replace(gate, passed=False,
                               reason=f"junction implausible: {realised:.1f} in "
                                      f"outside a {prior.radius_in:.0f} in prior")
            elif realised_m > ceiling_m:
                gate = replace(gate, passed=False,
                               reason=f"junction implausible: {realised_m:.0f} m "
                                      f"across a {kind.value if kind else '?'} junction")
            if not gate.passed:
                label = f"{label} (junction rejected)"
        # "best_effort" means a ladder was climbed and nothing on it held. A
        # split that never had a prior climbed nothing, so it keeps the label
        # saying what it was -- `gate_passed` already says it did not hold.
        if not gate.passed and prior is not None \
                and not any(a.gate.passed for a in attempts):
            # "best_effort" means a ladder was climbed and nothing on it held.
            # A screened seed candidate climbed one rung, which is not that.
            label = f"{how} (not placed)" if screen else "best_effort"

        outcome = _placed_from(rid, proposal, prepared, usable_trajectories[rid]) \
            if gate.passed else None
        results[rid] = ChainResult("ok", proposal=proposal, metres_per_inch=scale,
                                   trajectory=usable_trajectories.get(rid),
                                   context=dict(context, **chain, chain_how=label,
                                                **gate.as_row()),
                                   prepared=_keep(prepared))
        if outcome is not None:
            placed[rid] = outcome
        _release(rid)
        return outcome

    def charge(rid):
        """Count what a seed attempt actually spent, across both loops."""
        spent["solves"] += int(results[rid].context.get("chain_attempt", 0) or 0)

    def seed_from_fix(walk):
        """The capture's start point, from a fix outside the search.

        `walk` is the splits in walk order, and `geo_candidates` takes the
        EARLIEST of them carrying a fix as candidate one -- so the first split
        seeds the capture whenever it has one, and the rest chain outward from
        it. Returns the seeded reality id, or None.

        geo and oracle are the same thing -- somebody outside the search saying
        where a photograph was -- so they share this path and differ only in how
        far that claim may be wrong. `oracle` is a diagnostic, never production.
        """
        if policy.seed_source not in ("geo", "oracle") or geo_track_for is None:
            return None
        source = policy.seed_source
        radius = (RADIUS_ORACLE_IN if source == "oracle" else RADIUS_GEO_IN)
        for position, (rid, centre) in enumerate(geo_candidates(
                walk, by_id, usable_trajectories, geo_track_for,
                policy.max_geo_seed_attempts)):
            if spent["solves"] >= policy.max_seed_solves:
                break
            prior = PosePrior(np.asarray(centre, dtype=np.float64),
                              radius * policy.radius_scale, source)
            log(f"      seed {rid} from {source} (+-{prior.radius_in:.0f} in)"
                + ("  [screen]" if position else ""))
            outcome = run(rid, prior, f"seed ({source})", exhaustive=True,
                          screen=bool(position))
            charge(rid)
            if outcome is not None:
                return rid
        return None

    # ---- nothing to chain over -------------------------------------------
    if not orderable(usable_trajectories):
        why = ("one split" if len(usable_trajectories) < 2
               else "no image index")
        # A lone split has nothing to CHAIN with, but it still has a START
        # POINT, and those are different things. Returning here outright meant
        # 42 of the 134 test captures carrying a usable fix never read it: they
        # searched the whole sheet for a position already known to within
        # RADIUS_GEO_IN, 32-128 windows for what a prior places in 5+2.
        #
        # Walk order is unavailable by definition here -- that is what
        # `orderable` refused -- so the splits go in CSV order, which for the
        # one-split case this mostly serves is the only split there is.
        seed_from_fix(list(usable_trajectories))
        for case in members:
            if case.reality_id not in results:
                run(case.reality_id, None, f"independent ({why})")
        return results

    order = capture_order(usable_trajectories)
    for rid in trajectories:
        if rid not in order:
            run(rid, None, "independent (no trajectory)")

    # ---- the seed ---------------------------------------------------------
    seed_rid = seed_from_fix(order)
    if seed_rid is None and policy.seed_source != "none":
        # Most photographs first: more images is more of the walk, and the
        # inlier count that the gate turns on tracks it.
        by_images = sorted(order, key=lambda r: -len(usable_trajectories[r]))
        attempts = 0
        for rid in by_images:
            if attempts >= max(int(policy.max_seed_attempts), 1) \
                    or spent["solves"] >= policy.max_seed_solves:
                break
            if rid in results and results[rid].context.get("gate_passed"):
                seed_rid = rid
                break
            if rid in searched:
                # Its ladder already ended on an unconstrained search, so
                # repeating it costs solves and cannot find anything new. A
                # SCREENED candidate is not in this set -- it ran one rung --
                # so it is still offered here, which is the point.
                continue
            attempts += 1
            log(f"      seed {rid} by unconstrained search")
            outcome = run(rid, None, "seed (search)")
            charge(rid)
            if outcome is not None:
                seed_rid = rid
                break

    # ---- the walk ---------------------------------------------------------
    if seed_rid is None:
        # Nothing to lean on. Every split runs the search it would have run
        # anyway, and the first that gates re-seeds a walk over the rest.
        for rid in order:
            if rid in results:
                continue
            if run(rid, None, "independent (no seed)") is not None:
                # It passed the gate, so the walk starts here: it is the seed,
                # whatever it was called on the way in. Left as "independent"
                # this undercounts seeds and makes the chain look like it
                # started from nowhere.
                results[rid].context["chain_how"] = "seed (fallback)"
                seed_rid = rid
                break
    if seed_rid is None:
        for rid in order:
            if rid not in results:
                run(rid, None, "independent (no seed)")
        return results

    order_pos = {rid: position for position, rid in enumerate(order)}
    last_reach: dict = {}

    def walk_once(pass_no: int) -> int:
        """One walk outward from the seed. Returns how many splits it placed."""
        gained = 0
        for rid, frm, _walk_forward in walk_outward(order, order.index(seed_rid)):
            if rid in results and results[rid].context.get("gate_passed"):
                continue
            anchored = reach_anchor(rid, frm, order_pos, placed,
                                    max_reach=max(int(policy.max_reach), 1))
            if anchored is None:
                if pass_no == 1:
                    run(rid, None, "independent (no placed split in reach)",
                        seeded_from=frm, pass_no=pass_no)
                continue
            anchor, reach, forward = anchored
            if pass_no > 1 and rid in last_reach and reach >= last_reach[rid]:
                # Nothing improved for this split since it last ran: the same
                # anchor at the same distance would buy the same answer, and
                # a pass that re-solves every failure for nothing is just a
                # second bill.
                continue
            last_reach[rid] = reach
            earlier, later = (anchor, rid) if forward else (rid, anchor)
            link = classify_junction(usable_trajectories[earlier],
                                     usable_trajectories[later])
            prepared = prepare_once(rid)[0]
            bracket = (scale_bracket_from(placed[anchor].rig_height_m, prepared.views,
                                          prepared.metres_per_inch)
                       if policy.use_scale_bracket and prepared is not None else None)
            transfer = None
            prior = junction_prior(placed[anchor], link.kind, forward,
                                   policy.radius_scale, reach=reach,
                                   growth_in=policy.reach_radius_growth_in)
            how = "chained"
            if link.kind is Junction.SHARED and prepared is not None:
                # Distance in the walk says nothing about whether two splits
                # share photographs -- `classify_junction` decides that by name
                # -- and a transfer is the best junction in the data, so it is
                # still worth asking for against a far anchor.
                transfer = transfer_from(placed[anchor], prepared,
                                         usable_trajectories[rid])
                if transfer is not None:
                    prior = transfer_prior(transfer, prepared, policy.radius_scale)
                    how = "transferred_verified"
            if run(rid, prior, how, seeded_from=anchor, kind=link.kind,
                   bracket=bracket, transfer=transfer, forward=forward,
                   reach=reach, pass_no=pass_no,
                   # A reach-back that L1 could not place is not helped by
                   # tripling the disc; it goes straight to the unconstrained
                   # search it would have had without any of this.
                   widen=(reach <= 1)) is not None:
                gained += 1
        return gained

    for pass_no in range(1, max(int(policy.passes), 1) + 1):
        if pass_no > 1 and deadline() is not None and time.time() > deadline():
            break
        if walk_once(pass_no) == 0 and pass_no > 1:
            break               # a pass that placed nothing will not do better

    for rid in order:
        if rid not in results:
            run(rid, None, "independent (unreached)")
    return results


def _winner_from_prior(proposal) -> bool | None:
    """Whether the winning window came from the prior rather than wall content.

    The soft-constraint signal: a run whose winners keep coming from the
    insurance windows is a run whose priors are wrong, and that is worth reading
    off a column rather than inferring from the error.
    """
    for candidate in proposal.candidates:
        if (candidate.get("up_index") == proposal.up_index
                and candidate.get("window_index") == proposal.window_index
                and not candidate.get("refine")):
            return bool(candidate.get("from_prior", False))
    return None


def _trajectory_of(case):
    from ..utils import io
    return io.load_camera_trajectory(case)


def add_chain_arguments(parser) -> None:
    """The capture-chaining flags, identical on `infer` and `evaluate`.

    Every one of them is inert unless --chain is given, and --chain is off by
    default, so a command written before any of this existed runs exactly as it
    did. The gate thresholds are exposed because they are provisional: see
    `bevaligner/inference/gate.py` and `scripts/e11_chain_gate.py`.
    """
    group = parser.add_argument_group("capture chaining")
    group.add_argument("--chain", action="store_true",
                       help="place each capture's splits in walk order, every "
                            "split seeded from the one before it. Rows come out "
                            "capture-major rather than in CSV order")
    group.add_argument("--chain-seed", choices=["geo", "search", "none"],
                       default="geo",
                       help="where a capture's first pose comes from. geo: the "
                            "geo.txt fix, falling back to a search; search: an "
                            "unconstrained search on the longest split; none: no "
                            "seed, the first split that passes the gate starts "
                            "the chain")
    group.add_argument("--chain-radius-scale", type=float, default=1.0,
                       metavar="F", help="multiply every junction radius by this")
    group.add_argument("--chain-max-windows", type=int, default=5, metavar="N",
                       help="windows drawn around the prior, per gravity view")
    group.add_argument("--chain-insurance-windows", type=int, default=2,
                       metavar="N",
                       help="wall-richest windows kept whatever the prior says. "
                            "0 makes the prior a hard filter")
    group.add_argument("--chain-retries", type=int, default=2, metavar="N",
                       help="0: the prior only; 1: also a widened prior; 2: also "
                            "the full unconstrained search this flow runs today")
    group.add_argument("--chain-max-seed-attempts", type=int, default=3,
                       metavar="N")
    group.add_argument("--chain-max-reach", type=int, default=ChainPolicy.max_reach,
                       metavar="N",
                       help="how many walk positions back a split may look for a "
                            "gate-passing sibling to lean on. 1 is the immediate "
                            "neighbour only, which is what this replaced")
    group.add_argument("--chain-reach-radius-growth", type=float,
                       default=REACH_RADIUS_GROWTH_IN, metavar="IN",
                       help="paper inches added to the junction radius per extra "
                            f"walk position, saturating at {REACH_RADIUS_MAX_IN:.0f}")
    group.add_argument("--chain-passes", type=int, default=ChainPolicy.passes,
                       metavar="N",
                       help="walks over the capture. A second pass retries the "
                            "splits whose nearest anchor has since moved closer; "
                            "1 is a single walk")
    group.add_argument("--chain-max-geo-seeds", type=int,
                       default=ChainPolicy.max_geo_seed_attempts, metavar="N",
                       help="fix-bearing splits offered as the capture's seed. The "
                            "first gets the full ladder, the rest one rung each")
    group.add_argument("--chain-max-seed-solves", type=int,
                       default=ChainPolicy.max_seed_solves, metavar="N",
                       help="total searches a capture may spend finding a seed, "
                            "across the geo and unconstrained loops")
    group.add_argument("--chain-no-scale-bracket", action="store_true",
                       help="do not bracket a split's scale by its sibling's "
                            "solved rig height")
    group.add_argument("--chain-seconds-per-capture", type=float, default=0.0,
                       metavar="S",
                       help="past this, the remaining splits of a capture stop "
                            "retrying; 0 is unbounded")
    group.add_argument("--chain-gate-inlier-frac", type=float, default=0.60)
    group.add_argument("--chain-gate-min-inliers", type=int,
                       default=GateThresholds.min_inliers,
                       help="an absolute floor on support. The inlier "
                            "FRACTION is the signal that transfers "
                            "between matchers; this count does not")
    group.add_argument("--chain-gate-margin", type=float, default=0.05)


def add_oracle_seed_argument(parser) -> None:
    """`--chain-oracle-seed`, which reads ground truth and so is eval-only.

    The diagnostic that decides what to fix next. If an oracle-seeded run is far
    better than a geo-seeded one, the chain works and the seeding is the
    problem; if the two are alike, the propagation is -- the radii, the frames
    or the gate -- and a better seed would buy nothing.
    """
    parser.add_argument("--chain-oracle-seed", action="store_true",
                        help="seed each capture at its TRUE first camera instead "
                             "of at geo.txt; needs scoring, never production")


def chain_policy(args) -> ChainPolicy:
    return ChainPolicy(
        thresholds=GateThresholds(
            min_inlier_fraction=args.chain_gate_inlier_frac,
            min_inliers=args.chain_gate_min_inliers,
            min_margin=args.chain_gate_margin),
        radius_scale=args.chain_radius_scale,
        max_windows_with_prior=args.chain_max_windows,
        insurance_windows=args.chain_insurance_windows,
        retries=args.chain_retries,
        max_seed_attempts=args.chain_max_seed_attempts,
        use_scale_bracket=not args.chain_no_scale_bracket,
        seconds_per_capture=args.chain_seconds_per_capture,
        seed_source=args.chain_seed,
        max_reach=args.chain_max_reach,
        reach_radius_growth_in=args.chain_reach_radius_growth,
        passes=args.chain_passes,
        max_geo_seed_attempts=args.chain_max_geo_seeds,
        max_seed_solves=args.chain_max_seed_solves)
