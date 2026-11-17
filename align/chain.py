"""Sequential registration: order a capture's splits and carry a pose along them.

A capture is one continuous walk cut into pieces. The pieces are ordered by
image index, and where one ends the next begins -- so once any piece is placed,
its neighbour's start is known to within a few metres. That is a far better
constraint than the coarse score can currently produce on its own, which fails
specifically on translation.

Measured over the 68 adjacent pairs in the dataset, end-to-start distance in
paper inches:

    contiguous (consecutive frame indices)   40 pairs   median 0.18, max 1.22
    overlapping / nested                      9 pairs   median 0.81, max 8.58
    index gap (recording stopped)            19 pairs   median 1.04, max 9.54

against the GPS start point's median 8.7 m, p90 35 m, max 157 m. The GPS fix is
only the bootstrap; the chain is what actually pins things down.

Ordering needs image names and indices, which `Trajectory` already carries from
the same sanctioned shim that supplies positions -- no new reach into ground
truth.

Two things measured while building this, both worth keeping:

  The prior works, on translation. Given the TRUE start position, two of three
  test splits went from 64.5 m and 12.1 m of error down to 2.1 m and 4.3 m.
  Translation was the dimension the coarse score could not resolve, and a disc
  around a known point resolves it.

  It does not constrain YAW, and heading continuity cannot be used to. A
  single-point prior is satisfied just as well by a pose rotated about that
  point -- the third test split stayed wrong and flipped to 178 degrees out.
  The obvious fix, requiring the walk direction to carry across the seam, does
  not survive contact with the data: the turn at a contiguous junction has a
  median of 52.8 degrees and only 18 of 40 are under 45. People turn where the
  recording breaks. Yaw has to come from the wall term.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from align.frames import Sim2

# Search radius by junction kind, in METRES, taken from the measured spread
# above with headroom: roughly the worst case seen, not the median.
RADIUS_SHARED_M = 2.0
RADIUS_CONTIGUOUS_M = 5.0
RADIUS_GAP_M = 15.0

# The start point is measured in PAPER INCHES, because that is the frame it is
# actually expressed in and the error is proportionate to the drawing rather
# than to the ground. Against the camera each one names: georeferenced splits
# median 0.63 in, p90 2.03, max 2.99; paper-frame splits median 0.59, p90 1.35,
# with a single 33.7 outlier. Four inches covers p90 twice over.
#
# On the +-20 window the prior actually uses, 56 of 57 captures fall inside 4
# inches. The exception is RLT1788201055417 / STRNP423t7urx7q at 24 in (73 m),
# whose walk runs off its sheet and which E2 showed carries zero oracle inliers
# anyway -- a broken case to flag, not a reason to widen the radius for all.
RADIUS_GEO_IN = 4.0

# Cameras either side of the photograph the start point names, which together
# form the prior's reference set. See `search.TranslationPrior` for the table
# this comes from: the window beats the exact camera on error while keeping the
# admissible region five times tighter than the whole split. Do not widen it.
ANCHOR_WINDOW = 20


class Junction(str, Enum):
    SHARED = "shared"            # the two splits contain the same photographs
    CONTIGUOUS = "contiguous"    # consecutive frame indices, no overlap
    GAP = "gap"                  # recording stopped and restarted
    NONE = "none"                # not adjacent at all

    @property
    def radius_m(self) -> float:
        return {Junction.SHARED: RADIUS_SHARED_M,
                Junction.CONTIGUOUS: RADIUS_CONTIGUOUS_M,
                Junction.GAP: RADIUS_GAP_M}.get(self, RADIUS_GAP_M)


@dataclass
class Link:
    """One ordered pair of splits within a capture."""
    a: str                      # reality id, earlier by image index
    b: str                      # reality id, later
    kind: Junction
    index_gap: int              # frames between them; <= 0 means they overlap
    n_shared: int


def index_range(traj) -> tuple[int, int]:
    return int(np.min(traj.idx)), int(np.max(traj.idx))


def capture_order(trajectories: dict[str, "object"]) -> list[str]:
    """Reality ids ordered along the walk, by first image index."""
    return sorted(trajectories, key=lambda rid: index_range(trajectories[rid])[0])


def classify_junction(traj_a, traj_b) -> Link:
    """How two splits of one capture meet."""
    a0, a1 = index_range(traj_a)
    b0, b1 = index_range(traj_b)
    shared = set(traj_a.names) & set(traj_b.names)
    if shared:
        kind = Junction.SHARED
    elif b0 == a1 + 1:
        kind = Junction.CONTIGUOUS
    elif b0 <= a1:
        kind = Junction.SHARED if shared else Junction.GAP
    else:
        kind = Junction.GAP
    return Link(a="", b="", kind=kind, index_gap=int(b0 - a1),
                n_shared=len(shared))


def _umeyama_sim2(src: np.ndarray, dst: np.ndarray) -> Sim2:
    """Least-squares similarity taking `src` onto `dst`, both (N,2).

    Umeyama's closed form, restricted to the plane. Used where two splits share
    photographs: the same camera in both frames pins the transform between them
    outright.
    """
    src = np.asarray(src, dtype=np.float64)[:, :2]
    dst = np.asarray(dst, dtype=np.float64)[:, :2]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src)
    scale = float((D * np.diag(S)).sum() / var) if var > 0 else 1.0
    t = mu_d - scale * (R @ mu_s)
    return Sim2(s=scale, theta=float(np.arctan2(R[1, 0], R[0, 0])),
                tx=float(t[0]), ty=float(t[1]))


def transfer_pose(traj_a, traj_b, sim2_a: Sim2) -> tuple[Sim2, float] | None:
    """Place split B exactly, from shared photographs with a solved split A.

    The same image appears in both splits at the same place in the world, so
    fitting A's paper positions against B's arbitrary-frame positions gives B's
    pose with no search at all. Returns the pose and the RMS fit residual in
    paper units, or None when too few images are shared.
    """
    shared = sorted(set(traj_a.names) & set(traj_b.names))
    if len(shared) < 3:
        return None
    ia = {n: k for k, n in enumerate(traj_a.names)}
    ib = {n: k for k, n in enumerate(traj_b.names)}
    src = np.array([traj_b.positions[ib[n]][:2] for n in shared])
    dst = sim2_a.apply(np.array([traj_a.positions[ia[n]][:2] for n in shared]))
    sim = _umeyama_sim2(src, dst)
    resid = float(np.sqrt(((sim.apply(src) - dst) ** 2).sum(1).mean()))
    return sim, resid


def expected_start(traj_a, traj_b, sim2_a: Sim2, kind: Junction,
                   metres_per_inch: float) -> tuple[np.ndarray, float]:
    """Where split B should begin, in paper units, and how far to search.

    The walk is continuous, so B's first camera sits essentially where A's last
    one did. Which end of A matters: the splits are ordered by image index, so
    it is A's LAST camera that meets B's first.
    """
    last_a = traj_a.positions[int(np.argmax(traj_a.idx))][:2]
    centre = sim2_a.apply(np.asarray([last_a]))[0]
    return centre, float(kind.radius_m / max(metres_per_inch, 1e-9))


def geo_start(case, metres_per_inch: float) -> tuple[np.ndarray, float] | None:
    """The surveyed start point in paper units, with its search radius.

    It lives in the structure's TARGET frame, exactly like the camera positions:
    UTM metres where a design transform exists, paper inches where it does not.
    The `crs` field is set either way and is misleading on the second group --
    a paper-frame fix reads x=28.6, y=16.1 on a 42x30 sheet, with z=1.7 for
    every split, which is the rig height rather than an elevation.

    This was originally restricted to georeferenced structures on the
    assumption the fix was always UTM. It is not, and lifting that restriction
    takes the anchored splits from 28 to 60.
    """
    from align.frames import target_to_paper
    if case.geo is None:
        return None
    xy = target_to_paper(np.array([[case.geo.x, case.geo.y, case.geo.z]]),
                         case.structure.design_tm)[0, :2]
    return xy, float(RADIUS_GEO_IN)


@dataclass
class Anchor:
    """Where the capture's surveyed start point attaches to the walk."""
    reality_id: str             # the split that actually contains the image
    centre: np.ndarray          # the start point, in paper inches
    ref: np.ndarray             # (M, 2) cameras around it, in that split's frame
    radius: float               # paper inches
    image: str                  # the photograph the fix names
    window: int


def capture_geo(cases):
    """The start point belonging to a capture, from whichever split carries it.

    `Case.geo` is populated only where `geo_first_point.json`'s
    `source_reality_id` matches the split, so a capture's anchor can hang off
    any one of its members -- and the id it names is not reliably the split
    holding the photograph either. Capture-level code has to consult them all.
    """
    for case in cases:
        if case.geo is not None:
            return case.geo
    return None


def find_anchor(cases, trajectories: dict, geo, metres_per_inch: float,
                window: int = ANCHOR_WINDOW) -> Anchor | None:
    """Which split the start point anchors, and the cameras it may match.

    `geo_first_point.json` says which capture moment it describes twice over --
    `source_reality_id` names a split and `image` names a photograph -- and the
    two disagree: of the 59 splits owning a start point, 21 do not contain the
    image their own start point names, and 19 of those have it in a sibling.
    The photograph is the honest identifier, so it decides; the reality id
    decides only which capture the point belongs to.

    Returns None when no split contains the image, which is the correct answer
    for two captures. The previous code silently fell back to the split's first
    camera, which told the search "this camera is at the start point" about a
    camera that was not -- and inside a 4-inch radius that EXCLUDES the true
    pose, leaving it worse off than with no prior at all.
    """
    if geo is None:
        return None
    from align.frames import target_to_paper
    centre = target_to_paper(np.array([[geo.x, geo.y, geo.z]]),
                             cases[0].structure.design_tm)[0, :2]
    for case in cases:
        traj = trajectories.get(case.reality_id)
        if traj is None or geo.image not in traj.names:
            continue
        order = np.argsort(traj.idx)                 # walk order, not file order
        k = int(np.flatnonzero(order == traj.names.index(geo.image))[0])
        pick = order[max(0, k - window):k + window + 1]
        return Anchor(reality_id=case.reality_id, centre=centre,
                      ref=np.asarray(traj.positions)[pick][:, :2],
                      radius=float(RADIUS_GEO_IN), image=geo.image,
                      window=window)
    return None


def seed_index(order: list[str], cases_by_id: dict) -> int | None:
    """Which split in the order carries the GPS anchor.

    Not necessarily the first: of 60 geo-anchored captures only 39 have the
    anchor on the earliest split, so the chain starts wherever it is and walks
    outward in both directions.
    """
    for i, rid in enumerate(order):
        if cases_by_id[rid].geo is not None:
            return i
    return None


def walk_outward(order: list[str], seed: int):
    """Reality ids from the seed outward, each with the neighbour to lean on.

    Yields (reality_id, from_reality_id, forward). Forward first, then backward,
    so every step always has an already-placed neighbour adjacent to it.
    """
    for i in range(seed + 1, len(order)):
        yield order[i], order[i - 1], True
    for i in range(seed - 1, -1, -1):
        yield order[i], order[i + 1], False


# --------------------------------------------------------------------------
# the gate, and the capture-level solve
# --------------------------------------------------------------------------
# A pose is only worth propagating if the score actually preferred it. These
# are deliberately loose: the coarse score is weak, and a gate tuned tight here
# would reject nearly everything and quietly disable the chain.
GATE_MIN_MARGIN = 0.01      # best score over the best materially-different one
GATE_MIN_WALL = 0.02        # floor on the wall-agreement term
GATE_DISTINCT_M = 5.0       # how far apart two poses must be to count as rivals
GATE_DISTINCT_DEG = 10.0


@dataclass
class Gate:
    """Why a pose was or was not trusted enough to propagate."""
    passed: bool
    margin: float
    wall: float
    reason: str = ""


def gate_result(result, metres_per_inch: float) -> Gate:
    """Decide whether a solved split may seed the rest of the chain.

    A margin test, not a model: if the runner-up pose is somewhere materially
    different and scores about the same, the score did not actually choose, and
    propagating from it would carry a coin-flip into every split after it.
    """
    best = result.best if result else None
    if best is None:
        return Gate(False, 0.0, 0.0, "no hypothesis")
    rival = None
    for h in result.hypotheses[1:]:
        far = (np.hypot(h.sim2.tx - best.sim2.tx,
                        h.sim2.ty - best.sim2.ty) * metres_per_inch
               > GATE_DISTINCT_M)
        turned = abs(((h.yaw_deg - best.yaw_deg + 180) % 360) - 180) > GATE_DISTINCT_DEG
        if far or turned:
            rival = h
            break
    margin = float(best.score - rival.score) if rival is not None else float("inf")
    if best.wall < GATE_MIN_WALL:
        return Gate(False, margin, best.wall,
                    f"wall term {best.wall:.3f} below {GATE_MIN_WALL}")
    if margin < GATE_MIN_MARGIN:
        return Gate(False, margin, best.wall,
                    f"margin {margin:.3f} over a rival {GATE_DISTINCT_M:.0f} m away")
    return Gate(True, margin, best.wall)


def solve_capture(cases, solve_fn, metres_per_inch: float,
                  trajectories: dict | None = None) -> dict:
    """Register every split of one capture, in walk order, from the GPS seed.

    `solve_fn(case, prior) -> (result, pipeline)` does the per-split work; the
    sequencing lives here. Returns reality_id -> result, each carrying a `chain`
    record saying how its pose was obtained.

    The chain only ever leans on an already-placed neighbour, and only on one
    that passed the gate. Where the gate fails, that split keeps its own
    unchained result and the chain re-seeds from the next split that passes,
    rather than carrying a bad pose through the rest of the capture.
    """
    by_id = {c.reality_id: c for c in cases}
    trajs = dict(trajectories or {})
    results, placed = {}, {}

    def run(case, prior, how, frm=None, kind=None):
        result, p = solve_fn(case, prior)
        if result is not None:
            trajs.setdefault(case.reality_id, p.trajectory())
            g = gate_result(result, metres_per_inch)
            result.chain = dict(how=how, seeded_from=frm,
                                junction=kind.value if kind else None,
                                prior_radius_in=(prior.radius if prior else None),
                                gate_passed=g.passed, gate_margin=g.margin,
                                gate_reason=g.reason)
            if g.passed and result.best is not None:
                placed[case.reality_id] = result.best.sim2
        results[case.reality_id] = result
        return result

    if not trajs:
        raise ValueError("solve_capture needs trajectories to order the splits")
    order = capture_order(trajs)
    geo = capture_geo([by_id[rid] for rid in order])
    anchor = find_anchor([by_id[rid] for rid in order], trajs, geo,
                         metres_per_inch)

    if anchor is None:
        # Either the capture has no start point at all, or it has one naming a
        # photograph that is in none of its splits. Both mean there is nothing
        # to seed from; neither is a reason to invent an anchor.
        why = ("no geo anchor in this capture" if geo is None
               else f"start point names {geo.image}, which is in no split")
        for rid in order:
            run(by_id[rid], None, f"independent ({why})")
        return results

    # The seed is the split HOLDING the photograph, which for 19 of 59 is not
    # the one `Case.geo` hangs off. `walk_outward` already seeds from an
    # arbitrary index and propagates in both directions.
    seed = order.index(anchor.reality_id)
    seed_case = by_id[anchor.reality_id]
    from align.search import TranslationPrior
    prior = TranslationPrior(centre=anchor.centre, radius=anchor.radius,
                             ref=anchor.ref)
    run(seed_case, prior, f"seed (geo anchor, +-{anchor.window} images)")

    for rid, frm, _forward in walk_outward(order, seed):
        case = by_id[rid]
        if frm not in placed:
            # The neighbour we would lean on did not pass the gate. Solve this
            # one on its own and let it re-seed the chain if it does better.
            run(case, None, "independent (neighbour not trusted)", frm)
            continue
        link = classify_junction(trajs[frm], trajs[rid])
        sim_from = placed[frm]
        if link.kind is Junction.SHARED:
            transferred = transfer_pose(trajs[frm], trajs[rid], sim_from)
            if transferred is not None:
                sim, resid = transferred
                from align.pipeline import SearchResult
                from align.search import Hypothesis
                h = Hypothesis(sim2=sim, score=float("inf"), wall=float("nan"),
                               free=float("nan"), scale_prior=float("nan"),
                               scale=sim.s, yaw_deg=float(np.degrees(sim.theta)),
                               viewport=-1, rank=0)
                res = SearchResult(reality_id=rid, structure=case.structure.key,
                                   hypotheses=[h],
                                   metres_per_inch=metres_per_inch)
                res.chain = dict(how="transferred (shared images)",
                                 seeded_from=frm, junction=link.kind.value,
                                 n_shared=link.n_shared,
                                 fit_residual_in=resid, gate_passed=True,
                                 gate_margin=float("inf"), gate_reason="")
                results[rid] = res
                placed[rid] = sim
                continue
        centre, radius = expected_start(trajs[frm], trajs[rid], sim_from,
                                        link.kind, metres_per_inch)
        from align.search import TranslationPrior
        t = trajs[rid]
        ref = t.positions[int(np.argmin(t.idx))][:2]
        run(case, TranslationPrior(centre=centre, radius=radius, ref=ref),
            "chained", frm, link.kind)
    return results
