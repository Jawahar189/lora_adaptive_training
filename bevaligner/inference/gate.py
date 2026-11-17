"""Whether a solved pose is firm enough to seed the rest of its capture.

Chaining spends one split's answer on the next, so a pose that propagates is
worth more than a pose that is merely written: a wrong one is not one bad row
but a bad row per split after it. This module is the decision to propagate, and
nothing else -- `propose_pose` still returns whatever it found, every capture
still gets a transform, and a pose that fails here is written with its reason
rather than withheld.

`align.chain.gate_result` is the same decision for the classical solver, and
cannot be borrowed: it reads `result.hypotheses[*].score/.wall/.yaw_deg`, which
is a ranked list of correlation scores. The learned path has inlier counts.

WHAT ACTUALLY SEPARATES A GOOD PLACEMENT FROM A BAD ONE, measured over the 1073
scored rows of `out/bevaligner/eval/test_viz.jsonl` against `tpr@1.0 > 0.5`
(base rate 0.166):

    signal                 good median (p10)   bad median (p90)
    inlier_fraction        0.811 (0.623)       0.482 (0.882)
    num_inliers            390   (291)         226   (427)
    refined                1.0   (1.0)         0.0   (1.0)
    runner-up margin       0.338 (0.071)       0.134 (0.432)
    rig_height_m           2.01  (1.63)        1.53  (2.77)

No single column separates them -- every one of those bad p90s is above the
corresponding good p10. The conjunction does: all five together keep 216 rows at
precision 0.620 and recall 0.753, 3.7x the base rate, and that is what catches a
correct seed in 70 of the 79 captures holding one.

Re-fit over `out/bevaligner/chain/a_base` -- 1073 rows of
`datasets/splits/prior_bev_test.csv` under prior_bev_l4_feat, the first eval
carrying candidate poses, so the first where the margin is a real quantity
rather than `inf` everywhere. It is finite on 766 of 1073 rows, median 0.165.
Seeding, over the 79 multi-split captures that contain a correctly placed split:

    frac  margin  refined | captures seeded  seed precision | rows  prec  recall
    0.60  0.05    yes     |       95              0.663      | 236   0.619 0.820
    0.65  0.20    yes     |       85              0.694      | 189   0.661 0.702
    0.75  0.20    yes     |       80              0.700      | 152   0.684 0.584

The margin is what moves it; the ABSOLUTE inlier count moves nothing at all --
every row of the sweep is identical across floors of 0, 30, 60, 120 and 200,
because an inlier fraction above 0.6 already implies a high count under this
matcher. That is the same thing the keypoint-vs-dense gap says from the other
direction, and it is why `min_inliers` is a guard against a degenerate fit
rather than a threshold worth tuning.

The defaults below are the first row. They seed the most captures and, in
absolute terms, the most CORRECT ones (63 against 59 at 0.65/0.20) -- but also
the most wrong ones (32 against 26), and the costs are not symmetric: a missing
seed falls back to the unconstrained search, which is what the split would have
had anyway, while a wrong one is propagated. Whether to pay 10 captures for 3
points of seed precision is settled by what a chained run does to the rmse p90,
not by this table: a p90 that rises is bad seeds travelling, and is the signal
to take the 0.65/0.20 row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.metrics import RIG_PLAUSIBLE_M, rig_height_m

# How far apart two poses must be before one counts as the other's rival. Below
# these a "rival" is the same answer found twice, which is agreement, not doubt.
GATE_DISTINCT_M = 5.0
GATE_DISTINCT_DEG = 10.0


@dataclass(frozen=True)
class GateThresholds:
    """What a pose must clear to be propagated. See the module docstring."""
    min_inlier_fraction: float = 0.60
    # An absolute floor on support, to throw out a fit made of almost nothing.
    # It is deliberately low because THE COUNT DOES NOT TRANSFER BETWEEN
    # MATCHERS and the fraction does: the dense matcher proposes about 500
    # matches per window and the keypoint matcher about 80, so a correct
    # keypoint placement lands near 50 inliers where a correct dense one lands
    # near 400. A floor of 60, fitted on dense runs, rejects every keypoint
    # placement there is -- measured, on both of the two splits a keypoint run
    # over `datasets/splits/prior_bev_val.csv` placed correctly (46 and 51
    # inliers at fractions 0.63 and 0.64). Twice `PoseProposalOptions.min_inliers`
    # is the floor that means "more than a bare fit" under either matcher.
    min_inliers: int = 24
    min_margin: float = 0.05
    require_refined: bool = True
    distinct_m: float = GATE_DISTINCT_M
    distinct_deg: float = GATE_DISTINCT_DEG
    rig_plausible: tuple[float, float] = RIG_PLAUSIBLE_M


@dataclass
class Gate:
    """Why a pose was or was not trusted enough to seed the rest of the walk."""
    passed: bool
    margin: float = 0.0
    inlier_fraction: float = 0.0
    num_inliers: int = 0
    rig_height_m: float = float("nan")
    rival_distance_m: float = float("nan")
    reason: str = ""

    def as_row(self) -> dict:
        """The columns this decision contributes to the result row."""
        return dict(gate_passed=bool(self.passed), gate_reason=self.reason,
                    gate_margin=_finite(self.margin),
                    gate_inlier_frac=round(float(self.inlier_fraction), 3),
                    gate_rival_distance_m=_finite(self.rival_distance_m))


def _finite(value: float):
    """`inf` and `nan` are not JSON; None is, and reads as "there wasn't one"."""
    value = float(value)
    return round(value, 4) if np.isfinite(value) else None


def _staged(candidates) -> list[dict]:
    """Stage-one candidates carrying a pose.

    The refine pass appends a candidate of its own with no centre, and an early
    build wrote none at all, so a row from before this column existed degrades
    to "no rival found" rather than raising.
    """
    return [c for c in candidates
            if c.get("centre_x") is not None and not c.get("refine")]


def _winner(candidates, proposal) -> dict | None:
    """The stage-one candidate the proposal was built from."""
    for candidate in _staged(candidates):
        if (candidate.get("up_index") == proposal.up_index
                and candidate.get("window_index") == proposal.window_index):
            return candidate
    return None


def rival_margin(proposal, metres_per_inch: float,
                 thresholds: GateThresholds) -> tuple[float, float]:
    """`(margin, rival distance in metres)` over materially different poses.

    The margin is fractional -- the winner's lead over the best rival as a share
    of its own support -- because inlier counts are not comparable between
    captures: 40 of 60 is the same confidence as 400 of 600 and a tenth of the
    absolute gap.

    A rival is a candidate whose walk centroid sits more than `distinct_m` away
    or whose yaw differs by more than `distinct_deg`. Anything closer is the
    same answer found through an overlapping window, and counting it would mean
    the gate rejects precisely the poses several windows agree on.

    `inf` when nothing else was materially different -- which is the honest
    reading of a search that only ever found one answer.
    """
    winner = _winner(proposal.candidates, proposal)
    if winner is None:
        return float("inf"), float("nan")
    best = float(winner.get("num_inliers", proposal.num_inliers))
    centre = np.array([winner["centre_x"], winner["centre_y"]], dtype=float)
    yaw = float(winner.get("yaw_deg", 0.0))
    rival, rival_distance = None, float("nan")
    for candidate in _staged(proposal.candidates):
        if candidate is winner:
            continue
        other = np.array([candidate["centre_x"], candidate["centre_y"]], dtype=float)
        distance_m = float(np.linalg.norm(other - centre)) * metres_per_inch
        turned = abs(((float(candidate.get("yaw_deg", 0.0)) - yaw + 180) % 360)
                     - 180) > thresholds.distinct_deg
        if distance_m <= thresholds.distinct_m and not turned:
            continue
        if rival is None or candidate["num_inliers"] > rival["num_inliers"]:
            rival, rival_distance = candidate, distance_m
    if rival is None:
        return float("inf"), float("nan")
    return (best - float(rival["num_inliers"])) / max(best, 1.0), rival_distance


def gate_proposal(proposal, metres_per_inch: float,
                  thresholds: GateThresholds = GateThresholds(),
                  *, refine_was_on: bool = True) -> Gate:
    """Decide whether `proposal` may seed the rest of its capture.

    Tests run cheapest-and-most-damning first and the first failure wins, so
    `reason` names the one thing that was wrong rather than the last thing
    checked. `refine_was_on` exists because `require_refined` is a statement
    about the second pass having agreed, which is not a question when the caller
    turned the second pass off.
    """
    if proposal is None:
        return Gate(False, reason="no pose")
    rig = rig_height_m(proposal.camera_height_above_floor, proposal.sim2.s,
                       metres_per_inch)
    low, high = thresholds.rig_plausible
    margin, rival_distance = rival_margin(proposal, metres_per_inch, thresholds)
    gate = Gate(passed=False, margin=margin,
                inlier_fraction=proposal.inlier_fraction,
                num_inliers=proposal.num_inliers, rig_height_m=rig,
                rival_distance_m=rival_distance)

    if not np.isfinite(rig) or not low <= rig <= high:
        gate.reason = f"rig height {rig:.2f} m outside {low}-{high}"
        return gate
    if proposal.num_inliers < thresholds.min_inliers:
        gate.reason = (f"{proposal.num_inliers} inliers below "
                       f"{thresholds.min_inliers}")
        return gate
    if proposal.inlier_fraction < thresholds.min_inlier_fraction:
        gate.reason = (f"inlier fraction {proposal.inlier_fraction:.2f} below "
                       f"{thresholds.min_inlier_fraction}")
        return gate
    if thresholds.require_refined and refine_was_on and not proposal.refined:
        gate.reason = "the zoomed pass did not agree"
        return gate
    if margin < thresholds.min_margin:
        gate.reason = (f"margin {margin:.3f} over a rival "
                       f"{rival_distance:.0f} m away")
        return gate
    gate.passed = True
    return gate
