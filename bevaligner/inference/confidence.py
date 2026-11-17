"""How likely a placed reality is actually within 2 m, as a calibrated number.

`gate.py` answers a narrower question -- may this pose seed the rest of its
capture? -- as a boolean, and only on a chained run. This is the continuous
answer to the broader one, on every run: a consumer reading
`<stem>_predictions.jsonl` gets a transform for every reality, including the
ones that are badly wrong, and until now nothing in the row said which was
which. On the 1073 scored rows of `out/bevaligner/chain/a_base.jsonl` only 250
land within 2 m.

WHAT IT IS. A logistic regression over four signals the search already
computes, fitted against `rmse_m < 2 m`:

    confidence = sigmoid(-10.6448 + 6.9448 * inlier_fraction
                         +  2.3063 * refined
                         +  2.1086 * margin
                         +  2.9889 * rig_plausible)

Fitted on the 1073 scored rows of `out/bevaligner/chain/a_base.jsonl` -- an
UNCHAINED run, as the gate's own table is -- base rate 0.233. Scored by
five-fold cross-validation GROUPED BY CAPTURE, so no capture is ever on both
sides of the split; a per-row split would leak, because the splits of one
capture walk the same floor:

    AUC 0.959   Brier 0.0668 (base 0.179)

      confidence     n    mean p   actually within 2 m   median rmse
      [0.00,0.10)  658    0.016          0.006              37.65 m
      [0.10,0.30)   97    0.182          0.227              20.84 m
      [0.30,0.50)   75    0.383          0.347               4.28 m
      [0.50,0.70)   48    0.613          0.667               0.98 m
      [0.70,0.85)   85    0.783          0.753               0.97 m
      [0.85,1.01)  110    0.917          0.927               0.80 m

Held out on `out/bevaligner/chain/b_chain.jsonl`, the chained run over the
same splits: AUC 0.954, Brier 0.0728, and the bins hold to within 0.05. On
`out/bevaligner/eval/test_geo_first_main.jsonl`, a DIFFERENT configuration:
AUC 0.886, mean 0.451 against a base rate of 0.490.

WHY THE ABSOLUTE INLIER COUNT IS NOT HERE. Adding it scores marginally better
in-family and then collapses: a model carrying the count predicts mean 0.03 on
that geo-first run against its base rate of 0.39, because its matcher produces a
median of 41 inliers where this one produces 263. That is the same fact
`gate.py` records about `min_inliers` -- the count does not transfer between
matchers, the fraction does. Every signal above is scale-free for that reason,
and it costs nothing: 0.959 against the 0.959 a count-carrying fit scores here.

WHAT `margin == 1.0` MEANS. `rival_margin` returns `inf` when the search found
no materially different rival, and that clips to 1.0 here. It is NOT a
certainty: 307 of the 1073 rows are in that bucket and 32.6% of them land within
2 m, against 19.5% of the rows whose finite margin is below 0.9. There is no
training data with a MEASURED lead near 1.0 (one row), so the coefficient at 1.0
is learned from the no-rival bucket and means "nothing else disagreed", not "a
large measured lead". The fit is calibrated on it -- 0.351 predicted against
0.326 actual over those 307 rows, 0.190 against 0.196 over the other 766 -- but
the bucket is not one population: on `b_chain` the share of placed rows with no
rival runs from 0.08 for `best_effort` (median 11 windows searched) to 0.58 for
`chained` (median 8). "Nothing disagreed" after three windows is a weaker
statement than after thirty-two, and this score cannot tell them apart.

WHAT IS NOT MEASURED BY ANY NUMBER ABOVE.

  * A pose carried over by `chain.proposal_from_transfer`, which runs no search
    and so has no candidates, no matches and no inliers. `confidence_for`
    returns None for it rather than a number -- see `_NO_EVIDENCE`. There are no
    such rows in either the fitted or the held-out run, so a score for them
    would be an extrapolation from nothing. An empty cell is this repo's
    convention for "not measurable" (`utils.metrics`, `gate._finite`).
  * A run with the zoomed pass turned off. `CONFIDENCE_NO_REFINE` drops the
    `refined` term for that case (AUC 0.943, Brier 0.0793, grouped-CV on the
    same rows), but it is fitted on `refined = False` rows from runs where
    refinement was ON -- the zoomed pass DISAGREEING, not the pass being
    disabled. No `--no-refine` eval exists in `out/` to fit the real thing. The
    two are not the same population and the second set is an approximation
    until one does.

`scripts/e13_confidence_fit.py` re-derives every number here from a scored
JSONL. A refit will not reproduce a runtime score to the last digit: the script
reads `inlier_frac` as the row rounds it to 3 places, under 5e-4 in the fraction
and so under 0.004 in the logit.
"""
from __future__ import annotations

import numpy as np

from ..utils.metrics import rig_height_m
from .gate import GateThresholds, rival_margin

# bias, inlier_fraction, refined, margin, rig_plausible
CONFIDENCE = (-10.6448, 6.9448, 2.3063, 2.1086, 2.9889)
# The same, fitted without the `refined` term, for a run whose zoomed pass was
# off: bias, inlier_fraction, margin, rig_plausible.
CONFIDENCE_NO_REFINE = (-10.9241, 8.0939, 2.9668, 3.6688)

# What `confidence_for` returns when the pose came from somewhere other than a
# search, so none of the four signals describes it. None rather than 0.0: a zero
# asserts a measured impossibility, and nothing was measured.
_NO_EVIDENCE = None


def confidence_from(inlier_fraction: float, margin: float, refined: bool,
                    rig_plausible: bool, *, refined_term: bool = True) -> float:
    """The score itself, from four plain numbers.

    Separate from `confidence_for` so `scripts/e13_confidence_fit.py` can score
    a JSONL row -- which has no `PoseProposal` -- through exactly the arithmetic
    that runs in the pipeline, rather than a second copy of it.

    `margin` may be `inf` (no rival was found) and clips to 1.0 in both
    branches; without the clip the refine-off branch would return exactly 1.0
    for a bucket that is right about a third of the time.
    """
    margin = 1.0 if not np.isfinite(margin) else float(min(max(margin, 0.0), 1.0))
    fraction = float(min(max(inlier_fraction, 0.0), 1.0))
    if refined_term:
        bias, w_fraction, w_refined, w_margin, w_rig = CONFIDENCE
        logit = bias + w_fraction * fraction + w_refined * float(bool(refined))
    else:
        bias, w_fraction, w_margin, w_rig = CONFIDENCE_NO_REFINE
        logit = bias + w_fraction * fraction
    logit += w_margin * margin + w_rig * float(bool(rig_plausible))
    return round(float(1.0 / (1.0 + np.exp(-logit))), 3)


def confidence_for(proposal, metres_per_inch: float, *,
                   refine_was_on: bool | None = None,
                   thresholds: GateThresholds = GateThresholds()) -> float | None:
    """`P(this reality is placed within 2 m)`, or None when nothing was searched.

    The margin comes from `gate.rival_margin` rather than a second rival rule:
    one definition, or `confidence` and `gate_margin` would describe different
    searches in the same row. `thresholds` is passed through for the same
    reason -- the day `distinct_m` becomes a flag, both read it.

    `refine_was_on` defaults to what the proposal recorded about the run that
    made it, because whether the zoomed pass RAN is a property of that run and
    not of the caller asking for a score.
    """
    if proposal is None or not proposal.candidates:
        return _NO_EVIDENCE
    if refine_was_on is None:
        refine_was_on = bool(getattr(proposal, "refine_was_on", True))
    rig = rig_height_m(proposal.camera_height_above_floor, proposal.sim2.s,
                       metres_per_inch)
    low, high = thresholds.rig_plausible
    margin, _rival_distance = rival_margin(proposal, metres_per_inch, thresholds)
    return confidence_from(proposal.inlier_fraction, margin, proposal.refined,
                           np.isfinite(rig) and low <= rig <= high,
                           refined_term=refine_was_on)
