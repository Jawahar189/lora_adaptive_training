"""The per-reality confidence: the constants, the clip, and what it declines to score.

Nothing here loads a model or reads the dataset. The coefficients in
`bevaligner/inference/confidence.py` were fitted offline from a scored eval and
are constants in the source, so the thing a test can usefully protect is that
they still say what the module's table says they say -- one transposed digit is
otherwise a silent recalibration of every row the pipeline writes.

`scripts/e13_confidence_fit.py` re-derives them from the eval; these pin the
arithmetic that consumes them.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.eval.reality import ResultWriter
from bevaligner.inference.confidence import (CONFIDENCE, CONFIDENCE_NO_REFINE,
                                               confidence_for, confidence_from)
from bevaligner.utils.metrics import (SheetResultWriter,
                                        summarise_confidence)
from test_bevaligner_chain import FakeProposal


# --- the constants ----------------------------------------------------------

def test_the_reference_points_are_pinned():
    """The four rows quoted in the module docstring, to three places.

    These are not round numbers chosen for a test: the middle two are the median
    unrefined row and the median correct row of
    `out/bevaligner/chain/a_base.jsonl`, which is what makes them worth
    pinning -- they are where the real distribution sits.
    """
    assert confidence_from(0.00, 0.00, False, False) == 0.0
    assert confidence_from(0.55, 0.31, False, True) == 0.04
    assert confidence_from(0.82, 0.64, True, True) == 0.845
    assert confidence_from(1.00, 1.00, True, True) == 0.976


def test_the_coefficients_are_the_fitted_ones():
    """Guards the constants themselves, not just their product."""
    assert CONFIDENCE == (-10.6448, 6.9448, 2.3063, 2.1086, 2.9889)
    assert CONFIDENCE_NO_REFINE == (-10.9241, 8.0939, 2.9668, 3.6688)


def test_a_proposal_scores_what_its_signals_say():
    """`FakeProposal`'s defaults: fraction 0.9, refined, one candidate (so no
    rival), rig 2.0 m at `metres_per_inch` 1.0 -- all four signals favourable."""
    assert confidence_for(FakeProposal(), 1.0) == 0.953
    assert confidence_for(FakeProposal(refined=False), 1.0) == 0.669


# --- the refine-off variant ---------------------------------------------------

def test_the_refine_off_variant_drops_the_term_rather_than_zeroing_it():
    """A run with the zoomed pass off is not a run whose zoomed pass disagreed.

    Scoring it with the main coefficients would charge it 2.3063 of logit for a
    pass that never ran, so the second set drops the term -- and dropping it
    means `refined` cannot matter at all under that set, which is what the
    second assertion checks.
    """
    off = confidence_for(FakeProposal(), 1.0, refine_was_on=False)
    assert off == 0.952
    assert off != confidence_for(FakeProposal(), 1.0)
    assert confidence_for(FakeProposal(refined=False), 1.0,
                          refine_was_on=False) == off


def test_the_proposal_carries_whether_the_zoomed_pass_ran():
    """`propose_pose` records it, so the two row sites cannot disagree about it."""
    from bevaligner.inference.propose import PoseProposalOptions, PoseProposal

    assert PoseProposal.__dataclass_fields__["refine_was_on"].default is True
    assert PoseProposalOptions().refine_on_footprint is True

    proposal = FakeProposal()
    proposal.refine_was_on = False
    assert confidence_for(proposal, 1.0) == confidence_for(FakeProposal(), 1.0,
                                                           refine_was_on=False)


# --- the margin ---------------------------------------------------------------

def test_no_rival_clips_to_one_and_is_not_a_certainty():
    """`rival_margin` returns `inf` when nothing materially different was found.

    It clips to 1.0 rather than overflowing, and the clipped score stays well
    under 1.0: on the fitted run that bucket is right 32.6% of the time, so a
    score of 1.0 would be a lie in two rows out of three.
    """
    clipped = confidence_from(0.9, float("inf"), True, True)
    assert clipped == confidence_from(0.9, 1.0, True, True)
    assert clipped < 1.0
    # and in the branch that has no `refined` term, where an unclipped margin
    # would be multiplied by 2.9668 instead
    assert (confidence_from(0.9, float("inf"), False, True, refined_term=False)
            == confidence_from(0.9, 1.0, False, True, refined_term=False))


def test_a_lone_candidate_reads_as_no_rival():
    """`FakeProposal`'s single candidate is the winner itself, so margin is inf."""
    assert confidence_for(FakeProposal(), 1.0) == confidence_from(0.9, 1.0, True, True)


def test_the_score_rises_with_every_signal():
    base = dict(inlier_fraction=0.5, margin=0.2, refined=False, rig_plausible=False)
    low = confidence_from(**base)
    assert confidence_from(**dict(base, inlier_fraction=0.8)) > low
    assert confidence_from(**dict(base, margin=0.9)) > low
    assert confidence_from(**dict(base, refined=True)) > low
    assert confidence_from(**dict(base, rig_plausible=True)) > low


# --- what it declines to score ------------------------------------------------

def test_a_pose_no_search_produced_has_no_confidence():
    """`chain.proposal_from_transfer` carries a pose over from a sibling split.

    It runs no search: no candidates, no matches, no inliers. None of the four
    signals describes it, and there are no such rows in either the fitted or the
    held-out eval, so a number for it would be an extrapolation from nothing.
    """
    assert confidence_for(FakeProposal(candidates=[]), 1.0) is None
    assert confidence_for(None, 1.0) is None
    # the shape such a row actually has on disk (window -1, nothing matched)
    transferred = FakeProposal(candidates=[], inliers=0, fraction=0.0, refined=False)
    assert confidence_for(transferred, 1.0) is None


def test_an_implausible_rig_costs_the_term():
    """A fitted height outside `RIG_PLAUSIBLE_M` means the floor fit found
    something that is not the floor, which corrupts the scale as well."""
    assert (confidence_for(FakeProposal(height=0.01), 1.0)
            == confidence_from(0.9, 1.0, True, False))
    assert confidence_for(FakeProposal(height=0.01), 1.0) < confidence_for(
        FakeProposal(), 1.0)


def test_a_rig_height_that_is_not_a_number_does_not_raise():
    """`sim2.s` can come back NaN from a degenerate fit, and a comparison
    against NaN is False either way -- the term has to be dropped explicitly."""
    from align.frames import Sim2

    proposal = FakeProposal(sim2=Sim2(s=float("nan"), theta=0.0, tx=0.0, ty=0.0))
    assert confidence_for(proposal, 1.0) == confidence_from(0.9, 1.0, True, False)


# --- the row it ends up on ----------------------------------------------------

class _Case:
    reality_id, capture_id = "RLT1", "SNP1"

    class structure:
        key = "data_local/STR1"
        georeferenced = False


def test_the_column_is_on_every_learned_path_csv(tmp_path):
    """Not behind --chain: how likely a pose is to be right is a property of the
    pose, not of whether the run happened to chain."""
    writer = SheetResultWriter(tmp_path / "run")
    header = next(csv.reader(open(writer.paths["csv"])))
    writer.close()
    assert "confidence" in header


def test_the_confidence_travels_with_the_transform(tmp_path):
    """The predictions JSONL is the file a consumer reads, so the score has to
    be on it and not only in the report."""
    writer = SheetResultWriter(tmp_path / "run")
    writer.write(_Case(), {"predicted": dict(rmse_xy=1.0, confidence=0.84,
                                             tm=np.eye(4))})
    writer.close()
    row = next(csv.DictReader(open(writer.paths["csv"])))
    assert row["confidence"] == "0.84"
    prediction = json.loads(open(writer.paths["predictions"]).read().strip())
    assert prediction["confidence"] == 0.84
    assert prediction["reality_id"] == "RLT1" and len(prediction["tm"]) == 16


def test_an_unscored_pose_leaves_the_cell_empty(tmp_path):
    """An abstention, or a transferred pose: empty means NOT MEASURED. A 0.0
    would assert a measured impossibility."""
    writer = SheetResultWriter(tmp_path / "run")
    writer.write(_Case(), {})                                   # no pose at all
    writer.write(_Case(), {"predicted": dict(rmse_xy=1.0, confidence=None,
                                             tm=np.eye(4))})
    writer.close()
    rows = list(csv.DictReader(open(writer.paths["csv"])))
    assert [row["confidence"] for row in rows] == ["", ""]
    prediction = json.loads(open(writer.paths["predictions"]).read().strip())
    assert "confidence" not in prediction


def test_the_classical_predictions_file_is_untouched(tmp_path):
    """`align.eval.reality` is the classical pipeline's output contract: its
    predictions line stays exactly what every existing reader expects."""
    writer = ResultWriter(tmp_path / "run")
    writer.write(_Case(), {"predicted": dict(rmse_xy=1.0, confidence=0.84,
                                             tm=np.eye(4))})
    writer.close()
    prediction = json.loads(open(writer.paths["predictions"]).read().strip())
    assert set(prediction) == {"reality_id", "tm"}


def test_the_run_says_what_it_believes_about_itself():
    """An unscored run has no rmse and no tpr to print, so the confidence line
    is the only statement about quality it can make. Rows nothing scored are
    counted as unscored, not as zeroes."""
    rows = [{"confidence": 0.9}, {"confidence": 0.6}, {"confidence": 0.05},
            {"confidence": None}]
    text = summarise_confidence(rows)
    assert ">= 0.50: 2/4" in text and ">= 0.85: 1/4" in text
    assert "median 0.60" in text
    assert summarise_confidence([{"rmse_xy": 1.0}]) == "  confidence: none scored"
