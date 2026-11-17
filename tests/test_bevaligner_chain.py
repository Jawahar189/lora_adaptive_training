"""Capture-wise sequential placement, on synthetic input with a known answer.

Nothing here reads the dataset or loads a model. The searches are stubs scripted
to succeed or fail at chosen rungs, which is what makes the ladder and the
re-seeding testable at all -- a real search would make the test about the
matcher rather than about the sequencing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.frames import Sim2
from bevaligner.inference import chain as chainmod
from bevaligner.inference.chain import (ChainPolicy, Junction, PosePrior,
                                          gravity_trajectory, junction_prior,
                                          orderable, radius_for,
                                          scale_bracket_from)
from bevaligner.inference.gate import GateThresholds, gate_proposal, rival_margin
from bevaligner.inference.propose import (PoseProposalOptions, _tiled_windows,
                                            candidate_windows, windows_with_origin)


class FakeTraj:
    """Just enough of a Trajectory for the sequencing layer."""

    def __init__(self, start, n, positions=None):
        self.names = [f"VID_x_IMG_{i:05d}.JPG" for i in range(start, start + n)]
        self.idx = np.arange(start, start + n)
        self.positions = (np.zeros((n, 3)) if positions is None
                          else np.asarray(positions, dtype=np.float64))
        self.frame = "ply"

    def __len__(self):
        return len(self.names)


class FakeView:
    def __init__(self, height=2.0, rotation=None, n=8):
        self.camera_height_above_floor = height
        self.rotation_to_gravity_up = (np.eye(3) if rotation is None
                                       else np.asarray(rotation, float))
        self.camera_xy = np.zeros((n, 2))
        self.camera_z = np.zeros(n)


class FakePrepared:
    def __init__(self, trajectory, views=None, metres_per_inch=1.0):
        self.trajectory = trajectory
        self.views = views or {0: FakeView()}
        self.metres_per_inch = metres_per_inch
        self.pool = None


class FakeProposal:
    """A PoseProposal's gate-relevant surface."""

    def __init__(self, sim2=None, up_index=0, inliers=400, fraction=0.9,
                 refined=True, candidates=None, height=2.0, rotation=None):
        self.sim2 = sim2 or Sim2(s=1.0, theta=0.0, tx=0.0, ty=0.0)
        self.up_index = up_index
        self.window_index = 0
        self.num_inliers = inliers
        self.inlier_fraction = fraction
        self.num_matches = 500
        self.num_verified = None
        self.refined = refined
        self.camera_height_above_floor = height
        self.camera_z = np.zeros(4)
        self.rotation_to_gravity_up = (np.eye(3) if rotation is None
                                       else np.asarray(rotation, float))
        self.candidates = candidates if candidates is not None else [
            dict(up_index=0, window_index=0, num_inliers=inliers, centre_x=0.0,
                 centre_y=0.0, yaw_deg=0.0, scale=1.0, from_prior=True)]


def case(reality_id):
    return type("Case", (), {"reality_id": reality_id})()


# --- the no-prior guarantee ------------------------------------------------
class Pool:
    def __init__(self, points, extent):
        self.band_paper_xy = np.asarray(points, float)
        self.extent = extent


@pytest.fixture
def pool():
    rng = np.random.default_rng(0)
    return Pool(rng.uniform([0, 0], [100, 100], size=(20000, 2)), (0., 0., 100., 100.))


def test_no_prior_leaves_the_tiling_untouched(pool):
    """The preservation guarantee as a test, not a promise."""
    options = PoseProposalOptions()
    assert candidate_windows(pool, 20.0, options) == _tiled_windows(pool, 20.0, options)
    assert all(not from_prior
               for _w, from_prior in windows_with_origin(pool, 20.0, options))


def test_prior_windows_centre_on_the_prior_and_keep_insurance(pool):
    options = PoseProposalOptions(prior_centre=(50., 50.), prior_radius_in=8.,
                                  max_windows_with_prior=5, insurance_windows=2)
    windows = windows_with_origin(pool, 20.0, options)
    assert sum(1 for _w, p in windows if p) == 5
    assert sum(1 for _w, p in windows if not p) == 2
    assert windows[0][0] == pytest.approx((40., 40., 60., 60.))
    assert len({tuple(np.round(w, 6)) for w, _ in windows}) == len(windows)


def test_disc_off_the_prior_keeps_only_insurance(pool):
    far = PoseProposalOptions(prior_centre=(5000., 5000.), prior_radius_in=8.)
    assert all(not p for _w, p in windows_with_origin(pool, 20.0, far))
    hard = PoseProposalOptions(prior_centre=(5000., 5000.), prior_radius_in=8.,
                               insurance_windows=0)
    assert windows_with_origin(pool, 20.0, hard) == []


# --- frames ---------------------------------------------------------------
def test_gravity_trajectory_is_the_frame_sim2_maps_from():
    """The trap: a Sim2 applied to unrotated positions looks entirely plausible."""
    angle = np.deg2rad(35.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    raw = FakeTraj(0, 5, positions=np.c_[np.arange(5.), np.arange(5.) * 2, np.zeros(5)])
    rotated = gravity_trajectory(raw, rotation)
    expected = np.asarray(raw.positions) @ rotation.T
    assert rotated.positions == pytest.approx(expected)
    assert rotated.names == raw.names
    assert not np.allclose(rotated.positions, raw.positions)


# --- sequencing -----------------------------------------------------------
def test_orderable_refuses_splits_with_no_image_index():
    good = {"a": FakeTraj(0, 10), "b": FakeTraj(10, 10)}
    assert orderable(good)
    blind = FakeTraj(0, 10)
    blind.idx = np.full(10, -1)
    assert not orderable({"a": blind, "b": FakeTraj(10, 10)})
    assert not orderable({"a": FakeTraj(0, 10)})          # one split is not a chain


def test_junction_prior_takes_the_end_that_meets():
    positions = np.c_[np.arange(10.), np.zeros(10), np.zeros(10)]
    placed = chainmod.Placed("a", Sim2(1.0, 0.0, 0.0, 0.0), 0,
                             FakeTraj(0, 10, positions), 2.0)
    forward = junction_prior(placed, Junction.CONTIGUOUS, forward=True)
    backward = junction_prior(placed, Junction.CONTIGUOUS, forward=False)
    assert forward.centre == pytest.approx([9.0, 0.0])    # A's LAST camera
    assert backward.centre == pytest.approx([0.0, 0.0])   # A's FIRST camera
    assert forward.radius_in == radius_for(Junction.CONTIGUOUS)


def test_radii_are_paper_inches_not_metres():
    assert radius_for(Junction.CONTIGUOUS) < radius_for(Junction.SHARED)
    assert radius_for(Junction.SHARED) < radius_for(Junction.GAP)
    assert radius_for(Junction.GAP, scale=2.0) == 2 * radius_for(Junction.GAP)


# --- the scale bracket ----------------------------------------------------
def test_scale_bracket_round_trips_and_spans_both_hypotheses():
    views = {0: FakeView(height=2.0), 1: FakeView(height=1.0)}
    lo, hi = scale_bracket_from(2.0, views, metres_per_inch=1.0)
    assert lo <= 1.0 <= hi          # the true scale for the 2.0 m hypothesis
    assert lo <= 2.0 <= hi          # and for the 1.0 m one
    assert not lo <= 5.0 <= hi      # but not a scale 5x off
    assert scale_bracket_from(float("nan"), views, 1.0) is None
    assert scale_bracket_from(2.0, {}, 1.0) is None


# --- the gate -------------------------------------------------------------
def test_gate_reports_the_first_thing_that_was_wrong():
    thresholds = GateThresholds()
    assert gate_proposal(None, 1.0, thresholds).reason == "no pose"
    assert "rig height" in gate_proposal(
        FakeProposal(height=0.01), 1.0, thresholds).reason
    assert "inliers below" in gate_proposal(
        FakeProposal(inliers=10), 1.0, thresholds).reason
    assert "inlier fraction" in gate_proposal(
        FakeProposal(fraction=0.1), 1.0, thresholds).reason
    assert "zoomed pass" in gate_proposal(
        FakeProposal(refined=False), 1.0, thresholds).reason
    assert gate_proposal(FakeProposal(), 1.0, thresholds).passed


def test_require_refined_is_not_asked_when_refinement_was_off():
    gate = gate_proposal(FakeProposal(refined=False), 1.0, GateThresholds(),
                         refine_was_on=False)
    assert gate.passed


def test_an_overlapping_window_finding_the_same_pose_is_not_a_rival():
    """Stage-one windows overlap by half, so the runner-up is usually agreement."""
    near = [dict(up_index=0, window_index=0, num_inliers=400, centre_x=0.0,
                 centre_y=0.0, yaw_deg=0.0),
            dict(up_index=0, window_index=1, num_inliers=395, centre_x=1.0,
                 centre_y=0.0, yaw_deg=1.0)]
    margin, _ = rival_margin(FakeProposal(candidates=near), 1.0, GateThresholds())
    assert margin == float("inf")
    assert gate_proposal(FakeProposal(candidates=near), 1.0, GateThresholds()).passed

    far = near[:1] + [dict(up_index=0, window_index=1, num_inliers=395,
                           centre_x=90.0, centre_y=0.0, yaw_deg=0.0)]
    margin, distance = rival_margin(FakeProposal(candidates=far), 1.0, GateThresholds())
    assert margin == pytest.approx((400 - 395) / 400)
    assert distance == pytest.approx(90.0)
    assert "margin" in gate_proposal(FakeProposal(candidates=far), 1.0,
                                     GateThresholds()).reason


# --- the ladder and the walk ----------------------------------------------
def walk(trajectories, script, policy=None, geo=None, metres_per_inch=1.0):
    """Run a capture whose searches are scripted by `script[reality_id]`.

    Each entry is a list consumed one rung at a time: a `FakeProposal`, or None
    for "the search found nothing". A shorter list than the ladder asks for
    means every later rung also found nothing.
    """
    calls = {rid: 0 for rid in trajectories}
    prepared = {rid: FakePrepared(traj, metres_per_inch=metres_per_inch)
                for rid, traj in trajectories.items()}

    def prepare(case, context):
        context["num_views"] = 1
        return prepared[case.reality_id], None

    def solve(prep, options):
        rid = next(r for r, p in prepared.items() if p is prep)
        outcomes = script[rid]
        index = calls[rid]
        calls[rid] += 1
        return outcomes[index] if index < len(outcomes) else None

    results = chainmod.place_capture_chain(
        [case(rid) for rid in trajectories], prepare, solve,
        PoseProposalOptions(), policy or ChainPolicy(seed_source="search"),
        geo_track_for=geo, trajectory_for=lambda c: trajectories[c.reality_id],
        log=lambda *a: None)
    return results, calls


def contiguous(n=3, length=10):
    return {chr(ord("a") + i): FakeTraj(i * length, length) for i in range(n)}


def test_the_walk_seeds_once_then_chains():
    trajectories = contiguous()
    results, calls = walk(trajectories, {r: [FakeProposal()] * 3 for r in trajectories})
    hows = {r: results[r].context["chain_how"] for r in results}
    assert hows["a"].startswith("seed")
    assert hows["b"] == "chained" and hows["c"] == "chained"
    assert all(results[r].context["gate_passed"] for r in results)
    # one search each: the prior rung succeeded, so no rung below it ran
    assert calls == {"a": 1, "b": 1, "c": 1}
    assert results["b"].context["chain_seeded_from"] == "a"
    assert results["b"].context["chain_junction"] == "contiguous"


def test_a_split_the_prior_cannot_place_recovers_and_reseeds():
    trajectories = contiguous()
    script = {"a": [FakeProposal()],
              # fails the gate at L1 and L2, then an unconstrained search works
              "b": [FakeProposal(inliers=5), FakeProposal(inliers=5), FakeProposal()],
              "c": [FakeProposal()]}
    results, calls = walk(trajectories, script)
    assert results["b"].context["chain_how"] == "recovered"
    assert calls["b"] == 3 and results["b"].context["chain_attempt"] == 3
    assert results["b"].context["gate_passed"]
    # having recovered, b is trusted again and c leans on it
    assert results["c"].context["chain_how"] == "chained"
    assert results["c"].context["chain_seeded_from"] == "b"


def test_a_split_that_never_gates_is_written_but_never_propagated():
    trajectories = contiguous()
    script = {"a": [FakeProposal()],
              "b": [FakeProposal(inliers=5)] * 3,
              "c": [FakeProposal()]}
    results, _ = walk(trajectories, script)
    assert results["b"].context["chain_how"] == "best_effort"
    assert results["b"].context["gate_passed"] is False
    assert results["b"].status == "ok" and results["b"].proposal is not None
    # c is not allowed to lean on it -- it reaches PAST it, to the seed
    assert results["c"].context["chain_seeded_from"] == "a"
    assert results["c"].context["chain_reach"] == 2


def test_retries_bound_the_extra_searches():
    trajectories = contiguous(n=2)
    script = {"a": [FakeProposal()], "b": [FakeProposal(inliers=5)] * 3}
    _, calls = walk(trajectories, script,
                    ChainPolicy(seed_source="search", retries=0))
    assert calls["b"] == 1
    _, calls = walk(trajectories, script,
                    ChainPolicy(seed_source="search", retries=1))
    assert calls["b"] == 2


def test_a_broken_junction_is_refused_however_well_it_scored():
    """The 107-inch contiguous junctions in the data: a good pose in a bad place."""
    trajectories = contiguous(n=2)
    far = FakeProposal(sim2=Sim2(s=1.0, theta=0.0, tx=10_000.0, ty=0.0))
    results, _ = walk(trajectories, {"a": [FakeProposal()], "b": [far] * 3})
    assert "junction implausible" in results["b"].context["gate_reason"]
    assert results["b"].context["gate_passed"] is False
    assert results["b"].context["chain_junction_realised_in"] > 0


def test_a_capture_with_no_image_index_runs_independently():
    trajectories = contiguous(n=2)
    for traj in trajectories.values():
        traj.idx = np.full(len(traj.idx), -1)
    results, _ = walk(trajectories, {r: [FakeProposal()] for r in trajectories})
    assert all(r.context["chain_how"].startswith("independent")
               for r in results.values())
    assert all(r.context["chain_mode"] == "chain" for r in results.values())


def test_when_nothing_gates_every_row_is_the_search_it_would_have_had():
    trajectories = contiguous()
    results, _ = walk(trajectories, {r: [FakeProposal(inliers=5)] * 3
                                     for r in trajectories})
    assert all(r.context["gate_passed"] is False for r in results.values())
    assert all(r.proposal is not None for r in results.values())


def test_the_geo_seed_is_tried_before_any_search():
    trajectories = contiguous(n=2)

    class Track:
        def first_of(self, _traj):
            return np.array([3.0, 4.0])

    results, _ = walk(trajectories, {r: [FakeProposal()] for r in trajectories},
                      ChainPolicy(seed_source="geo"), geo=lambda _c: Track())
    assert results["a"].context["chain_how"] == "seed (geo)"
    assert results["a"].context["chain_prior_radius_in"] == chainmod.RADIUS_GEO_IN
    assert results["a"].context["chain_prior_centre_x"] == 3.0


# --- splits that share photographs ----------------------------------------
def bent(start, n):
    """A path that turns, so a reflection of it is not a rotation of it.

    A straight walk cannot tell the two gravity hypotheses apart -- a flipped
    line is still a line -- so the shared stretch has to bend for the transfer
    to resolve the sign. It generally does; this just makes the test honest.
    """
    t = np.arange(start, start + n, dtype=float)
    return np.c_[t, np.sin(t * 0.7) * 3.0, np.zeros(n)]


def overlapping():
    """Two splits whose last five and first five photographs are the same."""
    return {"a": FakeTraj(0, 10, bent(0, 10)), "b": FakeTraj(5, 10, bent(5, 10))}


def test_shared_photographs_place_the_split_and_a_search_verifies_it():
    trajectories = overlapping()
    results, _ = walk(trajectories, {"a": [FakeProposal()], "b": [FakeProposal()]})
    assert results["b"].context["chain_junction"] == "shared"
    assert results["b"].context["chain_how"] == "transferred_verified"
    assert results["b"].context["chain_n_shared"] == 5
    assert results["b"].context["chain_fit_residual_in"] == pytest.approx(0.0, abs=1e-6)
    # the prior is the tight one, and it came from the transfer
    assert results["b"].context["chain_prior_radius_in"] == radius_for(Junction.SHARED)


def test_a_transfer_the_search_cannot_verify_is_still_a_pose():
    """Umeyama on genuinely shared photographs is the best junction in the data."""
    trajectories = overlapping()
    results, _ = walk(trajectories, {"a": [FakeProposal()], "b": []})
    assert results["b"].context["chain_how"] == "transferred_unverified"
    assert results["b"].status == "ok"
    assert results["b"].proposal.num_inliers == 0      # no evidence of its own
    assert results["b"].proposal.method == "transfer"
    assert results["b"].context["gate_passed"] is True


def test_the_transfer_resolves_the_gravity_sign():
    """The up hypothesis whose fit is worse is the one pointing the wrong way."""
    trajectories = overlapping()
    flip = np.diag([1.0, -1.0, -1.0])
    prepared = FakePrepared(trajectories["b"],
                            views={0: FakeView(rotation=flip), 1: FakeView()})
    placed = chainmod.Placed("a", Sim2(1.0, 0.0, 0.0, 0.0), 0,
                             trajectories["a"], 2.0)
    transfer = chainmod.transfer_from(placed, prepared, trajectories["b"])
    assert transfer is not None
    assert transfer.up_index == 1                     # the unflipped one
    assert transfer.residual_in == pytest.approx(0.0, abs=1e-6)


def test_a_failed_geo_seed_does_not_buy_the_same_search_twice():
    """Its ladder already ended on an unconstrained search; repeating it is free of
    information and costs three solves."""
    trajectories = contiguous(n=2)

    class Track:
        def first_of(self, _traj):
            return np.array([0.0, 0.0])

    # nothing ever gates, so the seed loop runs out of candidates
    results, calls = walk(trajectories, {r: [FakeProposal(inliers=5)] * 6
                                         for r in trajectories},
                          ChainPolicy(seed_source="geo"), geo=lambda _c: Track())
    # a is geo candidate 1: the full ladder, three rungs, and NOT a fourth
    # search repeating the last of them -- it is in `searched`.
    assert calls["a"] == 3
    # b is geo candidate 2: screened at one rung. That rung is NOT an
    # unconstrained search, so b is still offered to the search loop afterwards.
    assert calls["b"] == 2
    assert results["a"].context["chain_how"] == "best_effort"
    assert results["b"].context["chain_how"] == "seed (search)"
    assert results["b"].context["gate_passed"] is False



def test_wall_bands_are_not_held_for_a_whole_long_capture():
    """`--qualitative` asks the chain to keep each split's Prepared; a 70-split
    capture must not mean 70 of them alive at once."""
    trajectories = contiguous(n=8, length=5)
    policy = ChainPolicy(seed_source="search", keep_prepared=True,
                         max_kept_prepared=3)
    results, _ = walk(trajectories, {r: [FakeProposal()] for r in trajectories},
                      policy)
    kept = sum(1 for r in results.values() if r.prepared is not None)
    assert 0 < kept <= policy.max_kept_prepared + 1
    # and with it off, none are held at all
    results, _ = walk(trajectories, {r: [FakeProposal()] for r in trajectories},
                      ChainPolicy(seed_source="search"))
    assert all(r.prepared is None for r in results.values())


# --- what a row says it is ------------------------------------------------
def test_a_seed_that_needed_a_wider_disc_is_still_the_seed():
    """The rung and the role are different facts. Labelling a widened seed
    "widened" loses the only thing about it the chain cares about."""
    trajectories = contiguous(n=2)

    class Track:
        def first_of(self, _traj):
            return np.array([0.0, 0.0])

    # the geo prior fails at L1 and succeeds at L2
    script = {"a": [FakeProposal(inliers=5), FakeProposal(), FakeProposal()],
              "b": [FakeProposal()]}
    results, _ = walk(trajectories, script, ChainPolicy(seed_source="geo"),
                      geo=lambda _c: Track())
    assert results["a"].context["chain_how"] == "seed (geo)"
    assert results["a"].context["chain_rung"] == "widened"
    assert results["a"].context["chain_seeded_from"] is None


def test_a_chained_split_that_recovers_still_says_so():
    """Where there WAS a neighbour, the fallback rung is the better label: it
    says the prior was abandoned and the chain re-seeded here."""
    trajectories = contiguous(n=3)
    script = {"a": [FakeProposal()],
              "b": [FakeProposal(inliers=5), FakeProposal(inliers=5), FakeProposal()],
              "c": [FakeProposal()]}
    results, _ = walk(trajectories, script)
    assert results["b"].context["chain_how"] == "recovered"
    assert results["b"].context["chain_rung"] == "recovered"
    assert results["b"].context["chain_seeded_from"] == "a"


def test_a_split_that_starts_the_chain_by_accident_is_named_a_seed():
    """No geo, no search seed gates, and the first split that passes on its own
    becomes the seed. Left as "independent" it undercounts seeds and makes the
    chain look like it started from nowhere."""
    trajectories = contiguous(n=3)
    # the two longest-first seed attempts fail, the third split passes
    script = {"a": [FakeProposal(inliers=5)], "b": [FakeProposal(inliers=5)],
              "c": [FakeProposal()]}
    results, _ = walk(trajectories, script,
                      ChainPolicy(seed_source="search", max_seed_attempts=2))
    assert results["c"].context["chain_how"] == "seed (fallback)"
    assert results["c"].context["gate_passed"] is True


# --- reaching past a split that could not be placed -----------------------
from bevaligner.inference.chain import (JUNCTION_MAX_M,  # noqa: E402
                                          MAX_PRIOR_RADIUS_IN,
                                          REACH_RADIUS_MAX_IN, reach_anchor,
                                          reach_radius_in)


def test_reach_anchor_takes_the_nearest_and_breaks_ties_toward_the_walk():
    pos = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    # equidistant either side: the side the walk is coming from wins
    assert reach_anchor("c", "b", pos, {"a": 1, "e": 1}, max_reach=5) == ("a", 2, True)
    assert reach_anchor("c", "d", pos, {"a": 1, "e": 1}, max_reach=5) == ("e", 2, False)
    # an adjacent placed split is always the unique winner, whatever the walk says
    assert reach_anchor("c", "b", pos, {"a": 1, "d": 1}, max_reach=5) == ("d", 1, False)
    # max_reach=1 is the behaviour this replaced: the neighbour, or nothing
    assert reach_anchor("c", "b", pos, {"a": 1}, max_reach=1) is None
    assert reach_anchor("c", "b", pos, {}, max_reach=5) is None


def test_the_reach_radius_grows_then_saturates_short_of_the_ladder_cap():
    assert reach_radius_in(Junction.GAP, 1) == radius_for(Junction.GAP)
    assert reach_radius_in(Junction.GAP, 10) == pytest.approx(33.0)
    assert reach_radius_in(Junction.GAP, 40) == REACH_RADIUS_MAX_IN
    # it must stay under the cap, or every deep reach-back is silently routed
    # to an unconstrained search instead of being guided
    assert REACH_RADIUS_MAX_IN < MAX_PRIOR_RADIUS_IN
    assert reach_radius_in(Junction.SHARED, 1) == radius_for(Junction.SHARED)


def test_the_meeting_end_follows_the_anchor_not_the_walk():
    """`forward` means the ANCHOR is earlier. Reached through walk() only in a
    second pass, so it is pinned against the helper directly."""
    positions = np.c_[np.arange(10.), np.zeros(10), np.zeros(10)]
    placed = chainmod.Placed("a", Sim2(1.0, 0.0, 0.0, 0.0), 0,
                             FakeTraj(0, 10, positions), 2.0)
    assert junction_prior(placed, Junction.GAP, forward=True,
                          reach=4).centre == pytest.approx([9.0, 0.0])
    assert junction_prior(placed, Junction.GAP, forward=False,
                          reach=4).centre == pytest.approx([0.0, 0.0])
    # and the radius widened with the reach while the centre did not move
    assert junction_prior(placed, Junction.GAP, True, reach=4).radius_in \
        == pytest.approx(radius_for(Junction.GAP) + 3.0)


def test_a_split_reaches_past_a_neighbour_that_could_not_be_placed():
    """The whole point of the change."""
    trajectories = contiguous(n=3)
    script = {"a": [FakeProposal()], "b": [FakeProposal(inliers=5)] * 3,
              "c": [FakeProposal()]}
    results, _ = walk(trajectories, script)
    assert results["c"].context["chain_how"] == "chained"
    assert results["c"].context["chain_seeded_from"] == "a"
    assert results["c"].context["chain_reach"] == 2
    assert results["c"].context["gate_passed"] is True


def test_reach_is_bounded_by_policy_and_one_is_the_old_behaviour():
    trajectories = contiguous(n=4)
    script = {"a": [FakeProposal()], "b": [FakeProposal(inliers=5)] * 3,
              "c": [FakeProposal(inliers=5)] * 3, "d": [FakeProposal()]}
    results, _ = walk(trajectories, script,
                      ChainPolicy(seed_source="search", max_reach=2, passes=1))
    assert results["d"].context["chain_how"].startswith("independent")
    assert results["d"].context["chain_reach"] is None
    results, _ = walk(trajectories, {r: [FakeProposal(inliers=5)] * 3
                                     if r != "a" else [FakeProposal()]
                                     for r in trajectories},
                      ChainPolicy(seed_source="search", max_reach=1, passes=1))
    for rid in ("c", "d"):
        assert results[rid].context["chain_how"].startswith("independent")


def test_the_adjacent_case_is_untouched():
    trajectories = contiguous(n=3)
    results, calls = walk(trajectories, {r: [FakeProposal()] * 3
                                         for r in trajectories})
    assert calls == {"a": 1, "b": 1, "c": 1}
    assert results["b"].context["chain_reach"] == 1
    assert results["c"].context["chain_seeded_from"] == "b"
    assert all(r.context["chain_pass"] == 1 for r in results.values())


def test_a_broken_reach_back_junction_is_refused_by_the_prior_ceiling():
    trajectories = contiguous(n=3)
    far = FakeProposal(sim2=Sim2(s=1.0, theta=0.0, tx=900.0, ty=0.0))
    results, _ = walk(trajectories, {"a": [FakeProposal()],
                                     "b": [FakeProposal(inliers=5)] * 3,
                                     "c": [far] * 3})
    assert "junction implausible" in results["c"].context["gate_reason"]
    assert results["c"].context["gate_passed"] is False


def test_the_physical_ceiling_is_metres_and_not_paper_inches():
    """A ceiling in paper inches is an order of magnitude looser on a coarse
    sheet than on a fine one -- `metres_per_inch` runs 2.36 to 30.4 across these
    structures -- which let a placement pass at 29.84 inches while being 100 m
    out. Here the SAME 20-inch drawing error is judged on two sheets."""
    trajectories = contiguous(n=2)

    def run_at(scale):
        # rig_height_m = camera_height x sim2.s x metres_per_inch, and the gate
        # wants it in (1.2, 3.5); hold it at 2 m so the SHEET SCALE is the only
        # thing changing between the two runs.
        height = 2.0 / scale
        ok = FakeProposal(height=height)
        off = FakeProposal(sim2=Sim2(s=1.0, theta=0.0, tx=20.0, ty=0.0), height=height)
        # widen the prior out of the way so only the physical ceiling can decide
        return walk(trajectories, {"a": [ok], "b": [off] * 3},
                    ChainPolicy(seed_source="search", radius_scale=20.0, passes=1),
                    metres_per_inch=scale)[0]

    fine = run_at(0.2)      # 20 in x 0.2 = 4 m: an ordinary contiguous junction
    assert fine["b"].context["chain_junction_realised_m"] == pytest.approx(4.0)
    assert fine["b"].context["gate_passed"] is True

    coarse = run_at(5.0)    # the same 20 in = 100 m, far outside any junction
    assert coarse["b"].context["chain_junction_realised_m"] == pytest.approx(100.0)
    assert coarse["b"].context["gate_passed"] is False
    assert " m across a " in coarse["b"].context["gate_reason"]


def test_the_ceiling_is_loosest_where_continuity_is_weakest():
    """A gap junction is 92% of all junctions and is not physically
    constrained -- the recording stopped and the person walked off."""
    assert JUNCTION_MAX_M[Junction.CONTIGUOUS] < JUNCTION_MAX_M[Junction.SHARED] \
        < JUNCTION_MAX_M[Junction.GAP]


def test_a_reach_back_does_not_also_pay_for_a_widened_disc():
    """Tripling a disc that was already the right place is not what a split
    whose matcher found nothing needs; it goes straight to the search it would
    have had without a prior at all."""
    trajectories = contiguous(n=3)
    script = {"a": [FakeProposal()], "b": [FakeProposal(inliers=5)] * 3,
              "c": [FakeProposal(inliers=5)] * 3}
    _results, calls = walk(trajectories, script,
                           ChainPolicy(seed_source="search", passes=1))
    # b is adjacent: L1, L2, L3. c reaches back: L1 then straight to L3.
    assert calls["b"] == 3
    assert calls["c"] == 2


# --- a capture with nothing to chain over still has a start point ----------
class _Fix:
    """A geo track whose one fix sits at (3, 4), wherever it is asked from."""

    def first_of(self, _traj):
        return np.array([3.0, 4.0])


def test_a_lone_split_is_seeded_from_its_fix():
    """One split cannot be CHAINED, but it can still be told where to look.

    Before this, `orderable` refused the capture and returned before the seed
    loop, so 42 of the 134 test captures carrying a usable fix searched the
    whole sheet for a position already known to within RADIUS_GEO_IN.
    """
    trajectories = {"a": FakeTraj(0, 10)}
    results, _ = walk(trajectories, {"a": [FakeProposal()]},
                      ChainPolicy(seed_source="geo"), geo=lambda _c: _Fix())
    assert results["a"].context["chain_how"] == "seed (geo)"
    assert results["a"].context["chain_prior_radius_in"] == chainmod.RADIUS_GEO_IN
    assert results["a"].context["chain_prior_centre_x"] == 3.0


def test_a_lone_split_without_a_fix_is_unchanged():
    """The shipped label, so a capture with no fix reads exactly as before."""
    trajectories = {"a": FakeTraj(0, 10)}
    results, _ = walk(trajectories, {"a": [FakeProposal()]},
                      ChainPolicy(seed_source="geo"), geo=lambda _c: None)
    assert results["a"].context["chain_how"] == "independent (one split)"
    assert not results["a"].context.get("chain_prior_radius_in")


def test_a_lone_split_still_gets_its_unconstrained_search():
    """The fix never COSTS the split the search it would have had.

    A seed runs `exhaustive`, so the ladder climbs L1 (the fix), L2 (widened)
    and L3 -- and L3 is byte for byte the unconstrained search the old
    `independent (one split)` path ran. Here the two prior rungs find nothing
    and L3 places it, which is the case that would silently regress if the
    fallback were ever dropped.

    It also means the fix cannot make this split WORSE: `best_attempt` takes
    the best-supported rung, and L3 is always among them.
    """
    trajectories = {"a": FakeTraj(0, 10)}
    results, calls = walk(trajectories, {"a": [None, None, FakeProposal()]},
                          ChainPolicy(seed_source="geo"), geo=lambda _c: _Fix())
    assert calls["a"] == 3                       # L1, L2, L3 -- not re-searched after
    assert results["a"].status == "ok"


def test_a_lone_splits_fix_costs_one_extra_ladder_not_one_extra_search():
    """What the change actually costs, written down so it cannot drift.

    Three rungs rather than one search: 5+2 windows, then 10+2, then the full
    `max_windows`. At --max-windows 128 that is 147 against 128, about 15% MORE
    -- the change buys consistency and the best-of-three rung, not speed.
    """
    trajectories = {"a": FakeTraj(0, 10)}
    _, calls = walk(trajectories, {"a": [FakeProposal()]},
                    ChainPolicy(seed_source="geo"), geo=lambda _c: _Fix())
    assert calls["a"] == 3
    _, plain = walk(trajectories, {"a": [FakeProposal()]},
                    ChainPolicy(seed_source="search"), geo=lambda _c: _Fix())
    assert plain["a"] == 1


def test_seed_source_search_ignores_a_fix_on_a_lone_split():
    """`--chain-seed search` means no external hint, one split or many."""
    trajectories = {"a": FakeTraj(0, 10)}
    results, _ = walk(trajectories, {"a": [FakeProposal()]},
                      ChainPolicy(seed_source="search"), geo=lambda _c: _Fix())
    assert results["a"].context["chain_how"] == "independent (one split)"


def test_the_first_split_seeds_and_the_rest_chain_from_it():
    """A fix on the capture's FIRST split seeds it; the others chain outward."""
    trajectories = contiguous(3)
    seen = {}

    def geo(case):
        seen[case.reality_id] = True
        return _Fix() if case.reality_id == "a" else None

    results, _ = walk(trajectories, {r: [FakeProposal()] for r in trajectories},
                      ChainPolicy(seed_source="geo"), geo=geo)
    assert results["a"].context["chain_how"] == "seed (geo)"
    assert results["b"].context["chain_how"] == "chained"
    assert results["b"].context["chain_seeded_from"] == "a"
    assert results["c"].context["chain_seeded_from"] == "b"
