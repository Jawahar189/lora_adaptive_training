"""Sequential registration, on synthetic input with a known answer.

None of this reads the dataset. The junction classification, the exact transfer
through shared images, and the translation prior are all checked against cases
constructed so the right answer is known exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.chain import (Junction, _umeyama_sim2, capture_geo, capture_order,
                         classify_junction, expected_start, find_anchor,
                         transfer_pose,
                         walk_outward)
from align.frames import Sim2
from align.search import (SearchGrid, TranslationPrior, get_backend, search)

RES = 0.05


class FakeTraj:
    """Just enough of a Trajectory for the sequencing layer."""

    def __init__(self, start, n, positions=None):
        self.names = [f"VID_x_IMG_{i:05d}.JPG" for i in range(start, start + n)]
        self.idx = np.arange(start, start + n)
        self.positions = (np.zeros((n, 3)) if positions is None
                          else np.asarray(positions, dtype=np.float64))


def room_outline(w=6.0, h=4.0, step=0.02):
    xs, ys = np.arange(0, w, step), np.arange(0, h, step)
    return np.concatenate([
        np.stack([xs, np.zeros_like(xs)], 1),
        np.stack([xs, np.full_like(xs, h)], 1),
        np.stack([np.zeros_like(ys), ys], 1),
        np.stack([np.full_like(ys, w), ys], 1)])


def stamp(points, origin, shape, res=RES):
    m = np.zeros(shape, bool)
    idx = np.floor((points - np.asarray(origin)) / res).astype(int)
    ok = ((idx[:, 0] >= 0) & (idx[:, 0] < shape[1])
          & (idx[:, 1] >= 0) & (idx[:, 1] < shape[0]))
    m[idx[ok, 1], idx[ok, 0]] = True
    return m


# --- sequencing ----------------------------------------------------------
def test_capture_order_follows_image_index():
    trajs = {"b": FakeTraj(500, 100), "a": FakeTraj(1, 100), "c": FakeTraj(900, 50)}
    assert capture_order(trajs) == ["a", "b", "c"]


def test_junction_kinds():
    a = FakeTraj(1, 100)                       # 1..100
    assert classify_junction(a, FakeTraj(101, 50)).kind is Junction.CONTIGUOUS
    assert classify_junction(a, FakeTraj(140, 50)).kind is Junction.GAP
    shared = classify_junction(a, FakeTraj(90, 50))
    assert shared.kind is Junction.SHARED and shared.n_shared == 11


def test_walk_outward_always_has_a_placed_neighbour():
    order = list("abcde")
    placed = {order[2]}
    for rid, frm, _ in walk_outward(order, 2):
        assert frm in placed, f"{rid} leans on {frm}, which is not placed yet"
        placed.add(rid)
    assert placed == set(order)


def test_radius_grows_with_junction_uncertainty():
    assert (Junction.SHARED.radius_m < Junction.CONTIGUOUS.radius_m
            < Junction.GAP.radius_m)


# --- exact transfer through shared images --------------------------------
def test_transfer_pose_is_exact():
    """Two splits sharing photographs: B's pose follows from A's with no search."""
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.normal(size=(160, 2)), axis=0)
    a = FakeTraj(1, 100, np.c_[walk[:100], np.zeros(100)])
    # B lives in its own arbitrary frame, related to A's by a known similarity
    b_frame = Sim2(s=0.4, theta=-1.1, tx=7.0, ty=-3.0)
    b_pos = b_frame.apply(walk[60:])
    b = FakeTraj(61, 100, np.c_[b_pos, np.zeros(len(b_pos))])
    assert classify_junction(a, b).kind is Junction.SHARED

    sim_a = Sim2(s=2.3, theta=0.4, tx=-5.0, ty=11.0)      # A's solved pose
    sim_b, resid = transfer_pose(a, b, sim_a)
    assert resid < 1e-9
    # B's own cameras must land where A's pose says the shared ones are
    got = sim_b.apply(b.positions[:40, :2])
    want = sim_a.apply(walk[60:100])
    assert np.abs(got - want).max() < 1e-8


def test_transfer_pose_needs_enough_shared_images():
    a, b = FakeTraj(1, 100), FakeTraj(99, 50)             # 2 shared
    assert transfer_pose(a, b, Sim2(1.0, 0.0, 0.0, 0.0)) is None


def test_expected_start_is_the_end_of_the_previous_split():
    walk = np.c_[np.linspace(0, 10, 50), np.zeros(50), np.zeros(50)]
    a = FakeTraj(1, 50, walk)
    b = FakeTraj(51, 50, walk)
    sim_a = Sim2(s=2.0, theta=0.0, tx=1.0, ty=1.0)
    centre, radius = expected_start(a, b, sim_a, Junction.CONTIGUOUS,
                                    metres_per_inch=5.0)
    assert np.allclose(centre, sim_a.apply(walk[-1:, :2])[0])
    assert radius == pytest.approx(Junction.CONTIGUOUS.radius_m / 5.0)


# --- the prior, inside the search ----------------------------------------
def test_prior_pulls_the_answer_to_the_right_copy():
    """Two identical rooms on one sheet. Only the prior can say which."""
    shape, origin = (300, 700), (0.0, 0.0)
    room = room_outline(w=5.0, h=3.0)
    left, right = np.array([2.0, 3.0]), np.array([22.0, 3.0])
    sheet = (stamp(room + left, origin, shape)
             | stamp(room + right, origin, shape)).astype(np.float32)
    grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([0.0]),
                      viewports=[0])
    bk = get_backend("numpy")

    for target in (left, right):
        prior = TranslationPrior(centre=target, radius=1.5,
                                 ref=np.array([0.0, 0.0]))
        best = search(room, room[::40], sheet, origin, RES, grid, backend=bk,
                      w_free=0.0, w_scale=0.0, prior=prior, top_k=1)[0]
        assert not best.outside_prior
        assert abs(best.sim2.tx - target[0]) <= 3 * RES, (
            f"prior at {target} did not select that copy")


def test_prior_falls_back_when_its_disc_is_empty():
    """A prior pointing off the sheet must not return nothing."""
    shape, origin = (200, 300), (0.0, 0.0)
    room = room_outline(w=4.0, h=3.0)
    sheet = stamp(room + np.array([1.0, 1.0]), origin, shape).astype(np.float32)
    grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([0.0]),
                      viewports=[0])
    hyps = search(room, room[::40], sheet, origin, RES, grid,
                  backend=get_backend("numpy"), w_free=0.0, w_scale=0.0,
                  prior=TranslationPrior(centre=np.array([900.0, 900.0]),
                                         radius=0.5,
                                         ref=np.array([0.0, 0.0])),
                  top_k=1)
    assert hyps, "fallback must still return a hypothesis"
    assert hyps[0].outside_prior, "and must flag that the disc was empty"


def test_prior_records_distance_from_its_centre():
    shape, origin = (200, 300), (0.0, 0.0)
    room = room_outline(w=4.0, h=3.0)
    sheet = stamp(room + np.array([1.0, 1.0]), origin, shape).astype(np.float32)
    grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([0.0]),
                      viewports=[0])
    best = search(room, room[::40], sheet, origin, RES, grid,
                  backend=get_backend("numpy"), w_free=0.0, w_scale=0.0,
                  prior=TranslationPrior(centre=np.array([1.0, 1.0]), radius=2.0,
                                         ref=np.array([0.0, 0.0])),
                  top_k=1)[0]
    assert best.prior_dist == pytest.approx(0.0, abs=3 * RES)


# --- the anchor ----------------------------------------------------------
#
# The start point names a capture moment twice over -- a reality id and a
# photograph -- and the two disagree on 21 of 59 splits. These pin which one
# decides, and pin that a name matching nothing produces no prior at all
# rather than a confidently wrong one.


class FakeGeo:
    def __init__(self, image, x, y, z=0.0):
        self.image, self.x, self.y, self.z = image, x, y, z


class FakeStructure:
    design_tm = None


class FakeCase:
    def __init__(self, rid, geo=None):
        self.reality_id = rid
        self.geo = geo
        self.structure = FakeStructure()


def _two_splits():
    """A capture cut in two: images 1..50 and 51..100, walking along +x."""
    a = FakeTraj(1, 50, np.c_[np.arange(50) * 0.1, np.zeros(50), np.zeros(50)])
    b = FakeTraj(51, 50, np.c_[5.0 + np.arange(50) * 0.1,
                               np.zeros(50), np.zeros(50)])
    return {"a": a, "b": b}


def test_anchor_follows_the_photograph_not_the_reality_id():
    """The 19-split case: the point hangs off 'a' but names an image in 'b'."""
    trajs = _two_splits()
    cases = [FakeCase("a", FakeGeo(trajs["b"].names[10], 6.0, 0.0)),
             FakeCase("b")]
    anchor = find_anchor(cases, trajs, cases[0].geo, metres_per_inch=1.0)
    assert anchor is not None
    assert anchor.reality_id == "b", "seeded the split that does not hold the image"
    assert anchor.centre == pytest.approx([6.0, 0.0])


def test_anchor_is_a_window_around_the_named_image():
    trajs = _two_splits()
    cases = [FakeCase("a", FakeGeo(trajs["a"].names[25], 2.5, 0.0)),
             FakeCase("b")]
    anchor = find_anchor(cases, trajs, cases[0].geo, metres_per_inch=1.0,
                         window=5)
    assert anchor.ref.shape == (11, 2), "expected the image and 5 either side"
    assert anchor.ref[:, 0] == pytest.approx(np.arange(20, 31) * 0.1)


def test_anchor_window_is_clipped_at_the_ends_of_a_split():
    trajs = _two_splits()
    cases = [FakeCase("a", FakeGeo(trajs["a"].names[1], 0.1, 0.0)), FakeCase("b")]
    anchor = find_anchor(cases, trajs, cases[0].geo, metres_per_inch=1.0,
                         window=20)
    assert len(anchor.ref) == 22, "window must stop at the first image"


def test_an_unfindable_image_yields_no_anchor():
    """No silent fallback. A wrong prior is worse than none: inside a 4-inch
    radius it excludes the true pose outright."""
    trajs = _two_splits()
    cases = [FakeCase("a", FakeGeo("VID_x_IMG_99999.JPG", 1.0, 2.0)),
             FakeCase("b")]
    assert find_anchor(cases, trajs, cases[0].geo, metres_per_inch=1.0) is None


def test_capture_geo_finds_a_point_on_any_member():
    trajs = _two_splits()
    cases = [FakeCase("a"), FakeCase("b", FakeGeo(trajs["a"].names[0], 0.0, 0.0))]
    assert capture_geo(cases) is cases[1].geo
    assert capture_geo([FakeCase("a"), FakeCase("b")]) is None


# --- the windowed prior ---------------------------------------------------


def test_windowed_prior_admits_a_pose_where_any_reference_lands_in_the_disc():
    """The union of discs, not the disc of the first point."""
    ref = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    prior = TranslationPrior(centre=np.array([10.0, 5.0]), radius=0.5, ref=ref)
    centres = prior.allowed_translations(scale=1.0, yaw_deg=0.0)
    assert centres.shape == (3, 2)
    assert centres[0] == pytest.approx([10.0, 5.0])
    assert centres[2] == pytest.approx([4.0, 5.0])


def test_single_point_prior_still_works():
    """A (2,) ref is a window of one, so nothing that passed one breaks."""
    prior = TranslationPrior(centre=np.array([2.0, 1.0]), radius=1.0,
                             ref=np.array([1.0, 0.0]))
    assert prior.ref.shape == (1, 2)
    assert prior.allowed_translation(1.0, 0.0) == pytest.approx([1.0, 1.0])


def test_windowed_prior_finds_the_copy_a_single_camera_would_miss():
    """Two identical rooms; the window brackets the right one.

    The reference camera nearest the surveyed point is NOT the one the point
    names -- which is the real failure mode, since the point describes a moment
    and the walk has moved on by the time the shutter fired. A single-camera
    prior centres its disc on the wrong copy; the window contains both.
    """
    res, shape = RES, (140, 300)
    origin = (0.0, 0.0)
    left = room_outline() + np.array([1.0, 1.0])
    right = room_outline() + np.array([8.0, 1.0])
    sheet = stamp(np.vstack([left, right]), origin, shape).astype(np.float32)

    cloud = room_outline()
    # The walk runs along the bottom wall; the surveyed point sits at its far
    # end, which is camera 40 of 41 rather than camera 0.
    walk = np.c_[np.linspace(0.0, 6.0, 41), np.zeros(41)]
    grid = SearchGrid(scales=np.array([1.0]), yaws_deg=np.array([0.0]),
                      viewports=[])
    truth = np.array([8.0, 1.0])                 # the RIGHT copy
    surveyed = truth + walk[-1]

    def best(prior):
        hyps = search(cloud, walk, sheet, origin, res, grid,
                      backend=get_backend(), prior=prior)
        return np.array([hyps[0].sim2.tx, hyps[0].sim2.ty])

    single = TranslationPrior(centre=surveyed, radius=1.0, ref=walk[0])
    windowed = TranslationPrior(centre=surveyed, radius=1.0, ref=walk)
    assert np.linalg.norm(best(windowed) - truth) < 0.5, "window missed the copy"
    assert np.linalg.norm(best(single) - truth) > 1.0, (
        "this test is only meaningful if the single-camera prior fails here")
