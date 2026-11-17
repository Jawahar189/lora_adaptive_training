"""The prior pool a current row is allowed to see, and what it is made of.

`align/priors.py` guards the classical solver's reads at runtime; the learned
path does not go through it, because the prior's transform reaches it through
the capture cache rather than through `tm.json`. The guarantee therefore has to
be re-asserted here, against the real manifests: no row may ever be handed a
split of its own capture.
"""
from pathlib import Path

import numpy as np
import pytest

from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex
from bevaligner.config import DataConfig
from bevaligner.data import prior_bev
from bevaligner.data.store import CaptureStore

from conftest import make_split

CURRENT = Path("/mnt/align-data/datasets/manifest_all_19_aug_23_sep_current_with_prior.csv")
PRIOR = Path("/mnt/align-data/datasets/manifest_all_19_aug_23_sep_prior.csv")


@pytest.fixture(scope="module")
def paired_store(tmp_path_factory):
    if not CURRENT.exists() or not PRIOR.exists():
        pytest.skip("the 19 Aug - 23 Sep manifests are not on this machine")
    index = PairedIndex.load(
        datasets_mod.resolve([str(CURRENT)], [str(PRIOR)], name="test"))
    store = CaptureStore(tmp_path_factory.mktemp("cache"), data_config=DataConfig())
    store.index = index
    return store, index


def test_no_row_can_see_a_split_of_its_own_capture(paired_store):
    """The one that matters. 1,457 capture ids appear on both sides of these
    manifests, so "it came out of the prior index" is not enough to make a
    split safe -- a sibling's registration is as much the answer as the row's.
    """
    store, index = paired_store
    offenders = []
    for case in index.current:
        siblings = set(index.siblings_of(case))
        for reality_id, capture_id, _date in store.pool_members(case):
            if reality_id in siblings or capture_id == case.capture_id:
                offenders.append((case.reality_id, reality_id))
    assert offenders == []


def test_every_pool_member_is_a_named_prior_capture(paired_store):
    store, index = paired_store
    for case in list(index.current)[:2000]:
        allowed = set(case.prior_capture_ids)
        for _reality_id, capture_id, _date in store.pool_members(case):
            assert capture_id in allowed


def test_the_join_reaches_the_rows_the_manifests_promise(paired_store):
    """A regression guard on the denominator, not on the code: if this moves,
    the dataset moved and every number quoted against it has to move too.

    It has already earned its keep once -- the export was refreshed on
    2026-09-28 and `prior_capture_id` changed on 43% of current rows, which this
    caught and nothing else would have.
    """
    _store, index = paired_store
    coverage = index.coverage()
    assert coverage["current"] == 11177
    assert coverage["prior"] == 10528
    assert coverage["with_priors"] == 11160


def test_a_mis_registered_split_is_dropped_from_the_pool():
    """One split disagreeing with the others by more than the tolerance poisons
    the whole target, and nothing downstream can tell that from a hard pair."""
    names = tuple(f"IMG_{i}.jpg" for i in range(6))
    good_xy = np.stack([np.arange(6.0), np.zeros(6)], axis=1)
    agrees = make_split("RLTa", camera_paper_xy=good_xy, names=names,
                        band_paper_xy=np.random.default_rng(0).normal(size=(500, 2)))
    also_agrees = make_split("RLTb", camera_paper_xy=good_xy + 0.01, names=names,
                             band_paper_xy=np.random.default_rng(1).normal(size=(500, 2)))
    # 40 inches out, against a 5-rig-height (about 5 m) tolerance
    disagrees = make_split("RLTc", camera_paper_xy=good_xy + 40.0, names=names,
                           band_paper_xy=np.random.default_rng(2).normal(size=(500, 2)))
    kept, dropped, _levels = prior_bev.select_splits([agrees, also_agrees, disagrees])
    assert {s.reality_id for s in kept} == {"RLTa", "RLTb"}
    assert dropped and dropped[0][0] == "RLTc"


def test_a_pool_never_leaves_one_structure(paired_store):
    """Which is why there is no floor resolution anywhere in this module.

    A structure carries exactly one floor plan (598 of 598 in this export) and a
    capture belongs to exactly one structure (1,944 of 1,944), so every split in
    a pool is on the same floor by construction. Clustering them by height only
    discarded the ones whose registration had drifted -- the camera-z spread
    within a structure has a median of 0.18 m.
    """
    _store, index = paired_store
    for case in index.current:
        keys = {s.structure_key for s in index.priors_for(case)}
        assert keys <= {case.structure.key}, case.reality_id


def test_splits_on_one_floor_are_all_kept():
    """Height is not a reason to drop a split, however far apart they sit."""
    rng = np.random.default_rng(0)
    splits = [make_split(f"RLT{i}", camera_paper_xy=rng.normal(size=(10, 2)),
                         band_paper_xy=rng.normal(size=(900, 2)),
                         camera_paper_z_median=z)
              for i, z in enumerate((100.0, 103.0, 106.0))]
    kept, dropped, levels = prior_bev.select_splits(splits)
    assert {s.reality_id for s in kept} == {"RLT0", "RLT1", "RLT2"}
    assert dropped == [] and levels == 1


def test_boxes_that_share_ground_pair_and_boxes_that_touch_do_not():
    """One predicate decides both the pool and the gate, and it takes no number."""
    walk = np.array([0.0, 0.0, 10.0, 10.0])
    assert prior_bev.boxes_overlap(walk, np.array([5.0, 5.0, 20.0, 20.0]))
    assert prior_bev.boxes_overlap(walk, np.array([-5.0, -5.0, 50.0, 50.0]))  # contained
    assert not prior_bev.boxes_overlap(walk, np.array([11.0, 0.0, 20.0, 10.0]))
    # sharing an edge is sharing no ground: area, not intersection
    assert not prior_bev.boxes_overlap(walk, np.array([10.0, 0.0, 20.0, 10.0]))
    # scale-free by construction -- the same geometry a hundred times over
    for other in ([5.0, 5.0, 20.0, 20.0], [11.0, 0.0, 20.0, 10.0]):
        assert prior_bev.boxes_overlap(walk, np.array(other)) == \
            prior_bev.boxes_overlap(walk * 100, np.array(other) * 100)


def test_a_box_ignores_a_stray_camera():
    """Trimmed at 2.5-97.5, because one walk's positions span 397 m and an
    untrimmed box would overlap half the site."""
    walk = np.zeros((100, 2))
    walk[:, 0] = np.linspace(0, 10, 100)
    walk[0] = [10_000.0, 10_000.0]
    box = prior_bev.trimmed_bbox(walk)
    assert box[2] < 20.0 and box[3] < 20.0
    assert not prior_bev.boxes_overlap(box, np.array([100.0, 100.0, 200.0, 200.0]))


def test_cameras_in_bbox_marks_what_the_window_anchors_on():
    walk = np.stack([np.linspace(0, 100, 101), np.zeros(101)], axis=1)
    mask = prior_bev.cameras_in_bbox(walk, np.array([-1.0, -1.0, 50.0, 1.0]))
    assert mask[:51].all() and not mask[51:].any()
    assert prior_bev.cameras_in_bbox(np.zeros((0, 2)),
                                     np.array([0.0, 0.0, 1.0, 1.0])).shape == (0,)




def test_the_pool_keeps_every_overlapping_split_not_the_best_one():
    """Pairing exactly one was measured and rejected: it costs 8.4pp of rows and
    takes median coverage 0.50 -> 0.30, because the box that contains the walk
    best is the one whose cameras actually track it only 82.6% of the time.
    Keeping all that overlap gives the singular answer anyway on 40.5% of rows.
    """
    import types
    boxes = {
        "RLTover1": np.array([0.0, 0.0, 12.0, 12.0]),      # overlaps
        "RLTover2": np.array([8.0, 8.0, 30.0, 30.0]),      # overlaps, much bigger
        "RLTaway":  np.array([90.0, 90.0, 99.0, 99.0]),    # elsewhere on the site
    }
    store = prior_bev.PriorPoolStore.__new__(prior_bev.PriorPoolStore)
    store._bboxes = dict(boxes)
    store.paths = types.SimpleNamespace(capture_npz=lambda r: Path("/nonexistent"))
    kept = store.overlapping(list(boxes), np.array([5.0, 5.0, 10.0, 10.0]))
    assert set(kept) == {"RLTover1", "RLTover2"}
    # a walk nowhere near any of them pairs with nothing, which is the gate
    assert store.overlapping(list(boxes), np.array([50.0, 50.0, 60.0, 60.0])) == ()
    # and no current box means no pool rather than every pool
    assert store.overlapping(list(boxes), None) == ()


def test_occupancy_sees_walls_where_boxes_and_cameras_do_not():
    """The two failures that made this the criterion, in miniature.

    A box around an L-shaped walk is mostly empty, so a walk sitting in that
    emptiness scores 1.00 on the box and shares no wall; and a prior that walked
    a corridor while the current walked a room puts their cameras far apart and
    their walls together.
    """
    # an L of wall, and a small walk in the empty corner of its bounding box
    arm = np.stack([np.linspace(0, 100, 400), np.zeros(400)], axis=1)
    prior = np.vstack([arm, arm[:, ::-1]])              # along x, then along y
    walk = np.random.default_rng(0).normal(size=(300, 2)) * 2 + [70.0, 70.0]

    box_says = prior_bev.boxes_overlap(prior_bev.trimmed_bbox(walk),
                                       prior_bev.trimmed_bbox(prior))
    _mask, occ = prior_bev.band_occupancy(walk, prior, cell=2.11)
    assert box_says and occ == 0.0        # the box is fooled, occupancy is not

    # walls that coincide score high however far apart the two walks' paths are
    _mask, same = prior_bev.band_occupancy(prior, prior, cell=2.11)
    assert same == 1.0


def test_occupancy_is_a_share_of_the_current_walk():
    """Not symmetric: a huge prior containing a small walk is a perfect target,
    which is the same reason IoU was rejected."""
    walk = np.stack([np.linspace(0, 20, 200), np.zeros(200)], axis=1)
    big = np.stack([np.linspace(-500, 500, 20_000), np.zeros(20_000)], axis=1)
    _mask, covered = prior_bev.band_occupancy(walk, big, cell=2.11)
    assert covered == 1.0
    _mask, reverse = prior_bev.band_occupancy(big, walk, cell=2.11)
    assert reverse < 0.1


def test_occupancy_marks_which_part_of_the_walk_is_covered():
    """The mask is what anchors the target window on the covered half."""
    walk = np.stack([np.linspace(0, 100, 101), np.zeros(101)], axis=1)
    prior = np.stack([np.linspace(0, 50, 200), np.zeros(200)], axis=1)
    mask, share = prior_bev.band_occupancy(walk, prior, cell=2.11)
    assert mask[:45].all() and not mask[-45:].any()
    assert 0.45 < share < 0.65
