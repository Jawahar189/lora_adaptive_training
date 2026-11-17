"""Two rows of the real CSV through the store to a training pair.

Skips unless the cache already holds those rows AND their prior splits (filling
it needs the point clouds on disk and a minute of compute);
`bevaligner.data.preprocess` or a training run creates it.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex
from bevaligner.config import Config
from bevaligner.data.dataset import DensityPriorPairDataset, collate_pairs
from bevaligner.data.store import CaptureStore
from bevaligner.utils import affine

REPO = Path(__file__).resolve().parents[2]
# Overridable, so the same tests can be pointed at a small cache built for one
# structure instead of waiting on the full one:
#   BEVALIGNER_TEST_CACHE=cache/smoke BEVALIGNER_TEST_CSV=... pytest ...
CSV = Path(os.environ.get("BEVALIGNER_TEST_CSV",
                          REPO / "datasets" / "100_prod_current.csv"))
PRIOR_CSV = Path(os.environ.get("BEVALIGNER_TEST_PRIOR_CSV",
                                REPO / "datasets" / "100_prod_prior.csv"))
CACHE = Path(os.environ.get("BEVALIGNER_TEST_CACHE",
                            REPO / "cache" / "bevaligner"))


@pytest.fixture(scope="module")
def two_cached_records():
    if not CSV.exists() or not PRIOR_CSV.exists() or not CACHE.exists():
        pytest.skip("no CSV or no cache")
    store = CaptureStore(CACHE)
    store.index = PairedIndex.load(
        datasets_mod.resolve([str(CSV)], [str(PRIOR_CSV)], name="test"))
    # `capture_records` is the gate -- cached, pooled AND covered -- so ask it
    # rather than reimplementing the same three conditions here
    records = []
    for case in store.index.current:
        records += store.capture_records([case])
        if len(records) == 2:
            break
    if len(records) < 2:
        pytest.skip("fewer than two cached rows with a usable prior pool")
    return records, store


@pytest.mark.parametrize("train", [False, True])
def test_samples_have_the_right_shape_and_targets_inside(two_cached_records, train):
    records, store = two_cached_records
    data_config = Config().data
    data_config.queries_per_sample = 256
    dataset = DensityPriorPairDataset(records, store, data_config, patch_size=14,
                                          train=train, length=6, seed=0)
    for index in range(6):
        sample = dataset[(index, 504)]
        assert sample["density"].shape == (3, 504, 504)
        assert sample["target"].shape == (3, 504, 504)
        assert sample["target_patch_valid"].shape == (36, 36)
        assert sample["query_px"].shape == (256, 2)
        assert sample["target_px"].shape == (256, 2)
        target_px = sample["target_px"].numpy()
        assert (target_px >= 0).all() and (target_px < 504).all()
        assert affine.determinant(sample["density_to_crop"].numpy()) > 0
    batch = collate_pairs([dataset[(0, 504)], dataset[(1, 504)]])
    assert batch["density"].shape[0] == 2 and len(batch["meta"]) == 2


def test_gt_cameras_land_where_the_chain_says(two_cached_records):
    """The retarget's load-bearing assertion, on real rows rather than synthetic.

    A camera carried through `density_to_crop` must land where `paper_to_crop`
    puts its ground-truth paper position. Only the OVERLAPPING cameras are
    required to be inside the window: the walk often covers ground the prior
    never did, and the window is deliberately placed on the part it did.
    """
    records, store = two_cached_records
    dataset = DensityPriorPairDataset(records, store, Config().data, patch_size=14,
                                      train=False)
    for index, record in enumerate(records):
        sample = dataset[(index, 1008)]
        density_px_to_world_xy, density_to_crop, paper_to_crop = (
            sample[name].numpy().astype(np.float64)
            for name in ("density_px_to_world_xy", "density_to_crop", "paper_to_crop"))
        cameras_in_density = affine.apply(affine.invert(density_px_to_world_xy),
                                          record.camera_xy)
        via_chain = affine.apply(density_to_crop, cameras_in_density)
        direct = affine.apply(paper_to_crop, record.camera_paper_xy)
        assert np.allclose(via_chain, direct, atol=0.05)
        anchors = affine.apply(paper_to_crop, record.anchor_paper_xy)
        inside = (anchors >= 0).all(1) & (anchors < 1008).all(1)
        assert inside.mean() > 0.8


def test_records_follow_csv_order(two_cached_records):
    records, store = two_cached_records
    position = {case.reality_id: i for i, case in enumerate(store.index.current)}
    assert position[records[0].reality_id] < position[records[1].reality_id]


def test_patch_validity_marks_where_the_prior_walked(two_cached_records):
    records, store = two_cached_records
    dataset = DensityPriorPairDataset(records, store, Config().data, patch_size=14,
                                          train=False)
    patch_valid = dataset[(0, 504)]["target_patch_valid"]
    # the prior's coverage, not padding: some of the window is always outside it
    assert patch_valid.dtype == torch.bool and patch_valid.any()


def _balanced_dataset(layout: dict[str, dict[str, int]], power, length: int = 20000):
    """`layout` is {structure: {capture: n_realities}}, so a test can set the
    skew at both levels independently."""
    from types import SimpleNamespace
    dataset = DensityPriorPairDataset.__new__(DensityPriorPairDataset)
    dataset.records = [SimpleNamespace(structure_key=s, capture_id=f"{s}/{c}")
                       for s, caps in layout.items()
                       for c, n in caps.items() for _ in range(n)]
    dataset.seed, dataset.epoch, dataset.length = 0, 0, length
    dataset.record_cdf = dataset._structure_balanced_cdf(power)
    return dataset


def _shares(dataset, attr, draws=30000):
    from collections import Counter
    got = Counter(getattr(dataset.records[dataset._record_for(i)], attr)
                  for i in range(draws))
    return {k: v / draws for k, v in got.items()}


def test_structure_balance_of_none_is_the_plain_walk():
    """The default must allocate nothing and choose exactly as before, or every
    run predating the knob stops reproducing."""
    dataset = _balanced_dataset({"a": {"c0": 5}, "b": {"c0": 1}}, power=None)
    assert dataset.record_cdf is None
    assert [dataset._record_for(i) for i in range(8)] == [0, 1, 2, 3, 4, 5, 0, 1]


def test_surveys_of_one_structure_are_drawn_equally_however_long_they_ran():
    """The inner draw, and the reason this is keyed on captures at all: a
    69-reality survey must not outvote its 3-reality sibling."""
    dataset = _balanced_dataset({"a": {"long": 69, "short": 3}}, power=1.0)
    by_capture = _shares(dataset, "capture_id")
    assert by_capture["a/long"] == pytest.approx(0.5, abs=0.02)
    assert by_capture["a/short"] == pytest.approx(0.5, abs=0.02)


def test_power_zero_draws_every_structure_equally():
    """`a` has 8 surveys to `b`'s 1 and 100 realities to its 1; at 0 neither
    buys anything, which is the paper's per-scene shape."""
    layout = {"a": {f"c{i}": 12 for i in range(8)} | {"c8": 4}, "b": {"c0": 1}}
    shares = _shares(_balanced_dataset(layout, power=0.0), "structure_key")
    assert shares["a"] == pytest.approx(0.5, abs=0.02)


def test_power_one_pays_for_surveys_not_for_their_length():
    """9 surveys against 1 is 9x. The 100 realities behind those 9 are not
    9x again -- that is the whole difference from weighting realities."""
    layout = {"a": {f"c{i}": 12 for i in range(8)} | {"c8": 4}, "b": {"c0": 1}}
    shares = _shares(_balanced_dataset(layout, power=1.0), "structure_key")
    assert shares["a"] == pytest.approx(9 / 10, abs=0.02)      # not 100/101


def test_structure_balance_interpolates_and_is_deterministic():
    layout = {"a": {f"c{i}": 12 for i in range(8)} | {"c8": 4}, "b": {"c0": 1}}
    half = _shares(_balanced_dataset(layout, power=0.5), "structure_key")["a"]
    assert 0.5 < half < 9 / 10
    # same index, same record: forked workers agree without set_epoch reaching them
    first, second = _balanced_dataset(layout, 1.0), _balanced_dataset(layout, 1.0)
    assert [first._record_for(i) for i in range(50)] == \
           [second._record_for(i) for i in range(50)]
