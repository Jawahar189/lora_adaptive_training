"""The prior-side boundary, checked at runtime.

`tests/test_no_leakage.py` pins the SURFACE -- which modules may name a
ground-truth file, which may import whom. It cannot pin the role, because the
prior's transform and the current split's answer are the same nine characters
and the only thing separating them is a value that arrives at runtime. These
tests are that half.
"""
import functools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex
from align.io.pairing import prior_reference_for
from align.priors import LeakageError, PriorReference, PriorSplit



@functools.lru_cache(maxsize=1)
def _index():
    try:
        spec = datasets_mod.from_env(require_prior=True)
        if spec is None:
            return None
        ix = PairedIndex.load(spec)
    except (FileNotFoundError, OSError, KeyError):
        return None
    return ix if len(ix.current) else None


def _cases():
    ix = _index()
    return list(ix.current) if ix else []


_SKIP = pytest.mark.skip(reason=f"set ${datasets_mod.ENV_CURRENT} to run against a dataset")
CASES = _cases() or [pytest.param(None, marks=_SKIP)]


def _ids(c):
    return getattr(c, "reality_id", "no-dataset")


def _fake(reality_id, capture_id="CAPother", date="2026-01-01", rlt="/tmp/other"):
    return PriorSplit(reality_id=reality_id, capture_id=capture_id,
                      structure_key="data/STR1", capture_date=date,
                      rlt_dir=Path(rlt), cloud_path=Path(rlt) / "point_cloud.ply",
                      shots_path=Path(rlt) / "shots.geojson")


def _ref(**kw):
    base = dict(current_reality_id="RLT_cur", current_capture_id="CAPcur",
                current_rlt_dir=Path("/tmp/cur"), current_capture_date="2026-06-10",
                structure_key="data/STR1", splits=[_fake("RLT_prior")],
                forbidden_reality_ids=frozenset({"RLT_cur", "RLT_sibling"}))
    base.update(kw)
    return PriorReference(**base)


# -- the guard, on fabricated input so every branch is reachable -------------
def test_refuses_the_current_split():
    with pytest.raises(LeakageError, match="is the current split"):
        _ref().tm("RLT_cur")


def test_refuses_a_sibling_of_the_current_capture():
    with pytest.raises(LeakageError, match="sibling"):
        _ref().tm("RLT_sibling")


def test_refuses_a_split_outside_the_pool():
    with pytest.raises(LeakageError, match="not in the prior pool"):
        _ref().tm("RLT_unknown")


def test_refuses_a_split_from_the_current_capture():
    ref = _ref(splits=[_fake("RLT_p", capture_id="CAPcur")])
    with pytest.raises(LeakageError, match="current capture"):
        ref.tm("RLT_p")


def test_refuses_a_split_aliased_to_the_current_directory():
    """The ids can look fine and the path still be the answer."""
    ref = _ref(splits=[_fake("RLT_p", rlt="/tmp/cur")])
    with pytest.raises(LeakageError, match="own directory"):
        ref.tm("RLT_p")


def test_refuses_a_later_capture():
    ref = _ref(splits=[_fake("RLT_p", date="2026-06-11")])
    with pytest.raises(LeakageError, match="not a prior"):
        ref.tm("RLT_p")


def test_assert_clean_catches_a_read_under_the_current_split():
    ref = _ref()
    ref._record(Path("/tmp/cur/tm.json"))          # what a bypass would look like
    with pytest.raises(LeakageError, match="touched the current split"):
        ref.assert_clean()


# -- the same invariants over the real dataset -------------------------------
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_pool_is_disjoint_from_the_current_capture(case):
    ix = _index()
    ref = prior_reference_for(case, ix)
    if ref is None:
        pytest.skip("no prior with a point cloud")
    siblings = set(ix.siblings_of(case))
    assert case.reality_id not in ref.reality_ids
    assert not (set(ref.reality_ids) & siblings)
    assert case.capture_id not in ref.capture_ids


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_pool_matches_the_csv(case):
    ix = _index()
    ref = prior_reference_for(case, ix)
    if ref is None:
        pytest.skip("no prior with a point cloud")
    assert set(ref.capture_ids) <= set(case.prior_capture_ids)
    for split in ref.splits:
        assert split.structure_key == case.structure.key
        if split.capture_date and case.capture_date:
            assert split.capture_date <= case.capture_date


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_guard_refuses_the_case_itself(case):
    ix = _index()
    ref = prior_reference_for(case, ix)
    if ref is None:
        pytest.skip("no prior with a point cloud")
    with pytest.raises(LeakageError):
        ref.tm(case.reality_id)


def test_exactly_one_case_has_no_usable_prior():
    """Pinned as a number so a data regression is visible rather than absorbed."""
    ix = _index()
    if ix is None:
        pytest.skip(f"set ${datasets_mod.ENV_CURRENT} to run this")
    without = [c.reality_id for c in ix.current
               if prior_reference_for(c, ix) is None]
    assert without == ["RLT1788195295881"], without
