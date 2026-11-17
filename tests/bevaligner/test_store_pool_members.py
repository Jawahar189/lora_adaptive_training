"""`CaptureStore.pool_members` for a reality that is a current row and a prior row at once."""
from types import SimpleNamespace

from bevaligner.data.store import CaptureStore


class Index:
    """Priors by named capture; the bare minimum `pool_members` asks of an index."""

    def __init__(self, priors_by_capture):
        self.priors_by_capture = priors_by_capture

    def siblings_of(self, case):
        return []

    def priors_for(self, case):
        return [split for capture in case.prior_capture_ids
                for split in self.priors_by_capture.get(capture, [])]


def test_preparing_the_prior_side_first_does_not_empty_the_current_side(tmp_path):
    """`infer` prepares the prior rows before placing: a reality that is also a
    prior row (with no priors of its own) must not leave its current row an empty pool."""
    split = SimpleNamespace(reality_id="RLT_OLD", capture_id="SNP_OLD", capture_date="2026-09-01")
    store = CaptureStore(tmp_path)
    store.index = Index({"SNP_OLD": [split]})
    as_prior = SimpleNamespace(reality_id="RLT_X", prior_capture_ids=())
    as_current = SimpleNamespace(reality_id="RLT_X", prior_capture_ids=("SNP_OLD",))
    assert store.pool_members(as_prior) == ()
    assert store.pool_members(as_current) == (("RLT_OLD", "SNP_OLD", "2026-09-01"),)
