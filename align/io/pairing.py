"""The only bridge between a current case and the priors it may see.

`align.priors` deliberately cannot import `align.io.cases`, so it can never be
handed the current split by accident. That leaves somebody having to put the two
together, and this is that somebody -- one small module, easy to read in full,
where the pool a case is allowed to align against is assembled and the
forbidden set is computed.

The forbidden set is every reality in the case's OWN capture, not just the case
itself. A sibling split's registration is as much the answer as the case's own,
and 49 reality ids appear on both sides of the dataset, so "it came out of the
prior index" is not on its own enough to make a split safe.
"""
from __future__ import annotations

from align.io.cases import Case, PairedIndex
from align.priors import PriorReference


def prior_reference_for(case: Case, index: PairedIndex,
                        metres_per_inch: float = float("nan")
                        ) -> PriorReference | None:
    """The reference for one current case, or None when it has no usable prior.

    None is a real outcome, not an error: one of the 131 current rows names only
    prior splits whose point cloud is missing from the export.
    """
    splits = [s for s in index.priors_for(case) if s.cloud_path is not None]
    if not splits:
        return None
    return PriorReference(
        current_reality_id=case.reality_id,
        current_capture_id=case.capture_id,
        current_rlt_dir=case.rlt_dir,
        current_capture_date=case.capture_date,
        structure_key=case.structure.key,
        splits=splits,
        forbidden_reality_ids=frozenset(index.siblings_of(case)),
        design_tm=case.structure.design_tm,
        metres_per_inch=metres_per_inch,
    )
