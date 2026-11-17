"""What the solver actually opens, on real data.

The strongest statement available: run the real thing and watch the filesystem.
Static tests pin names; this pins behaviour, and it is the only check that would
catch a read performed inside a C extension or through a path assembled at
runtime.
"""
import functools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import reference as R
from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex
from align.io.pairing import prior_reference_for
from align.io.ply import read_ply
from align.io.shots import load_shots

# Every file that describes where the CURRENT split already is.
ANSWER = {"tm.json", "images_full.json", "images.json", "geo.txt"}


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


def _solvable(n):
    ix = _index()
    if ix is None:
        return []
    out = []
    for case in ix.current:
        if case.has_cloud and prior_reference_for(case, ix) is not None:
            out.append(case)
        if len(out) == n:
            break
    return out


_SKIP = pytest.mark.skip(reason=f"set ${datasets_mod.ENV_CURRENT} to run against a dataset")
CASES = _solvable(3) or [pytest.param(None, marks=_SKIP)]


@pytest.mark.parametrize("case", CASES, ids=lambda c: getattr(c, "reality_id", "none"))
def test_building_a_reference_opens_nothing_of_the_current_split(case, audit_reads):
    """The prior side, watched.

    Builds the reference from scratch -- cache bypassed, or the test would watch
    an npz being read instead of the work it stands for.
    """
    ix = _index()
    ref = prior_reference_for(case, ix)
    with audit_reads(case.rlt_dir, ANSWER) as hits:
        assert R.build(ref, case.structure, stride=24) is not None
    assert not hits, f"reference build opened {[str(h) for h in hits]}"
    ref.assert_clean()


@pytest.mark.parametrize("case", CASES, ids=lambda c: getattr(c, "reality_id", "none"))
def test_reading_the_current_split_opens_only_its_inputs(case, audit_reads):
    """The current side: a cloud and an SfM trajectory, and nothing else."""
    with audit_reads(case.rlt_dir, ANSWER) as hits:
        read_ply(case.cloud_path, stride=64)
        load_shots(case.rlt_dir)
    assert not hits, f"solver inputs opened {[str(h) for h in hits]}"


def test_the_audit_would_notice(audit_reads):
    """The hook is worthless if it cannot see a read; prove it can."""
    ix = _index()
    if ix is None:
        pytest.skip(f"set ${datasets_mod.ENV_CURRENT} to run this")
    case = _solvable(1)[0]
    with audit_reads(case.rlt_dir, ANSWER) as hits:
        (Path(case.rlt_dir) / "tm.json").read_text()
    assert hits, "the audit hook saw nothing; the other two tests prove nothing"
