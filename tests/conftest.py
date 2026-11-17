"""A read audit, for the half of the boundary that cannot be read off the source.

`tests/test_no_leakage.py` scans string literals and imports. It cannot see a
read performed by a C extension -- PIL opening a sheet, `align/io/ply.py`
memory-mapping a cloud -- and it cannot see WHICH split a path refers to, which
is the whole question now that a prior's transform is an input and the current
split's is the answer.

`sys.addaudithook` sees every open by value. The hook is process-global and
cannot be removed once installed, so it is a no-op unless armed, the armed state
is thread-local, and a session finalizer asserts it was put back.
"""
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

_state = threading.local()
_installed = False


def _hook(event, args):
    if event != "open":
        return
    watch = getattr(_state, "watch", None)
    if not watch:
        return
    path = args[0]
    if not isinstance(path, (str, bytes, Path)):
        return                      # an fd, not a path
    try:
        resolved = Path(path if isinstance(path, (str, Path))
                        else path.decode()).resolve()
    except (ValueError, OSError):
        return
    for root, names, hits in watch:
        if resolved.name in names and (root == resolved.parent or root in resolved.parents):
            hits.append(resolved)


def _install():
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True


@contextmanager
def forbid_reads_under(root, names):
    """Record any open of `names` under `root` while the block runs."""
    _install()
    hits: list[Path] = []
    entry = (Path(root).resolve(), set(names), hits)
    watch = getattr(_state, "watch", None)
    if watch is None:
        watch = _state.watch = []
    watch.append(entry)
    try:
        yield hits
    finally:
        watch.remove(entry)


@pytest.fixture
def audit_reads():
    return forbid_reads_under


@pytest.fixture(scope="session", autouse=True)
def _audit_is_disarmed_at_the_end():
    yield
    assert not getattr(_state, "watch", None), "a read audit was left armed"
