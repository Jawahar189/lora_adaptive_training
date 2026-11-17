"""Enforces the ground-truth boundary -- the half of it that is syntactic.

The boundary used to be one rule: `align/groundtruth.py` is the only module
that may open tm.json or images_full.json. That rule died when the solver
started aligning to the prior capture, because a PRIOR split's tm.json is a
legitimate input -- the prior is already registered in production -- while the
CURRENT split's is the answer, and the two are the same nine characters. No
static property separates them: the discriminator is a value that arrives at
runtime.

So the boundary is now two halves, and this file is only the first:

  * HERE, statically: which modules may NAME a ground-truth file, which may
    IMPORT a module that reads one, and -- for the prior reader -- which files
    it may name at all. Exactly two modules may name tm.json, and that is
    asserted as an equality so a third fails loudly.

  * `tests/test_priors.py`, at runtime: that the prior reader refuses the
    current split, refuses its siblings, refuses anything outside the declared
    pool, and records what it opened. That is where the actual guarantee lives.

Read the static tests as pinning a convention. The one that cannot be talked
around is `PriorReference._check`, and its tests are next door.
"""
import ast
import re
from pathlib import Path

import pytest

ALIGN = Path(__file__).resolve().parents[1] / "align"

# The current split's own answer, in every shape it takes on disk:
#   tm.json              the transform being solved for
#                        -- but NOT design_tm.json or tm_local_to_global.json,
#                           which are sheet georeferencing and legitimate input
#   images_full.json     camera positions already in the target frame
#   images.json          production's localisation output, also target frame,
#                        and a baseline this project has to beat rather than read
#
# geo.txt is deliberately ABSENT. It looked like ground truth and is not: one z
# value per file, 2-42 m from images_full, and on paper-frame structures it is
# in paper inches with z = 1.7 -- the per-image form of the single surveyed
# point the old regime treated as a legitimate bootstrap. It is quarantined
# behind an opt-in flag rather than banned, pending a provenance decision, and
# nothing reads it today.
FORBIDDEN = re.compile(r"(?<![\w])tm\.json"
                       r"|images_full"
                       r"|(?<![\w])images\.json")

# May NAME a ground-truth file.
#   groundtruth.py  reads the current split's, for evaluation only
#   priors.py       reads a PRIOR's tm.json, and only through its own guard
MAY_NAME = {"groundtruth.py", "priors.py"}

# May IMPORT a module that reads one. Evaluation only -- and note this is a
# SEPARATE set from MAY_NAME. They used to be the same set, which silently let
# align/io/paths.py import the evaluator.
MAY_IMPORT = {"groundtruth.py", "metrics.py"}

# Modules that do the actual solving. If any of these can see ground truth, a
# result from this pipeline means nothing.
SOLVING_MODULE_NAMES = {"cloud.py", "search.py", "sheet.py", "segments.py",
                        "gravity.py", "frames.py", "scales.py", "ply.py",
                        "viz.py", "pipeline.py", "rerank.py", "chain.py",
                        "shots.py", "traj.py", "layout.py", "datasets.py",
                        "cases.py", "paths.py", "pairing.py", "visual.py"}

ALL_MODULES = sorted(p for p in ALIGN.rglob("*.py") if "eval" not in p.parts)
PRIORS = ALIGN / "priors.py"


def _docstring_nodes(tree):
    """Docstrings describe the boundary; only executable literals can cross it."""
    out = set()
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                          ast.AsyncFunctionDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _named(path, pattern=FORBIDDEN):
    """Non-docstring string literals in `path` matching `pattern`."""
    tree = ast.parse(Path(path).read_text())
    docs = _docstring_nodes(tree)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs and pattern.search(n.value)]


def _imports(path):
    tree = ast.parse(Path(path).read_text())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            out.append(n.module)
        elif isinstance(n, ast.Import):
            out += [a.name for a in n.names]
    return out


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda p: p.name)
def test_no_groundtruth_filenames(path):
    if path.name in MAY_NAME:
        pytest.skip("permitted to name ground truth")
    hits = _named(path)
    assert not hits, f"{path.name} references ground-truth files: {hits}"


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda p: p.name)
def test_no_groundtruth_import(path):
    if path.name in MAY_IMPORT:
        pytest.skip("permitted importer")
    bad = [m for m in _imports(path) if "groundtruth" in m]
    assert not bad, f"{path.name} imports ground truth: {bad}"


@pytest.mark.parametrize("name", sorted(SOLVING_MODULE_NAMES))
def test_no_solving_module_names_a_groundtruth_file(name):
    """A solving module may never NAME a ground-truth file.

    Separate from the scan above because this list is explicit: a module that
    disappears from it should fail rather than silently stop being checked.
    """
    matches = [p for p in ALIGN.rglob("*.py") if p.name == name]
    assert matches, f"{name} not found; update SOLVING_MODULE_NAMES"
    for path in matches:
        hits = _named(path)
        assert not hits, f"{name} references ground-truth files: {hits}"


def test_exactly_two_modules_name_the_transform():
    """Equality, not subset: a third reader of tm.json must fail loudly.

    `align.groundtruth` reads the current split's transform for scoring;
    `align.priors` reads a prior's as an input. There is no third role.
    """
    tm_only = re.compile(r"(?<![\w])tm\.json")
    readers = {p.name for p in ALIGN.rglob("*.py") if _named(p, tm_only)}
    assert readers == MAY_NAME, (
        f"expected exactly {sorted(MAY_NAME)} to name tm.json, got {sorted(readers)}")


def test_priors_names_the_registration_and_nothing_else():
    """The prior reader may read a prior's TRANSFORM, never anyone's poses.

    Its camera positions come from shots.geojson pushed through that transform,
    which is why images_full is named nowhere in the solver at all.
    """
    for pattern, label in [(re.compile(r"images_full"), "images_full"),
                           (re.compile(r"(?<![\w])images\.json"), "images.json")]:
        hits = _named(PRIORS, pattern)
        assert not hits, f"priors.py names {label}: {hits}"
    assert _named(PRIORS, re.compile(r"(?<![\w])tm\.json")), (
        "priors.py should name tm.json; if it no longer does, this file is stale")


def test_priors_cannot_reach_a_current_case():
    """The structural half of the role split.

    `align.priors` imports neither the evaluator nor the case index, so a
    current `Case` is not reachable from inside it and the guard cannot be
    handed the split it exists to refuse. `align.io.pairing` is the only bridge,
    and it is small enough to read in one sitting.
    """
    bad = [m for m in _imports(PRIORS)
           if "groundtruth" in m or m.endswith("io.cases") or m.endswith("pairing")]
    assert not bad, f"priors.py imports {bad}; the guard becomes bypassable"


def test_groundtruth_module_is_the_only_reader():
    readers = {p.name for p in ALIGN.rglob("*.py") if _named(p)}
    assert readers <= MAY_NAME, (
        f"unexpected ground-truth readers: {sorted(readers - MAY_NAME)}")
