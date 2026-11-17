"""Which CSVs make up a dataset. The paths come from the caller, never from here.

The index used to hardcode two filenames in two places, then five named sets in
one. Both are the same mistake at different scales: a path baked into the
library is a path that goes stale silently, and a default that still resolves
after the data has moved produces results for the wrong dataset rather than an
error.

So nothing here knows any path. A dataset is a current-side index and,
optionally, the prior-side index it references; both are supplied. Every script
takes the same three flags through `add_arguments`, so a pair of CSVs can be
handed to any stage without remembering which one spells it differently.

The environment variables ALIGN_CURRENT_CSV and ALIGN_PRIOR_CSV stand in where
a path cannot be passed as an argument -- the test suite, mostly, which skips
rather than guesses when they are unset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CURRENT = "ALIGN_CURRENT_CSV"
ENV_PRIOR = "ALIGN_PRIOR_CSV"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    current: tuple[str, ...]
    prior: tuple[str, ...] = ()

    @property
    def paired(self) -> bool:
        return bool(self.prior)

    def paths(self, side: str, repo_root=None) -> list[Path]:
        root = Path(repo_root or REPO_ROOT)
        names = self.current if side == "current" else self.prior
        return [p if (p := Path(n)).is_absolute() else root / n for n in names]


def _as_paths(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        out: tuple[str, ...] = ()
        for v in value:
            out += _as_paths(v)
        return out
    text = str(value).strip()
    if not text:
        return ()
    # A single variable may carry several paths, comma- or colon-separated.
    parts = [t for chunk in text.split(",") for t in chunk.split(os.pathsep)]
    return tuple(t.strip() for t in parts if t.strip())


def resolve(current, prior=None, name: str | None = None) -> DatasetSpec:
    """A dataset from explicit CSV paths."""
    if isinstance(current, DatasetSpec):
        return current
    cur = _as_paths(current)
    if not cur:
        raise ValueError("no current-side CSV given; pass --current-csv "
                         f"or set {ENV_CURRENT}")
    if name is None:
        name = Path(cur[0]).stem if len(cur) == 1 else "custom"
    return DatasetSpec(name, cur, _as_paths(prior))


def from_env(require_prior: bool = False) -> DatasetSpec | None:
    """The dataset named by the environment, or None when it is not set."""
    current = _as_paths(os.environ.get(ENV_CURRENT))
    if not current:
        return None
    prior = _as_paths(os.environ.get(ENV_PRIOR))
    if require_prior and not prior:
        return None
    return resolve(current, prior)


def from_args(args) -> DatasetSpec:
    """The dataset a script was asked for: its flags, else the environment."""
    current = getattr(args, "current_csv", None) or os.environ.get(ENV_CURRENT)
    prior = getattr(args, "prior_csv", None) or os.environ.get(ENV_PRIOR)
    return resolve(current, prior)


def add_arguments(ap, prior: bool = True, required: bool = True) -> None:
    """The dataset flags, identical across every script.

    Required unless the environment supplies them, so a run always states which
    data it ran on -- in the command, or in the environment it inherited, but
    never by falling back to something a library remembered.
    """
    has_env = bool(os.environ.get(ENV_CURRENT))
    ap.add_argument("--current-csv", nargs="+", metavar="CSV",
                    required=required and not has_env,
                    default=None,
                    help=f"current-side index; defaults to ${ENV_CURRENT}")
    if prior:
        ap.add_argument("--prior-csv", nargs="+", metavar="CSV", default=None,
                        help="prior-side index, the captures to align against; "
                             f"defaults to ${ENV_PRIOR}")
