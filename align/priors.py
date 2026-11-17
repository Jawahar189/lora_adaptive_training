"""The prior capture's registration -- a solver INPUT, and the guard that keeps it one.

A current split is aligned to the bird's-eye view of the capture that came
before it, a median of one day earlier. That prior capture is already
registered in production, so each of its splits carries a known transform into
the structure's target frame, and using it is not leakage: it is the same
information the production system has when a new capture arrives.

The current split's transform, sitting in a file of the same name, IS the
answer. Nothing distinguishes the two syntactically -- `rlt_dir / "tm.json"` is
the same nine characters either way -- so the boundary cannot be a filename
rule any more. It is a runtime one, and it lives here:

    * this module is the only one besides `align.groundtruth` that names
      `tm.json`, and the only one that may read a PRIOR's;
    * every read goes through `_check`, which refuses the current split, any
      SIBLING of the current capture, anything outside the declared pool, any
      path that resolves to the current split's own directory, and any split
      captured after the current one;
    * every path actually opened is recorded, so a caller can assert after the
      fact that nothing under the current split was touched.

This module imports neither `align.groundtruth` nor `align.io.cases`. That is
deliberate and load-bearing: a current `Case` is not reachable from in here, so
there is no accidental path by which the guard could be handed the very split
it exists to refuse. `align.io.pairing` is the only bridge between the two.

Positions come from `shots.geojson` and are pushed into the target frame with
the prior's transform, rather than read from the prior's target-frame pose file.
The two agree to 8e-10 m, so it costs nothing -- and it means `images_full` is
named nowhere in the solver at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import json
import numpy as np

from align.frames import apply as apply4
from align.frames import is_similarity, target_to_paper
from align.io.ply import read_ply
from align.io.shots import load_shots
from align.io.traj import Trajectory

TM_NAME = "tm.json"


class LeakageError(RuntimeError):
    """A prior-side read was asked for something it may not see."""


@dataclass(frozen=True)
class PriorSplit:
    """One split of an already-registered capture.

    No transform path: see the module docstring. The path is derived inside
    `PriorReference`, after the split has been checked.
    """
    reality_id: str
    capture_id: str
    structure_key: str
    capture_date: str
    rlt_dir: Path
    cloud_path: Path | None
    shots_path: Path


def _load_tm(path: Path) -> np.ndarray:
    raw = json.loads(Path(path).read_text())
    return np.asarray(raw["tm"], dtype=np.float64).reshape(4, 4)


class PriorReference:
    """The splits one current split is allowed to align against.

    Everything it returns is in PAPER INCHES, not the raw target frame. The
    target frame is UTM metres for some structures and paper inches for others,
    so a reference built "in the target frame" would make `COARSE_CELL_M = 0.20`
    mean two different things on one dataset -- and would do it silently. Paper
    is the frame the solver already works in, and `align.eval` composes back out
    of it unchanged.
    """

    def __init__(self, *, current_reality_id: str, current_capture_id: str,
                 current_rlt_dir: Path, current_capture_date: str,
                 structure_key: str, splits: Sequence[PriorSplit],
                 forbidden_reality_ids: frozenset[str] = frozenset(),
                 design_tm: np.ndarray | None = None,
                 metres_per_inch: float = float("nan")) -> None:
        self.current_reality_id = current_reality_id
        self.current_capture_id = current_capture_id
        self.current_rlt_dir = Path(current_rlt_dir).resolve()
        self.current_capture_date = current_capture_date
        self.structure_key = structure_key
        self.design_tm = design_tm
        self.metres_per_inch = metres_per_inch
        self._splits = {s.reality_id: s for s in splits}
        self._forbidden = frozenset(forbidden_reality_ids) | {current_reality_id}
        self._reads: list[Path] = []
        self._tm_cache: dict[str, np.ndarray] = {}
        # Trajectories are asked for more than once per reference -- once to cut
        # each split's wall band, again to check the splits agree with each
        # other -- and re-reading a file the guard has already cleared is pure
        # cost. Clouds are NOT cached: they are tens of megabytes each and are
        # wanted exactly once.
        self._traj_cache: dict[str, Trajectory] = {}

    # -- identity ----------------------------------------------------------
    @property
    def splits(self) -> tuple[PriorSplit, ...]:
        return tuple(self._splits.values())

    @property
    def reality_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._splits))

    @property
    def capture_ids(self) -> tuple[str, ...]:
        return tuple(sorted({s.capture_id for s in self._splits.values()}))

    @property
    def same_day(self) -> bool:
        """A prior captured the same day may be the same walk cut differently."""
        return any(s.capture_date == self.current_capture_date
                   for s in self._splits.values() if s.capture_date)

    @property
    def key(self) -> str:
        """Identifies the reference by its inputs, for caching."""
        import hashlib
        h = hashlib.sha1("|".join(self.reality_ids).encode()).hexdigest()[:16]
        return h

    def reads(self) -> tuple[Path, ...]:
        return tuple(self._reads)

    def assert_clean(self) -> None:
        """Nothing under the current split was opened through this object."""
        bad = [p for p in self._reads
               if self.current_rlt_dir in Path(p).resolve().parents
               or Path(p).resolve() == self.current_rlt_dir]
        if bad:
            raise LeakageError(
                f"{self.current_reality_id}: prior reads touched the current "
                f"split: {[str(p) for p in bad]}")

    # -- the guard ---------------------------------------------------------
    def _check(self, reality_id: str) -> PriorSplit:
        if reality_id in self._forbidden:
            why = ("is the current split" if reality_id == self.current_reality_id
                   else "is a sibling in the current capture")
            raise LeakageError(f"{reality_id} {why}; it may not be read as a prior")
        split = self._splits.get(reality_id)
        if split is None:
            raise LeakageError(
                f"{reality_id} is not in the prior pool of "
                f"{self.current_reality_id} ({len(self._splits)} splits)")
        if split.capture_id == self.current_capture_id:
            raise LeakageError(
                f"{reality_id} belongs to the current capture "
                f"{self.current_capture_id}")
        if Path(split.rlt_dir).resolve() == self.current_rlt_dir:
            raise LeakageError(
                f"{reality_id} resolves to the current split's own directory")
        if (split.capture_date and self.current_capture_date
                and split.capture_date > self.current_capture_date):
            raise LeakageError(
                f"{reality_id} was captured on {split.capture_date}, after "
                f"{self.current_capture_date}; a later capture is not a prior")
        return split

    def _record(self, path: Path) -> Path:
        self._reads.append(Path(path))
        return Path(path)

    # -- guarded reads -----------------------------------------------------
    def tm(self, reality_id: str) -> np.ndarray:
        """The prior split's registration, ply -> target frame."""
        if reality_id in self._tm_cache:
            return self._tm_cache[reality_id]
        split = self._check(reality_id)
        M = _load_tm(self._record(Path(split.rlt_dir) / TM_NAME))
        if not is_similarity(M, tol=1e-6):
            raise ValueError(f"{reality_id}: prior transform is not a similarity")
        self._tm_cache[reality_id] = M
        return M

    def trajectory(self, reality_id: str) -> Trajectory:
        """The prior split's cameras in PAPER INCHES."""
        if reality_id in self._traj_cache:
            return self._traj_cache[reality_id]
        split = self._check(reality_id)
        traj = load_shots(self._record(split.shots_path).parent)
        pos = target_to_paper(apply4(self.tm(reality_id), traj.positions),
                              self.design_tm)
        out = Trajectory(traj.names, traj.idx, pos, "paper")
        self._traj_cache[reality_id] = out
        return out

    def points(self, reality_id: str, stride: int = 6) -> np.ndarray:
        """The prior split's cloud in PAPER INCHES."""
        split = self._check(reality_id)
        if split.cloud_path is None:
            raise ValueError(f"{reality_id} names no point cloud")
        xyz = read_ply(self._record(split.cloud_path), stride=stride,
                       min_source_frames=2).xyz
        return target_to_paper(apply4(self.tm(reality_id), xyz), self.design_tm)

    def split_dir(self, reality_id: str) -> Path:
        """The prior split's directory, for its panoramas -- through the guard,
        like every other read of a prior, and recorded like one."""
        return self._record(Path(self._check(reality_id).rlt_dir))

    def with_clouds(self) -> tuple[PriorSplit, ...]:
        return tuple(s for s in self._splits.values() if s.cloud_path is not None)
