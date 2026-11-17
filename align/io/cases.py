"""CSV-driven index of structures and reality splits.

Only production inputs are exposed here. `Case` deliberately has no field that
reaches the CURRENT split's tm.json or images_full.json -- those live behind
`align.groundtruth`. A `PriorSplit` does carry a tm path, because a prior
capture is already registered and its transform is a legitimate input; it is
only ever dereferenced through `align.priors`, which checks it is not the
current split.

Dataset quirks handled:
  * the CSV generations have different column sets, and the older one has a
    leading-space column name " residue images, images_full";
  * 9 of 322 rows leave `panovggt_point_cloud_path` BLANK, so directories are
    resolved from `tm_path` (populated on every row) and a missing cloud is a
    property of the case rather than a crash;
  * the directory shape differs between generations -- see `align.io.layout`;
  * STR1782849228036 appears under BOTH roots, so structures stay keyed by
    (root, structure_id);
  * `design_tm_status` disagrees with the `design_tm` column on 3 rows, saying
    no_world_transform while naming a real file. The file wins; the CSV's claim
    is kept as `status_csv` so the disagreement stays visible;
  * a blank `design_tm` must be tested for TRUTHINESS, never with exists():
    Path("") is Path("."), which exists, and load_design_tm then dies on a
    directory;
  * geo_first_point.json exists only in the older generation. The newer one
    names geo.txt, which is a per-image GNSS track rather than one surveyed
    point -- it is NOT loaded here, pending a provenance decision.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from align.frames import SheetGeometry, load_design_tm
from align.io import datasets as datasets_mod
from align.io.layout import detect_from_sheet
from align.priors import PriorSplit

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GeoPoint:
    crs: str
    image: str
    x: float
    y: float
    z: float
    source_reality_id: str


@dataclass
class Structure:
    structure_id: str
    project_id: str
    root: str                      # "data" | "data2" | "data_v"
    dir: Path
    sheet_path: Path
    paper_w: float
    paper_h: float
    design_tm: np.ndarray | None
    status: str                    # "ok" | "no_world_transform"
    status_csv: str = ""           # what the CSV claimed, when it disagrees

    _geom: SheetGeometry | None = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return f"{self.root}/{self.structure_id}"

    @property
    def georeferenced(self) -> bool:
        return self.design_tm is not None

    def geometry(self) -> SheetGeometry:
        if self._geom is None:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(self.sheet_path) as im:
                w, h = im.size
            self._geom = SheetGeometry(self.paper_w, self.paper_h, w, h)
            self._geom.check()
        return self._geom


@dataclass
class Case:
    reality_id: str
    capture_id: str
    project_id: str
    structure: Structure
    rlt_dir: Path
    cloud_path: Path | None
    shots_path: Path
    geo: GeoPoint | None = None       # older generation only
    capture_date: str = ""
    prior_capture_ids: tuple[str, ...] = ()
    prior_capture_date: str = ""
    role: str = "current"
    parent: str = ""                  # prior side: the reality this one was split from
    imu_scale: float | None = None    # metres per cloud unit, from the IMU (bevaligner/data/imu.py)

    @property
    def key(self) -> str:
        return self.reality_id

    @property
    def has_cloud(self) -> bool:
        return self.cloud_path is not None


def _load_geo(cap_dir: Path | None, reality_id: str) -> GeoPoint | None:
    if cap_dir is None:
        return None
    p = cap_dir / "geo_first_point.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    if j.get("source_reality_id") != reality_id:
        return None            # anchors a different split; useless here
    return GeoPoint(
        crs=j.get("crs", ""), image=j["image"],
        x=float(j["x"]), y=float(j["y"]), z=float(j["z"]),
        source_reality_id=j["source_reality_id"],
    )


def _norm_row(row: dict) -> dict:
    return {k.strip(): v for k, v in row.items() if k is not None}


def _path(value) -> Path | None:
    """A column that is blank means ABSENT, not the working directory."""
    value = (value or "").strip()
    return Path(value) if value else None


def _prior_ids(value) -> tuple[str, ...]:
    value = (value or "").strip()
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return (value,)
    if not isinstance(parsed, list):
        parsed = [parsed]
    return tuple(str(p).strip() for p in parsed if str(p).strip())


def _structure_for(row: dict, layout, cache: dict) -> Structure:
    sid = row["structure_id"]
    key = f"{layout.root}/{sid}"
    if key in cache:
        return cache[key]
    # The column names the transform in both generations; fall back to probing
    # only for a row that leaves it blank but has one on disk anyway.
    dtm_path = _path(row.get("design_tm"))
    if dtm_path is None:
        for name in ("design_tm.json", "tm_local_to_global.json"):
            if (layout.structure_dir / name).exists():
                dtm_path = layout.structure_dir / name
                break
    design_tm = None
    if dtm_path is not None and dtm_path.is_file():
        design_tm = load_design_tm(dtm_path)
        if np.allclose(design_tm, np.eye(4)):
            # the identity written for a sheet with no georeference: no metres in it, and read as one
            # it would claim 1 m per paper inch -- the sheet's scale comes from its priors' rig instead
            design_tm = None
    claimed = (row.get("design_tm_status") or "").strip()
    actual = "ok" if design_tm is not None else "no_world_transform"
    cache[key] = Structure(
        structure_id=sid,
        project_id=row.get("project_id", ""),
        root=layout.root,
        dir=layout.structure_dir,
        sheet_path=Path(row["design_floor_plan_path"]),
        paper_w=float(row["paper_width"]),
        paper_h=float(row["paper_height"]),
        design_tm=design_tm,
        status=actual,
        status_csv=claimed if claimed != actual else "",
    )
    return cache[key]


def _rows(csv_path: Path):
    with open(csv_path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = _norm_row(raw)
            if row.get("reality_id"):
                yield row


def _layout_for(row: dict):
    rlt_dir = Path(row["tm_path"]).parent
    return detect_from_sheet(rlt_dir, row["design_floor_plan_path"])


class CaseIndex:
    def __init__(self, cases: list[Case]):
        self.cases = cases

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def get(self, reality_id: str) -> Case:
        for c in self.cases:
            if c.reality_id == reality_id:
                return c
        raise KeyError(reality_id)

    def roots(self) -> list[str]:
        return sorted({c.structure.root for c in self.cases})

    @classmethod
    def load(cls, csv_path, repo_root: Path | None = None,
             role: str = "current") -> "CaseIndex":
        repo_root = Path(repo_root or REPO_ROOT)
        csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            csv_path = repo_root / csv_path
        structures: dict[str, Structure] = {}
        cases: list[Case] = []
        for row in _rows(csv_path):
            layout = _layout_for(row)
            cases.append(Case(
                reality_id=row["reality_id"],
                capture_id=row.get("capture_id", ""),
                project_id=row.get("project_id", ""),
                structure=_structure_for(row, layout, structures),
                rlt_dir=layout.rlt_dir,
                cloud_path=_path(row.get("panovggt_point_cloud_path")),
                shots_path=Path(row["shots_path"]),
                geo=_load_geo(layout.capture_dir, row["reality_id"]),
                capture_date=(row.get("capture_date") or "").strip(),
                prior_capture_ids=_prior_ids(row.get("prior_capture_id")),
                prior_capture_date=(row.get("prior_capture_date") or "").strip(),
                role=role,
                parent=(row.get("parent_reality_id") or "").strip(),
                imu_scale=float(row["imu_scale"]) if (row.get("imu_scale") or "").strip() else None,
            ))
        return cls(cases)

    @classmethod
    def load_dataset(cls, spec, repo_root=None,
                     side: str = "current") -> "CaseIndex":
        spec = datasets_mod.resolve(spec)
        seen, merged = set(), []
        for path in spec.paths(side, repo_root):
            if not path.exists():
                continue
            for c in cls.load(path, repo_root, role=side):
                if c.reality_id not in seen:
                    seen.add(c.reality_id)
                    merged.append(c)
        return cls(merged)

    @classmethod
    def load_all(cls, repo_root: Path | None = None, spec=None) -> "CaseIndex":
        """Every current-side split of a dataset the CALLER names.

        There is no default. A library that remembers a path keeps resolving it
        after the data moves and answers about the wrong dataset instead of
        failing, so the paths arrive from a flag or from the environment.
        """
        spec = spec or datasets_mod.from_env()
        if spec is None:
            raise ValueError(
                "no dataset given: pass --current-csv, or set "
                f"${datasets_mod.ENV_CURRENT}")
        return cls.load_dataset(spec, repo_root, side="current")

    def by_capture(self) -> dict[str, list[Case]]:
        out: dict[str, list[Case]] = {}
        for c in self.cases:
            out.setdefault(c.capture_id, []).append(c)
        return out

    def by_structure(self) -> dict[str, list[Case]]:
        out: dict[str, list[Case]] = {}
        for c in self.cases:
            out.setdefault(c.structure.key, []).append(c)
        return out


class PairedIndex:
    """A current-side index, a prior-side index, and the edges between them.

    Two indexes rather than a `priors` field on `Case`: 49 reality ids appear on
    BOTH sides -- a split is a current row for its own capture and a prior for a
    later one -- so a case holding its own priors is a cycle. "Is this reality a
    prior?" is not a well-formed question; only "is it a prior OF THIS CASE?" is,
    and that is what `priors_for` answers.
    """

    def __init__(self, current: CaseIndex, prior: CaseIndex, name: str = ""):
        self.current = current
        self.prior = prior
        self.name = name
        self._prior_by_capture = prior.by_capture()
        self._current_by_capture = current.by_capture()

    @classmethod
    def load(cls, spec=None, repo_root=None) -> "PairedIndex":
        spec = datasets_mod.resolve(spec) if spec is not None \
            else datasets_mod.from_env()
        if spec is None:
            raise ValueError(
                "no dataset given: pass --current-csv/--prior-csv, or set "
                f"${datasets_mod.ENV_CURRENT} and ${datasets_mod.ENV_PRIOR}")
        current = CaseIndex.load_dataset(spec, repo_root, side="current")
        prior = (CaseIndex.load_dataset(spec, repo_root, side="prior")
                 if spec.paired else CaseIndex([]))
        return cls(current, prior, spec.name)

    def __len__(self) -> int:
        return len(self.current)

    def __iter__(self):
        return iter(self.current)

    def priors_for(self, case: Case) -> list[PriorSplit]:
        """The prior splits this case is allowed to see, cloud-bearing only."""
        out = []
        for cap in case.prior_capture_ids:
            for p in self._prior_by_capture.get(cap, []):
                if p.cloud_path is None:
                    continue
                out.append(PriorSplit(
                    reality_id=p.reality_id, capture_id=p.capture_id,
                    structure_key=p.structure.key, capture_date=p.capture_date,
                    rlt_dir=p.rlt_dir, cloud_path=p.cloud_path,
                    shots_path=p.shots_path))
        return out

    def siblings_of(self, case: Case) -> list[str]:
        """Every reality in the case's OWN capture, current side and prior side.

        A sibling's registration is as much the answer as the case's own, so the
        prior guard has to refuse all of them, not just the case itself.
        """
        ids = {c.reality_id for c in self._current_by_capture.get(case.capture_id, [])}
        ids |= {c.reality_id for c in self._prior_by_capture.get(case.capture_id, [])}
        return sorted(ids)

    def pairs(self):
        for case in self.current:
            yield case, self.priors_for(case)

    def coverage(self) -> dict:
        n_with = sum(1 for _, p in self.pairs() if p)
        return dict(current=len(self.current), prior=len(self.prior),
                    with_priors=n_with, without=len(self.current) - n_with)
