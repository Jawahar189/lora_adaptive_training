"""Look every file for a split up by reality_id.

A convenience resolver for notebooks and one-off analysis: give it a reality id
and it tells you where everything for that split lives, across both dataset
generations and both directory layouts.

It no longer names the current split's ground truth. It used to carry `tm` and
`images_full` fields and was permitted to do so on the grounds that naming a
path is not reading it -- but nothing needed those fields that was not already
going through `align.groundtruth`, which resolves them itself from `rlt_dir`.
Dropping them takes this module out of the leakage allow-list entirely.

Quirks:
  * 9 of 322 rows leave `panovggt_point_cloud_path` blank, so `cloud` is
    optional and directories come from `tm_path`;
  * `geo` points at geo_first_point.json in the older generation and at geo.txt
    in the newer -- different things. See `align.io.cases`;
  * the older generation names `shots_path`/`images_prod_path` for files that
    are not on disk; the newer one has both.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from align.io import datasets as datasets_mod
from align.io.layout import detect_from_sheet

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CasePaths:
    reality_id: str
    capture_id: str
    project_id: str
    structure_id: str
    root: str                       # "data" | "data2" | "data_v"

    floorplan: Path                 # the sheet, PNG
    cloud: Path | None              # panovggt point cloud, PLY -- 9 rows have none
    shots: Path                     # per-image SfM poses, ply frame
    geo: Path | None
    design_tm: Path | None          # structure-level, None when not georeferenced
    images_dir: Path | None         # images_processed/

    structure_dir: Path
    capture_dir: Path | None        # absent in the flattened layout
    rlt_dir: Path

    paper_w: float                  # inches
    paper_h: float
    design_tm_status: str
    georeferenced: bool
    capture_date: str
    prior_capture_ids: tuple[str, ...]
    role: str
    csv: str

    def as_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self).items()}

    def missing(self) -> list[str]:
        """Named files that are not on disk."""
        return [k for k, v in asdict(self).items()
                if isinstance(v, Path) and not Path(v).exists()]


def _rows(csv_path: Path):
    with open(csv_path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {k.strip(): v for k, v in raw.items() if k is not None}
            if row.get("reality_id"):
                yield row


def _path(value) -> Path | None:
    """Blank means absent. Path("") is Path("."), which exists -- never test that."""
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
    return tuple(str(p).strip() for p in parsed if str(p).strip())


def _sides(spec):
    yield "current", spec.current
    yield "prior", spec.prior


def case_paths(reality_id: str, repo_root=None, dataset=None) -> CasePaths:
    """Every path for one split, keyed by reality_id.

        p = case_paths("RLT1788195772679")
        p.floorplan, p.cloud, p.shots, p.paper_w
    """
    repo_root = Path(repo_root or REPO_ROOT)
    spec = (datasets_mod.resolve(dataset) if dataset is not None
            else datasets_mod.from_env())
    if spec is None:
        raise ValueError("no dataset given: pass `dataset=`, or set "
                         f"${datasets_mod.ENV_CURRENT}")
    searched = []
    for role, names in _sides(spec):
        for name in names:
            path = repo_root / name
            searched.append(name)
            if not path.exists():
                continue
            for row in _rows(path):
                if row["reality_id"] != reality_id:
                    continue
                rlt_dir = Path(row["tm_path"]).parent
                layout = detect_from_sheet(rlt_dir, row["design_floor_plan_path"])
                dtm = _path(row.get("design_tm"))
                if dtm is None:
                    for cand in ("design_tm.json", "tm_local_to_global.json"):
                        if (layout.structure_dir / cand).exists():
                            dtm = layout.structure_dir / cand
                            break
                geo = _path(row.get("geo_path"))
                if geo is None and layout.capture_dir is not None:
                    geo = layout.capture_dir / "geo_first_point.json"
                return CasePaths(
                    reality_id=reality_id,
                    capture_id=row.get("capture_id", ""),
                    project_id=row.get("project_id", ""),
                    structure_id=row.get("structure_id", ""),
                    root=layout.root,
                    floorplan=Path(row["design_floor_plan_path"]),
                    cloud=_path(row.get("panovggt_point_cloud_path")),
                    shots=Path(row["shots_path"]),
                    geo=geo if (geo is not None and geo.exists()) else None,
                    design_tm=dtm if (dtm is not None and dtm.is_file()) else None,
                    images_dir=_path(row.get("images_path")),
                    structure_dir=layout.structure_dir,
                    capture_dir=layout.capture_dir,
                    rlt_dir=layout.rlt_dir,
                    paper_w=float(row["paper_width"]),
                    paper_h=float(row["paper_height"]),
                    design_tm_status=row.get("design_tm_status", ""),
                    georeferenced=dtm is not None and dtm.is_file(),
                    capture_date=(row.get("capture_date") or "").strip(),
                    prior_capture_ids=_prior_ids(row.get("prior_capture_id")),
                    role=role,
                    csv=name,
                )
    raise KeyError(f"{reality_id} is in none of {', '.join(searched)}")


def all_reality_ids(repo_root=None, dataset=None,
                    side: str = "current") -> list[str]:
    repo_root = Path(repo_root or REPO_ROOT)
    spec = (datasets_mod.resolve(dataset) if dataset is not None
            else datasets_mod.from_env())
    if spec is None:
        raise ValueError("no dataset given: pass `dataset=`, or set "
                         f"${datasets_mod.ENV_CURRENT}")
    names = spec.current if side == "current" else spec.prior
    seen: list[str] = []
    for name in names:
        p = repo_root / name
        if p.exists():
            seen += [r["reality_id"] for r in _rows(p) if r["reality_id"] not in seen]
    return seen
