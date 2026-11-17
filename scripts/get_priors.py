"""The prior captures for a set of current captures: select, fetch what is missing, write.

    python scripts/get_priors.py --current datasets/x/current.csv --prior datasets/x/prior.csv \\
        [--mode multi] [--look-back-days 20] [--data-root data_priors]

Reads `--current` (one row per current split: project_id, structure_id, capture_id,
capture_date, reality_id and the design columns) and leaves the two CSVs
`python -m bevaligner.infer` reads:

    --current   the same rows, `prior_capture_id` / `prior_capture_date` filled in
                (rewritten in place; a rerun just rewrites them)
    --prior     one row per prior split, its files on local disk

Selection, once per current snapshot (`--mode`):

  single  the previous snapshot, by production's rule:
          `Track3dAPI.get_previous_reality_s3_path`.
  multi   `t3d_helpers`' FPCoverageRetriever over the `--look-back-days` before the
          capture (end_date = capture date - 1): the fewest realities, newest first,
          that cover the floor. A `boundary.json` smaller than `--min-boundary-m2` is
          a placeholder, so the whole sheet is the boundary instead. The previous
          snapshot is added too (unless --no-previous): measured on the 1,118 test
          captures, the selection alone drops it outright for 93 and keeps only some
          of its splits for 853.

Every prior split of every selected date is written; `bevaligner.infer` merges the splits
a row names into one pool, by default as a mosaic (each 5 m cell keeps the walls of the newest
date that has them; `--no-prior-mosaic` stacks every split instead). Each prior row carries
`parent_reality_id`, the reality it was split from (the API's `parent`): such a split keeps its
parent's whole cloud, so the pool cuts it back to its own walls (`prior_bev.cut_split_priors`).

A prior reality is used only when the API has it active and reconstructed by one of
`--pipeline-versions`. A complete copy under `--search-roots` or `--data-root`
(`<root>/<PRJ>__<STR>/<RLT>/` with point_cloud.ply, shots.geojson and tm.json) is reused;
otherwise those three are downloaded -- tm.json from the reality's `pointcloud/`, where
production reads it -- and `images_full.json` written. A reality without them on S3 is
skipped. Credentials: `--env` (the Track3D API login and ROLE_ARN; S3 is reached by
assuming that role with the machine's own identity, no AWS keys are read).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def default_workers() -> int:
    """Threads for the network-bound work: every core but two, and at least one.
    Not a fixed number: machines differ, and two cores stay free for everything else."""
    return max(1, (os.cpu_count() or 4) - 2)
BUCKET = "constructn-projects-us"
S3_PATH = re.compile(r"(PRJ\w+)/structures/(STR\w+)/snapshots/(SNP\w+)/reality/(RLT\d+)")


@dataclass(frozen=True)
class PriorRef:
    project_id: str
    structure_id: str
    snapshot_id: str
    reality_id: str
    date: str                       # the snapshot's date, YYYY-MM-DD
    parent: str = ""                # the reality this one was split from (the API's `parent`)

    @property
    def s3_prefix(self) -> str:
        return (f"{self.project_id}/structures/{self.structure_id}/snapshots/"
                f"{self.snapshot_id}/reality/{self.reality_id}")


# ----------------------------------------------------------------------------
# Track3D API and S3

def connect(env: str):
    """`(Track3dAPI, S3_AWS_DATA)` from the `.env` at `env` (repo-relative or absolute).

    Only ROLE_ARN is exported: the S3 client assumes that role from the machine's
    identity, and a stale AWS_* session copied into the environment would win.
    """
    from dotenv import dotenv_values
    from t3d_helpers.access.api_handler import Track3dAPI
    from t3d_helpers.access.data_aws import S3_AWS_DATA
    env_path = REPO / env
    role_arn = dotenv_values(env_path).get("ROLE_ARN")
    if role_arn:
        os.environ["ROLE_ARN"] = role_arn        # t3d_helpers' optimizer reads it here
    return Track3dAPI(env_path=str(env_path)), S3_AWS_DATA(role_arn=role_arn)


class StructureSnapshots:
    """One structure's snapshots from the API: dates, and each reality's record."""

    def __init__(self, api, project_id: str, structure_id: str):
        url = (f"{api.base_url}{api.api_all_snapshot_version}"
               f"/projects/{project_id}/structures/{structure_id}/snapshots")
        response = api._request_with_auth_retry(
            "GET", url=url, request_name="get_all_snapshots",
            headers={"authorization": f"Bearer {api.auth_token}", "Connection": "close"})
        response.raise_for_status()
        result = response.json().get("result") or {}
        self.date, self.snapshot_of, self.record = {}, {}, {}
        for snap in result.get("mSnapshots", []):
            if snap.get("isDeleted") or not snap.get("date"):
                continue
            self.date[snap["_id"]] = str(snap["date"])[:10]
            for rec in snap.get("reality", []):
                self.snapshot_of[rec["_id"]] = snap["_id"]
                self.record[rec["_id"]] = rec

    def accepted(self, reality_id: str, pipeline_versions: list[str]) -> bool:
        """Active, not deleted, and reconstructed by one of `pipeline_versions`."""
        rec = self.record.get(reality_id) or {}
        version = str(rec.get("pipelineVersion") or "").lower()
        return (rec.get("status") == "active" and not rec.get("isDeleted")
                and version in {v.lower() for v in pipeline_versions})

    def ref(self, project_id: str, structure_id: str, snapshot_id: str, reality_id: str,
            date: str) -> PriorRef:
        parent = str((self.record.get(reality_id) or {}).get("parent") or "")
        return PriorRef(project_id, structure_id, snapshot_id, reality_id, date=date, parent=parent)


def download_batch(s3, local_root: Path, remote_root: str, files: list[tuple[str, str]],
                   workers: int) -> None:
    """Many files in ONE t3d_helpers `download_files_threaded` call.

    Production's own download function, given the whole list instead of one file per call:
    it builds its client once and spreads the files over `workers` threads (88 priors, 264
    files, 2.9 GB: 12 s; one file per call: ~4-5 min). `files` are (path under `remote_root`,
    path under `local_root`); a non-empty file already on disk is not fetched again, and a
    missing one is reported by t3d_helpers and simply absent afterwards.
    """
    todo = [(remote, local) for remote, local in files
            if not ((local_root / local).is_file() and (local_root / local).stat().st_size > 0)]
    if not todo:
        return
    for _, local in todo:                       # it writes into folders, it does not make them
        (local_root / local).parent.mkdir(parents=True, exist_ok=True)
    s3.download_files_threaded(local_path=str(local_root), remote_path=remote_root,
                               file_lis=[remote for remote, _ in todo],
                               name_lis=[local for _, local in todo],
                               num_threads=max(1, workers))


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


# ----------------------------------------------------------------------------
# selection

def previous_snapshot(api, snapshots: StructureSnapshots, current: dict, args) -> list[PriorRef]:
    """The previous snapshot's accepted realities, by production's own call."""
    source = (f"s3://{BUCKET}/{current['project_id']}/structures/{current['structure_id']}/"
              f"snapshots/{current['snapshot_id']}/reality/{current['reality_id']}")
    refs = []
    for path in api.get_previous_reality_s3_path(source, verbose=False) or []:
        match = S3_PATH.search(path)
        if match and snapshots.accepted(match.group(4), args.pipeline_versions):
            refs.append(snapshots.ref(*match.groups(), date=snapshots.date.get(match.group(3), "")))
    return refs


def _tolerate_verbose_kwarg() -> None:
    """t3d_helpers 2.0.9's optimizer passes `verbose=` to S3 helpers that take none."""
    from t3d_helpers.access.data_aws import S3_AWS_DATA

    def without_verbose(method):
        def call(self, *a, verbose=None, **kw):
            return method(self, *a, **kw)
        call.drops_verbose = True
        return call

    for name in ("download_file", "download_folder_threaded"):
        method = getattr(S3_AWS_DATA, name)
        if not getattr(method, "drops_verbose", False):
            setattr(S3_AWS_DATA, name, without_verbose(method))


def _use_sheet_when_boundary_is_placeholder(retriever, min_area_m2: float) -> None:
    """`boundary.json` when it is a real footprint, else the whole sheet in world coords."""
    from shapely.geometry import Polygon
    path = retriever.bounds_json_path
    if path is not None and Path(path).exists():
        points = json.loads(Path(path).read_text()).get("boundaryPoints", [])
        if len(points) >= 3 and Polygon([(p["x"], p["y"]) for p in points]).area >= min_area_m2:
            return
    w, h = retriever._paper_width, retriever._paper_height
    corners = np.array([[0, 0, 0, 1], [w, 0, 0, 1], [w, h, 0, 1], [0, h, 0, 1]], float)
    world = corners @ retriever._design_tm.T
    sheet = retriever._downloads_root_dir / "sheet_bounds.json"
    sheet.write_text(json.dumps({"boundaryPoints": [dict(x=float(x), y=float(y), z=float(z), w=1)
                                                    for x, y, z, _ in world]}))
    retriever.bounds_json_path = sheet


def _seed_design(api, current: dict, output_dir: Path) -> None:
    """The floor map we already have, where the optimizer looks for it. Without it the
    optimizer downloads the design's whole folder (thousands of tiles) to read that file."""
    sheet = Path(current.get("design_floor_plan_path") or "")
    if not sheet.is_file():
        return
    design_id = api.get_design_id(project_id=current["project_id"],
                                  structure_id=current["structure_id"], verbose=False)
    target = output_dir / "data" / "design_assets" / str(design_id) / "floormap" / "floormap.png"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with suppress(FileExistsError):
            target.symlink_to(sheet.resolve())


def _prefetch_images_json(retriever, workers: int) -> tuple[int, int]:
    """Every candidate's images.json, in one production download call, where the optimizer
    reads it from.

    The optimizer fetches them one at a time as it maps each date (~0.6 s apiece, ten
    minutes on a busy site); they are small and an old reality's never changes, so they are
    fetched together and kept. Returns (on disk, missing on S3).
    """
    root = retriever._reality_images_cache_dir                 # <...>/snapshots/
    wanted = [f"{snapshot_id}/reality/{reality_id}/images.json"
              for date in retriever._sorted_dates
              for snapshot_id, reality_ids in retriever._date_to_snapshot_realities[date]
              for reality_id in reality_ids]
    download_batch(retriever._s3, root,
                   f"{retriever.project_id}/structures/{retriever.structure_id}/snapshots",
                   [(path, path) for path in wanted], workers)
    present = sum(nonempty(root / path) for path in wanted)
    return present, len(wanted) - present


def coverage_selection(api, current: dict, capture_date: str, args,
                       snapshots: StructureSnapshots) -> list[PriorRef]:
    """FPCoverageRetriever's realities over the look-back window, accepted ones only."""
    from t3d_helpers.common.fp_coverage_optimizer import FPCoverageRetriever
    end_date = datetime.strptime(capture_date, "%Y-%m-%d") - timedelta(days=1)
    output_dir = REPO / args.data_root / "_coverage" / current["structure_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_design(api, current, output_dir)
    retriever = FPCoverageRetriever(
        current["structure_id"], end_date=end_date.strftime("%Y-%m-%d"),
        project_id=current["project_id"], env_path=str(REPO / args.env),
        n_dates_to_look_back=args.look_back_days, output_dir=output_dir)
    retriever.setup()
    if not retriever._pipeline_ready:
        return []
    t0 = time.time()
    fetched, missing = _prefetch_images_json(retriever, args.prefetch_workers)
    print(f"    {current['structure_id']}: {fetched} candidate images.json fetched in "
          f"{time.time() - t0:.0f} s ({missing} missing on S3)", flush=True)
    _use_sheet_when_boundary_is_placeholder(retriever, args.min_boundary_m2)
    retriever._generate_grids()
    try:
        selected = retriever._run_optimizer()["selected_realities"]
    except RuntimeError as error:              # no candidate passed the optimizer's thresholds
        print(f"    coverage selection: {error}")
        return []
    return [snapshots.ref(current["project_id"], current["structure_id"], r["snapshot_id"],
                          r["reality_id"], date=str(r["date"])[:10])
            for r in selected if snapshots.accepted(r["reality_id"], args.pipeline_versions)]


def select_snapshot(api, args, snapshots: StructureSnapshots, snapshot_id: str,
                    row: dict) -> list[PriorRef]:
    """One current snapshot's prior realities, the previous snapshot's first, no repeats."""
    current = dict(row, snapshot_id=snapshot_id)
    refs = []
    if args.mode == "single" or args.previous:
        refs += previous_snapshot(api, snapshots, current, args)
    if args.mode == "multi":
        date = snapshots.date.get(snapshot_id) or row["capture_date"][:10]
        refs += coverage_selection(api, current, date, args, snapshots)
    return list({ref.reality_id: ref for ref in reversed(refs)}.values())[::-1]


def _select_structure(job) -> dict:
    """One structure's snapshots, in turn, with this process's own API client."""
    args, snapshots, keyed_rows = job
    api, _ = connect(args.env)
    found = {}
    for key, row in keyed_rows:
        try:
            found[key] = select_snapshot(api, args, snapshots, key[2], row)
        except Exception as error:                                  # noqa: BLE001
            print(f"    {key[1]} {key[2]}: selection failed: {type(error).__name__}: {error}")
            found[key] = []
    return found


def select(api, args, currents: list[dict]):
    """(snapshot key of each current row, prior refs per snapshot key).

    The work is network-bound, so structures run `--workers` at a time, each in its
    own process: t3d_helpers' clients serialize on their own locks, so threads
    sharing one wait on each other. A structure's snapshots stay on one worker, in
    turn, sharing the optimizer's `<data_root>/_coverage/<structure>/`.
    """
    structures: dict[tuple, StructureSnapshots] = {}
    key_of, first_row = {}, {}
    for row in currents:
        structure = (row["project_id"], row["structure_id"])
        if structure not in structures:
            structures[structure] = StructureSnapshots(api, *structure)
        snapshot = structures[structure].snapshot_of.get(row["reality_id"]) or row["capture_id"]
        key_of[row["reality_id"]] = (*structure, snapshot)
        first_row.setdefault((*structure, snapshot), row)
    if args.mode == "multi":
        _tolerate_verbose_kwarg()               # once, before the workers fork

    keys_of = defaultdict(list)
    for key in first_row:
        keys_of[key[:2]].append(key)
    jobs = [(args, structures[structure], [(key, first_row[key]) for key in keys])
            for structure, keys in keys_of.items()]
    refs_of: dict[tuple, list[PriorRef]] = {}
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        for found in pool.map(_select_structure, jobs):
            for key, refs in found.items():
                refs_of[key] = refs
                print(f"[{len(refs_of)}/{len(first_row)}] {key[1]} {key[2]}: {len(refs)} prior "
                      f"realities from {len({r.snapshot_id for r in refs})} snapshot(s)", flush=True)
    return key_of, refs_of


# ----------------------------------------------------------------------------
# local copies

def structure_dir(root: Path, project_id: str, structure_id: str) -> Path:
    """`<root>/<PRJ>__<STR>`, the layout `align.io.cases` reads."""
    return root / f"{project_id}__{structure_id}"


def write_images_full(rlt_dir: Path) -> Path:
    """Every photograph in the target frame, from the split's shots and tm.json --
    `t3d_helpers`' `convert_shots`, as production writes it."""
    from t3d_helpers.sfm.shots2images import convert_shots
    path = rlt_dir / "images_full.json"
    if not path.is_file():
        convert_shots(str(rlt_dir / "shots.geojson"), str(rlt_dir / "tm.json"), str(path))
    return path


def find_local(ref: PriorRef, roots: list[Path]) -> Path | None:
    """A complete copy of the reality under one of `roots`, beside its sheet."""
    for root in roots:
        rlt_dir = structure_dir(root, ref.project_id, ref.structure_id) / ref.reality_id
        if all((rlt_dir / name).is_file() for name in ("point_cloud.ply", "shots.geojson", "tm.json")) \
                and (rlt_dir.parent / "floormap.png").is_file():
            return rlt_dir
    return None


# (path under the reality on S3, local name)
PRIOR_FILES = (("point_cloud.ply", "point_cloud.ply"), ("shots.geojson", "shots.geojson"),
               ("pointcloud/tm.json", "tm.json"))


def link_sheet(rlt_dir: Path, sheet_dir: Path) -> None:
    """The loader wants the split under its sheet: floormap.png and the design tm beside it."""
    rlt_dir.mkdir(parents=True, exist_ok=True)
    for name in ("floormap.png", "tm_local_to_global.json"):
        link, source = rlt_dir.parent / name, sheet_dir / name
        if source.is_file() and not link.exists():
            with suppress(FileExistsError):
                link.symlink_to(source.resolve())


def prior_row(ref: PriorRef, rlt_dir: Path, current: dict, fields: list[str]) -> dict:
    """The prior split as a dataset row: its own files, its structure's design."""
    row = dict.fromkeys(fields, "")
    row.update(reality_id=ref.reality_id, capture_id=ref.snapshot_id, capture_date=ref.date,
               project_id=ref.project_id, structure_id=ref.structure_id,
               panovggt_point_cloud_path=str(rlt_dir / "point_cloud.ply"),
               shots_path=str(rlt_dir / "shots.geojson"), tm_path=str(rlt_dir / "tm.json"),
               images_full_path=str(write_images_full(rlt_dir)),
               design_floor_plan_path=str(rlt_dir.parent / "floormap.png"),
               design_tm=current.get("design_tm", ""),
               paper_width=current.get("paper_width", ""), paper_height=current.get("paper_height", ""),
               parent_reality_id=ref.parent)
    return row


def fetch(refs_of: dict, currents: list[dict], args, s3, fields: list[str]):
    """One prior row per reality on local disk, and the reality ids that could not be.

    A complete copy under a search root is reused; every other prior's three files come in
    one production download call per structure. A reality without all three is skipped (many
    keep only the viewer's converted copy of the cloud).
    """
    data_root = REPO / args.data_root
    roots = [Path(root) for root in args.search_roots] + [data_root]
    design_of = {(r["project_id"], r["structure_id"]): r for r in currents}
    unique = list({ref.reality_id: ref for refs in refs_of.values() for ref in refs}.values())
    found = {ref.reality_id: find_local(ref, roots) for ref in unique}

    by_structure: dict[tuple, list] = defaultdict(list)
    for ref in unique:
        if found[ref.reality_id] is None:
            by_structure[(ref.project_id, ref.structure_id)].append(ref)
    for (project_id, structure_id), group in by_structure.items():
        local_root = structure_dir(data_root, project_id, structure_id)
        sheet_dir = Path(design_of[(project_id, structure_id)]["design_floor_plan_path"]).parent
        for ref in group:
            link_sheet(local_root / ref.reality_id, sheet_dir)
        download_batch(s3, local_root, f"{project_id}/structures/{structure_id}/snapshots",
                       [(f"{ref.snapshot_id}/reality/{ref.reality_id}/{remote}", f"{ref.reality_id}/{name}")
                        for ref in group for remote, name in PRIOR_FILES], args.workers)
        for ref in group:
            rlt_dir = local_root / ref.reality_id
            if all(nonempty(rlt_dir / name) for _, name in PRIOR_FILES):
                found[ref.reality_id] = rlt_dir

    rows, skipped = {}, []
    for ref in unique:
        rlt_dir = found[ref.reality_id]
        try:
            if rlt_dir is None:
                raise FileNotFoundError("no point_cloud.ply, shots.geojson and tm.json on S3")
            rows[ref.reality_id] = prior_row(ref, rlt_dir, design_of[(ref.project_id, ref.structure_id)], fields)
        except Exception as error:                                  # noqa: BLE001
            if rlt_dir is not None:
                print(f"    {ref.reality_id}: {type(error).__name__}: {error}")
            skipped.append(ref.reality_id)
    return rows, skipped


# ----------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", required=True, help="the current CSV; priors are filled in")
    parser.add_argument("--prior", required=True, help="the prior CSV to write")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi",
                        help="single: the previous snapshot; multi: + coverage-selected dates")
    parser.add_argument("--no-previous", dest="previous", action="store_false",
                        help="multi: do not add the previous snapshot")
    parser.add_argument("--look-back-days", type=int, default=20)
    parser.add_argument("--min-boundary-m2", type=float, default=100.0)
    parser.add_argument("--pipeline-versions", nargs="+", default=["v9", "v10"], metavar="VERSION",
                        help="keep only prior realities reconstructed by these pipelines, e.g. "
                             "'v10' or 'v9 v10' (default: v9 v10, as production)")
    parser.add_argument("--data-root", default="data_priors",
                        help="where downloaded prior realities go: <root>/<PRJ>__<STR>/<RLT>/")
    parser.add_argument("--search-roots", nargs="*", default=[],
                        help="data roots searched first; a complete reality there is reused")
    parser.add_argument("--env", default="third_party/.env", help="Track3D API login and ROLE_ARN")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="download threads, and structures selected at once (default: CPU cores - 2)")
    parser.add_argument("--prefetch-workers", type=int, default=default_workers(),
                        help="multi: candidates' images.json fetched at once before the optimizer "
                             "runs (default: CPU cores - 2)")
    args = parser.parse_args(argv)

    current_csv, prior_csv = Path(args.current), Path(args.prior)
    with open(current_csv) as handle:
        reader = csv.DictReader(handle)
        fields, currents = list(reader.fieldnames), list(reader)
    api, s3 = connect(args.env)
    t0 = time.time()
    key_of, refs_of = select(api, args, currents)
    t1 = time.time()
    priors, skipped = fetch(refs_of, currents, args, s3, fields)
    t2 = time.time()

    for row in currents:
        refs = [r for r in refs_of[key_of[row["reality_id"]]] if r.reality_id in priors]
        row["prior_capture_id"] = json.dumps(sorted({r.snapshot_id for r in refs}))
        row["prior_capture_date"] = json.dumps(sorted({r.date for r in refs}))
    prior_fields = list(dict.fromkeys(fields + ["parent_reality_id"]))
    for path, rows, columns in ((prior_csv, list(priors.values()), prior_fields),
                                (current_csv, currents, fields)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(priors)} prior realities -> {prior_csv}  "
          f"({len(skipped)} without complete files on S3, skipped)")
    print(f"{len(currents)} current rows, priors filled in -> {current_csv}")
    print(f"timing: selection {t1 - t0:.0f} s, download {t2 - t1:.0f} s")


if __name__ == "__main__":
    main()
