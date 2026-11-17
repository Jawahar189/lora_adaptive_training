"""The IMU's metric scale of every current split, as the `imu_scale` column of the current CSV.

Three steps, each reusing what is already on disk:

  fetch     each split's IMU files, from S3 to beside the splits (`bevaligner/data/imu_download.py`)
  resolve   the resolve stage of imu_based_scale_resolution (external/, a submodule pinned to a
            release tag), over all the splits at once: per video it registers the splits together to
            time the frames against the IMU, then measures each split on its own cameras (a split it
            cannot register, alone): metres per reconstruction unit of each split's shots.geojson --
            the units its cloud and cameras are in
  write     that number as `imu_scale`, blank where the IMU gave none

`align.io.cases.Case.imu_scale` reads the column, and the aligner bands the cloud at the camera height
it implies (`gravity.metric_scale_views`). Production calls, once per job and after the current CSV
is written,

    add_imu_scale(current_csv, s3, api)       # a boto3 S3 client and a t3d_helpers Track3dAPI

and for a dataset `scripts/fetch_dataset.py` does the same, or by hand:

    python -m bevaligner.data.imu <current.csv> [--env third_party/.env] [--offline]

Nothing here fails a run: a split without an IMU scale, for whatever reason, is left blank and placed
on its floor fit as before. Each call logs how many splits got one.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

from .imu_download import IMU_FILES, clients, fetch, video_of, warn
from .imu_download import snapshot_realities  # noqa: F401 -- ml-jarvis bev/imu_scale.py calls imu.snapshot_realities

REPO_ROOT = Path(__file__).resolve().parents[2]
IMU_REPO = REPO_ROOT / "external" / "imu_based_scale_resolution"
COLUMN = "imu_scale"


def resolve_command(table: Path, out: Path) -> list[str]:
    """run.py's resolve command at the pinned tag -- its flags and defaults, so a new tag's change comes
    with it -- run by this interpreter instead of the .venv run.py expects inside the IMU repo."""
    spec = importlib.util.spec_from_file_location("imu_run", IMU_REPO / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    argv, sys.argv = sys.argv, ["run.py", str(table), out.name, "--stage", "resolve", "--out", str(out)]
    try:
        args = run.parse_arguments()
    finally:
        sys.argv = argv
    return [sys.executable, *run.stage_command("resolve", table, out, args)[1:]]


def resolve(rows: list[dict], imu_dirs: dict[str, Path], work: Path) -> dict[str, float]:
    """{reality_id: metres per reconstruction unit} from the IMU repo's resolve stage.

    The IMU repo reads one video per capture_id and only the folder of `Gyro_data`, so a job whose
    splits come from more than one video is given one capture per video, and a folder holding more
    than one video's IMU is left out -- the IMU repo would read whichever sorts first. Paths are
    handed over absolute: the IMU repo runs from its own src/.
    """
    work = Path(work).absolute()
    work.mkdir(parents=True, exist_ok=True)
    inputs = [(row, video_of(row["shots_path"])) for row in rows
              if row["reality_id"] in imu_dirs
              and len(list(imu_dirs[row["reality_id"]].glob(f"*_{IMU_FILES[0]}.csv"))) == 1]
    if not inputs:
        return {}
    videos: dict[str, set] = {}
    for row, video in inputs:
        videos.setdefault(row["capture_id"], set()).add(video)
    table = work / "imu_inputs.csv"
    with open(table, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reality_id", "capture_id", "shots_path", "Gyro_data"])
        for row, video in inputs:
            capture = row["capture_id"] if len(videos[row["capture_id"]]) == 1 else f"{row['capture_id']}__{video}"
            writer.writerow([row["reality_id"], capture, Path(row["shots_path"]).absolute(),
                             imu_dirs[row["reality_id"]].absolute() / f"{video}_{IMU_FILES[0]}.csv"])
    out, log = work / "resolve", work / "resolve.log"
    with open(log, "w") as handle:
        done = subprocess.run(resolve_command(table, out), cwd=IMU_REPO / "src",
                              stdout=handle, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(f"the IMU repo's resolve stage exited {done.returncode}, see {log}")
    scales = {}
    with open(out / "imu_scale_results.csv", newline="") as handle:
        for result in csv.DictReader(handle):
            try:
                scale = float(result.get(COLUMN) or "nan")
            except ValueError:
                continue
            if math.isfinite(scale) and scale > 0:
                scales[result["reality_id"]] = scale
    return scales


def read_rows(csv_path) -> tuple[list[str], list[dict]]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_column(csv_path, scales: dict[str, float], out=None) -> None:
    """The CSV with `imu_scale` set from `scales` (blank for the rest); every other column as it was."""
    columns, rows = read_rows(csv_path)
    if COLUMN not in columns:
        columns.append(COLUMN)
    for row in rows:
        row[COLUMN] = repr(scales[row["reality_id"]]) if row["reality_id"] in scales else ""
    target = Path(out or csv_path)
    partial = target.with_name(target.name + ".part")
    with open(partial, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, target)


def add_imu_scale(csv_path, s3=None, api=None, work=None, out=None) -> dict[str, float]:
    """fetch, resolve and write, over the splits of `csv_path`. Never raises: whatever goes wrong is a
    warning and a blank `imu_scale`, and a blank is the floor fit -- the IMU can only add to a run,
    never stop one."""
    rows, found, scales = [], {}, {}
    try:
        _, rows = read_rows(csv_path)
        found = fetch(rows, s3, api)
        scales = resolve(rows, found, Path(work) if work else REPO_ROOT / "cache" / "imu" / Path(csv_path).stem)
    except Exception as error:                                              # noqa: BLE001
        warn(f"no imu_scale: {type(error).__name__}: {error}")
    try:
        write_column(csv_path, scales, out)
    except Exception as error:                                              # noqa: BLE001
        warn(f"{csv_path} left as it was: {type(error).__name__}: {error}")
    print(f"[imu] imu_scale for {len(scales)} of {len(rows)} splits (IMU files for {len(found)}) "
          f"-> {out or csv_path}")
    return scales


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", help="current CSV: reality_id, capture_id, project_id, shots_path")
    parser.add_argument("--out", default=None, help="CSV to write (default: the input, in place)")
    parser.add_argument("--work", default=None,
                        help="the resolve stage's folder (default: cache/imu/<csv stem>)")
    parser.add_argument("--env", default=str(REPO_ROOT / "third_party" / ".env"),
                        help=".env with ROLE_ARN and the Track3D API login (default: %(default)s)")
    parser.add_argument("--offline", action="store_true", help="use only the IMU files already on disk")
    args = parser.parse_args(argv)

    s3 = api = None
    if not args.offline:
        try:
            s3, api = clients(args.env)
        except Exception as error:                                          # noqa: BLE001
            warn(f"no S3 or API, IMU files on disk only: {type(error).__name__}: {error}")
    add_imu_scale(args.csv, s3, api, args.work, args.out)


if __name__ == "__main__":
    main()
