"""The IMU files of every current split, found through the API and downloaded from S3.

A split carries no IMU of its own. Its photographs are frames of one 360 video and their names say
which one (VID_20261022_135751_00_298_IMG_00286.JPG is frame 286 of VID_20261022_135751_00_298). That
video's IMU is stored with the reality it was uploaded and stitched as -- the job's reality with
`isStitching`, in the split's snapshot -- under insv_metadata/:

    <video>_GyroData.csv                  gyro and accelerometer, ~1 kHz
    <video>_timelapseTimestampData.csv    when each frame was taken

`fetch` puts those two files in <PRJ>__<STR>/<stitched RLT>/insv_metadata/ beside the splits, the layout
of the rest of the current data, and reuses them when they are already there. `bevaligner/data/imu.py`
hands them to the IMU repo.
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "constructn-projects-us"
IMU_FILES = ("GyroData", "timelapseTimestampData")      # all the resolve stage reads
_FRAME = re.compile(r"^(.+)_IMG_\d+\.\w+$")


def warn(message: str) -> None:
    print(f"[imu] WARNING {message}", file=sys.stderr)


def video_of(shots_path) -> str | None:
    """The video a split's frames were cut from, or None where their names disagree or say nothing."""
    features = json.loads(Path(shots_path).read_text()).get("features") or []
    videos = {m.group(1) for f in features
              if (m := _FRAME.match(str((f.get("properties") or {}).get("filename") or "")))}
    return videos.pop() if len(videos) == 1 else None


def local_imu_dir(shots_path, video: str) -> Path | None:
    """The folder beside the split already holding the video's IMU files, if one does."""
    for gyro in sorted(Path(shots_path).parents[1].glob(f"*/insv_metadata/{video}_{IMU_FILES[0]}.csv")):
        if all((gyro.parent / f"{video}_{tag}.csv").is_file() for tag in IMU_FILES):
            return gyro.parent
    return None


def snapshot_realities(api, project_id: str, snapshot_id: str) -> list[dict]:
    """Every reality record of a snapshot. `Track3dAPI.get_realities_snapshot` asks the same endpoint
    but keeps only ids, of active realities, which cannot say which reality is the stitched one."""
    url = f"{api.base_url}{api.api_reality_version}/projects/{project_id}/reality?snapshot={snapshot_id}"
    response = api._request_with_auth_retry(
        "GET", url=url, request_name="snapshot_realities",
        headers={"authorization": f"Bearer {api.auth_token}", "Connection": "close"})
    response.raise_for_status()
    return response.json().get("result") or []


class StitchedRealities:
    """The s3 prefixes of a split's stitched realities -- its job's, newest first. The job comes from the
    reality's own details (the CSV's capture_id is the snapshot); a snapshot's records are asked once.

    A job processed since the IMU began to be stored has one, holding the IMU. A job reprocessed since
    then has one per run, and only the later ones hold it.
    """

    def __init__(self, api):
        self.api = api
        self._records: dict[str, list[dict]] = {}

    def prefixes(self, project_id: str, reality_id: str) -> list[str]:
        details = self.api.get_reality_details(project_id, reality_id, verbose=False)
        snapshot_id = details["snapshot_id"]
        if snapshot_id not in self._records:
            self._records[snapshot_id] = snapshot_realities(self.api, project_id, snapshot_id)
        return [r["mediaPath"] for r in sorted(self._records[snapshot_id], key=lambda r: r["_id"], reverse=True)
                if r.get("job") == details["job_id"] and r.get("isStitching") and r.get("mediaPath")]


def fetch(rows: list[dict], s3=None, api=None, bucket: str = BUCKET, workers: int = 8) -> dict[str, Path]:
    """{reality_id: folder holding its video's IMU files}, downloading what is not on disk yet.

    Each video is looked up and downloaded once, for all of its splits. Without `s3` and `api` only
    what is already on disk is used. A video whose IMU cannot be found or fetched is left out, with a
    warning, and the rest carry on.
    """
    stitched = StitchedRealities(api) if s3 is not None and api is not None else None
    with ThreadPoolExecutor(workers) as pool:
        videos = list(pool.map(lambda row: video_of(row["shots_path"]), rows))
    keyed = [((Path(row["shots_path"]).parents[1], video), row) for row, video in zip(rows, videos) if video]
    first = dict(reversed(keyed))

    def one(item):
        (structure_dir, video), row = item
        folder = local_imu_dir(row["shots_path"], video)
        if folder is not None or stitched is None:
            return folder
        try:
            for prefix in stitched.prefixes(row["project_id"], row["reality_id"]):
                keys = [f"{prefix}/insv_metadata/{video}_{tag}.csv" for tag in IMU_FILES]
                listed = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/insv_metadata/{video}_")
                if not set(keys) <= {obj["Key"] for obj in listed.get("Contents", [])}:
                    continue
                folder = structure_dir / prefix.rstrip("/").rsplit("/", 1)[-1] / "insv_metadata"
                folder.mkdir(parents=True, exist_ok=True)
                for key in keys:
                    s3.download_file(bucket, key, str(folder / key.rsplit("/", 1)[-1]))
                return folder
            warn(f"{row['capture_id']} {video}: no IMU in the job's stitched realities")
        except Exception as error:                                          # noqa: BLE001
            warn(f"{row['capture_id']} {video}: {type(error).__name__}: {error}")
        return None

    with ThreadPoolExecutor(workers) as pool:
        folders = dict(zip(first, pool.map(one, first.items())))
    return {row["reality_id"]: folders[key] for key, row in keyed if folders[key] is not None}


def clients(env) -> tuple:
    """(boto3 S3 client, Track3dAPI) from a .env carrying ROLE_ARN and the API login, connected the
    way scripts/get_priors.py connects."""
    from dotenv import dotenv_values
    from t3d_helpers.access.api_handler import Track3dAPI
    from t3d_helpers.access.data_aws import S3_AWS_DATA
    role_arn = dotenv_values(env).get("ROLE_ARN")
    if role_arn:
        os.environ["ROLE_ARN"] = role_arn
    return S3_AWS_DATA(role_arn=role_arn).download_bucket.meta.client, Track3dAPI(env_path=str(env))
