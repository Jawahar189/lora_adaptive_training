#!/usr/bin/env python3
"""reality ids in -> data on disk and the two local CSVs `run_alignment.py` eats.

`third_party/build_auto_reg_dataset.py` does this already, but only from the
auto_reg sheet and only through `t3d_helpers.Track3dAPI`, which is not on every
machine. This asks for less: a list of reality ids, and it works out the rest.

    python3 scripts/fetch_dataset.py --realities-csv datasets/single_reality.csv \
        --data-output /home/ubuntu/workdir/data_local \
        --csv-output datasets --name single_reality

The input CSV, which is `datasets/single_reality.csv`:

    project_id,structure_id,snapshot_id,reality_id
    PRJ515831,STRdlowjI6lFt1L,SNPMgXEfLPQyFRLVnfh,RLT1789815995825

Those four columns are the general case. `reality_id` on its own also works, but
ONLY for an id that --index-csv already carries -- the bucket has 4610 project
prefixes and no way to search by reality id, so an unknown id with nothing
beside it cannot be located at all. snapshot_id is optional where the
structure's snapshots can be listed; project_id and structure_id are not.

    python3 scripts/fetch_dataset.py --realities RLT1788195772679 --plan
    python3 scripts/fetch_dataset.py --realities-csv ... --offline   # no S3

Three stages, each resumable, same shape as the auto_reg script:

  resolve    for every reality: project, structure, capture, capture date, the
             PRIOR captures and their splits, and the s3 prefix of each. Held in
             memory and handed to the next stage; --write-plan also saves it to
             <csv-output>/<name>_plan.json, and a later stage run on its own
             reads that file when it is there and re-resolves when it is not.
  download   fetch each split's point_cloud.ply, shots.geojson and tm.json, plus
             the structure's floormap.png and tm_local_to_global.json.
             Resume-safe: an object whose local size already matches s3 is skipped.
  localize   generate images_full.json from shots + tm (the vendored converter in
             align/io/floormap.py, which tests/test_floormap.py pins against
             production's own output), then write the two 24-column CSVs, and
             the current one's imu_scale column (bevaligner/data/imu.py: each
             video's IMU from s3, into <PRJ>__<STR>/<stitched RLT>/insv_metadata/).
             -> <csv-output>/<name>_current.csv, <name>_prior.csv

WHERE THE FACTS COME FROM, in priority order. Each reality takes the first that
answers, so a machine with the API is not required and a machine with neither
the API nor the index still works as long as it can list the bucket:

  1. columns on the input CSV      project_id, structure_id, snapshot_id and
                                   paper_width / paper_height, when present
  2. --index-csv                   the shipped datasets/dataset_all_local_*.csv:
                                   identity, capture, dates, paper size, priors
  3. the bucket itself             walk <pid>/structures/<sid>/snapshots/*/reality/*

Rule 3 needs to know which snapshot is the PRIOR one, which the API answers with
snapshot dates. There are none in s3 -- but a reality id is epoch milliseconds
(RLT1788195772679 -> 2026-08-29), so snapshots order by the newest reality they
contain and the prior is the nearest strictly older one. That is the same rule
Track3dAPI.get_previous_reality_s3_path applies, derived from the ids instead of
from the API.

Paper size is the one fact neither s3 nor the ids carry. It comes from the index,
from a design JSON in s3 if one is there, or from --paper; a reality with no
paper size is reported and skipped rather than written as a row that makes
`float(row["paper_width"])` explode three stages later.

Layout written is the flattened one (`align/io/layout.py`), which is what the
current export uses:

    <data-output>/<PRJ>__<STR>/floormap.png
    <data-output>/<PRJ>__<STR>/tm_local_to_global.json
    <data-output>/<PRJ>__<STR>/<RLT>/{point_cloud.ply,shots.geojson,tm.json,images_full.json}
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# AWS CREDENTIALS -- paste a fresh SSO session here when the old one expires.
# Leave the strings empty to fall back, in order, to: --creds-from (a sibling
# script carrying its own block), the environment, then <work>/.env.
# These are short-lived session tokens; do not commit this file anywhere public.
# ---------------------------------------------------------------------------

AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_SESSION_TOKEN=""
import argparse
import collections
import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BUCKET = "constructn-projects-us"
URI = f"s3://{BUCKET}/"
WORKERS = 16
# Per structure, how many of its snapshots are listed at once during resolve.
SNAP_WORKERS = 8
IST = timezone(timedelta(hours=5, minutes=30))

# The default index: the CSVs shipped with the repo. Their PATH columns point at
# a mount that may not exist here -- only their identity columns are read.
DEFAULT_INDEX = ("datasets/dataset_all_local_current.csv",
                 "datasets/dataset_all_local_prior.csv")
DEFAULT_CREDS_FROM = "third_party/build_auto_reg_dataset.py"

# artifact -> (suffix under the reality prefix, local filename). tm.json lives
# one level down in s3 and flat on disk, which is why this is a mapping and not
# a list of names.
REALITY_ART = {
    "panovggt_point_cloud_path": ("point_cloud.ply", "point_cloud.ply"),
    "shots_path":                ("shots.geojson", "shots.geojson"),
    "tm_path":                   ("pointcloud/tm.json", "tm.json"),
    "images_prod_path":          ("images.json", "images.json"),
    "geo_path":                  ("images/geo.txt", "geo.txt"),
}
# Only these are needed to solve; the other two are best-effort.
REQUIRED_ART = ("panovggt_point_cloud_path", "shots_path", "tm_path")

# plan key -> (local filename, required). The plan holds the s3 uri under the
# first, because which design a structure resolves to is a lookup, not a suffix.
STRUCTURE_ART = {"floormap": ("floormap.png", True),
                 "design_tm": ("tm_local_to_global.json", False)}

# The 24 columns of dataset_*_local_current.csv, in order. The twelfth carries a
# comma inside its own name; it is written blank and nothing reads it.
LOCAL_COLS = ["reality_id", "capture_id", "project_id", "structure_id",
              "panovggt_point_cloud_path", "design_floor_plan_path", "shots_path",
              "tm_path", "images_prod_path", "images_full_path", "geo_path",
              " residue images, images_full", "localisation_status",
              "localisation tpr@1m", "design_tm", "paper_width", "paper_height",
              "design_tm_status", "images_path", "capture_date",
              "current_upload_date", "prior_capture_id", "prior_capture_date",
              "prior_upload_date"]

_RLT_MS = re.compile(r"RLT(\d{13})")

# Columns the input CSV may carry beside reality_id. The first three locate the
# split in s3 outright, which is the only way a reality outside --index-csv can
# be found: the bucket has 4610 project prefixes and no way to search by id.
# The last two are the one fact NEITHER s3 nor the id encodes -- every threshold
# in align/sheet.py is in paper inches, so without it there is no run.
HINT_COLS = ("project_id", "structure_id", "snapshot_id", "design_id",
             "paper_width", "paper_height")


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
#: Searched in order when neither the block above nor --env names a session.
ENV_FILES = ("third_party/.env", ".env", "../.env")

AWS_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def arm_credentials(creds_from: str | None, env_file: str | None = None) -> str:
    """Put a session into the environment and say where it came from.

    Never prints a key. The order is deliberate: a session pasted into the block
    above wins, then a .env, then another script's own AWS_* block, then whatever
    boto3's chain already has. The .env sits ABOVE the script block because the
    block is the thing that goes stale -- a paste from last week beating a .env
    refreshed this morning is the wrong way round, and an expired token fails as
    `ExpiredToken` several seconds into a listing rather than at startup.
    """
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
        if AWS_SESSION_TOKEN:
            os.environ["AWS_SESSION_TOKEN"] = AWS_SESSION_TOKEN
        return "the block at the top of this script"

    candidates = [env_file] if env_file else list(ENV_FILES)
    for name in candidates:
        if not name:
            continue
        path = Path(name)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            continue
        try:
            from dotenv import dotenv_values
        except ImportError:
            break
        values = dotenv_values(path)
        if values.get("AWS_ACCESS_KEY_ID") and values.get("AWS_SECRET_ACCESS_KEY"):
            # override: a session already in the environment may be the stale one.
            for key in AWS_KEYS:
                if values.get(key):
                    os.environ[key] = values[key]
                elif key == "AWS_SESSION_TOKEN":
                    os.environ.pop(key, None)
            return f"{path}"

    if creds_from:
        path = Path(creds_from)
        if not path.is_absolute():
            path = REPO / path
        if path.exists():
            src = path.read_text()
            found = {}
            for key in AWS_KEYS:
                m = re.search(rf'^{key}\s*=\s*"([^"]*)"', src, re.M)
                if m and m.group(1):
                    found[key] = m.group(1)
            if "AWS_ACCESS_KEY_ID" in found and "AWS_SECRET_ACCESS_KEY" in found:
                os.environ.update(found)
                return f"the AWS_* block in {path}"

    return "the ambient environment (~/.aws, env vars, instance role)"


def s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", config=Config(
        max_pool_connections=WORKERS * SNAP_WORKERS + 8,
        retries={"max_attempts": 5, "mode": "standard"}))


def preflight(s3, source: str) -> None:
    """One listing call, so a dead session says so now rather than mid-walk.

    An expired token otherwise surfaces as a botocore traceback several seconds
    into the resolve, after the index has loaded and with nothing said about
    which of the four credential sources was actually used.
    """
    from botocore.exceptions import ClientError, NoCredentialsError
    try:
        s3.list_objects_v2(Bucket=BUCKET, Prefix="", Delimiter="/", MaxKeys=1)
        return
    except NoCredentialsError:
        code = "NoCredentials"
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
    sys.exit(
        f"\ns3: {code} using credentials from {source}.\n"
        f"    Paste a fresh SSO session into the AWS_* block at the top of this\n"
        f"    script, or into a .env it searches ({', '.join(ENV_FILES)}), or\n"
        f"    export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN.\n"
        f"    --offline resolves the pairing from --index-csv without the network.")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def reality_epoch(reality_id: str) -> int:
    """Milliseconds encoded in a reality id, or 0.

    RLT1788195772679 is 2026-08-29. This is what orders snapshots when no API is
    available to hand over their dates.
    """
    m = _RLT_MS.search(reality_id or "")
    return int(m.group(1)) if m else 0


def reality_date(reality_id: str) -> str:
    ms = reality_epoch(reality_id)
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, IST).strftime("%Y-%m-%d")


def read_reality_ids(csv_path=None, inline=()) -> tuple[list[str], dict]:
    """The ids to fetch, and any hints the input CSV carried alongside them."""
    ids, hints = list(inline), {}
    if csv_path:
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            rid = row.get("reality_id")
            if not rid:
                continue
            ids.append(rid)
            hint = {k: row[k] for k in HINT_COLS if row.get(k)}
            if hint:
                hints[rid] = hint
    seen, out = set(), []
    for rid in ids:
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out, hints


def load_index(paths) -> dict[str, dict]:
    """reality_id -> row, over every index CSV given. Identity columns only."""
    index = {}
    for name in paths:
        path = Path(name)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            print(f"  index {path} is not there; skipping")
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                row = {(k or "").strip(): (v or "") for k, v in row.items()}
                if row.get("reality_id"):
                    index.setdefault(row["reality_id"], row)
    return index


def prior_capture_ids(cell: str) -> list[str]:
    cell = (cell or "").strip()
    if not cell:
        return []
    try:
        parsed = json.loads(cell)
    except json.JSONDecodeError:
        return [cell]
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [str(p).strip() for p in parsed if str(p).strip()]


# --------------------------------------------------------------------------
# the design: which drawing this structure is, and how big the paper is
# --------------------------------------------------------------------------
# Neither fact is in s3. A structure can carry HUNDREDS of designs, each with
# its own floormap.png (PRJ515831__STRdlowjI6lFt1L has 156), and nothing in the
# bucket says which one is current -- picking wrong does not raise, it aligns
# against the wrong drawing and returns a plausible answer in the wrong place.
# Paper size is not in the bucket at all, and every threshold in align/sheet.py
# is in inches, so there is no run without it.
#
# `Track3dAPI.get_design_id` + `get_design_details` answer both, and are the
# same two calls production's own downloader makes. Where the package is not
# installed the s3 guess is still there, but it is a guess and says so.
def api_client(env_path: str | None):
    """Track3dAPI against the .env that carries API_EMAIL / API_PASSWORD, or None."""
    try:
        from t3d_helpers.access.api_handler import Track3dAPI
    except ImportError:
        return None
    import logging
    quiet = logging.getLogger("fetch_dataset.api")
    quiet.setLevel(logging.WARNING)
    for name in ([env_path] if env_path else list(ENV_FILES)):
        path = Path(name)
        if not path.is_absolute():
            path = REPO / path
        if path.exists():
            return Track3dAPI(env_path=str(path), logger=quiet)
    return Track3dAPI(logger=quiet)


_REALITY_CACHE: dict[tuple[str, str], dict] = {}


def api_realities(api, pid: str, sid: str) -> dict[str, dict]:
    """Every reality record this structure has, keyed by id.

    Carries `pipelineVersion`, which is the only place the reconstruction
    pipeline is recorded -- it is on neither the s3 layout nor the id. Note it
    is ABSENT on older realities rather than set to an older value: measured
    over one structure's 4688 records, 695 say v9, 2 say v2, and 3991 say
    nothing at all.
    """
    key = (pid, sid)
    if key in _REALITY_CACHE:
        return _REALITY_CACHE[key]
    url = (f"{api.base_url}{api.api_all_snapshot_version}"
           f"/projects/{pid}/structures/{sid}/snapshots")
    r = api._request_with_auth_retry(
        "GET", url=url, request_name="get_all_snapshots",
        headers={"authorization": f"Bearer {api.auth_token}",
                 "Connection": "close"})
    if not r.ok:
        raise RuntimeError(f"get_all_snapshots -> HTTP {r.status_code}")
    data = r.json()
    out = {}
    if data.get("success") and data.get("result"):
        for snap in data["result"]["mSnapshots"]:
            for rec in snap.get("reality", []):
                out[rec["_id"]] = rec
    _REALITY_CACHE[key] = out
    return out


def pipeline_version(rec: dict | None) -> str:
    """The pipeline a reality was reconstructed by; "unknown" when unrecorded."""
    return str((rec or {}).get("pipelineVersion") or "unknown")


def mongo_pipeline_lookup():
    """`get_reality_wise_pipeline_info` as a per-reality callable, or None.

    Mongo is the authoritative source and production reads it directly. It is
    not always reachable -- the helper package has to be new enough to carry the
    function, and MONGO_URL has to resolve -- so this returns None rather than
    failing, and the caller falls back to the same field off the snapshots API.
    Each miss is "unknown", never an assumption, which is how the reference
    implementation treats it too.
    """
    try:
        from t3d_helpers.db_access.mongodb.db_data import \
            get_reality_wise_pipeline_info
    except ImportError:
        return None

    def lookup(reality_id: str) -> str:
        try:
            df = get_reality_wise_pipeline_info(reality_id=reality_id)
        except Exception:                                         # noqa: BLE001
            return "unknown"
        try:
            if df is None or (hasattr(df, "empty") and df.empty):
                return "unknown"
            return str(df["pipelineVersion"].iloc[0]).strip() or "unknown"
        except Exception:                                         # noqa: BLE001
            return "unknown"

    try:                        # one probe: a dead Mongo should not cost a call
        lookup("RLT0000000000000")                                # per reality
    except Exception:                                             # noqa: BLE001
        return None
    return lookup


_DESIGN_CACHE: dict[tuple[str, str, str], dict] = {}


def design_from_api(api, pid: str, sid: str, design_id: str = "") -> dict:
    """The structure's current design: its id, floormap, world transform, paper."""
    key = (pid, sid, design_id)
    if key not in _DESIGN_CACHE:
        _DESIGN_CACHE[key] = _design_from_api(api, pid, sid, design_id)
    return dict(_DESIGN_CACHE[key])


def _design_from_api(api, pid: str, sid: str, design_id: str = "") -> dict:
    import numpy as np
    did = design_id or api.get_design_id(project_id=pid, structure_id=sid,
                                         verbose=False)
    det = api.get_design_details(project_id=pid, structure_id=sid,
                                 design_id=did, verbose=False) or {}
    tm = det.get("tm")
    # A design places in world space only with a real, non-identity transform --
    # the same test build_auto_reg_dataset.py applies.
    ok = tm is not None and not np.allclose(
        np.asarray(tm, dtype=float).reshape(4, 4), np.identity(4))
    return dict(
        design_id=did,
        floormap=f"{URI}{pid}/structures/{sid}/designs/{did}/floormap/floormap.png",
        design_tm="", design_tm_obj=(list(tm) if ok else None),
        paper_w=str(det.get("paper_width", "")),
        paper_h=str(det.get("paper_height", "")),
        status="ok" if ok else "no_world_transform")


# --------------------------------------------------------------------------
# stage: resolve
# --------------------------------------------------------------------------
class Bucket:
    """The listing calls, memoized. One structure costs a handful of them."""

    def __init__(self, s3):
        self.s3 = s3
        self._prefixes: dict[str, list[str]] = {}
        self._snapshots: dict[tuple[str, str], list] = {}

    def children(self, prefix: str) -> list[str]:
        """The immediate sub-prefixes of `prefix`, without their trailing slash."""
        if prefix in self._prefixes:
            return self._prefixes[prefix]
        out = []
        pages = self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=prefix, Delimiter="/")
        for page in pages:
            for cp in page.get("CommonPrefixes", []):
                out.append(cp["Prefix"][len(prefix):].rstrip("/"))
        self._prefixes[prefix] = out
        return out

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.s3.head_object(Bucket=BUCKET, Key=key)
            return True
        except ClientError:
            return False

    def size(self, key: str) -> int:
        from botocore.exceptions import ClientError
        try:
            return int(self.s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"])
        except ClientError:
            return 0

    # -- the structure-shaped questions ------------------------------------
    def snapshots(self, pid: str, sid: str) -> list[tuple[int, str, list[str]]]:
        """[(newest reality epoch, snapshot_id, [reality ids]), ...], newest first."""
        if (pid, sid) in self._snapshots:
            return self._snapshots[(pid, sid)]
        base = f"{pid}/structures/{sid}/snapshots/"
        snaps = self.children(base)
        with ThreadPoolExecutor(SNAP_WORKERS) as ex:
            listed = list(ex.map(lambda s: self.children(f"{base}{s}/reality/"), snaps))
        out = []
        for snap, reals in zip(snaps, listed):
            if reals:
                out.append((max(reality_epoch(r) for r in reals), snap, reals))
        out.sort(reverse=True)
        self._snapshots[(pid, sid)] = out
        return out

    def design(self, pid: str, sid: str) -> dict:
        """The structure's floormap, world transform and paper size, from s3."""
        base = f"{pid}/structures/{sid}/designs/"
        best = dict(floormap="", design_tm="", paper_w="", paper_h="")
        for did in self.children(base):
            fm = f"{base}{did}/floormap/floormap.png"
            if not self.exists(fm):
                continue
            tm = f"{base}{did}/tm.json"
            cand = dict(floormap=f"{URI}{fm}",
                        design_tm=f"{URI}{tm}" if self.exists(tm) else "",
                        **self._paper(f"{base}{did}/"))
            # Prefer a design that is georeferenced; otherwise the first with ink.
            if cand["design_tm"] or not best["floormap"]:
                best = cand
            if cand["design_tm"]:
                break
        return best

    def _paper(self, design_prefix: str) -> dict:
        """Paper size from any small JSON sitting beside the design, if one has it."""
        pages = self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=design_prefix)
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".json") or obj["Size"] > 200_000:
                    continue
                if key.endswith("/tm.json"):
                    continue
                try:
                    body = self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
                    doc = json.loads(body)
                except Exception:                                   # noqa: BLE001
                    continue
                if isinstance(doc, dict) and doc.get("paper_width"):
                    return dict(paper_w=str(doc["paper_width"]),
                                paper_h=str(doc.get("paper_height", "")))
        return dict(paper_w="", paper_h="")


def reality_prefixes(bucket, pid, sid) -> dict[str, tuple[str, str]]:
    """reality_id -> (s3 prefix, snapshot_id) for one structure."""
    out = {}
    base = f"{pid}/structures/{sid}/snapshots/"
    for _epoch, snap, reals in bucket.snapshots(pid, sid):
        for rid in reals:
            out[rid] = (f"{base}{snap}/reality/{rid}", snap)
    return out


def art_uris(prefix: str) -> dict[str, str]:
    return {col: f"{URI}{prefix}/{suffix}" for col, (suffix, _) in REALITY_ART.items()}


class Progress:
    """One line every `every` seconds: n/total, rate, ETA, plus whatever `extra` says."""

    def __init__(self, label: str, total: int, every: float = 10.0):
        self.label, self.total, self.every = label, total, every
        self.t0 = self.last = time.monotonic()

    def tick(self, n: int, extra: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < self.every and n != self.total:
            return
        self.last = now
        dt = now - self.t0
        rate = n / dt if dt else 0.0
        eta = (self.total - n) / rate if rate else 0.0
        print(f"  {self.label} [{n}/{self.total}] {100 * n / max(self.total, 1):.1f}%  "
              f"{rate:.1f}/s  elapsed {timedelta(seconds=int(dt))}  "
              f"eta {timedelta(seconds=int(eta))}{'  ' + extra if extra else ''}",
              flush=True)


def prefetch(reality_ids, hints, index, bucket, api, versions) -> None:
    """Warm every per-structure cache resolve() reads, WORKERS structures at once.

    resolve() itself stays serial and unchanged; this only makes each of its
    network calls a cache hit. A lookup that fails here is left uncached, so
    resolve() repeats it and records the problem exactly as before.
    """
    todo = {}
    for rid in reality_ids:
        idx, hint = index.get(rid, {}), hints.get(rid, {})
        pid = hint.get("project_id") or idx.get("project_id", "")
        sid = hint.get("structure_id") or idx.get("structure_id", "")
        if pid and sid:
            todo.setdefault((pid, sid), hint.get("design_id", ""))

    def warm(item):
        (pid, sid), did = item
        for call in ((lambda: design_from_api(api, pid, sid, did)) if api else None,
                     (lambda: bucket.snapshots(pid, sid)) if bucket else None,
                     (lambda: api_realities(api, pid, sid)) if api and versions else None):
            if call is None:
                continue
            try:
                call()
            except Exception:                                     # noqa: BLE001
                pass

    prog = Progress("prefetch structures", len(todo))
    with ThreadPoolExecutor(WORKERS) as ex:
        for n, _ in enumerate(as_completed([ex.submit(warm, i) for i in todo.items()]), 1):
            prog.tick(n)


def resolve(reality_ids, hints, index, bucket, paper_override=None, api=None,
            versions=(), mongo=None) -> dict:
    """The whole plan: every current split, its priors, and where each lives in s3.

    `bucket` is None when running --offline, in which case the plan carries
    identities and pairings but no s3 prefixes, which is enough to see what a
    run WOULD fetch.
    """
    plan = {"current": [], "prior": {}, "structures": {}, "problems": []}
    prior_by_capture: dict[str, list[dict]] = {}
    for row in index.values():
        if row.get("capture_id"):
            prior_by_capture.setdefault(row["capture_id"], []).append(row)

    prefetch(reality_ids, hints, index, bucket, api, versions)
    prog = Progress("resolve", len(reality_ids))
    for n, rid in enumerate(reality_ids, 1):
        prog.tick(n - 1, f"structures {len(plan['structures'])}  "
                         f"current {len(plan['current'])}  prior {len(plan['prior'])}  "
                         f"problems {len(plan['problems'])}")
        idx = index.get(rid, {})
        hint = hints.get(rid, {})
        pid = hint.get("project_id") or idx.get("project_id", "")
        sid = hint.get("structure_id") or idx.get("structure_id", "")
        if not (pid and sid):
            plan["problems"].append(
                f"{rid}: no project/structure -- add project_id and structure_id "
                f"columns to the input CSV, or pass an --index-csv that has it")
            continue

        key = f"{pid}__{sid}"
        if key not in plan["structures"]:
            design = dict(design_id="", floormap="", design_tm="",
                          design_tm_obj=None, paper_w="", paper_h="", status="")
            if api is not None:
                try:
                    design = design_from_api(api, pid, sid, hint.get("design_id", ""))
                except Exception as exc:                          # noqa: BLE001
                    plan["problems"].append(f"{key}: design lookup failed: {exc!r}"[:160])
                    design["status"] = ""
            if not design["floormap"] and bucket:
                # No API: the bucket's own guess, which is only safe on a
                # structure with one design. Said out loud for that reason.
                guess = bucket.design(pid, sid)
                design.update(floormap=guess["floormap"], design_tm=guess["design_tm"],
                              paper_w=guess["paper_w"], paper_h=guess["paper_h"],
                              status="ok" if guess["design_tm"] else "")
                if guess["floormap"]:
                    plan["problems"].append(
                        f"{key}: design chosen by listing s3, not by the API -- "
                        f"check it is the right drawing")
            paper_w = (paper_override[0] if paper_override
                       else hint.get("paper_width") or design["paper_w"]
                       or idx.get("paper_width", ""))
            paper_h = (paper_override[1] if paper_override
                       else hint.get("paper_height") or design["paper_h"]
                       or idx.get("paper_height", ""))
            plan["structures"][key] = dict(
                project_id=pid, structure_id=sid,
                design_id=design["design_id"],
                floormap=design["floormap"], design_tm=design["design_tm"],
                design_tm_obj=design["design_tm_obj"],
                paper_width=str(paper_w), paper_height=str(paper_h),
                design_tm_status=(design["status"]
                                  or idx.get("design_tm_status", "")))
        st = plan["structures"][key]
        if not st["paper_width"] or not st["paper_height"]:
            # Every threshold in align/sheet.py is in inches and every metric
            # conversion goes through this, so a blank one is not a degraded row,
            # it is a ValueError in align/io/cases.py three stages later.
            plan["problems"].append(
                f"{rid}: no paper size for {key} -- add paper_width and "
                f"paper_height columns, or pass --paper W H")
            plan["structures"].pop(key, None)
            continue

        prefixes = reality_prefixes(bucket, pid, sid) if bucket else {}
        # A snapshot_id on the input CSV names the prefix outright, which is the
        # one case that needs no listing at all -- used as the fallback so a
        # structure whose snapshots cannot be listed still resolves.
        if bucket and rid not in prefixes and hint.get("snapshot_id"):
            snap = hint["snapshot_id"]
            prefixes[rid] = (f"{pid}/structures/{sid}/snapshots/{snap}"
                             f"/reality/{rid}", snap)

        # -- the priors ----------------------------------------------------
        priors, prior_caps, prior_date = [], [], idx.get("prior_capture_date", "")
        caps = prior_capture_ids(idx.get("prior_capture_id", ""))
        if caps:                                   # the index knows the pairing
            prior_caps = caps
            for cap in caps:
                for prow in prior_by_capture.get(cap, []):
                    priors.append(dict(reality_id=prow["reality_id"],
                                       capture_id=cap,
                                       capture_date=prow.get("capture_date", "")))
        elif bucket:                               # derive it from the ids
            snaps = bucket.snapshots(pid, sid)
            here = next((n for n, (_, _, reals) in enumerate(snaps) if rid in reals),
                        None)
            if here is None:
                plan["problems"].append(f"{rid}: in no snapshot of {key}")
            elif here + 1 >= len(snaps):
                plan["problems"].append(f"{rid}: no snapshot older than its own")
            else:
                _epoch, snap, reals = snaps[here + 1]
                prior_caps = [snap]                # the snapshot stands in for the
                prior_date = reality_date(max(reals, key=reality_epoch))
                for prid in reals:                 # capture: one id, consistently
                    priors.append(dict(reality_id=prid, capture_id=snap,
                                       capture_date=reality_date(prid)))
        # Keep only the priors built by an accepted pipeline. A v2 split next to
        # v9 ones is not a worse reference, it is a differently-made one, and it
        # is merged into the same raster where nothing downstream can tell them
        # apart. Only the API records this, so an --no-api run cannot filter.
        if versions and priors:
            if api is None:
                plan["problems"].append(
                    f"{rid}: --pipeline-version needs the API; kept all priors")
            else:
                try:
                    recs = api_realities(api, pid, sid)
                except Exception as exc:                          # noqa: BLE001
                    plan["problems"].append(
                        f"{key}: reality lookup failed, kept all priors: {exc!r}"[:160])
                else:
                    def version_of(rid):
                        if mongo is not None:
                            v = mongo(rid)
                            if v != "unknown":
                                return v.lower()
                        return pipeline_version(recs.get(rid)).lower()

                    seen = collections.Counter(
                        version_of(p["reality_id"]) for p in priors)
                    priors = [p for p in priors
                              if version_of(p["reality_id"]) in versions]
                    rejected = {v: n for v, n in seen.items() if v not in versions}
                    if rejected:
                        plan["problems"].append(
                            f"{rid}: dropped {sum(rejected.values())} prior split(s) "
                            f"by pipeline {dict(rejected)}, kept {len(priors)}")
        if not priors:
            plan["problems"].append(f"{rid}: no prior splits found")

        capture_id = idx.get("capture_id") or (prefixes.get(rid, ("", ""))[1]) \
            or hint.get("snapshot_id", "")
        plan["current"].append(dict(
            reality_id=rid, capture_id=capture_id, project_id=pid,
            structure_id=sid, structure=key,
            prefix=prefixes.get(rid, ("", ""))[0],
            capture_date=idx.get("capture_date") or reality_date(rid),
            upload_date=idx.get("current_upload_date", ""),
            prior_capture_id=json.dumps(prior_caps),
            prior_capture_date=prior_date,
            prior_upload_date=idx.get("prior_upload_date", "")))

        for p in priors:
            if p["reality_id"] in plan["prior"]:
                continue
            pidx = index.get(p["reality_id"], {})
            plan["prior"][p["reality_id"]] = dict(
                reality_id=p["reality_id"],
                capture_id=p["capture_id"], project_id=pid, structure_id=sid,
                structure=key,
                prefix=prefixes.get(p["reality_id"], ("", ""))[0],
                capture_date=(p["capture_date"] or pidx.get("capture_date", "")
                              or reality_date(p["reality_id"])),
                upload_date=pidx.get("current_upload_date", ""))

    prog.tick(len(reality_ids), f"structures {len(plan['structures'])}  "
              f"current {len(plan['current'])}  prior {len(plan['prior'])}  "
              f"problems {len(plan['problems'])}", force=True)
    # A split with no prefix cannot be fetched. Drop it rather than carry a row
    # whose every path cell will come back empty.
    for rid in [r for r, row in plan["prior"].items() if bucket and not row["prefix"]]:
        plan["problems"].append(f"{rid}: no prefix under any snapshot in s3")
        plan["prior"].pop(rid)
    keep = []
    for row in plan["current"]:
        if bucket and not row["prefix"]:
            plan["problems"].append(
                f"{row['reality_id']}: no prefix under any snapshot in s3 -- "
                f"add a snapshot_id column, or check the id belongs to "
                f"{row['structure']}")
        else:
            keep.append(row)
    plan["current"] = keep
    return plan


# --------------------------------------------------------------------------
# stage: download
# --------------------------------------------------------------------------
def write_design_tm(plan, dest: Path) -> int:
    """Save the world transform the API returned, where s3 carries no tm.json.

    `align.frames.load_design_tm` accepts a bare list of 16, which is what the
    API hands back and what production's own downloader writes.
    """
    n = 0
    for key, st in plan["structures"].items():
        tm = st.get("design_tm_obj")
        if not tm:
            continue
        out = dest / key / "tm_local_to_global.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tm))
        n += 1
    return n


def tasks_for(plan, dest: Path) -> list[tuple[str, Path, bool]]:
    """(s3 uri, local path, required) for everything the plan names."""
    out, seen = [], set()
    # A split can be current for one reality and prior for another; list it once.
    for row in plan["current"] + list(plan["prior"].values()):
        if not row["prefix"] or row["reality_id"] in seen:
            continue
        seen.add(row["reality_id"])
        rdir = dest / row["structure"] / row["reality_id"]
        for col, uri in art_uris(row["prefix"]).items():
            out.append((uri, rdir / REALITY_ART[col][1], col in REQUIRED_ART))
    for key, st in plan["structures"].items():
        for field, (fname, required) in STRUCTURE_ART.items():
            if st.get(field):
                out.append((st[field], dest / key / fname, required))
    return out


def plan_sizes(s3, tasks) -> None:
    from botocore.exceptions import ClientError

    # Objects sharing an s3 folder are sized by ONE delimited listing of that
    # folder instead of a HEAD each: a reality's point_cloud.ply, shots.geojson
    # and images.json all sit at its root. A folder holding only one wanted key
    # (pointcloud/tm.json, images/geo.txt -- images/ is thousands of objects
    # deep) is still a HEAD, which is cheaper than listing it.
    by_dir: dict[str, list] = {}
    for uri, _local, required in tasks:
        key = uri[len(URI):]
        by_dir.setdefault(key.rsplit("/", 1)[0] + "/", []).append((uri, key, required))

    def size_dir(item):
        folder, wanted = item
        if len(wanted) == 1:
            uri, key, required = wanted[0]
            try:
                n = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
            except ClientError:
                n = 0
            return [(uri, n, required)]
        found = {}
        for page in s3.get_paginator("list_objects_v2").paginate(
                Bucket=BUCKET, Prefix=folder, Delimiter="/"):
            for obj in page.get("Contents", []):
                found[obj["Key"]] = obj["Size"]
        return [(uri, found.get(key, 0), required) for uri, key, required in wanted]

    prog = Progress("size", len(by_dir))
    sizes, running = [], 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for n, got in enumerate(ex.map(size_dir, by_dir.items()), 1):
            sizes += got
            running += sum(g[1] for g in got)
            prog.tick(n, f"{len(sizes)}/{len(tasks)} objects  "
                         f"{running / 1e9:.2f} GB so far")
    total = sum(n for _, n, _ in sizes)
    absent = [(u, r) for u, n, r in sizes if not n]
    print(f"  {sum(1 for _, n, _ in sizes if n)}/{len(sizes)} objects present, "
          f"{total / 1e9:.2f} GB")
    if absent:
        print(f"  {len(absent)} not in s3 "
              f"({sum(1 for _, r in absent if r)} of them required):")
        for uri, required in absent[:10]:
            print(f"    {'REQUIRED ' if required else '         '}{uri}")
    import shutil
    free = shutil.disk_usage(Path(tasks[0][1]).anchor).free
    print(f"  free on disk {free / 1e9:.0f} GB")


def download(s3, tasks) -> dict:
    from botocore.exceptions import BotoCoreError, ClientError
    lock = threading.Lock()
    done = {"got": 0, "skipped": 0, "missing": 0, "failed": 0, "bytes": 0}

    def fetch(task):
        # A 4th element is the size from a listing, which saves a HEAD per object.
        uri, local, _required = task[:3]
        key = uri[len(URI):]
        if len(task) > 3:
            head = {"ContentLength": task[3]}
        else:
            try:
                head = s3.head_object(Bucket=BUCKET, Key=key)
            except ClientError:
                with lock:
                    done["missing"] += 1
                return
        if local.exists() and local.stat().st_size == head["ContentLength"]:
            with lock:
                done["skipped"] += 1
            return
        # Per process and thread, so two runs over the same set cannot collide.
        tmp = local.with_name(
            f"{local.name}.{os.getpid()}.{threading.get_ident()}.part")
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, key, str(tmp))
            tmp.replace(local)
        except (ClientError, BotoCoreError, OSError) as exc:
            if tmp.exists():
                tmp.unlink()
            # Another run fetching the same object finished it first: not a failure.
            if local.exists() and local.stat().st_size == head["ContentLength"]:
                with lock:
                    done["skipped"] += 1
                return
            with lock:
                done["failed"] += 1
            print(f"  FAIL {local}: {exc}", flush=True)
            return
        with lock:
            done["got"] += 1
            done["bytes"] += head["ContentLength"]

    with ThreadPoolExecutor(WORKERS) as ex:
        futures = [ex.submit(fetch, t) for t in tasks]
        for n, _ in enumerate(as_completed(futures), 1):
            if n % max(10, len(tasks) // 200) == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] {done}", flush=True)
    return done


# --------------------------------------------------------------------------
# stage: localize
# --------------------------------------------------------------------------
def write_images_full(rlt_dir: Path) -> bool:
    """Per-image poses in the target frame, from this split's shots and transform.

    The same file production ships, rebuilt by the vendored converter in
    `align/io/floormap.py` -- which `tests/test_floormap.py` pins against real
    deliveries (positions exact, angles to 1e-12 degrees). It is the split's
    ground truth and is written here so scoring has something to score against;
    a deployment run with --no-score does not need it.
    """
    from align.io.floormap import poses_to_images, shot_poses
    import numpy as np
    tm_path = rlt_dir / "tm.json"
    if not (rlt_dir / "shots.geojson").exists() or not tm_path.exists():
        return False
    tm = np.asarray(json.loads(tm_path.read_text())["tm"], dtype=float).reshape(4, 4)
    images = poses_to_images(shot_poses(rlt_dir), tm)
    if not images:
        return False
    (rlt_dir / "images_full.json").write_text(json.dumps(images, indent=2))
    return True


def localize(plan, dest: Path, out_dir: Path, name: str) -> tuple[Path, Path]:
    """Write the two CSVs, with a path cell EMPTY wherever the file is not on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = (out_dir / f"{name}_current.csv", out_dir / f"{name}_prior.csv")
    sides = (("current", plan["current"], paths[0]),
             ("prior", list(plan["prior"].values()), paths[1]))

    for side, rows, path in sides:
        made, out_rows, skipped = 0, [], []
        for row in rows:
            st = plan["structures"][row["structure"]]
            sdir = dest / row["structure"]
            rdir = sdir / row["reality_id"]
            try:
                made += bool(write_images_full(rdir))
            except Exception as exc:                              # noqa: BLE001
                print(f"  {row['reality_id']}: images_full failed: {exc!r}"[:140])

            def keep(p: Path, is_dir=False) -> str:
                ok = (p.is_dir() and any(p.iterdir())) if is_dir else p.is_file()
                return str(p) if ok else ""

            rec = {c: "" for c in LOCAL_COLS}
            rec.update(
                reality_id=row["reality_id"], capture_id=row["capture_id"],
                project_id=row["project_id"], structure_id=row["structure_id"],
                panovggt_point_cloud_path=keep(rdir / "point_cloud.ply"),
                design_floor_plan_path=keep(sdir / "floormap.png"),
                shots_path=keep(rdir / "shots.geojson"),
                tm_path=keep(rdir / "tm.json"),
                images_prod_path=keep(rdir / "images.json"),
                images_full_path=keep(rdir / "images_full.json"),
                geo_path=keep(rdir / "geo.txt"),
                design_tm=keep(sdir / "tm_local_to_global.json"),
                paper_width=st["paper_width"], paper_height=st["paper_height"],
                design_tm_status=("ok" if (sdir / "tm_local_to_global.json").is_file()
                                  else "no_world_transform"),
                images_path=keep(rdir / "images_processed", True),
                capture_date=row["capture_date"],
                current_upload_date=row.get("upload_date", ""),
                prior_capture_id=(row.get("prior_capture_id", "[]")
                                  if side == "current" else "[]"),
                prior_capture_date=(row.get("prior_capture_date", "")
                                    if side == "current" else ""),
                prior_upload_date=(row.get("prior_upload_date", "")
                                   if side == "current" else ""))
            # `align.io.cases` derives rlt_dir from tm_path on EVERY row, so a
            # blank one is not a degraded row -- Path("").parent is ".", and the
            # whole index fails to load. A split missing any of the three the
            # solver needs cannot be used anyway: no transform is no reference,
            # no shots is no trajectory, no cloud is nothing to match.
            missing = [c for c in REQUIRED_ART if not rec[c]]
            if missing:
                skipped.append((row["reality_id"], ", ".join(missing)))
                continue
            out_rows.append(rec)

        if skipped:
            print(f"\n  {len(skipped)} {side} split(s) dropped, artifact not in s3:")
            for rid, why in skipped[:8]:
                print(f"    {rid}  no {why}")
            if len(skipped) > 8:
                print(f"    ... and {len(skipped) - 8} more")
        if side == "current" and not out_rows:
            sys.exit("no current split has all of point_cloud.ply, shots.geojson "
                     "and tm.json; there is nothing to run")

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOCAL_COLS)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nwrote {path}   {len(out_rows)} rows, "
              f"images_full generated {made}/{len(out_rows)}")
        for col in ("panovggt_point_cloud_path", "shots_path", "tm_path",
                    "design_floor_plan_path", "design_tm", "images_full_path",
                    "paper_width"):
            n = sum(1 for r in out_rows if str(r[col]).strip())
            flag = "" if n == len(out_rows) else "   <-- gaps"
            print(f"    {col:26} {n:3}/{len(out_rows)}{flag}")
    return paths


# --------------------------------------------------------------------------
# stage: images
# --------------------------------------------------------------------------
IMAGES_DIR = "images_processed"


def image_tasks(s3, rows) -> list[tuple[str, Path, bool]]:
    """(s3 uri, local path, required) for every object under each split's images.

    Works from the written CSVs, not the plan: the s3 prefix is fully named by a
    row's project, structure, capture (snapshot) and reality, and the local
    reality directory is wherever that row's tm.json already sits. So nothing is
    re-resolved and the priors paired in are exactly the ones in the CSV.
    """
    def lst(row):
        prefix = (f"{row['project_id']}/structures/{row['structure_id']}/snapshots/"
                  f"{row['capture_id']}/reality/{row['reality_id']}/{IMAGES_DIR}/")
        local = Path(row["tm_path"]).parent / IMAGES_DIR
        out = []
        for page in s3.get_paginator("list_objects_v2").paginate(
                Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/"):
                    continue
                out.append((f"{URI}{obj['Key']}", local / obj["Key"][len(prefix):],
                            False, obj["Size"]))
        return out

    prog = Progress("list", len(rows))
    listed, running = [], 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for n, got in enumerate(ex.map(lst, rows), 1):
            listed.append(got)
            running += len(got)
            prog.tick(n, f"{running} objects so far")
    empty = [r["reality_id"] for r, got in zip(rows, listed) if not got]
    flat = [t for got in listed for t in got]
    print(f"  {len(flat)} image objects across {len(rows) - len(empty)}/{len(rows)} "
          f"splits, {sum(t[3] for t in flat) / 1e9:.2f} GB")
    if empty:
        print(f"  {len(empty)} split(s) with no {IMAGES_DIR}/ in s3: "
              f"{' '.join(empty[:8])}{' ...' if len(empty) > 8 else ''}")
    return flat


def fill_images_path(path: Path) -> None:
    """Set images_path on every row whose images_processed/ is on disk, in place.

    Only that one column changes; every other cell, and the row order, is kept.
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        d = Path(row["tm_path"]).parent / IMAGES_DIR
        row["images_path"] = str(d) if d.is_dir() and any(d.iterdir()) else ""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    n = sum(1 for r in rows if r["images_path"])
    print(f"  updated {path}   images_path {n}/{len(rows)}"
          f"{'' if n == len(rows) else '   <-- gaps'}")


def fetch_images(s3, out_dir: Path, name: str, sides, plan_only: bool,
                 only=()) -> None:
    paths = {"current": out_dir / f"{name}_current.csv",
             "prior": out_dir / f"{name}_prior.csv"}
    todo = [paths[s] for s in sides]
    for p in todo:
        if not p.exists():
            sys.exit(f"{p} not found; run the other stages first")
    rows = []
    for p in todo:
        with open(p, newline="") as fh:
            rows += list(csv.DictReader(fh))
    if only:
        rows = [r for r in rows if r["reality_id"] in set(only)]
        if not rows:
            sys.exit(f"none of {' '.join(only)} is in {', '.join(map(str, todo))}")
    # A split can be current in one pairing and prior in another; fetch it once.
    rows = list({r["reality_id"]: r for r in rows}.values())
    print(f"listing {IMAGES_DIR}/ for {len(rows)} split(s) from "
          f"{', '.join(str(p) for p in todo)}")
    tasks = image_tasks(s3, rows)
    if plan_only:
        import shutil
        free = shutil.disk_usage(Path(rows[0]["tm_path"]).anchor).free
        print(f"  free on disk {free / 1e9:.0f} GB")
        return
    print(f"\ndownloading {len(tasks)} image objects")
    print(f"  {download(s3, tasks)}")
    for p in todo:
        fill_images_path(p)


# --------------------------------------------------------------------------
def describe(plan) -> None:
    print(f"\n{len(plan['current'])} current, {len(plan['prior'])} prior splits, "
          f"{len(plan['structures'])} structure(s)")
    priors_by_structure = {}
    for row in plan["prior"].values():
        priors_by_structure.setdefault(row["structure"], []).append(row["reality_id"])
    for row in plan["current"]:
        st = plan["structures"][row["structure"]]
        mine = [p["reality_id"] for p in plan["prior"].values()
                if p["structure"] == row["structure"]]
        print(f"  {row['reality_id']}  {row['structure']}  "
              f"{row['capture_date'] or '?'}  paper {st['paper_width'] or '?'}"
              f"x{st['paper_height'] or '?'}  "
              f"{'georeferenced' if st['design_tm_status'] == 'ok' else 'paper frame'}")
        print(f"      priors: {len(mine)}  {' '.join(mine[:6])}"
              f"{' ...' if len(mine) > 6 else ''}")
    for why in plan["problems"]:
        print(f"  ! {why}")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("input")
    g.add_argument("--realities-csv", default=None,
                   help="a CSV with a reality_id column; project_id, structure_id "
                        "and snapshot_id are used when present")
    g.add_argument("--realities", nargs="+", default=(),
                   help="reality ids straight on the command line")
    g.add_argument("--index-csv", nargs="+", default=list(DEFAULT_INDEX),
                   help="CSVs to look identity, dates, paper size and the prior "
                        "pairing up in before asking s3")
    g.add_argument("--paper", nargs=2, type=float, default=None,
                   metavar=("W", "H"), help="paper size, when nothing else has it")

    g = ap.add_argument_group("output")
    g.add_argument("--data-output", default="/home/ubuntu/workdir/data_local",
                   help="where the artifacts land")
    g.add_argument("--csv-output", default="datasets",
                   help="directory for <name>_current.csv and <name>_prior.csv")
    g.add_argument("--name", default=None,
                   help="stem for the two CSVs; default: the input CSV's stem")

    g = ap.add_argument_group("run")
    g.add_argument("--stage", choices=["resolve", "download", "localize", "all",
                                       "images"],
                   default="all",
                   help="'images' is separate from 'all': it reads the existing "
                        "<name>_current/_prior.csv, fetches each split's "
                        "images_processed/ (~1 GB each) and fills images_path "
                        "in place. Nothing else is re-fetched or re-resolved")
    g.add_argument("--images-side", nargs="+", choices=["current", "prior"],
                   default=["current", "prior"],
                   help="with --stage images: which CSV's splits to fetch")
    g.add_argument("--images-realities", nargs="+", default=(), metavar="RLT",
                   help="with --stage images: fetch only these reality ids "
                        "(e.g. to test one before the full run)")
    g.add_argument("--plan", action="store_true",
                   help="size every object and download nothing")
    g.add_argument("--pipeline-version", nargs="+", default=["v9"],
                   metavar="V",
                   help="keep only prior splits reconstructed by these "
                        "pipelines. 'unknown' keeps the ones whose record does "
                        "not say (most older realities), 'all' disables the "
                        "filter. Needs the API; default: v9")
    g.add_argument("--no-api", action="store_true",
                   help="do not call Track3dAPI; fall back to guessing the "
                        "design by listing s3, which is only safe where the "
                        "structure has exactly one")
    g.add_argument("--offline", action="store_true",
                   help="resolve from the index alone; touches no network")
    g.add_argument("--write-plan", action="store_true",
                   help="also save <name>_plan.json: which snapshot each split "
                        "came from and which priors were paired in. Off by "
                        "default; every stage re-resolves without it")
    g.add_argument("--replan", action="store_true",
                   help="resolve again even where <name>_plan.json exists; "
                        "needed after changing --pipeline-version or the input")
    g.add_argument("--workers", type=int, default=WORKERS)
    g.add_argument("--env", default=None,
                   help=f"a .env carrying AWS_*; default searches {', '.join(ENV_FILES)}")
    g.add_argument("--creds-from", default=DEFAULT_CREDS_FROM,
                   help="a script whose AWS_* block to reuse when nothing else has one")
    return ap


def main():
    global WORKERS
    args = build_parser().parse_args()
    WORKERS = args.workers
    dest = Path(args.data_output)
    out_dir = Path(args.csv_output)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    name = args.name or (Path(args.realities_csv).stem if args.realities_csv
                         else "realities")
    plan_path = out_dir / f"{name}_plan.json"

    ids, hints = read_reality_ids(args.realities_csv, args.realities)
    if not ids and args.stage in ("resolve", "all"):
        sys.exit("no reality ids: pass --realities-csv or --realities")

    bucket = None
    if not args.offline:
        source = arm_credentials(args.creds_from, args.env)
        print(f"aws: credentials from {source}")
        client = s3_client()
        preflight(client, source)
        bucket = Bucket(client)

    if args.stage == "images":
        if bucket is None:
            sys.exit("--offline cannot download")
        fetch_images(bucket.s3, out_dir, name, args.images_side, args.plan,
                     args.images_realities)
        return

    # The plan is an in-memory hand-off between the stages. It is written only
    # when asked for: a later stage run on its own re-resolves rather than
    # requiring a file, so nothing downstream depends on it existing.
    if (args.stage in ("download", "localize") or args.plan) \
            and not args.replan and plan_path.exists():
        plan = json.loads(plan_path.read_text())
        print(f"plan: {plan_path}  (saved; --replan to resolve again)")
    else:
        print(f"resolving {len(ids)} reality id(s)"
              f"{' (offline)' if args.offline else ''}")
        index = load_index(args.index_csv)
        print(f"  index: {len(index)} realities")
        api = None if args.no_api else api_client(args.env)
        print(f"  design source: {'Track3dAPI' if api else 's3 listing (a guess)'}")
        versions = (() if "all" in args.pipeline_version
                    else tuple(v.lower() for v in args.pipeline_version))
        if versions:
            print(f"  prior pipeline: keeping {', '.join(versions)}")
        mongo = mongo_pipeline_lookup() if versions else None
        if versions:
            print(f"  pipeline source: "
                  f"{'mongo' if mongo else 'snapshots API (mongo unavailable)'}")
        plan = resolve(ids, hints, index, bucket, paper_override=args.paper,
                       api=api, versions=versions, mongo=mongo)
        describe(plan)
        if not plan["current"]:
            # Writing headers with no rows, and then printing a run command that
            # cannot work, is worse than stopping: the failure only shows up two
            # commands later as an empty run.
            sys.exit("\nnothing to fetch: no reality resolved to a location in s3.\n"
                     "    Give the input CSV project_id, structure_id and\n"
                     "    snapshot_id columns, or an --index-csv carrying the id.")
        if args.write_plan:
            out_dir.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(plan, indent=2))
            print(f"\nwrote {plan_path}")

    tasks = tasks_for(plan, dest)
    if args.plan:
        if not tasks:
            sys.exit("nothing to fetch; resolve found no s3 prefixes")
        print(f"\nplan: {len(tasks)} objects")
        plan_sizes(bucket.s3, tasks)
        return

    if args.stage in ("download", "all"):
        if bucket is None:
            sys.exit("--offline cannot download")
        wrote = write_design_tm(plan, dest)
        if wrote:
            print(f"wrote {wrote} tm_local_to_global.json from the API")
        print(f"\ndownloading {len(tasks)} objects into {dest}")
        print(f"  {download(bucket.s3, tasks)}")

    if args.stage in ("localize", "all"):
        current, prior = localize(plan, dest, out_dir, name)
        # Each split's IMU scale, as the current CSV's imu_scale column. Never fatal: a split
        # without one is blank and keeps its floor fit.
        from bevaligner.data.imu import add_imu_scale
        add_imu_scale(current, bucket.s3 if bucket is not None else None,
                      None if args.offline or args.no_api else api_client(args.env))
        print(f"\nrun it with:\n"
              f"  python3 scripts/run_alignment.py \\\n"
              f"      --current-csv {current} --prior-csv {prior} \\\n"
              f"      --stem out/deploy/run_alignment --visualise "
              f"--viz-dir out/deploy/viz \\\n"
              f"      --images --images-gt --reference-force")


if __name__ == "__main__":
    main()
