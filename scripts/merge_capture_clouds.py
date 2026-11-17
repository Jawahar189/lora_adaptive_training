"""Merge every reality's point cloud of one capture into a single PLY.

Each row of the dataset CSV is one reality (RLT) with its own `point_cloud.ply`
in the local SfM frame and a `tm.json` that maps that frame to absolute UTM
(row-major 4x4, applied as tm @ [x, y, z, 1]; the `offset` field is NOT applied,
matching `align.frames.load_design_tm` and `align.priors`). Realities sharing a
`capture_id` were scanned in the same session, so after tm they share a frame
and can simply be concatenated.

Output, per capture:
    <out>/<capture_id>.ply   binary little-endian; double x,y,z (UTM coordinates
                             are ~6e5 / 3e6 m, float32 would round to ~0.1 m),
                             uchar rgb, ushort reality_idx, uint observation_count,
                             uint source_frame_count
    <out>/<capture_id>.json  manifest: project/structure, the realities in
                             reality_idx order, per-reality point counts and tm.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from align.io.ply import _parse_header  # noqa: E402

OUT_DTYPE = np.dtype([
    ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("reality_idx", "<u2"),
    ("observation_count", "<u4"), ("source_frame_count", "<u4"),
])
PLY_TYPE = {"f8": "double", "u1": "uchar", "u2": "ushort", "u4": "uint"}


def read_raw(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        n, dt, header_len = _parse_header(fh)
    expected = header_len + n * dt.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: size {path.stat().st_size} != header-implied {expected}")
    return np.fromfile(path, dtype=dt, count=n, offset=header_len)


def load_tm(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text())
    return np.asarray(raw["tm"], dtype=np.float64).reshape(4, 4)


def transform(raw: np.ndarray, tm: np.ndarray, idx: int) -> np.ndarray:
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
    xyz = xyz @ tm[:3, :3].T + tm[:3, 3]
    out = np.zeros(len(raw), dtype=OUT_DTYPE)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    names = raw.dtype.names
    for c in ("red", "green", "blue"):
        if c in names:
            out[c] = raw[c]
    out["reality_idx"] = idx
    for c in ("observation_count", "source_frame_count"):
        if c in names:
            out[c] = raw[c]
    return out


def write_ply(path: Path, pts: np.ndarray, comments: list[str]) -> None:
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {c}" for c in comments]
    lines.append(f"element vertex {len(pts)}")
    for name, (dt, _) in OUT_DTYPE.fields.items():
        lines.append(f"property {PLY_TYPE[dt.str.lstrip('<|')]} {name}")
    lines.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode())
        pts.tofile(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--force", action="store_true", help="rewrite existing outputs")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="")))
    by_capture: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("panovggt_point_cloud_path") and r.get("tm_path"):
            by_capture[r["capture_id"]].append(r)
        else:
            print(f"skip {r['reality_id']}: no point cloud or tm path", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)

    for cid, rs in sorted(by_capture.items()):
        ply_out = args.out / f"{cid}.ply"
        if ply_out.exists() and not args.force:
            print(f"{cid}: exists, skipping")
            continue
        rs = sorted(rs, key=lambda r: r["reality_id"])
        parts, manifest = [], []
        for idx, r in enumerate(rs):
            tm = load_tm(Path(r["tm_path"]))
            raw = read_raw(Path(r["panovggt_point_cloud_path"]))
            parts.append(transform(raw, tm, idx))
            manifest.append({"reality_idx": idx, "reality_id": r["reality_id"],
                             "n_points": int(len(raw)),
                             "point_cloud_path": r["panovggt_point_cloud_path"],
                             "tm_path": r["tm_path"], "tm": tm.reshape(-1).tolist()})
        pts = np.concatenate(parts)
        write_ply(ply_out, pts, [f"capture_id {cid}",
                                 f"project_id {rs[0]['project_id']}",
                                 f"structure_id {rs[0]['structure_id']}",
                                 "frame UTM (tm applied, offset not applied)"])
        (args.out / f"{cid}.json").write_text(json.dumps({
            "capture_id": cid, "project_id": rs[0]["project_id"],
            "structure_id": rs[0]["structure_id"], "source_csv": str(args.csv),
            "n_points": int(len(pts)), "realities": manifest}, indent=1))
        print(f"{cid}: {len(rs):3d} realities, {len(pts):>10,d} points -> {ply_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
