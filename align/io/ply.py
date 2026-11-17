"""Binary PLY reader driven by the file's own header.

Two schemas occur in this dataset (checked across all 132 clouds):
  131 files: float x,y,z + uchar rgb + uint observation_count, source_frame_count,
             source_window_count, quarantined_count + float mean_confidence
    1 file : double x,y,z + uchar rgb only
so the dtype is always built from the header, never assumed.

Note on the per-point quality fields: `mean_confidence` is 1.0 and
`quarantined_count` is 0 on every point of every cloud -- both are useless.
`observation_count` (p50 12, max 2336) and `source_frame_count` (p50 3, max 33)
do carry real dynamic range.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NP_TYPE = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


@dataclass
class PointCloud:
    xyz: np.ndarray            # (N, 3) float64
    obs: np.ndarray | None     # (N,) observation_count, or None
    sframes: np.ndarray | None  # (N,) source_frame_count, or None
    path: Path

    def __len__(self) -> int:
        return len(self.xyz)


def _parse_header(fh) -> tuple[int, np.dtype, int]:
    """Return (n_vertices, vertex dtype, header byte length)."""
    magic = fh.readline()
    if magic.strip() != b"ply":
        raise ValueError("not a PLY file")
    n_bytes = len(magic)
    fmt = None
    n_vertices = None
    props: list[tuple[str, str]] = []
    in_vertex = False
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("unterminated PLY header")
        n_bytes += len(line)
        tok = line.split()
        if not tok:
            continue
        key = tok[0]
        if key == b"format":
            fmt = tok[1].decode()
        elif key == b"element":
            in_vertex = tok[1] == b"vertex"
            if in_vertex:
                n_vertices = int(tok[2])
        elif key == b"property" and in_vertex:
            if tok[1] == b"list":
                raise ValueError("list properties are not supported on vertices")
            props.append((tok[2].decode(), tok[1].decode()))
        elif key == b"end_header":
            break
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt!r}")
    if n_vertices is None:
        raise ValueError("no vertex element in header")
    dt = np.dtype([(name, _NP_TYPE[ptype]) for name, ptype in props])
    return n_vertices, dt, n_bytes


def read_ply(path, stride: int = 1, min_source_frames: int = 0) -> PointCloud:
    """Read a binary PLY.

    stride            -- keep every Nth point (cheap uniform subsample)
    min_source_frames -- drop points seen by fewer than this many source frames;
                         ignored when the schema lacks the field.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        n, dt, header_len = _parse_header(fh)
        expected = header_len + n * dt.itemsize
        actual = path.stat().st_size
        if actual != expected:
            raise ValueError(f"{path}: size {actual} != header-implied {expected}")
        arr = np.memmap(path, dtype=dt, mode="r", offset=header_len, shape=(n,))
        if stride > 1:
            arr = arr[::stride]
        names = dt.names
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        obs = np.asarray(arr["observation_count"]) if "observation_count" in names else None
        sfr = np.asarray(arr["source_frame_count"]) if "source_frame_count" in names else None

    if min_source_frames > 1 and sfr is not None:
        keep = sfr >= min_source_frames
        # Never filter away everything; fall back to unfiltered if too aggressive.
        if keep.sum() >= max(100, 0.02 * len(keep)):
            xyz, obs, sfr = xyz[keep], (obs[keep] if obs is not None else None), sfr[keep]

    return PointCloud(xyz=xyz, obs=obs, sframes=sfr, path=path)


def ply_schema(path) -> tuple[str, ...]:
    """Property names of the vertex element, without reading the body."""
    with open(path, "rb") as fh:
        _, dt, _ = _parse_header(fh)
    return dt.names
