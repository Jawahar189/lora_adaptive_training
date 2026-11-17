"""Look at the pairs before training on them, and check the shortcut they rest on.

Two jobs, both cheap, both worth doing before a GPU is committed.

PICTURES. Four panels per row, through the same code the DataModule uses:

    whole prior | the target window | the query | the two overlaid at truth

The overlay is the one to read. Red is the prior, green the current capture at
its ground-truth pose; where they agree it is yellow. A pair whose overlay is
red beside green rather than yellow on top of yellow is a pair the matcher
cannot learn from, and the overlap gate should have caught it -- if it did not,
the gate is wrong, not the picture.

THE CROSS-CHECK (`--cross-check`). The learned path takes a prior split's
transform from its cached `gt_sim2`, which was fitted by Umeyama on its cameras.
The classical path reads `tm.json` through `align.priors.PriorReference`, which
is guarded. The two should agree; this measures by how much, per split, and is
the evidence that the cheap route is the same route.

    .venv/bin/python scripts/probe_prior_pairs.py --limit 12 --cross-check
    .venv/bin/python scripts/probe_prior_pairs.py --realities RLT... --out out/pairs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from align.io import datasets as datasets_mod                      # noqa: E402
from align.io.cases import PairedIndex                             # noqa: E402
from align.io.pairing import prior_reference_for                   # noqa: E402
from bevaligner.config import Config, REPO_ROOT, load_config     # noqa: E402
from bevaligner.data import prior_bev                            # noqa: E402
from bevaligner.data.density import render_density_map, square_extent  # noqa: E402
from bevaligner.data.datamodule import build_capture_store       # noqa: E402
from bevaligner.utils import affine                              # noqa: E402


def panel(rgb01: np.ndarray, caption: str, size: int, strip: int = 34) -> Image.Image:
    tile = Image.new("RGB", (size, size + strip), (12, 12, 16))
    tile.paste(Image.fromarray((np.clip(rgb01, 0, 1) * 255).astype(np.uint8)), (0, 0))
    draw = ImageDraw.Draw(tile)
    for i, line in enumerate(caption.split("\n")[:2]):
        draw.text((6, size + 3 + i * 14), line, fill=(215, 215, 225))
    return tile


def grey(image: np.ndarray) -> np.ndarray:
    return np.stack([image] * 3, axis=-1)


def figure(path: Path, title: str, record, pool, data_config, size: int = 560) -> None:
    whole = prior_bev.render_target(pool, pool.extent, size, None, data_config)
    window = prior_bev.choose_window(pool, record.anchor_paper_xy, None, data_config,
                                     size, 16)
    target = prior_bev.render_target(pool, window, size, None, data_config)
    if whole is None or target is None:
        return
    # the query, rendered into the SAME window so the overlay is at truth
    query_paper = affine.apply(record.gt_sim2_affine, record.band_xy)
    query = render_density_map(query_paper, size, extent=target.window, gamma=0.5)

    walk = affine.apply(whole.paper_to_crop, record.camera_paper_xy)
    keep = ((walk[:, 0] >= 0) & (walk[:, 0] < size)
            & (walk[:, 1] >= 0) & (walk[:, 1] < size))
    first = grey(whole.image) * np.array([1.0, 0.45, 0.30])
    if keep.any():
        track = np.zeros((size, size), dtype=np.float32)
        np.add.at(track, (walk[keep][:, 1].astype(int), walk[keep][:, 0].astype(int)), 1.0)
        first[..., 1] = np.maximum(first[..., 1], np.clip(track, 0, 1) * 0.95)

    overlay = np.zeros((size, size, 3), dtype=np.float32)
    overlay[..., 0] = target.image                     # prior   -> red
    overlay[..., 1] = query.image                      # current -> green

    mpi = record.metres_per_inch
    tiles = [
        panel(first, f"whole merged prior, {pool.span * mpi:,.0f} m across\n"
                     f"green = the walk at GT ({keep.mean():.0%} inside)", size),
        panel(grey(target.image),
              f"TARGET: the window, {(window[2] - window[0]) * mpi:,.0f} m across\n"
              f"{target.valid_mask.mean():.0%} of it is valid", size),
        panel(grey(query.image), "QUERY: the current walk, same window\n"
                                 "(training renders it in its own square)", size),
        panel(overlay, "overlay at ground truth\n"
                       "red prior, green current, yellow = agree", size),
    ]
    gap, top = 10, 24
    sheet = Image.new("RGB", (4 * size + 5 * gap, size + 34 + top + gap), (12, 12, 16))
    ImageDraw.Draw(sheet).text((gap, 6), title, fill=(245, 245, 250))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (gap + i * (size + gap), top))
    sheet.save(path)


def cross_check(case, index, record, pool) -> dict:
    """The cached `gt_sim2` route against the guarded `tm.json` route.

    Both should put a prior split's cloud in the same paper inches. The number
    reported is the median distance between the two, per split, in metres --
    the evidence that reading the cache instead of `align.priors` costs nothing.
    """
    reference = prior_reference_for(case, index)
    if reference is None:
        return {}
    out = {}
    for split in pool.splits:
        try:
            guarded = reference.trajectory(split.reality_id).positions[:, :2]
        except Exception as error:                                  # noqa: BLE001
            out[split.reality_id] = f"{type(error).__name__}"
            continue
        cached = split.camera_paper_xy
        if len(guarded) != len(cached):
            n = min(len(guarded), len(cached))
            guarded, cached = guarded[:n], cached[:n]
        if not len(guarded):
            continue
        delta = np.linalg.norm(guarded - cached, axis=1)
        out[split.reality_id] = float(np.median(delta) * pool.metres_per_inch) \
            if np.isfinite(pool.metres_per_inch) else float(np.median(delta))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="bevaligner/configs/prior_bev.yaml")
    ap.add_argument("--realities", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--cache-dir", default=None,
                    help="override data.cache_dir, e.g. a smoke cache")
    ap.add_argument("--cross-check", action="store_true")
    ap.add_argument("--out", default="out/prior_pairs")
    args = ap.parse_args(argv)

    config = load_config(REPO_ROOT / args.config) if args.config else Config()
    if args.cache_dir:
        config.data.cache_dir = args.cache_dir
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = PairedIndex.load(datasets_mod.resolve(
        [str(REPO_ROOT / p) for p in config.data.csv_files],
        [str(REPO_ROOT / p) for p in config.data.prior_csv_files], name="probe"))
    store = build_capture_store(config)
    store.index = index
    print(f"index: {index.coverage()}")

    cases = list(index.current)
    if args.realities:
        wanted = set(args.realities)
        cases = [c for c in cases if c.reality_id in wanted]

    rows, made = [], 0
    for case in cases:
        if made >= args.limit:
            break
        result = store.capture_result(case.reality_id)
        if result is None or not result.prepared:
            continue
        records = store.capture_records([case], {case.reality_id: result})
        if not records:
            continue
        record = records[0]
        pool = store.pool_for(record)
        if pool is None:
            continue
        name = (f"{made:02d}_{'geo' if case.structure.georeferenced else 'nogeo'}"
                f"_ov{record.overlap:.2f}_{case.reality_id}.png")
        title = (f"{case.structure.key}  {case.reality_id}   "
                 f"overlap {record.overlap:.0%}   {len(pool.splits)} prior splits   "
                 f"scale {record.metres_per_inch_source} "
                 f"({record.metres_per_inch:.2f} m/inch)"
                 + ("   SAME DAY" if record.same_day else ""))
        figure(out / name, title, record, pool, config.data, args.size)
        row = dict(reality_id=case.reality_id, overlap=round(record.overlap, 3),
                   splits=len(pool.splits), span_m=round(pool.span * record.metres_per_inch),
                   scale=record.metres_per_inch_source, same_day=record.same_day,
                   figure=name)
        if args.cross_check:
            errors = cross_check(case, index, record, pool)
            numeric = [v for v in errors.values() if isinstance(v, float)]
            row["tm_vs_cached_m"] = (round(float(np.median(numeric)), 6)
                                     if numeric else None)
        rows.append(row)
        made += 1
        print(f"[{made}] {name}")

    if not rows:
        print("nothing to draw: is the cache built for these rows?")
        return
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "summary.csv", index=False)
    print("\n" + frame.to_string(index=False))
    if args.cross_check and "tm_vs_cached_m" in frame:
        values = frame["tm_vs_cached_m"].dropna()
        if len(values):
            print(f"\ncached gt_sim2 vs guarded tm.json, metres: "
                  f"median {values.median():.2e}  max {values.max():.2e}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
