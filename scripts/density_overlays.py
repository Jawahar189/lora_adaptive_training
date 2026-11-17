#!/usr/bin/env python3
"""One picture per capture: its density map laid over the floor plan.

The pose is the ground truth from the cache, so these say whether the truth
itself is right -- a correct one draws the cloud's walls on top of the
drawing's, a wrong one draws them beside. Written as <capture_id>.png, one per
capture, so a folder of them can be flicked through.

A capture id covers several realities (31 of 49 in `100_prod_current.csv` do),
each a separately reconstructed part of the same visit. They are merged here:
every reality's band points go through its own ground-truth similarity into
paper inches, and the density is counted once over all of them. So the picture
is the whole capture on its drawing, not one split of it. Each reality is drawn
in its own colour, named in a legend under the caption, so one whose truth is
off shows as a coloured copy beside the others.

    .venv/bin/python scripts/density_overlays.py
    .venv/bin/python scripts/density_overlays.py --csv datasets/100_prod_current.csv \
        --out out/bevaligner/density_overlays --limit 20
"""
from __future__ import annotations

import argparse
import functools
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevaligner.config import REPO_ROOT                       # noqa: E402
from bevaligner.data import density                           # noqa: E402
from bevaligner.data.floorplan import patch_valid_mask        # noqa: F401,E402
from bevaligner.data.dataset import CaptureRecord             # noqa: E402
from bevaligner.data.store import CaptureStore                # noqa: E402
from bevaligner.utils import affine, io                       # noqa: E402
from bevaligner.utils.gravity import BAND_HI, BAND_LO         # noqa: E402
from bevaligner.viz import render_density_overlay             # noqa: E402


@functools.lru_cache(maxsize=1)
def _sheet(path: str):
    """Sheets are 70 megapixels; captures are processed by structure so one is
    open at a time."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def paper_points(case, store):
    """One reality's wall band, in paper inches through its ground-truth pose."""
    result = store.capture_result(case.reality_id)
    if result is None or not result.ok:
        return None, (result.status if result else "not cached"), None
    if not store.has_prepared_sheet(case.structure.key):
        return None, "no sheet", None
    record = CaptureRecord.load(store.paths.capture_npz(case.reality_id),
                                result.viewport_index,
                                store.sheet_index(case.structure.key))
    band = ((record.band_relative_height > BAND_LO)
            & (record.band_relative_height < BAND_HI))
    points = record.band_xy[band]
    if len(points) < 200:
        return None, "thin band", None
    return affine.apply(record.gt_sim2_affine, points), result.residual_m, record


def sample_correspondence_targets(density_map, density_to_sheet, count: int, seed: int = 0):
    """Where the training targets land, in sheet pixels.

    The same rule the dataset uses: queries drawn from the density in proportion
    to mass, with sub-pixel jitter, pushed through the chain. Sampled here from
    the capture's merged map rather than one reality's, which changes the
    distribution not at all and the frame only by the similarity already applied.
    """
    if count <= 0:
        return None
    rng = np.random.default_rng(seed)
    queries = density.sample_query_pixels(density_map.image, count, rng)
    return affine.apply(density_to_sheet, queries)


# One colour per reality, in order. No blue: the correspondence targets are.
REALITY_COLOURS = [(200, 30, 30), (30, 150, 40), (235, 125, 0), (150, 40, 190),
                   (0, 155, 155), (185, 150, 0), (220, 60, 150), (115, 70, 30)]


def overlay_for(cases, store, resolution: int, long_side: int, correspondences: int,
                correspondence_radius: float = 4.5):
    """Every usable reality of one capture, on one drawing.

    The realities are counted into a single density map, which frames the
    picture and is where the correspondence targets are sampled. Each reality
    is then drawn from its own map on that same grid, in its own colour, so a
    reality whose truth is off stands out from the rest.
    """
    parts, residuals, record, reality_ids = [], [], None, []
    reasons = []
    for case in cases:
        points, detail, loaded = paper_points(case, store)
        if points is None:
            reasons.append(detail)
            continue
        parts.append(points)
        residuals.append(detail)
        reality_ids.append(case.reality_id)
        record = record or loaded
    if not parts:
        return None, (reasons[0] if reasons else "no usable reality")
    paper_xy = np.vstack(parts)
    density_map = density.render_density_map(paper_xy, resolution)
    density_to_sheet = affine.chain(density_map.density_px_to_world_xy,
                                    record.paper_to_sheet_px)
    structure = cases[0].structure
    caption = (f"{cases[0].capture_id}  {structure.key}  "
               f"{len(parts)}/{len(cases)} realities  "
               f"GT resid max {max(residuals):.3f} m  "
               f"{record.metres_per_inch:.2f} m/inch")
    targets = sample_correspondence_targets(density_map, density_to_sheet, correspondences)
    if targets is not None:
        caption += f"  {len(targets)} correspondence targets"
    colours = [REALITY_COLOURS[i % len(REALITY_COLOURS)] for i in range(len(parts))]
    layers = [(density.render_density_map(points, resolution,
                                          extent=density_map.extent).image, colour)
              for points, colour in zip(parts, colours)]
    legend = [(f"{reality_id}  {len(points)} pts  resid {residual:.3f} m", colour)
              for reality_id, points, residual, colour
              in zip(reality_ids, parts, residuals, colours)]
    image = render_density_overlay(_sheet(str(structure.sheet_path)), density_map.image,
                                   density_to_sheet, long_side=long_side,
                                   correspondence_px=targets,
                                   correspondence_radius=correspondence_radius,
                                   caption=caption, density_layers=layers, legend=legend)
    return image, "ok"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", default=["datasets/100_prod_current.csv"])
    parser.add_argument("--out", default="out/bevaligner/density_overlays")
    parser.add_argument("--cache-dir", default="cache/bevaligner")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N captures")
    parser.add_argument("--realities", nargs="*", default=None)
    parser.add_argument("--resolution", type=int, default=1024,
                        help="density map side, in pixels, before it is warped")
    parser.add_argument("--long-side", type=int, default=1600, help="output long side")
    parser.add_argument("--correspondences", type=int, default=250, metavar="N",
                        help="draw N sampled training targets on the overlay (0 = none)")
    parser.add_argument("--correspondence-radius", type=float, default=4.5,
                        help="radius of a target dot, in output pixels")
    parser.add_argument("--force", action="store_true", help="redraw ones already written")
    args = parser.parse_args(argv)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    store = CaptureStore(REPO_ROOT / args.cache_dir)
    cases = io.load_cases([REPO_ROOT / path for path in args.csv])
    if args.realities:
        wanted = set(args.realities)
        cases = [c for c in cases if c.reality_id in wanted]
    # grouped by capture, and by structure within it, so one sheet is open at a time
    by_capture: dict[str, list] = {}
    for case in cases:
        by_capture.setdefault(case.capture_id, []).append(case)
    captures = sorted(by_capture.items(), key=lambda item: (item[1][0].structure.key, item[0]))
    if args.limit:
        captures = captures[:args.limit]

    written, skipped = 0, {}
    for position, (capture_id, capture_cases) in enumerate(captures, 1):
        path = out_dir / f"{capture_id}.png"
        if path.exists() and not args.force:
            skipped["already drawn"] = skipped.get("already drawn", 0) + 1
            continue
        try:
            image, status = overlay_for(capture_cases, store, args.resolution,
                                        args.long_side, args.correspondences,
                                        args.correspondence_radius)
        except Exception as error:                                     # noqa: BLE001
            image, status = None, f"{type(error).__name__}: {error}"
        if image is None:
            skipped[status] = skipped.get(status, 0) + 1
            continue
        image.save(path)
        written += 1
        if written % 10 == 0 or position == len(captures):
            print(f"[{position}/{len(captures)}] {written} written", flush=True)
    print(f"{written} overlays ({len(captures)} captures) in {out_dir}")
    for reason, count in sorted(skipped.items(), key=lambda item: -item[1]):
        print(f"  skipped {count:3d}  {reason}")


if __name__ == "__main__":
    main()
