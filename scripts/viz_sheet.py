"""Render the extracted wall map with the ground-truth trajectory on top."""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
Image.MAX_IMAGE_PIXELS = None

from align import sheet as sheetmod
from align.frames import target_to_paper
from align.groundtruth import gt_trajectory
from align.io.cases import CaseIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--out", default="out/viz")
    ap.add_argument("--width", type=int, default=1500)
    args = ap.parse_args()

    index = CaseIndex.load_all()
    cases = [c for c in index if args.structure in c.structure.key]
    if not cases:
        raise SystemExit(f"no cases for {args.structure}")
    st = cases[0].structure
    f = sheetmod.extract(st.sheet_path, st.paper_w, st.paper_h, verbose=True)

    canvas = np.full(f.wall.shape + (3,), 255, dtype=np.uint8)
    canvas[f.ink] = (215, 215, 215)        # everything that was dropped, faint
    canvas[f.wall] = (0, 0, 0)             # surviving wall line-work
    img = Image.fromarray(canvas)
    dr = ImageDraw.Draw(img)

    for v in f.viewports:
        dr.rectangle([v.x0, v.y0, v.x1, v.y1], outline=(0, 140, 255),
                     width=max(f.wall.shape[0] // 400, 3))

    g = st.geometry()
    colors = [(220, 0, 0), (0, 160, 0), (190, 0, 190), (255, 140, 0)]
    for i, c in enumerate(cases):
        xy = target_to_paper(gt_trajectory(c.rlt_dir).positions, st.design_tm)[:, :2]
        px = g.paper_to_pixel(xy)
        r = max(f.wall.shape[0] // 300, 4)
        for x, y in px:
            dr.ellipse([x - r, y - r, x + r, y + r], fill=colors[i % len(colors)])

    Path(args.out).mkdir(parents=True, exist_ok=True)
    img.thumbnail((args.width, args.width))
    dest = Path(args.out) / f"{st.structure_id}_wall.png"
    img.save(dest)
    print(f"wrote {dest}  ({len(cases)} trajectories)")


if __name__ == "__main__":
    main()
