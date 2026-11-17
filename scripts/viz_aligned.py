#!/usr/bin/env python
"""One plot per structure: every solved reality on its floormap.

The per-split numbers cannot answer the question this picture answers. Two
splits both 15 m out in the SAME direction is a structure-level offset -- the
sheet scale, or the viewport -- and two splits 15 m out in opposite directions
is the search failing independently twice. Those need different fixes and look
identical in a table.

Each reality's walk is drawn in its own colour with ground truth beneath in
grey, so the offset between them is visible per reality; seeds are marked
distinctly from chained splits, so a capture that drifted along its chain reads
as drift rather than as scattered error.

    python scripts/viz_aligned.py --predictions out/predictions.jsonl \
        --csv SELECTED.csv --out out/aligned
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import align.viz as V
from align.frames import target_to_paper
from align.groundtruth import gt_trajectory, load_tm
from align.io.cases import CaseIndex
from align.pipeline import RealityPipeline

WIDTH = 1800
# Ground truth has to read against the wall map, which is itself grey. A darker
# warm grey separates from the plan's cool grey without competing with the
# reality colours for attention.
GT_GREY = (110, 104, 98)


def palette(n: int):
    """Distinct hues, held at one saturation so no reality reads as louder."""
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.72, 0.80))
            for i in range(n)]


def wrapped_legend(entries, width: int, pad: int = 12, row_h: int = 26):
    """A legend that WRAPS. `viz.legend` lays out on one row, which silently
    runs a 16-reality structure off the right edge."""
    from PIL import ImageFont
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        pass
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    rows, cur, x = [], [], pad
    for label, color in entries:
        w = int(probe.textlength(label, font=font)) + 38
        if cur and x + w > width - pad:
            rows.append(cur)
            cur, x = [], pad
        cur.append((label, color, x))
        x += w
    if cur:
        rows.append(cur)

    out = Image.new("RGB", (width, row_h * len(rows) + pad), V.PAPER)
    d = ImageDraw.Draw(out)
    for r, row in enumerate(rows):
        y = pad // 2 + r * row_h + row_h // 2
        for label, color, x in row:
            d.rectangle([x, y - 5, x + 11, y + 6], fill=color)
            d.text((x + 17, y - 7), label, font=font, fill=V.INK)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="out/predictions.jsonl")
    ap.add_argument("--metrics", default=None,
                    help="defaults to solve_metrics.jsonl beside --predictions")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="out/aligned")
    args = ap.parse_args()

    cases, seen = [], set()
    for name in args.csv:
        for c in CaseIndex.load(name).cases:
            if c.reality_id not in seen:
                seen.add(c.reality_id)
                cases.append(c)
    by_id = {c.reality_id: c for c in cases}

    preds = {}
    with open(args.predictions) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                preds[r["reality_id"]] = np.asarray(r["tm"]).reshape(4, 4)

    metrics = {}
    mpath = Path(args.metrics or
                 Path(args.predictions).with_name("solve_metrics.jsonl"))
    if mpath.exists():
        with open(mpath) as fh:
            for line in fh:
                if line.strip():
                    m = json.loads(line)
                    metrics[m["reality_id"]] = m

    by_structure = defaultdict(list)
    for rid in preds:
        if rid in by_id:
            by_structure[by_id[rid].structure.key].append(rid)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(preds)} predictions over {len(by_structure)} structures")

    for skey, rids in sorted(by_structure.items()):
        rids.sort()
        struct = by_id[rids[0]].structure
        feats = None
        try:
            feats = RealityPipeline(by_id[rids[0]]).sheet()
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {skey}: sheet unavailable ({exc!r})"[:110])
            continue
        geom = struct.geometry()
        scale = WIDTH / feats.wall.shape[1]
        base = V.mask_rgb(V.downsample_mask(feats.wall, WIDTH),
                          fg=(150, 158, 168)).convert("RGB")

        colors = palette(len(rids))
        entries, drawn = [], 0
        for color, rid in zip(colors, rids):
            case = by_id[rid]
            try:
                p = RealityPipeline(case)
                sfm = np.asarray(p.trajectory().positions, dtype=np.float64)
                tm = preds[rid]
                pred = sfm @ tm[:3, :3].T + tm[:3, 3]
                pred_px = geom.paper_to_pixel(
                    target_to_paper(pred, struct.design_tm)[:, :2]) * scale
                gt_px = geom.paper_to_pixel(target_to_paper(
                    gt_trajectory(case.rlt_dir).positions,
                    struct.design_tm)[:, :2]) * scale
            except Exception as exc:                              # noqa: BLE001
                print(f"  {skey}/{rid}: {exc!r}"[:110])
                continue

            base = V.draw_path(base, gt_px, color=GT_GREY, width=2)
            base = V.draw_path(base, pred_px, color=color, width=3)

            chain = (metrics.get(rid) or {}).get("chain") or {}
            how = chain.get("how", "")
            seed = how.startswith("seed") or how.startswith("independent")
            if len(pred_px):
                d = ImageDraw.Draw(base)
                x, y = pred_px[0]
                r = 9 if seed else 5
                # A seed is where the chain STARTS; everything else inherited a
                # pose from a neighbour, so drift along a capture is readable
                # as distance from the one filled marker.
                if seed:
                    d.ellipse([x - r, y - r, x + r, y + r], fill=color,
                              outline=(255, 255, 255), width=2)
                else:
                    d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)

            m = metrics.get(rid) or {}
            tpr = m.get("tpr_1m")
            med = (m.get("utm") or m.get("paper") or {}).get("median")
            unit = "m" if m.get("utm") else "in"
            entries.append((f"{rid}  tpr@1m {tpr:.2f}" if tpr is not None
                            else f"{rid}  tpr@1m n/a", color))
            if med is not None:
                entries[-1] = (entries[-1][0] + f"  median {med:.1f} {unit}", color)
            drawn += 1

        if not drawn:
            continue
        entries.append(("ground truth", GT_GREY))
        leg = wrapped_legend(entries, width=base.width)
        img = V.grid([V.panel(base, skey,
                              f"{drawn} realities · filled marker = chain seed"),
                      leg], cols=1)
        path = V.save(img, out_dir / f"{skey.replace('/', '_')}.png")
        print(f"  {path}  ({drawn} realities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
