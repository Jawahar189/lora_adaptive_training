"""The predicted cameras, drawn on the architectural sheet itself.

`viz_prior` draws the solver's own world -- wall bands and reference rasters --
which is the right picture for asking why a match landed where it did. This is
the other picture: the drawing an operator already knows, with the walk on top,
for asking whether the answer is right at all. A placement that is thirty metres
out or in the wrong wing is obvious here and invisible in a raster of walls.

Ground truth is drawn beneath in grey when the caller passes it and simply left
out when it does not, so the same panel serves a scored run and a deployment run
that never opened the answer.

The sheet is often enormous -- some are 11M pixels -- so it is downsampled once
for display and every coordinate is scaled by the same factor. Nothing is
recomputed per point.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from align import viz as V
from align.frames import target_to_paper

Image.MAX_IMAGE_PIXELS = None

PRED = V.RED             # this capture, placed
GT = (110, 104, 98)      # the truth: a warm grey, so it reads against the plan
START = (43, 93, 138)    # where the walk begins


def _sheet(path, width: int):
    """The floormap, shrunk for display, with the factor used to shrink it."""
    with Image.open(path) as im:
        img = im.convert("RGB")
    if img.width <= width:
        return img, 1.0
    scale = width / img.width
    return img.resize((width, max(1, round(img.height * scale))),
                      Image.LANCZOS), scale


def _pixels(target_xyz, structure, scale: float) -> np.ndarray:
    """Target-frame points onto the displayed sheet."""
    paper = target_to_paper(np.asarray(target_xyz, dtype=np.float64),
                            structure.design_tm)[:, :2]
    return structure.geometry().paper_to_pixel(paper) * scale


def render(case, images: dict, out_path, gt_target_xyz=None,
           width: int = 1800, subtitle: str = "") -> Path:
    """Draw one reality's predicted cameras on its floormap and save it.

    `images` is the per-image record this run generated -- positions already in
    the target frame -- so the picture is of the file that was written, not of
    an intermediate that resembles it.
    """
    base, scale = _sheet(case.structure.sheet_path, width)
    pred = np.array([v["position"] for v in images.values()], dtype=np.float64)
    if not len(pred):
        raise ValueError(f"{case.reality_id}: no predicted positions to draw")

    if gt_target_xyz is not None and len(gt_target_xyz):
        base = V.draw_path(base, _pixels(gt_target_xyz, case.structure, scale),
                           color=GT, width=2)

    px = _pixels(pred, case.structure, scale)
    base = V.draw_path(base, px, color=PRED, width=3, dot=2)

    # The first camera, marked distinctly: a walk that is right but reversed
    # looks identical to one that is right until you can see which end is which.
    from PIL import ImageDraw
    d = ImageDraw.Draw(base)
    x, y = px[0]
    d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=START,
              outline=(255, 255, 255), width=2)

    what = f"{len(px)} cameras"
    if gt_target_xyz is not None and len(gt_target_xyz):
        what += ", ground truth in grey"
    return V.save(V.panel(base, case.reality_id,
                          subtitle or f"{what} on {case.structure.key}",
                          width=width),
                  out_path)
