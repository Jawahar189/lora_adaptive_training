"""Architectural sheet -> wall map.

A `floormap.png` is a full CAD sheet, not a clean floor plan: title block,
notes paragraphs, dimension strings, leader lines, grid bubbles, hatching, and
frequently two or more plan viewports of which only one matches the capture.
Correlating against raw ink finds the densest region -- the title block -- every
time, so the cleaning here is what makes matching possible at all.

Two rules govern the implementation:

  * Every threshold is in INCHES and converted with the sheet's own ppi.
    Measured ppi across the 37 structures is 87.9, 150, 200, 300 and 600, so a
    hardcoded pixel radius would be off by up to 7x.

  * Cleaning happens at FULL resolution, before any downsampling. At
    1/8"=1'-0" a 1/8-inch glyph is ~3 real metres, so on a metric working grid
    text strokes and walls are the same size and can no longer be separated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from align.frames import design_scale

Image.MAX_IMAGE_PIXELS = None

# Two kinds of threshold, and mixing them up is the easiest way to destroy a
# sheet. Text is drawn at a fixed size ON PAPER regardless of drawing scale, so
# glyph thresholds are in INCHES. Walls are sized in the WORLD, so wall length
# and thickness are in METRES and converted through the sheet's scale -- 0.5
# inch is 2.4 m at 1/16"=1'-0" but 0.6 m at 1/4"=1'-0".
GLYPH_MAX = 0.22           # inches; smaller components are text / tick marks
SYMBOL_MAX = 1.00          # inches; filled blobs this size are symbols
DENSITY_WIN = 1.20         # inches; window for the local-ink-density test
DENSITY_MAX = 0.35         # above this the region is schedule / dense hatch
TEXT_BLOCK_WIN = 1.00      # inches; window for counting glyph components
TEXT_BLOCK_COUNT = 25      # glyphs per square inch that marks a text panel
TEXT_BLOCK_LINE_MAX = 0.04 # ...but only where long line-work is this sparse
TEXT_HALO = 0.15           # inches; modest halo, just the rules boxing a panel
BORDER_RULE_FRAC = 0.45    # a row/col this full of ink is a drawing rule

MIN_LINE_RUN_M = 0.80      # metres; a stroke shorter than this is not a wall
WALL_PAIR_RUN_M = 1.00     # metres; both sides of a wall pair at least this long
WALL_GAP_MIN_M = 0.05      # metres; plausible wall thickness range
WALL_GAP_MAX_M = 0.60
DEFAULT_METRES_PER_INCH = 3.0   # used only when the sheet scale is unknown


@dataclass
class Viewport:
    x0: int
    y0: int
    x1: int
    y1: int
    wall_mass: float

    @property
    def area_in(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass
class SheetFeatures:
    ink: np.ndarray          # bool, full res: raw binarized ink
    wall: np.ndarray         # bool, full res: surviving wall line-work
    ppi: float
    paper_w: float
    paper_h: float
    viewports: list[Viewport]
    photo_backed: bool
    ink_fraction: float
    border: tuple[int, int, int, int] = (0, 0, 0, 0)   # x0, y0, x1, y1


def _border_box(ink: np.ndarray) -> tuple[int, int, int, int]:
    """The drawing area inside the sheet border.

    Sheet borders and the column-grid strip are full-length rules; everything
    outside the innermost such pair is frame, not building.
    """
    h, w = ink.shape
    rows = np.flatnonzero(ink.mean(axis=1) > BORDER_RULE_FRAC)
    cols = np.flatnonzero(ink.mean(axis=0) > BORDER_RULE_FRAC)
    y0 = int(rows[rows < h * 0.25].max()) if (rows < h * 0.25).any() else 0
    y1 = int(rows[rows > h * 0.75].min()) if (rows > h * 0.75).any() else h
    x0 = int(cols[cols < w * 0.25].max()) if (cols < w * 0.25).any() else 0
    x1 = int(cols[cols > w * 0.75].min()) if (cols > w * 0.75).any() else w
    if x1 - x0 < w * 0.3 or y1 - y0 < h * 0.3:
        return 0, 0, w, h
    return x0, y0, x1, y1


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask, with a fallback for sheets drawn in very light grey.

    STR1782849228036 has under 1% ink and its darkest tones are ~250, so a
    fixed threshold returns almost nothing.
    """
    ink = gray < 160
    if ink.mean() < 0.005:
        thr = np.percentile(gray, 0.5) + 20
        ink = gray < thr
    return ink


def _runs_at_least(mask: np.ndarray, min_px: int, axis: int) -> np.ndarray:
    """Keep pixels belonging to a run of `True` at least `min_px` long."""
    m = mask if axis == 1 else mask.T
    out = np.zeros_like(m)
    h, w = m.shape
    padded = np.zeros((h, w + 2), dtype=bool)
    padded[:, 1:-1] = m
    d = np.diff(padded.astype(np.int8), axis=1)
    for row in range(h):
        starts = np.flatnonzero(d[row] == 1)
        ends = np.flatnonzero(d[row] == -1)
        if len(starts) == 0:
            continue
        long = (ends - starts) >= min_px
        for s, e in zip(starts[long], ends[long]):
            out[row, s:e] = True
    return out if axis == 1 else out.T


def _drop_small_components(ink: np.ndarray, ppi: float) -> tuple[np.ndarray, np.ndarray]:
    """Split ink into (kept, glyph-like) by connected-component size."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    glyph_px = GLYPH_MAX * ppi
    symbol_px = SYMBOL_MAX * ppi
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    longest = np.maximum(w, h)
    fill = area / np.maximum(w * h, 1)
    small = longest < glyph_px
    blob = (fill > 0.55) & (longest < symbol_px)
    drop = small | blob
    drop[0] = False                        # label 0 is background
    drop_mask = drop[labels]
    return ink & ~drop_mask, ink & drop_mask


def _text_blocks(glyphs: np.ndarray, ppi: float,
                 long_ink: np.ndarray | None = None) -> np.ndarray:
    """Pure text panels: notes, schedules, the title block.

    Density of glyphs alone is not enough -- a floor plan is covered in room
    tags and dimension numerals, and masking everywhere they are dense deletes
    the building. A text PANEL is dense in glyphs *and* sparse in long
    line-work; a tagged floor plan is dense in both.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        glyphs.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(glyphs)
    win = max(int(TEXT_BLOCK_WIN * ppi), 8)
    small = np.zeros(glyphs.shape, dtype=np.float32)
    cy = np.clip(centroids[1:, 1].astype(int), 0, glyphs.shape[0] - 1)
    cx = np.clip(centroids[1:, 0].astype(int), 0, glyphs.shape[1] - 1)
    np.add.at(small, (cy, cx), 1.0)
    # components per square inch
    dens = cv2.boxFilter(small, -1, (win, win), normalize=False) / (TEXT_BLOCK_WIN ** 2)
    hot = dens > TEXT_BLOCK_COUNT
    if long_ink is not None:
        line_dens = cv2.boxFilter(long_ink.astype(np.float32), -1, (win, win))
        hot &= line_dens < TEXT_BLOCK_LINE_MAX
    return hot


def _double_line_walls(ink: np.ndarray, ppi: float, mpi: float) -> np.ndarray:
    """Walls are drawn as a PAIR of parallel lines a wall-thickness apart;
    dimension, leader, grid and match lines are single. This is the strongest
    single discriminator available on a CAD sheet.

    For each axis, find white gaps whose width is a plausible wall thickness and
    whose two bounding ink runs are both long, then mark those runs.
    """
    gap_lo = max(int(WALL_GAP_MIN_M / mpi * ppi), 1)
    gap_hi = max(int(WALL_GAP_MAX_M / mpi * ppi), gap_lo + 1)
    run_px = max(int(WALL_PAIR_RUN_M / mpi * ppi), 2)
    out = np.zeros_like(ink)
    for axis in (0, 1):
        long_runs = _runs_at_least(ink, run_px, axis=1 - axis)
        m = long_runs if axis == 0 else long_runs.T
        # A pixel is wall-like if a long run sits within [gap_lo, gap_hi] of it
        # on the opposite side across white space.
        near = np.zeros_like(m)
        for shift in range(gap_lo, gap_hi + 1):
            a = np.zeros_like(m)
            a[:, shift:] = m[:, :-shift]
            b = np.zeros_like(m)
            b[:, :-shift] = m[:, shift:]
            near |= (m & b) | (m & a)
        out |= near if axis == 0 else near.T
    return out


CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "sheet"


def prepare(structure, metres_per_inch: float | None = None,
            force: bool = False, verbose: bool = False) -> SheetFeatures:
    """Wall extraction for one structure, straight from a `Structure`.

    The stage entry point: `extract` and `extract_cached` take loose arguments
    because they predate the case index, and assembling five of them by hand at
    every call site is how they drift apart.

    `metres_per_inch` is exact from the design transform where the structure has
    one. Where it does not, the sheet's scale is otherwise unknown and every
    threshold in the extraction depends on it, so pass the value from
    `align.scales.SheetScales` rather than letting it fall back to a constant.
    """
    mpi = metres_per_inch
    if mpi is None:
        mpi = (design_scale(structure.design_tm)
               if structure.design_tm is not None else DEFAULT_METRES_PER_INCH)
    if force:
        dest = _cache_path(structure.key, mpi)
        dest.unlink(missing_ok=True)
    return extract_cached(structure.sheet_path, structure.paper_w,
                          structure.paper_h, structure.key, mpi, verbose)


def _cache_path(key: str, metres_per_inch: float | None) -> Path:
    mpi = float(metres_per_inch or DEFAULT_METRES_PER_INCH)
    return CACHE_DIR / f"{key.replace('/', '_')}_mpi{mpi:.2f}.npz"


def extract_cached(sheet_path, paper_w: float, paper_h: float, key: str,
                   metres_per_inch: float | None = None,
                   verbose: bool = False) -> SheetFeatures:
    """Cache wall extraction per structure: sheets take up to 42 s and are
    shared by as many as 19 splits."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _cache_path(key, metres_per_inch)
    if dest.exists():
        z = np.load(dest, allow_pickle=False)
        vp = [Viewport(int(a), int(b), int(c), int(d), float(m))
              for a, b, c, d, m in z["viewports"]] if len(z["viewports"]) else []
        return SheetFeatures(
            ink=np.unpackbits(z["ink"], count=int(z["n"])).astype(bool).reshape(
                tuple(z["shape"])),
            wall=np.unpackbits(z["wall"], count=int(z["n"])).astype(bool).reshape(
                tuple(z["shape"])),
            ppi=float(z["ppi"]), paper_w=float(z["paper_w"]),
            paper_h=float(z["paper_h"]), viewports=vp,
            photo_backed=bool(z["photo_backed"]),
            ink_fraction=float(z["ink_fraction"]),
            border=tuple(int(v) for v in z["border"]))
    f = extract(sheet_path, paper_w, paper_h, metres_per_inch, verbose=verbose)
    np.savez_compressed(
        dest, ink=np.packbits(f.ink), wall=np.packbits(f.wall),
        shape=np.array(f.ink.shape), n=f.ink.size, ppi=f.ppi,
        paper_w=f.paper_w, paper_h=f.paper_h, photo_backed=f.photo_backed,
        ink_fraction=f.ink_fraction, border=np.array(f.border),
        viewports=np.array([[v.x0, v.y0, v.x1, v.y1, v.wall_mass]
                            for v in f.viewports], dtype=np.float64).reshape(-1, 5))
    return f


def extract(sheet_path, paper_w: float, paper_h: float,
            metres_per_inch: float | None = None,
            verbose: bool = False) -> SheetFeatures:
    mpi = float(metres_per_inch or DEFAULT_METRES_PER_INCH)
    with Image.open(sheet_path) as im:
        gray = np.asarray(im.convert("L"))
        rgb_small = np.asarray(im.convert("RGB").resize(
            (max(im.width // 8, 1), max(im.height // 8, 1))), dtype=np.float32)
    ih, iw = gray.shape
    ppi = iw / paper_w

    ink = _binarize(gray)
    ink_fraction = float(ink.mean())

    # Photo-backed sheets (site plan over aerial imagery) have large smooth
    # mid-grey regions; pure line-work does not.
    g = rgb_small.mean(2)
    lv = cv2.boxFilter(g, -1, (9, 9))
    l2 = cv2.boxFilter(g * g, -1, (9, 9))
    std = np.sqrt(np.clip(l2 - lv * lv, 0, None))
    photo = float((((g > 60) & (g < 225)) & (std < 18)).mean())

    keep, glyphs = _drop_small_components(ink, ppi)

    run_px = max(int(MIN_LINE_RUN_M / mpi * ppi), 2)
    long_ink = _runs_at_least(keep, run_px, axis=1) | _runs_at_least(keep, run_px, axis=0)

    text = _text_blocks(glyphs, ppi, long_ink)
    # A modest halo removes the rules boxing a panel without eating the plan.
    halo = max(int(TEXT_HALO * ppi), 3)
    text = cv2.dilate(text.astype(np.uint8), np.ones((halo, halo), np.uint8)).astype(bool)

    dwin = max(int(DENSITY_WIN * ppi), 8)
    dens = cv2.boxFilter(keep.astype(np.float32), -1, (dwin, dwin))
    dense = dens > DENSITY_MAX

    wall = long_ink & ~text & ~dense
    pairs = _double_line_walls(wall, ppi, mpi)
    if pairs.sum() > 0.02 * wall.sum():
        wall = pairs           # prefer confirmed wall pairs when there are enough

    bx0, by0, bx1, by1 = _border_box(ink)
    frame = np.ones_like(wall)
    frame[by0:by1, bx0:bx1] = False
    wall &= ~frame

    feats = SheetFeatures(ink=ink, wall=wall, ppi=ppi, paper_w=paper_w,
                          paper_h=paper_h, viewports=[], photo_backed=photo > 0.10,
                          ink_fraction=ink_fraction, border=(bx0, by0, bx1, by1))
    feats.viewports = find_viewports(feats)
    if verbose:
        print(f"  ppi {ppi:.1f}  mpi {mpi:.2f}  ink {ink_fraction:.3%} -> wall "
              f"{wall.mean():.3%}  photo {photo:.3f}  viewports {len(feats.viewports)}")
    return feats


def resample_to(feats: SheetFeatures, geom, bounds, res: float,
                shape: tuple[int, int]) -> np.ndarray:
    """The sheet's wall map resampled onto a paper-frame raster grid.

    `bounds` is (x0, y0, x1, y1) in paper inches, `res` inches per pixel, and
    `shape` the target (rows, cols) with row 0 at MINIMUM paper y -- matching
    `segments.rasterize`, so a cloud raster and this line up index for index.

    Downsampling thin line-work, so pool with `any` rather than nearest: a wall
    one sheet-pixel wide must survive landing in a cell six pixels across.
    """
    x0, y0, x1, y1 = bounds
    ih, iw = feats.wall.shape
    p0 = geom.paper_to_pixel(np.array([[x0, y1]]))[0]      # top-left in pixels
    p1 = geom.paper_to_pixel(np.array([[x1, y0]]))[0]      # bottom-right
    u0 = np.floor(p0).astype(int)                          # what was asked for,
    u1 = np.ceil(p1).astype(int)                           # sheet edges ignored
    if u1[0] <= u0[0] or u1[1] <= u0[1]:
        return np.zeros(shape, bool)
    c0 = np.clip(u0, 0, [iw, ih])                          # what actually exists
    c1 = np.clip(u1, 0, [iw, ih])
    if c1[0] <= c0[0] or c1[1] <= c0[1]:
        return np.zeros(shape, bool)

    h, w = shape
    # Bounds routinely hang off the sheet -- a cloud placed near an edge asks
    # for paper that is not there. Resizing the CLIPPED crop to the full shape
    # would stretch the plan to fit, silently misregistering it by however much
    # was missing, so pad instead: the crop goes where it belongs and the rest
    # stays blank.
    fx, fy = w / (u1[0] - u0[0]), h / (u1[1] - u0[1])
    ox0, ox1 = int(round((c0[0] - u0[0]) * fx)), int(round((c1[0] - u0[0]) * fx))
    oy0, oy1 = int(round((c0[1] - u0[1]) * fy)), int(round((c1[1] - u0[1]) * fy))
    ox0, oy0 = max(ox0, 0), max(oy0, 0)
    ox1, oy1 = min(ox1, w), min(oy1, h)
    if ox1 <= ox0 or oy1 <= oy0:
        return np.zeros(shape, bool)

    crop = feats.wall[c0[1]:c1[1], c0[0]:c1[0]].astype(np.float32)
    pooled = cv2.resize(crop, (ox1 - ox0, oy1 - oy0), interpolation=cv2.INTER_AREA)
    out = np.zeros(shape, bool)
    # INTER_AREA averages; any non-zero cell contained wall ink.
    out[oy0:oy1, ox0:ox1] = pooled > 0
    return np.flipud(out)                                   # paper y is up


def find_viewports(feats: SheetFeatures, max_depth: int = 3,
                   min_area_in2: float = 4.0) -> list[Viewport]:
    """Recursive X-Y cut on wall mass -- the standard document-layout split,
    and exactly right for sheets laid out as whitespace-separated viewports."""
    ppi = feats.ppi
    cell = max(int(0.5 * ppi), 1)
    small = cv2.resize(feats.wall.astype(np.float32),
                       (max(feats.wall.shape[1] // cell, 1),
                        max(feats.wall.shape[0] // cell, 1)),
                       interpolation=cv2.INTER_AREA)
    gap_cells = max(int(0.8 * ppi / cell), 1)
    bx0, by0, bx1, by1 = feats.border
    if bx1 > bx0:
        m = np.zeros_like(small, dtype=bool)
        m[by0 // cell:by1 // cell, bx0 // cell:bx1 // cell] = True
        small = small * m

    def cut(x0, y0, x1, y1, depth):
        block = small[y0:y1, x0:x1]
        if block.size == 0 or depth >= max_depth:
            return [(x0, y0, x1, y1)]
        for axis in (0, 1):
            prof = block.sum(axis=1 - axis)
            empty = prof <= max(prof.max() * 0.06, 1e-9)
            runs, start = [], None
            for i, e in enumerate(empty):
                if e and start is None:
                    start = i
                elif not e and start is not None:
                    runs.append((start, i))
                    start = None
            if start is not None:
                runs.append((start, len(empty)))
            interior = [(a, b) for a, b in runs
                        if b - a >= gap_cells and a > 0 and b < len(empty)]
            if interior:
                a, b = max(interior, key=lambda r: r[1] - r[0])
                mid = (a + b) // 2
                if axis == 0:
                    return (cut(x0, y0, x1, y0 + mid, depth + 1)
                            + cut(x0, y0 + mid, x1, y1, depth + 1))
                return (cut(x0, y0, x0 + mid, y1, depth + 1)
                        + cut(x0 + mid, y0, x1, y1, depth + 1))
        return [(x0, y0, x1, y1)]

    boxes = cut(0, 0, small.shape[1], small.shape[0], 0)
    out = []
    for bx0, by0, bx1, by1 in boxes:
        sub = small[by0:by1, bx0:bx1]
        if sub.size == 0:
            continue
        ys, xs = np.nonzero(sub > 0)
        if len(xs) < 10:
            continue
        # tighten to the ink actually present
        x0 = (bx0 + xs.min()) * cell
        x1 = (bx0 + xs.max() + 1) * cell
        y0 = (by0 + ys.min()) * cell
        y1 = (by0 + ys.max() + 1) * cell
        w_in, h_in = (x1 - x0) / ppi, (y1 - y0) / ppi
        if w_in * h_in < min_area_in2 or max(w_in, h_in) / max(min(w_in, h_in), 1e-6) > 8:
            continue
        out.append(Viewport(x0, y0, x1, y1, float(sub.sum())))
    out.sort(key=lambda v: -v.wall_mass)
    out = out[:4]

    # Insurance: the union of all retained wall ink, always present. Without it
    # a sheet whose plan gets cut into pieces that each fail the area/aspect
    # filter would yield no viewport at all and be unsolvable for a mechanical
    # reason rather than a data one (STR1782849228036 did exactly this).
    ys, xs = np.nonzero(feats.wall)
    if len(xs):
        union = Viewport(int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1, float(len(xs)))
        if not any(v.x0 == union.x0 and v.y0 == union.y0 and v.x1 == union.x1
                   and v.y1 == union.y1 for v in out):
            out.append(union)
    return out
