"""Rendering helpers for stage-by-stage inspection.

Every panel is a plain RGB image with a caption bar, so the stages can be laid
out side by side and compared without a plotting library.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

PAPER = (247, 248, 250)
INK = (22, 25, 29)
MUTED = (200, 206, 213)
RED = (200, 57, 43)
GREEN = (47, 122, 90)
AMBER = (181, 130, 42)
BLUE = (43, 93, 138)


def _font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def panel(img: Image.Image, title: str, subtitle: str = "",
          width: int = 900, bar: int = 46) -> Image.Image:
    """Scale an image to `width` and stack a caption bar above it.

    The bar grows to as many lines as the subtitle needs. These captions carry
    the legend and, where one is drawn, the ground-truth comparison, so they
    outgrow a single line routinely and a fixed two-line bar still clipped them.
    """
    img = img.convert("RGB")
    if img.width != width:
        img = img.resize((width, max(1, round(img.height * width / img.width))),
                         Image.LANCZOS)
    ft, fs = _font(17, True), _font(14)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    lines = []
    if subtitle:
        inline_x = 12 + probe.textlength(title, font=ft) + 14
        if inline_x + probe.textlength(subtitle, font=fs) <= width - 12:
            lines = [(inline_x, 12, subtitle)]
        else:
            y = 34
            avail = width - 24
            words, current = subtitle.split(" "), ""
            for word in words:
                trial = f"{current} {word}".strip()
                if probe.textlength(trial, font=fs) <= avail or not current:
                    current = trial
                else:
                    lines.append((12, y, current))
                    y += 18
                    current = word
            if current:
                lines.append((12, y, current))
            bar = max(bar, y + 26)

    out = Image.new("RGB", (width, img.height + bar), PAPER)
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    d.text((12, 9 if len(lines) <= 1 else 8), title, font=ft, fill=INK)
    for x, y, text in lines:
        d.text((x, y), text, font=fs, fill=(110, 118, 129))
    d.line([(0, bar - 1), (width, bar - 1)], fill=MUTED, width=1)
    return out


def grid(panels: list[Image.Image], cols: int = 2, gap: int = 14,
         pad: int = 14) -> Image.Image:
    """Tile panels into one image; rows size to their tallest member.

    Used for contact_sheet.png -- a contact sheet in the darkroom sense, where a
    whole roll is printed at thumbnail size on one page so it can be reviewed at
    a glance. Here it puts all of a split's stages side by side, which is how you
    see that, say, the wall band went empty two stages before the match failed.
    """
    rows = [panels[i:i + cols] for i in range(0, len(panels), cols)]
    w = max(sum(p.width for p in r) + gap * (len(r) - 1) for r in rows)
    h = sum(max(p.height for p in r) for r in rows) + gap * (len(rows) - 1)
    out = Image.new("RGB", (w + 2 * pad, h + 2 * pad), PAPER)
    y = pad
    for r in rows:
        x = pad
        for p in r:
            out.paste(p, (x, y))
            x += p.width + gap
        y += max(p.height for p in r) + gap
    return out


def mask_rgb(mask: np.ndarray, fg=INK, bg=(255, 255, 255)) -> Image.Image:
    a = np.empty(mask.shape + (3,), np.uint8)
    a[:] = bg
    a[mask] = fg
    return Image.fromarray(a)


def layers(shape, *pairs) -> Image.Image:
    """Paint (mask, colour) pairs in order over white."""
    a = np.full(shape + (3,), 255, np.uint8)
    for m, c in pairs:
        a[m] = c
    return Image.fromarray(a)


def downsample_mask(mask: np.ndarray, width: int) -> np.ndarray:
    """Shrink a binary mask keeping thin lines visible (any-pooling)."""
    import cv2
    if mask.shape[1] <= width:
        return mask
    h = max(1, round(mask.shape[0] * width / mask.shape[1]))
    return cv2.resize(mask.astype(np.float32), (width, h),
                      interpolation=cv2.INTER_AREA) > 0


PATH = (43, 93, 138)          # ground-truth walk, distinct from every verdict colour


def raster_to_display(xy: np.ndarray, origin, res: float, height_px: int,
                      flipped: bool, scale: float = 1.0) -> np.ndarray:
    """Paper-frame points -> display pixels for a panel.

    Panels reach the screen two different ways: some arrays are row-reversed
    when built, others are flipped after drawing. `flipped=True` means the
    caller flips afterwards, so draw in raster order; otherwise invert here.
    """
    p = np.asarray(xy, dtype=np.float64)[:, :2]
    x = (p[:, 0] - origin[0]) / res * scale
    y = (p[:, 1] - origin[1]) / res * scale
    if not flipped:
        y = height_px - y
    return np.stack([x, y], axis=1)


def draw_path(base: Image.Image, pts: np.ndarray, color=PATH, width: int = 2,
              dot: int = 0) -> Image.Image:
    """Draw the walk as a polyline, optionally with a marker at each pose."""
    out = base.convert("RGB")
    d = ImageDraw.Draw(out)
    if len(pts) > 1:
        d.line([tuple(q) for q in pts], fill=color, width=width, joint="curve")
    if dot:
        for x, y in pts:
            d.ellipse([x - dot, y - dot, x + dot, y + dot], fill=color)
    if len(pts):
        x, y = pts[0]
        r = max(width * 2, 4)
        d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=max(width, 2))
    return out


def draw_segments(base: Image.Image, segs: np.ndarray, scale: float,
                  colors, width: int = 2) -> Image.Image:
    out = base.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for i, s in enumerate(segs):
        c = colors[i] if not isinstance(colors, tuple) else colors
        d.line([(s[0] * scale, s[1] * scale), (s[2] * scale, s[3] * scale)],
               fill=c, width=width)
    return out


def legend(entries, width: int = 900, height: int = 34) -> Image.Image:
    out = Image.new("RGB", (width, height), PAPER)
    d = ImageDraw.Draw(out)
    x = 12
    for label, color in entries:
        d.rectangle([x, height // 2 - 5, x + 10, height // 2 + 5], fill=color)
        x += 16
        d.text((x, height // 2 - 8), label, font=_font(13), fill=INK)
        x += int(d.textlength(label, font=_font(13))) + 22
    return out


def save(img: Image.Image, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


RANK_COLORS = [(200, 57, 43), (214, 110, 40), (181, 150, 42), (120, 150, 70),
               (60, 140, 120), (55, 110, 160), (90, 90, 170), (130, 80, 160)]


def rank_color(rank: int, n: int = 8):
    """Warm for the top-ranked hypothesis, cooling as rank falls."""
    return RANK_COLORS[min(rank, len(RANK_COLORS) - 1)] if n else RANK_COLORS[0]


def score_bars(rows, width: int = 900, row_h: int = 24,
               labels=("wall", "walk", "scale", "aspect"), notes=None,
               highlight: int | None = None) -> Image.Image:
    """Each score term's CONTRIBUTION, one row per hypothesis.

    Contributions, not raw term values: the walk term is stored negative and
    enters the score negated, so plotting it raw draws a bar pointing the wrong
    way and misreads as the opposite of what it did. The four bars here sum to
    the printed total.
    """
    n = len(rows)
    top, foot = 36, 28
    lab_w = 150
    note_w = 132 if notes else 0
    img = Image.new("RGB", (width, top + n * row_h + foot), PAPER)
    d = ImageDraw.Draw(img)
    d.text((12, 9), "score contributions by rank", font=_font(15, True), fill=INK)

    x0, x1 = lab_w, width - 92 - note_w
    contribs = [(r[0], r[1], -r[2], r[3], r[4]) for r in rows]
    lim = max(max(abs(v) for c in contribs for v in c[1:]), 1e-6)
    zero = x0 + (x1 - x0) * 0.5
    scale = (x1 - x0) * 0.5 / lim
    colors = [BLUE, RED, GREEN, AMBER]

    d.line([(zero, top - 6), (zero, top + n * row_h + 2)], fill=MUTED, width=1)
    for i, (label, *terms) in enumerate(contribs):
        y = top + i * row_h
        if highlight is not None and i == highlight:
            d.rectangle([4, y, width - 6, y + row_h - 2], fill=(233, 241, 236))
            d.text((4, y + 5), "\u25b8", font=_font(12, True), fill=GREEN)
        d.text((16, y + 5), label[:24], font=_font(12), fill=(110, 118, 129))
        h = max((row_h - 8) // len(terms), 2)
        for j, v in enumerate(terms):
            yy = y + 3 + j * h
            xx = float(np.clip(zero + v * scale, x0, x1))
            lo, hi = sorted((zero, xx))
            if hi - lo < 1:
                hi = lo + 1
            d.rectangle([lo, yy, hi, yy + h - 1], fill=colors[j])
        d.text((x1 + 8, y + 5), f"{sum(terms):+.3f}", font=_font(12), fill=INK)
        if notes and i < len(notes):
            colour = GREEN if (highlight is not None and i == highlight) else \
                (110, 118, 129)
            d.text((x1 + 70, y + 5), notes[i], font=_font(12), fill=colour)

    lx = 12
    for label, color in zip(labels, colors):
        d.rectangle([lx, img.height - 19, lx + 9, img.height - 10], fill=color)
        lx += 14
        d.text((lx, img.height - 21), label, font=_font(12), fill=INK)
        lx += int(d.textlength(label, font=_font(12))) + 18
    d.text((lx, img.height - 21), "(contributions, signed as they enter the score)",
           font=_font(11), fill=(140, 147, 156))
    return img
