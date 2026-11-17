"""Pictures of a match and of a placement.

Two drawings, both from arrays a caller already has, so nothing here reads
ground truth or reaches back into the model:

    render_match_image      density map | sheet crop, with correspondences
    render_placement_image  the capture's points drawn on the sheet itself

The first answers "did the matcher find the right patches"; the second answers
"did the pose put the walk in the right room". A failure usually looks obvious
in exactly one of them, which is how they divide the work.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .utils import affine

CORRECT_COLOUR = (80, 220, 120)      # within tolerance of the truth, or a RANSAC inlier
WRONG_COLOUR = (240, 90, 90)         # beyond it, or an outlier
REJECTED_COLOUR = (120, 125, 165)    # thrown out by geometric verification, so
                                     # RANSAC never saw it -- neither inlier nor
                                     # outlier, and drawn as a third thing
TRUTH_COLOUR = (250, 210, 90)        # where the truth says it should have gone
CLOUD_COLOUR = (60, 120, 220)        # the capture's own points
CAPTION_STRIP = 14                   # px reserved above the panels


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """(H,W) float in [0,1] or (H,W,3) uint8 -> (H,W,3) uint8."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        return np.repeat(array[..., None], 3, axis=2)
    return array.astype(np.uint8)


def render_match_image(density: np.ndarray, target: np.ndarray, query_px: np.ndarray,
                       target_px: np.ndarray, predicted_px: np.ndarray | None = None,
                       confidence: np.ndarray | None = None, num_lines: int = 40,
                       correct_within_px: float | None = None,
                       correct: np.ndarray | None = None,
                       rejected: np.ndarray | None = None,
                       caption: str = "") -> Image.Image:
    """Density map | prior crop, with correspondences drawn across the seam.

    With `predicted_px` the line goes from the query to what the matcher chose,
    and the truth is ringed so the error is visible as a gap. Without it the
    line goes to the truth, which is what the training pairs are.

    A pair is drawn green when `correct` says so, or when `correct_within_px` is
    given and the prediction landed inside it; red otherwise. `rejected`
    overrides both in slate, for pairs geometric verification removed before the
    fit ran -- they are not outliers, they are matches RANSAC was never offered.
    """
    density, target = _to_rgb(density), _to_rgb(target)
    density_size, target_size = density.shape[0], target.shape[0]
    canvas = Image.new("RGB", (density_size + target_size,
                               max(density_size, target_size) + CAPTION_STRIP), (40, 40, 40))
    canvas.paste(Image.fromarray(density), (0, CAPTION_STRIP))
    canvas.paste(Image.fromarray(target), (density_size, CAPTION_STRIP))
    draw = ImageDraw.Draw(canvas)
    if caption:
        draw.text((4, 2), caption, fill=(235, 235, 235))

    drawn = np.linspace(0, len(query_px) - 1, min(num_lines, len(query_px))).astype(int)
    for i in drawn:
        query_x, query_y = float(query_px[i, 0]), float(query_px[i, 1]) + CAPTION_STRIP
        truth_x = float(target_px[i, 0]) + density_size
        truth_y = float(target_px[i, 1]) + CAPTION_STRIP
        if predicted_px is None:
            colour = tuple(int(c) for c in np.random.default_rng(i).integers(80, 255, 3))
            end_x, end_y = truth_x, truth_y
        else:
            end_x = float(predicted_px[i, 0]) + density_size
            end_y = float(predicted_px[i, 1]) + CAPTION_STRIP
            if correct is not None:
                is_correct = bool(correct[i])
            else:
                error = float(np.linalg.norm(predicted_px[i] - target_px[i]))
                is_correct = correct_within_px is not None and error <= correct_within_px
            if rejected is not None and bool(rejected[i]):
                colour = REJECTED_COLOUR
            else:
                colour = CORRECT_COLOUR if is_correct else WRONG_COLOUR
            draw.ellipse([truth_x - 3, truth_y - 3, truth_x + 3, truth_y + 3],
                         outline=TRUTH_COLOUR)
            draw.line([(truth_x, truth_y), (end_x, end_y)], fill=TRUTH_COLOUR, width=1)
        width = 1
        if confidence is not None:
            width = 1 + int(round(2 * float(confidence[i])
                                  / max(float(np.max(confidence)), 1e-9)))
        draw.line([(query_x, query_y), (end_x, end_y)], fill=colour, width=width)
        draw.ellipse([query_x - 2, query_y - 2, query_x + 2, query_y + 2], outline=colour)
        draw.ellipse([end_x - 2, end_y - 2, end_x + 2, end_y + 2], outline=colour)
    return canvas


def _sample(pool: np.ndarray, count: int) -> np.ndarray:
    """`count` indices spread evenly through `pool`."""
    if count <= 0 or not len(pool):
        return np.empty(0, int)
    return pool[np.linspace(0, len(pool) - 1, count).astype(int)]


def render_inference_matches(density_image: np.ndarray, crop_rgb: np.ndarray,
                             query_px: np.ndarray, target_crop_px: np.ndarray,
                             inlier_mask: np.ndarray | None = None,
                             confidence: np.ndarray | None = None,
                             verified_mask: np.ndarray | None = None,
                             num_lines: int = 60, caption: str = "") -> Image.Image:
    """The matches a pose was fitted from, green where RANSAC kept them.

    Every group is drawn in proportion to how many there are, so a fit supported
    by a tenth of its matches looks like one.

    With `verified_mask` there are three groups, not two, because geometric
    verification removes matches BEFORE RANSAC: an inlier, a match RANSAC saw
    and rejected, and a match RANSAC was never offered. Collapsing the last two
    would hide the thing the picture is drawn to show -- whether verification
    threw away the correspondences that disagreed, or the ones that were right.
    """
    total = len(query_px)
    if inlier_mask is None:
        chosen = np.linspace(0, total - 1, min(num_lines, total)).astype(int)
        rejected_of_chosen = None
    else:
        inlier_mask = np.asarray(inlier_mask, dtype=bool)
        if verified_mask is None:
            seen_but_rejected = ~inlier_mask
            never_seen = np.zeros(total, bool)
        else:
            verified = np.asarray(verified_mask, dtype=bool)
            seen_but_rejected = verified & ~inlier_mask
            never_seen = ~verified
        inliers = np.flatnonzero(inlier_mask)
        outliers = np.flatnonzero(seen_but_rejected)
        dropped = np.flatnonzero(never_seen)
        # Proportional, but an inlier set that exists is always shown: a fit
        # with three supporting matches is exactly the one worth looking at.
        share = int(round(num_lines * len(inliers) / max(total, 1)))
        take_in = min(len(inliers), max(share, min(3, len(inliers))))
        remaining = num_lines - take_in
        rest = len(outliers) + len(dropped)
        take_out = min(len(outliers), int(round(remaining * len(outliers) / max(rest, 1))))
        take_dropped = min(len(dropped), remaining - take_out)
        chosen = np.concatenate([_sample(inliers, take_in), _sample(outliers, take_out),
                                 _sample(dropped, take_dropped)]).astype(int)
        rejected_of_chosen = never_seen[chosen]
    chosen = chosen.astype(int)
    correct = (np.ones(len(chosen), bool) if inlier_mask is None
               else np.asarray(inlier_mask)[chosen])
    return render_match_image(
        density_image, crop_rgb, np.asarray(query_px)[chosen],
        np.asarray(target_crop_px)[chosen], predicted_px=np.asarray(target_crop_px)[chosen],
        confidence=None if confidence is None else np.asarray(confidence)[chosen],
        correct=correct, rejected=rejected_of_chosen, caption=caption)


def _fit_box(points: np.ndarray, margin: float, width: int, height: int,
             min_side: int = 400) -> tuple[int, int, int, int]:
    x0, y0 = points.min(0)
    x1, y1 = points.max(0)
    pad = margin * max(x1 - x0, y1 - y0, min_side)
    box = (max(int(x0 - pad), 0), max(int(y0 - pad), 0),
           min(int(x1 + pad), width), min(int(y1 + pad), height))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return (0, 0, width, height)
    return box


def render_placement_image(backdrop: Image.Image, cloud_sheet_px: np.ndarray,
                           camera_sheet_px: np.ndarray,
                           gt_camera_sheet_px: np.ndarray | None = None,
                           long_side: int = 1400, margin: float = 0.35,
                           max_points: int = 60000, caption: str = "") -> Image.Image:
    """The capture drawn on the reference it was placed against.

    `backdrop` is that reference as an image -- the prior capture's bird's-eye
    view here, a drawing in the runs this replaced. Every coordinate argument is
    in ITS pixels: `cloud_sheet_px` are the wall-band points through the
    predicted pose, `camera_sheet_px` the predicted camera walk, and
    `gt_camera_sheet_px`, when a caller has it, the truth to compare against.
    The view is cropped to what was drawn, with a margin, so the capture is
    legible on a reference that is mostly empty.
    """
    width, height = backdrop.size
    drawn = [np.asarray(cloud_sheet_px, dtype=np.float64)[:, :2],
             np.asarray(camera_sheet_px, dtype=np.float64)[:, :2]]
    if gt_camera_sheet_px is not None and len(gt_camera_sheet_px):
        drawn.append(np.asarray(gt_camera_sheet_px, dtype=np.float64)[:, :2])
    box = _fit_box(np.vstack(drawn), margin, width, height)

    region = backdrop.crop(box).convert("RGB")
    scale = long_side / max(region.size)
    if scale < 1:
        region = region.resize((max(int(region.size[0] * scale), 1),
                                max(int(region.size[1] * scale), 1)), Image.LANCZOS)
    else:
        scale = 1.0
    canvas = region.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def to_canvas(points):
        pts = np.asarray(points, dtype=np.float64)[:, :2]
        return (pts - [box[0], box[1]]) * scale

    cloud = to_canvas(cloud_sheet_px)
    if len(cloud) > max_points:
        step = int(np.ceil(len(cloud) / max_points))
        cloud = cloud[::step]
    for x, y in cloud:
        draw.point((x, y), fill=CLOUD_COLOUR + (110,))

    if gt_camera_sheet_px is not None and len(gt_camera_sheet_px) > 1:
        gt = to_canvas(gt_camera_sheet_px)
        draw.line([tuple(p) for p in gt], fill=TRUTH_COLOUR + (220,), width=3)
    cameras = to_canvas(camera_sheet_px)
    if len(cameras) > 1:
        draw.line([tuple(p) for p in cameras], fill=CORRECT_COLOUR + (230,), width=3)
    for x, y in cameras[:: max(1, len(cameras) // 40)]:
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=CORRECT_COLOUR + (255,))

    flat = Image.alpha_composite(canvas, overlay).convert("RGB")
    legend = ImageDraw.Draw(flat)
    legend.rectangle([0, 0, flat.size[0], CAPTION_STRIP + 4], fill=(28, 28, 28))
    legend.text((4, 3), caption or "placement", fill=(235, 235, 235))
    keys = [("points", CLOUD_COLOUR), ("predicted cameras", CORRECT_COLOUR)]
    if gt_camera_sheet_px is not None and len(gt_camera_sheet_px):
        keys.append(("ground truth", TRUTH_COLOUR))
    offset = flat.size[0] - 8
    for label, colour in reversed(keys):
        text_width = 6 * len(label) + 18
        offset -= text_width
        legend.rectangle([offset, 6, offset + 10, 12], fill=colour)
        legend.text((offset + 14, 3), label, fill=(235, 235, 235))
    return flat

def render_density_overlay(sheet: Image.Image, density_image: np.ndarray,
                           density_to_sheet_px: np.ndarray, long_side: int = 1600,
                           margin: float = 0.35, colour: tuple[int, int, int] = (200, 30, 30),
                           max_alpha: float = 0.85, gamma: float = 0.65,
                           correspondence_px: np.ndarray | None = None,
                           correspondence_colour: tuple[int, int, int] = (20, 130, 255),
                           correspondence_radius: float = 4.5,
                           caption: str = "",
                           density_layers: list | None = None,
                           legend: list | None = None) -> Image.Image:
    """The density map itself, warped onto the sheet and laid over the drawing.

    `density_to_sheet_px` is the 2x3 affine taking a density-map pixel to a
    sheet pixel -- the ground-truth chain, or a predicted one. The map is warped
    rather than drawn as points, so a wall in the cloud appears as a stroke of
    the same thickness it has in the map, and a half-metre offset from the
    drawing's own wall is visible as a doubled line.

    `correspondence_px` are training targets in sheet pixels: the places the
    network is asked to predict, sampled from the density in proportion to mass
    and pushed through the same chain. They are drawn on top, so it is visible
    whether supervision lands on drawing content or on blank paper.

    `density_layers`, a list of (image, colour), replaces the single map with
    several, each in its own colour and composited in order -- one per reality
    of a capture, say. Every image must share `density_image`'s grid, since one
    `density_to_sheet_px` places them all. `legend`, a list of (label, colour),
    is drawn as a row of swatches under the caption.
    """
    width, height = sheet.size
    rows, cols = density_image.shape[:2]
    corners = np.array([[0.0, 0.0], [cols, 0.0], [0.0, rows], [cols, rows]])
    on_sheet = affine.apply(density_to_sheet_px, corners)
    box = _fit_box(on_sheet, margin, width, height)

    region = sheet.crop(box).convert("RGB")
    scale = min(long_side / max(region.size), 1.0)
    if scale < 1:
        region = region.resize((max(int(region.size[0] * scale), 1),
                                max(int(region.size[1] * scale), 1)), Image.LANCZOS)
    else:
        scale = 1.0

    # density px -> canvas px, then inverted, because PIL maps output to input
    to_canvas = affine.compose(to_canvas_from_sheet(box, scale), density_to_sheet_px)
    to_density = affine.invert(to_canvas)
    composite = region.convert("RGBA")
    for image, layer_colour in (density_layers or [(density_image, colour)]):
        warped = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8), mode="L")
        warped = warped.transform(region.size, Image.AFFINE, tuple(to_density.reshape(-1)),
                                  resample=Image.BILINEAR, fillcolor=0)
        alpha = (np.asarray(warped, dtype=np.float32) / 255.0) ** gamma * max_alpha
        layer = np.zeros((region.size[1], region.size[0], 4), dtype=np.uint8)
        layer[..., 0], layer[..., 1], layer[..., 2] = layer_colour
        layer[..., 3] = (alpha * 255).astype(np.uint8)
        composite = Image.alpha_composite(composite, Image.fromarray(layer, mode="RGBA"))
    flat = composite.convert("RGB")
    draw = ImageDraw.Draw(flat)
    if correspondence_px is not None and len(correspondence_px):
        on_canvas = affine.apply(to_canvas_from_sheet(box, scale), correspondence_px)
        inside = ((on_canvas[:, 0] >= 0) & (on_canvas[:, 0] < flat.size[0])
                  & (on_canvas[:, 1] >= 0) & (on_canvas[:, 1] < flat.size[1]))
        radius = max(float(correspondence_radius), 1.0)
        for x, y in on_canvas[inside]:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill=correspondence_colour + (255,), outline=(255, 255, 255),
                         width=1 if radius < 4 else 2)
    if caption:
        draw.rectangle([0, 0, flat.size[0], CAPTION_STRIP + 4], fill=(28, 28, 28))
        draw.text((4, 3), caption, fill=(235, 235, 235))
    if legend:
        _draw_legend(draw, legend, top=(CAPTION_STRIP + 8) if caption else 4,
                     width=flat.size[0])
    return flat


def _draw_legend(draw: ImageDraw.ImageDraw, legend, top: int, width: int,
                 swatch: int = 12, gap: int = 14) -> None:
    """A row of colour swatches and labels, wrapped to the canvas width, on a
    pale backing so it reads over any drawing."""
    row_height = swatch + 6
    entries, x, y = [], 6, top
    for label, entry_colour in legend:
        entry_width = swatch + 4 + int(draw.textlength(label)) + gap
        if x + entry_width > width and x > 6:
            x, y = 6, y + row_height
        entries.append((x, y, label, entry_colour))
        x += entry_width
    draw.rectangle([0, top - 3, width, y + row_height], fill=(255, 255, 255))
    for x, y, label, entry_colour in entries:
        draw.rectangle([x, y, x + swatch, y + swatch], fill=tuple(entry_colour),
                       outline=(40, 40, 40))
        draw.text((x + swatch + 4, y), label, fill=(20, 20, 20))


def to_canvas_from_sheet(box, scale: float) -> np.ndarray:
    """Sheet pixels -> the cropped, scaled canvas the overlay draws on."""
    return affine.chain(affine.translation(-box[0], -box[1]), affine.scaling(scale))
