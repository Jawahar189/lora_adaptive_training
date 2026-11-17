"""The two inference pictures: what they draw, and that they stay inside their frame."""
import numpy as np
from PIL import Image

from bevaligner.viz import (CAPTION_STRIP, CLOUD_COLOUR, CORRECT_COLOUR, TRUTH_COLOUR,
                              WRONG_COLOUR, render_inference_matches, render_placement_image)

DENSITY = 64
CROP = 96


def _colours(image):
    return {tuple(c) for c in np.asarray(image.convert("RGB")).reshape(-1, 3)}


def _matches(num=40, rng=np.random.default_rng(0)):
    query_px = rng.uniform(2, DENSITY - 2, (num, 2))
    target_px = rng.uniform(2, CROP - 2, (num, 2))
    return query_px, target_px


def test_match_picture_has_both_panels_and_a_caption_strip():
    query_px, target_px = _matches()
    image = render_inference_matches(np.zeros((DENSITY, DENSITY), np.float32),
                                     np.zeros((CROP, CROP, 3), np.uint8),
                                     query_px, target_px, caption="RLT0001")
    assert image.size == (DENSITY + CROP, CROP + CAPTION_STRIP)


def test_inliers_are_green_and_outliers_red():
    query_px, target_px = _matches()
    all_in = render_inference_matches(np.zeros((DENSITY, DENSITY), np.float32),
                                      np.zeros((CROP, CROP, 3), np.uint8), query_px, target_px,
                                      inlier_mask=np.ones(len(query_px), bool))
    assert CORRECT_COLOUR in _colours(all_in) and WRONG_COLOUR not in _colours(all_in)

    mask = np.zeros(len(query_px), bool)
    mask[::4] = True
    mixed = render_inference_matches(np.zeros((DENSITY, DENSITY), np.float32),
                                     np.zeros((CROP, CROP, 3), np.uint8), query_px, target_px,
                                     inlier_mask=mask)
    assert {CORRECT_COLOUR, WRONG_COLOUR} <= _colours(mixed)


def test_both_kinds_are_drawn_when_inliers_are_rare():
    # a fit supported by a tenth of its matches must still show some inliers
    query_px, target_px = _matches(num=200)
    mask = np.zeros(200, bool)
    mask[:20] = True
    image = render_inference_matches(np.zeros((DENSITY, DENSITY), np.float32),
                                     np.zeros((CROP, CROP, 3), np.uint8), query_px, target_px,
                                     inlier_mask=mask, num_lines=40)
    assert {CORRECT_COLOUR, WRONG_COLOUR} <= _colours(image)


def test_placement_crops_to_what_was_drawn_and_fits_the_long_side():
    sheet = Image.new("RGB", (4000, 3000), (255, 255, 255))
    rng = np.random.default_rng(1)
    cloud = rng.uniform([1000, 800], [1400, 1100], (3000, 2))
    cameras = np.stack([np.linspace(1050, 1350, 40), np.linspace(850, 1050, 40)], axis=1)
    image = render_placement_image(sheet, cloud, cameras, long_side=600)
    assert max(image.size) <= 600                      # scaled down
    assert image.size[0] < 4000 and image.size[1] < 3000   # cropped, not the whole sheet
    colours = _colours(image)
    assert CLOUD_COLOUR[2] in {c[2] for c in colours}  # the points were drawn
    assert CORRECT_COLOUR in colours                   # so was the predicted walk


def test_placement_draws_ground_truth_when_it_is_given():
    sheet = Image.new("RGB", (2000, 1500), (255, 255, 255))
    cameras = np.stack([np.linspace(400, 900, 30), np.full(30, 600.0)], axis=1)
    ground_truth = cameras + [0, 120]
    cloud = cameras + [0, 10]
    without = render_placement_image(sheet, cloud, cameras, long_side=500)
    with_gt = render_placement_image(sheet, cloud, cameras, ground_truth, long_side=500)
    assert TRUTH_COLOUR not in _colours(without)
    assert TRUTH_COLOUR in _colours(with_gt)


def test_placement_survives_a_capture_at_the_sheet_edge():
    sheet = Image.new("RGB", (1200, 900), (255, 255, 255))
    cameras = np.array([[2.0, 2.0], [8.0, 6.0], [14.0, 3.0]])
    image = render_placement_image(sheet, cameras + 1, cameras, long_side=400)
    assert image.size[0] > 0 and image.size[1] > 0


# --- geometric verification adds a third state ----------------------------

def test_verification_rejects_are_drawn_as_a_third_colour():
    """A match verification removed is not an outlier -- RANSAC never saw it --
    so it must not be drawn in the colour that means "RANSAC disagreed"."""
    from bevaligner.viz import REJECTED_COLOUR
    query_px, target_px = _matches(num=60)
    inlier = np.zeros(len(query_px), bool)
    verified = np.zeros(len(query_px), bool)
    inlier[:20] = True                 # kept by RANSAC
    verified[:40] = True               # 20..40 seen by RANSAC and rejected
    #                                    40..60 never offered to it
    image = render_inference_matches(np.zeros((DENSITY, DENSITY), np.float32),
                                     np.zeros((CROP, CROP, 3), np.uint8),
                                     query_px, target_px, inlier_mask=inlier,
                                     verified_mask=verified)
    colours = _colours(image)
    assert CORRECT_COLOUR in colours
    assert WRONG_COLOUR in colours
    assert REJECTED_COLOUR in colours


def test_without_a_verified_mask_the_picture_is_unchanged():
    """The two-state path is what every existing caller uses."""
    from bevaligner.viz import REJECTED_COLOUR
    query_px, target_px = _matches(num=60)
    inlier = np.zeros(len(query_px), bool)
    inlier[:20] = True
    image = render_inference_matches(np.zeros((DENSITY, DENSITY), np.float32),
                                     np.zeros((CROP, CROP, 3), np.uint8),
                                     query_px, target_px, inlier_mask=inlier)
    assert REJECTED_COLOUR not in _colours(image)


def test_a_fully_rejected_set_still_draws():
    """Verification refusing everything is a picture worth having, not a crash."""
    query_px, target_px = _matches(num=30)
    image = render_inference_matches(np.zeros((DENSITY, DENSITY), np.float32),
                                     np.zeros((CROP, CROP, 3), np.uint8),
                                     query_px, target_px,
                                     inlier_mask=np.zeros(len(query_px), bool),
                                     verified_mask=np.zeros(len(query_px), bool))
    assert image.size == (DENSITY + CROP, CROP + CAPTION_STRIP)
