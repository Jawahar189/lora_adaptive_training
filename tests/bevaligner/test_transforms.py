"""Augmentation: the window the target is cut from, and the ops on the points."""
import numpy as np
import torch
from types import SimpleNamespace

from bevaligner.config import DataConfig
from bevaligner.data import prior_bev, transforms
from bevaligner.utils import affine

from conftest import make_pool


def _pool(span=200.0, seed=0):
    rng = np.random.default_rng(seed)
    cameras = rng.uniform(-span / 2, span / 2, size=(40, 2))
    band = cameras[rng.integers(0, 40, 6000)] + rng.normal(scale=span / 40, size=(6000, 2))
    return make_pool(band, cameras)


class _Record:
    """The three fields `subsample_points` reads."""

    def __init__(self, num_cameras, num_points, overlap_mask=None, seed=0):
        rng = np.random.default_rng(seed)
        self.camera_xy = rng.normal(size=(num_cameras, 2))
        self.band_camera_index = rng.integers(0, num_cameras, num_points)
        self.overlap_mask = (np.zeros(0, bool) if overlap_mask is None
                             else np.asarray(overlap_mask))


def test_choose_window_holds_the_anchors_and_spans_a_range():
    pool = _pool()
    config = DataConfig()
    rng = np.random.default_rng(0)
    anchors = np.array([[10.0, -5.0], [26.0, 3.0], [18.0, 9.0]])
    sides = []
    for _ in range(200):
        lo_x, lo_y, hi_x, hi_y = prior_bev.choose_window(pool, anchors, rng, config,
                                                         size=512, patch_size=16)
        assert lo_x <= anchors[:, 0].min() and hi_x >= anchors[:, 0].max()
        assert lo_y <= anchors[:, 1].min() and hi_y >= anchors[:, 1].max()
        assert np.isclose(hi_x - lo_x, hi_y - lo_y)     # square
        sides.append(hi_x - lo_x)
    sides = np.array(sides)
    anchor_side = 16.0
    assert sides.min() >= anchor_side * config.zoom_footprint_margin - 1e-9
    # never so far out that the walk falls under min_footprint_patches patches
    assert sides.max() <= anchor_side * 512 / (config.min_footprint_patches * 16) + 1e-9
    assert sides.max() / sides.min() > 2      # it really does sweep a range


def test_choose_window_is_deterministic_and_tight_for_evaluation():
    pool = _pool()
    anchors = np.array([[10.0, -5.0], [26.0, 3.0]])
    first = prior_bev.choose_window(pool, anchors, None, DataConfig(), 512, 16)
    second = prior_bev.choose_window(pool, anchors, None, DataConfig(), 512, 16)
    assert first == second
    lo_x, lo_y, hi_x, hi_y = first
    # centred on the anchors, at the whole pool or the cap, whichever is smaller
    assert np.isclose((lo_x + hi_x) / 2, anchors[:, 0].mean(), atol=1e-6)


def test_rotate_target_keeps_its_affine_consistent():
    pool = _pool()
    crop = prior_bev.render_target(pool, pool.extent, 256, None, DataConfig())
    rotated = prior_bev.rotate_target(crop, 1)
    point = np.array([[3.0, -2.0]])
    before = affine.apply(crop.paper_to_crop, point)[0]
    after = affine.apply(rotated.paper_to_crop, point)[0]
    # np.rot90 with one turn: (u, v) -> (v, S - u)
    assert np.allclose(after, [before[1], 256 - before[0]])
    assert rotated.valid_mask.shape == crop.valid_mask.shape


def test_camera_window_is_biased_into_the_overlap():
    """The half-covered walk is the common case; the window has to find the half."""
    overlap = np.zeros(40, bool)
    overlap[:10] = True                       # the prior only walked the first quarter
    record = _Record(40, 4000, overlap_mask=overlap)
    config = DataConfig()
    mask = np.ones(4000, bool)
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(200):
        kept = transforms.subsample_points(
            record, mask, rng, camera_window_probability=1.0,
            camera_window_min_fraction=0.1, point_keep_fraction=(1.0, 1.0),
            overlap_bias=config.camera_window_overlap_bias)
        indices = record.band_camera_index[kept]
        covered.append(float((indices < 10).mean()) if len(indices) else 0.0)
    biased = float(np.mean(covered))

    rng = np.random.default_rng(0)
    unbiased = float(np.mean([
        (lambda i: float((i < 10).mean()) if len(i) else 0.0)(
            record.band_camera_index[transforms.subsample_points(
                record, mask, rng, 1.0, 0.1, (1.0, 1.0), overlap_bias=0.0)])
        for _ in range(200)]))
    assert biased > unbiased + 0.1


def test_photometric_jitter_keeps_range_and_shape():
    """Only brightness/contrast survives: saturation, hue and the rgb wrapper
    were sheet-era code with no caller and no config field to feed them."""
    rng = np.random.default_rng(0)
    image = torch.rand(3, 32, 32)
    jittered = transforms.jitter_brightness_contrast(image, rng, 0.5, 0.5)
    assert jittered.shape == image.shape and jittered.min() >= 0 and jittered.max() <= 1
    unchanged = transforms.jitter_brightness_contrast(image, rng, 0.0, 0.0)
    assert torch.allclose(unchanged, image)


def _relative_yaws(density_degrees, target_degrees, draws=300):
    """The yaw the matcher actually has to solve, sample by sample."""
    from align.frames import Sim2
    from bevaligner.data import prior_bev
    from bevaligner.utils import affine
    from conftest import make_pool

    rng = np.random.default_rng(4)
    cameras = rng.uniform(-5, 5, size=(30, 2))
    band = cameras[rng.integers(0, 30, 4000)] + rng.normal(scale=1.5, size=(4000, 2))
    sim2 = Sim2(s=1.933, theta=np.deg2rad(-76.8), tx=40.0, ty=25.0)
    record = SimpleNamespace(band_xy=band,
                             band_relative_height=np.zeros(len(band), dtype=np.float32),
                             band_camera_index=np.zeros(len(band), dtype=np.int64),
                             camera_xy=cameras, overlap_mask=np.zeros(0, dtype=bool))
    config = DataConfig(camera_window_probability=0.0, point_keep_fraction=(1.0, 1.0),
                        random_crop_scale=(1.0, 1.0), zoom_crops=False,
                        density_gammas=[0.5], density_min_counts=[1],
                        band_low_range=(-1e6, -1e6 + 1), band_high_range=(1e6 - 1, 1e6),
                        density_rotation_degrees=density_degrees,
                        target_rotation_degrees=target_degrees)
    paper_cameras = sim2.apply(cameras)
    pool = make_pool(sim2.apply(band), paper_cameras)

    yaws = []
    for seed in range(draws):
        draw = np.random.default_rng(seed)
        density_map = transforms.render_record_density(record, 128, draw, config)
        rotation = prior_bev.draw_target_rotation(draw, config, paper_cameras)
        anchors = (paper_cameras if rotation is None
                   else affine.apply(rotation, paper_cameras))
        window = prior_bev.choose_window(pool, anchors, draw, config, 128, 16)
        crop = prior_bev.render_target(pool, window, 128, draw, config,
                                       paper_rotation=rotation)
        chain = affine.chain(density_map.density_px_to_world_xy,
                             affine.from_sim2(sim2), crop.paper_to_crop)
        yaws.append(affine.to_sim2(chain).theta)
    return np.asarray(yaws) % (2 * np.pi)


def test_the_continuous_yaw_fills_the_gaps_the_quarter_turns_leave():
    """Off, the relative yaw is four cardinals and nothing between -- the state
    this augmentation exists to replace. At 45 it is the whole circle, because
    a triangular kernel of half-width 90 degrees summed on a 90 degree lattice
    is flat."""
    off = _relative_yaws(0.0, 0.0)
    assert len(np.unique(np.round(off, 6))) == 4, "quarter turns alone: four angles"

    on = np.sort(_relative_yaws(45.0, 0.0))
    gaps = np.diff(np.concatenate([on, on[:1] + 2 * np.pi]))
    assert gaps.max() < 0.3, f"a hole of {np.degrees(gaps.max()):.0f} deg in the circle"
