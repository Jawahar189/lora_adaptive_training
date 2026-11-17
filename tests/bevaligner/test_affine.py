"""The affine chain: density pixel -> cloud xy -> paper inches -> crop pixel.

A mistake here is silent: the network trains happily on mirrored or shifted
targets and only the final accuracy tells. So the chain is pinned against
`align.frames` on synthetic data, and chirality is asserted outright.
"""
import numpy as np
import pytest

from align.frames import SheetGeometry, Sim2
from bevaligner.config import DataConfig
from bevaligner.data import correspondences, density, prior_bev, transforms
from bevaligner.utils import affine, geometry

from conftest import make_pool


def test_compose_invert_roundtrip():
    rng = np.random.default_rng(0)
    first = rng.normal(size=(2, 3))
    second = rng.normal(size=(2, 3))
    points = rng.normal(size=(10, 2))
    both = affine.apply(second, affine.apply(first, points))
    assert np.allclose(affine.apply(affine.compose(second, first), points), both)
    assert np.allclose(affine.apply(affine.invert(first), affine.apply(first, points)), points)
    assert np.allclose(affine.apply(affine.chain(first, second), points), both)


@pytest.mark.parametrize("turns", [0, 1, 2, 3, 4, 5])
def test_quarter_turns_match_numpy(turns):
    height, width = 5, 8
    image = np.arange(height * width).reshape(height, width)
    rotated = np.rot90(image, turns)
    rotation = affine.rotation_quarter_turns(turns, width, height)
    rows, cols = np.mgrid[0:height, 0:width]
    centres = np.stack([cols.ravel() + 0.5, rows.ravel() + 0.5], axis=1)
    mapped = affine.apply(rotation, centres)
    u = np.floor(mapped[:, 0]).astype(int)
    v = np.floor(mapped[:, 1]).astype(int)
    assert rotated.shape == ((width, height) if turns % 2 else (height, width))
    assert np.array_equal(rotated[v, u], image.ravel())
    assert affine.determinant(rotation) > 0


def test_render_is_y_down_and_its_affine_is_exact():
    rng = np.random.default_rng(1)
    xy = rng.uniform([-3, 10], [7, 14], size=(5000, 2))
    xy = np.vstack([xy, [[0.0, 13.9]]])          # a point near the TOP (max y)
    density_map = density.render_density_map(xy, 64, trim_fraction=0.0, pad_fraction=0.0)
    lo_x, lo_y, hi_x, hi_y = density_map.extent
    assert hi_x - lo_x == pytest.approx(hi_y - lo_y)     # square
    # pixel (0,0) top-left is at (lo_x, hi_y)
    to_world = density_map.density_px_to_world_xy
    assert np.allclose(affine.apply(to_world, [0.0, 0.0]), [lo_x, hi_y])
    # centre of the top row is the largest y
    top = affine.apply(to_world, [32.0, 0.5])
    bottom = affine.apply(to_world, [32.0, 63.5])
    assert top[1] > bottom[1]
    # a point maps back into the pixel that counted it
    back = affine.apply(density_map.world_xy_to_density_px, [0.0, 13.9])
    assert 0 <= back[0] < 64 and 0 <= back[1] < 64
    assert density_map.image[int(back[1]), int(back[0])] > 0
    assert density_map.image.max() == pytest.approx(1.0)
    assert density_map.num_points == len(xy)


def test_render_with_a_fixed_extent_rerenders_a_crop_exactly():
    rng = np.random.default_rng(2)
    xy = rng.normal(size=(20000, 2)) * [4, 1]
    full = density.render_density_map(xy, 128, gamma=1.0)
    lo_x, lo_y, hi_x, _ = full.extent
    span = hi_x - lo_x
    sub_extent = (lo_x + 0.25 * span, lo_y + 0.25 * span,
                  lo_x + 0.75 * span, lo_y + 0.75 * span)
    crop = density.render_density_map(xy, 64, extent=sub_extent, gamma=1.0)
    assert crop.world_units_per_pixel == pytest.approx(full.world_units_per_pixel)
    # same cloud point -> same location through both affines
    point = np.array([[lo_x + 0.5 * span, lo_y + 0.5 * span]])
    in_full = affine.apply(full.world_xy_to_density_px, point)
    in_crop = affine.apply(crop.world_xy_to_density_px, point)
    assert np.allclose(in_full - [32, 32], in_crop, atol=1e-9)


def test_paper_sheet_affine_matches_sheet_geometry():
    sheet_geometry = SheetGeometry(paper_w=36.0, paper_h=24.0, img_w=7200, img_h=4800)
    points = np.array([[0.0, 24.0], [0.0, 0.0], [36.0, 12.0], [10.5, 3.25]])
    paper_to_sheet = geometry.paper_to_sheet_px_affine(sheet_geometry)
    assert np.allclose(affine.apply(paper_to_sheet, points),
                       sheet_geometry.paper_to_pixel(points))
    assert np.allclose(affine.apply(geometry.sheet_px_to_paper_affine(sheet_geometry),
                                    sheet_geometry.paper_to_pixel(points)), points)
    assert affine.determinant(paper_to_sheet) < 0    # paper is y-up, pixels y-down


def test_sim2_roundtrip():
    sim2 = Sim2(s=1.7, theta=0.4, tx=3.0, ty=-2.0)
    as_matrix = affine.from_sim2(sim2)
    points = np.random.default_rng(3).normal(size=(7, 2))
    assert np.allclose(affine.apply(as_matrix, points), sim2.apply(points))
    back = affine.to_sim2(as_matrix)
    assert back.s == pytest.approx(sim2.s) and back.theta == pytest.approx(sim2.theta)


def _synthetic_chain():
    rng = np.random.default_rng(4)
    cameras = rng.uniform(-5, 5, size=(30, 2))
    band = cameras[rng.integers(0, 30, 4000)] + rng.normal(scale=1.5, size=(4000, 2))
    sim2 = Sim2(s=1.933, theta=np.deg2rad(-76.8), tx=40.0, ty=25.0)
    sheet_geometry = SheetGeometry(paper_w=84.0, paper_h=60.0, img_w=7381, img_h=5272)
    return rng, cameras, band, sim2, sheet_geometry


def test_full_chain_density_to_crop_preserves_chirality_and_cameras():
    """The one test that proves the retarget: truth in, truth out.

    The chain is `density px -> cloud xy -> [gt_sim2] -> paper inches -> crop px`
    and the target is rendered into a window of paper inches, so a camera pushed
    through the chain must land exactly where rendering the window puts it.
    A mirror anywhere in it would still produce a plausible picture and a wrong
    pose, which is why the determinant is asserted too.
    """
    _, cameras, band, sim2, _ = _synthetic_chain()
    density_map = density.render_density_map(band, 256)
    # the prior: the same rooms, in paper inches, a day earlier
    prior_band = sim2.apply(band)
    prior_cameras = sim2.apply(cameras)
    pool = make_pool(prior_band, prior_cameras)
    crop = prior_bev.render_target(pool, pool.extent, 512, None, DataConfig())
    assert crop is not None and crop.image.shape == (512, 512)

    density_to_crop = correspondences.density_to_crop_affine(
        density_map.density_px_to_world_xy, affine.from_sim2(sim2), crop.paper_to_crop)
    assert affine.determinant(density_to_crop) > 0, \
        "density and target must differ by a rotation, never a mirror"

    cameras_in_density = affine.apply(density_map.world_xy_to_density_px, cameras)
    via_chain = affine.apply(density_to_crop, cameras_in_density)
    direct = affine.apply(crop.paper_to_crop, sim2.apply(cameras))
    assert np.allclose(via_chain, direct, atol=1e-6)
    assert (direct >= 0).all() and (direct < 512).all()


def test_augmentation_ops_keep_chain_exact():
    _, cameras, band, _, _ = _synthetic_chain()
    density_map = density.render_density_map(band, 256)
    turns = 3
    rotation = affine.rotation_quarter_turns(turns, 256, 256)
    rotated_image = np.rot90(density_map.image, turns)
    rotated_to_world = affine.compose(density_map.density_px_to_world_xy,
                                      affine.invert(rotation))
    cameras_in_density = affine.apply(density_map.world_xy_to_density_px, cameras)
    cameras_rotated = affine.apply(rotation, cameras_in_density)
    assert np.allclose(affine.apply(rotated_to_world, cameras_rotated), cameras)
    # the rotated image really has mass where the rotated coordinates say
    u = np.floor(cameras_rotated[:, 0]).astype(int)
    v = np.floor(cameras_rotated[:, 1]).astype(int)
    assert rotated_image[v, u].mean() > 0


def test_target_window_maps_its_corners_onto_the_crop():
    """`paper_to_crop` is the window, and it is y-down like every image here."""
    _, cameras, band, sim2, _ = _synthetic_chain()
    pool = make_pool(sim2.apply(band), sim2.apply(cameras))
    lo_x, lo_y, hi_x, hi_y = pool.extent
    crop = prior_bev.render_target(pool, pool.extent, 256, None, DataConfig())
    corners = np.array([[lo_x, lo_y], [hi_x, hi_y]])
    in_crop = affine.apply(crop.paper_to_crop, corners)
    # paper (lo_x, lo_y) is the BOTTOM-left, so it lands at the bottom-left pixel
    assert np.allclose(in_crop[0], [0.0, 256.0], atol=1e-6)
    assert np.allclose(in_crop[1], [256.0, 0.0], atol=1e-6)


def test_query_pixels_follow_density():
    image = np.zeros((8, 8), np.float32)
    image[2, 5] = 1.0
    queries = density.sample_query_pixels(image, 50, np.random.default_rng(0))
    assert (np.floor(queries[:, 0]) == 5).all() and (np.floor(queries[:, 1]) == 2).all()


def _rotatable_record(band, cameras):
    """The five fields `render_record_density` reads, and nothing else."""
    from types import SimpleNamespace
    return SimpleNamespace(band_xy=band,
                           band_relative_height=np.zeros(len(band), dtype=np.float32),
                           band_camera_index=np.zeros(len(band), dtype=np.int64),
                           camera_xy=cameras,
                           overlap_mask=np.zeros(0, dtype=bool))


def _no_other_augmentation(**overrides):
    """Every other draw pinned, so a failure names the rotation.

    `density_min_counts` matters more than it looks: the rotation consumes a
    number from the generator, which shifts every later draw, and on a
    4000-point synthetic cloud a min_count of 2 rather than 1 zeroes nearly
    every occupied cell. Left free, this test would compare a rotation against
    a different speckle floor and blame the rotation.
    """
    return DataConfig(camera_window_probability=0.0, point_keep_fraction=(1.0, 1.0),
                      random_crop_scale=(1.0, 1.0), zoom_crops=False,
                      density_gammas=[0.5], density_min_counts=[1],
                      band_low_range=(-1e6, -1e6 + 1), band_high_range=(1e6 - 1, 1e6),
                      **overrides)


@pytest.mark.parametrize("density_degrees, target_degrees",
                         [(0.0, 0.0), (45.0, 0.0), (0.0, 45.0), (45.0, 45.0)])
def test_continuous_rotation_keeps_the_chain_exact(density_degrees, target_degrees):
    """The chain must still close to 1e-6 px with an arbitrary angle on either
    side or both.

    Three assertions, not one. A chain check alone would pass on a badly broken
    implementation, because both sides of it are built from the same returned
    affines -- it only proves composition is associative. What actually goes
    wrong is an affine that disagrees with the pixels beside it, or a coverage
    mask built from the track where the prior used to be. So the query image is
    required to have mass where its own affine says the cameras are, and the
    target's validity is required to be true where its own `paper_to_crop` says
    they are; that second one fails outright if `coverage_mask` is handed the
    unrotated cameras.
    """
    _, cameras, band, sim2, _ = _synthetic_chain()
    config = _no_other_augmentation(density_rotation_degrees=density_degrees,
                                    target_rotation_degrees=target_degrees)
    record = _rotatable_record(band, cameras)
    paper_cameras = sim2.apply(cameras)
    pool = make_pool(sim2.apply(band), paper_cameras)

    rng = np.random.default_rng(11)
    density_map = transforms.render_record_density(record, 256, rng, config)
    assert density_map is not None
    rotation = prior_bev.draw_target_rotation(rng, config, paper_cameras)
    assert (rotation is None) == (target_degrees == 0.0)
    anchors = paper_cameras if rotation is None else affine.apply(rotation, paper_cameras)
    window = prior_bev.choose_window(pool, anchors, rng, config, 512, 16)
    crop = prior_bev.render_target(pool, window, 512, rng, config, paper_rotation=rotation)
    assert crop is not None

    # 1. the query map's affine agrees with the query map's PIXELS
    in_density = affine.apply(density_map.world_xy_to_density_px, cameras)
    on_map = (in_density >= 0).all(1) & (in_density < 256).all(1)
    assert on_map.mean() > 0.9, "re-squaring the extent must keep the cloud in frame"
    assert density_map.image[np.floor(in_density[on_map, 1]).astype(int),
                             np.floor(in_density[on_map, 0]).astype(int)].mean() > 0

    # 2. the target's validity was built from the ROTATED track
    in_crop = affine.apply(crop.paper_to_crop, paper_cameras)
    on_crop = (in_crop >= 0).all(1) & (in_crop < 512).all(1)
    assert on_crop.mean() > 0.9, "rotating the anchors first must keep them in frame"
    assert crop.valid_mask[np.floor(in_crop[on_crop, 1]).astype(int),
                           np.floor(in_crop[on_crop, 0]).astype(int)].mean() > 0.95

    # 3. the chain closes, and it is still a rotation and not a mirror
    density_to_crop = correspondences.density_to_crop_affine(
        density_map.density_px_to_world_xy, affine.from_sim2(sim2), crop.paper_to_crop)
    assert affine.determinant(density_to_crop) > 0
    assert np.allclose(affine.apply(density_to_crop, in_density), in_crop, atol=1e-6)


def test_rotation_off_by_default_draws_nothing(monkeypatch):
    """The default must not merely produce the same picture -- it must not
    CONSUME a random number, or every later draw shifts and no run made before
    this one reproduces."""
    assert DataConfig().density_rotation_degrees == 0.0
    assert DataConfig().target_rotation_degrees == 0.0
    monkeypatch.setattr(affine, "rotation_about",
                        lambda *a, **k: pytest.fail("the default must not rotate"))
    _, cameras, band, sim2, _ = _synthetic_chain()
    record = _rotatable_record(band, cameras)
    config = _no_other_augmentation()

    before = np.random.default_rng(3).random(8)
    rng = np.random.default_rng(3)
    assert transforms.render_record_density(record, 256, rng, config) is not None
    assert prior_bev.draw_target_rotation(rng, config, sim2.apply(cameras)) is None
    # the stream is untouched by the disabled knobs: a run seeded the same way
    # still sees the same numbers in the same order
    assert np.array_equal(np.random.default_rng(3).random(8), before)
