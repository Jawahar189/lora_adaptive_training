"""`prior_bev.cut_split_priors`: a split of a parent keeps only the walls it walked itself."""
import numpy as np

from bevaligner.data.prior_bev import PriorSplit, cut_split_priors

# one shared cloud: a wall along x from 0 to 100, walked by a split at each end
WALL = np.column_stack([np.arange(0.0, 101.0), np.full(101, 5.0)])


def split(reality_id, cameras_x, parent="RLTparent", capture_id="SNPa", xy=WALL):
    cameras = np.column_stack([cameras_x, np.zeros(len(cameras_x))])
    return PriorSplit(reality_id=reality_id, capture_id=capture_id, capture_date="2026-08-20",
                      band_paper_xy=xy, band_relative_height=np.zeros(len(xy)),
                      camera_paper_xy=cameras, camera_paper_z_median=0.0,
                      camera_names=(), rig_height_in=1.0, parent=parent)


def test_siblings_share_the_cloud_out_by_nearest_camera():
    out = {s.reality_id: s for s in cut_split_priors([split("left", [0, 10]),
                                                      split("right", [90, 100])])}
    assert out["left"].band_paper_xy[:, 0].max() == 50
    assert out["right"].band_paper_xy[:, 0].min() == 50       # a tie stays with both


def test_a_split_without_a_parent_is_left_whole():
    whole = split("whole", [0, 10], parent="")
    out = {s.reality_id: s for s in cut_split_priors([whole, split("right", [90, 100])])}
    assert out["whole"] is whole


def test_only_the_same_capture_competes():
    alone = split("alone", [0, 10])
    out = cut_split_priors([alone, split("later", [90, 100], capture_id="SNPb")])
    assert out[0] is alone


def test_a_split_with_no_wall_of_its_own_is_dropped():
    far = split("far", [500, 510], xy=WALL[:10])
    out = cut_split_priors([far, split("near", [0, 10])])
    assert [s.reality_id for s in out] == ["near"]
