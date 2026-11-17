"""Ground-truth pairs: density pixels and where they truly sit on the crop.

    density px --> cloud xy --Sim2--> paper inches --> crop px

Every link is a 2x3 affine, so the whole chain is one matrix, and the only
thing sampled is *which* density pixels: those with mass, in proportion to
it, exactly as at inference.

One link shorter than the floor-plan chain, which had to pass through the
sheet's own pixel grid on the way. The prior's window is chosen in paper inches
and rendered at the crop's resolution, so paper lands on crop directly.
"""
from __future__ import annotations

import numpy as np

from ..utils import affine
from . import density


def density_to_crop_affine(density_px_to_world_xy: np.ndarray, gt_sim2_affine: np.ndarray,
                           paper_to_crop: np.ndarray) -> np.ndarray:
    return affine.chain(density_px_to_world_xy, gt_sim2_affine, paper_to_crop)


def sample_ground_truth_pairs(density_image: np.ndarray, density_to_crop: np.ndarray,
                              crop_valid_mask: np.ndarray, num_pairs: int,
                              rng: np.random.Generator, min_inside: int = 64,
                              oversample: int = 2):
    """`num_pairs` (query, target) pairs whose target lands where the prior walked.

    Draws `oversample * num_pairs` candidates, keeps those inside, and fills up
    to `num_pairs` by resampling the survivors. Returns (query_px, target_px) as
    (num_pairs, 2) float32 each, or None when fewer than `min_inside` survive.

    `crop_valid_mask` is the prior's coverage, so this is also what makes a
    partially-overlapping walk trainable: the queries whose truth falls where
    the prior never went are dropped rather than taught as correspondences to
    empty space.
    """
    crop_size = crop_valid_mask.shape[0]
    query_px = density.sample_query_pixels(density_image, oversample * num_pairs, rng)
    target_px = affine.apply(density_to_crop, query_px)
    inside = ((target_px[:, 0] >= 0) & (target_px[:, 0] < crop_size)
              & (target_px[:, 1] >= 0) & (target_px[:, 1] < crop_size))
    inside_px = target_px[inside]
    on_content = np.zeros(len(target_px), dtype=bool)
    on_content[inside] = crop_valid_mask[inside_px[:, 1].astype(int),
                                         inside_px[:, 0].astype(int)]
    if on_content.sum() < min_inside:
        return None
    keep = np.flatnonzero(on_content)
    if len(keep) >= num_pairs:
        keep = keep[:num_pairs]
    else:
        keep = np.concatenate([keep, rng.choice(keep, num_pairs - len(keep), replace=True)])
    return query_px[keep].astype(np.float32), target_px[keep].astype(np.float32)
