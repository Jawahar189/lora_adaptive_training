"""Run the matcher on one density map and one crop; keep the reliable pairs.

Queries are drawn in proportion to density, the top half by confidence is
kept, and a pair survives only if its query is also the best query for the
patch it chose (mutual nearest neighbours). Coordinates come back in the two
frames RANSAC wants: gravity-up cloud xy and paper inches.

Two matchers share that contract. `dense` (`match_density_to_crop`) lets every
query land anywhere on the prior: the soft-argmax over its patches. `keypoints`
(`match_density_keypoints_to_crop`) draws points on the prior the same way the
queries are drawn on the density map and matches point to point, so every match
lands on a prior point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data import density as density_module
from ..data import prior_bev
from ..models.matcher import sample_features_at_pixels
from ..utils import affine
from ..utils import image as image_utils

MATCHERS = ("dense", "keypoints")


def inference_autocast(device):
    """bf16 autocast where the GPU has native bf16 (compute capability >= 8: A100,
    L4); plain fp32 everywhere else (T4 is 7.5, and CPU). The matcher trains in
    bf16-mixed, so bf16 is its own precision; fp16 is never used."""
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)


@dataclass
class Matches:
    """Surviving correspondences, in every frame the caller needs."""

    source_world_xy: np.ndarray     # (K, 2) gravity-up cloud xy
    target_paper_xy: np.ndarray     # (K, 2) paper inches
    target_crop_px: np.ndarray      # (K, 2) crop pixels
    confidence: np.ndarray          # (K,)
    query_px: np.ndarray            # (K, 2) density-map pixels
    num_queries: int


def density_to_input_tensor(density_map: density_module.DensityMap) -> torch.Tensor:
    return image_utils.imagenet_normalize(image_utils.rgb_to_tensor(density_map.image))


def crop_to_input_tensors(crop: prior_bev.TargetCrop,
                          patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (image_utils.imagenet_normalize(image_utils.rgb_to_tensor(crop.image)),
            torch.from_numpy(prior_bev.patch_valid_mask(crop.valid_mask, patch_size)))


@torch.no_grad()
def match_density_to_crop(model, density_map: density_module.DensityMap,
                          crop: prior_bev.TargetCrop, *, num_queries: int = 1024,
                          keep_fraction: float = 0.5, mutual_check: bool = True,
                          mutual_radius_patches: float = 1.5,
                          rng: np.random.Generator | None = None, device=None,
                          density_tensor: torch.Tensor | None = None,
                          crop_tensors=None) -> Matches:
    rng = rng or np.random.default_rng(0)
    device = device or next(model.parameters()).device
    query_px = density_module.sample_query_pixels(density_map.image, num_queries, rng,
                                                  cdf=density_map.query_cdf)
    density_input = (density_to_input_tensor(density_map) if density_tensor is None
                     else density_tensor)[None].to(device)
    target_input, patch_valid = (crop_to_input_tensors(crop, model.patch_size)
                                    if crop_tensors is None else crop_tensors)
    target_input, patch_valid = target_input[None].to(device), patch_valid[None].to(device)
    query_tensor = torch.from_numpy(query_px.astype(np.float32))[None].to(device)

    with inference_autocast(device):
        output = model(density_input, target_input, query_tensor,
                       target_patch_valid=patch_valid)
    confidence = output.confidence[0].float().cpu().numpy()
    predicted_crop_px = output.predicted_px[0].float().cpu().numpy()

    keep = np.ones(len(query_px), dtype=bool)
    if keep_fraction < 1:
        num_kept = max(int(len(query_px) * keep_fraction), 3)
        keep[np.argsort(-confidence)[num_kept:]] = False
    if mutual_check:
        # Mutual nearest neighbours, at the granularity the features have. Many
        # queries share one prior patch (a capture spans a few dozen patches
        # and there are a thousand queries), so an exact reverse-argmax check
        # keeps one query per patch at best. Instead the patch's favourite query
        # must lie within `mutual_radius_patches` of this query on the density map.
        logits = output.similarity_logits[0].float()             # (Q, hw)
        best_patch_per_query = logits.argmax(dim=1)
        best_query_per_patch = logits.argmax(dim=0)
        favourite_query_px = query_px[
            best_query_per_patch[best_patch_per_query].cpu().numpy()]
        distance = np.linalg.norm(favourite_query_px - query_px, axis=1)
        keep &= distance <= mutual_radius_patches * model.patch_size

    target_crop_px = predicted_crop_px[keep]
    target_paper_xy = affine.apply(affine.invert(crop.paper_to_crop), target_crop_px)
    return Matches(
        source_world_xy=affine.apply(density_map.density_px_to_world_xy, query_px[keep]),
        target_paper_xy=target_paper_xy, target_crop_px=target_crop_px,
        confidence=confidence[keep], query_px=query_px[keep], num_queries=len(query_px))


@torch.no_grad()
def match_density_keypoints_to_crop(model, density_map: density_module.DensityMap,
                                    crop: prior_bev.TargetCrop, *, num_queries: int = 1024,
                                    keep_fraction: float = 0.5,
                                    rng: np.random.Generator | None = None,
                                    device=None) -> Matches:
    """Point to point: density-sampled points on BOTH maps, mutual nearest neighbours.

    The queries are drawn on the density map exactly as `match_density_to_crop`
    draws them, and as many again on the crop, the same way. Both are described
    by the encoder's features at those pixels, compared by cosine at the model's
    temperature, and a pair survives only if each is the other's best -- exactly,
    since a prior point can be chosen once. Of the survivors the most confident
    are kept, at most `keep_fraction` of the queries. The match is the prior
    point itself, so `target_crop_px` is always a pixel with prior content.
    """
    rng = rng or np.random.default_rng(0)
    device = device or next(model.parameters()).device
    query_px = density_module.sample_query_pixels(density_map.image, num_queries, rng,
                                                  cdf=density_map.query_cdf)
    if float(np.asarray(crop.image).sum()) <= 0:              # nothing on the prior to match
        empty = np.zeros((0, 2))
        return Matches(source_world_xy=empty, target_paper_xy=empty, target_crop_px=empty,
                       confidence=np.zeros(0), query_px=empty, num_queries=len(query_px))
    prior_px = density_module.sample_query_pixels(crop.image, num_queries, rng)

    density_input = density_to_input_tensor(density_map)[None].to(device)
    target_input = crop_to_input_tensors(crop, model.patch_size)[0][None].to(device)
    with inference_autocast(device):
        density_features, target_features = model.encode_density_and_target(
            density_input, target_input)
    query_features = sample_features_at_pixels(
        density_features, torch.from_numpy(query_px.astype(np.float32))[None].to(device),
        density_input.shape[-2:])[0].float()
    prior_features = sample_features_at_pixels(
        target_features, torch.from_numpy(prior_px.astype(np.float32))[None].to(device),
        target_input.shape[-2:])[0].float()

    logits = query_features @ prior_features.T / model.temperature       # (Q, P)
    confidence_t, best_t = logits.softmax(dim=1).max(dim=1)
    mutual = logits.argmax(dim=0)[best_t] == torch.arange(len(query_px), device=device)
    confidence, best, keep = (confidence_t.cpu().numpy(), best_t.cpu().numpy(),
                              mutual.cpu().numpy())
    if keep_fraction < 1 and keep.any():
        allowed = max(int(len(query_px) * keep_fraction), min(3, int(keep.sum())))
        ranked = np.argsort(-np.where(keep, confidence, -1.0))
        keep[ranked[allowed:]] = False

    target_crop_px = prior_px[best[keep]]
    return Matches(
        source_world_xy=affine.apply(density_map.density_px_to_world_xy, query_px[keep]),
        target_paper_xy=affine.apply(affine.invert(crop.paper_to_crop), target_crop_px),
        target_crop_px=target_crop_px, confidence=confidence[keep],
        query_px=query_px[keep], num_queries=len(query_px))
