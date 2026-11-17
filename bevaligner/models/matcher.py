"""From two feature maps to correspondences.

For each query on the density map, its feature is sampled bilinearly, compared
by cosine to every prior patch, and the softmax at `temperature` becomes a
distribution over patches. Its expectation over patch centres is the predicted
prior coordinate and its peak the confidence -- the paper's soft-argmax
head, with no learned parameters beyond the LoRA in the encoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DinoPatchEncoder


@dataclass
class MatchOutput:
    density_features: torch.Tensor        # (B, C, h, w) density-map patch features
    target_features: torch.Tensor      # (B, C, h', w') prior patch features
    query_features: torch.Tensor          # (B, Q, C) features at the queries
    similarity_logits: torch.Tensor       # (B, Q, h'w') cosine / temperature, -inf on padding
    patch_probabilities: torch.Tensor     # (B, Q, h'w')
    predicted_px: torch.Tensor            # (B, Q, 2) predicted prior crop pixels
    confidence: torch.Tensor              # (B, Q) peak probability
    patch_centres_px: torch.Tensor        # (h'w', 2) patch centres in crop pixels


def patch_centre_pixels(rows: int, cols: int, patch_size: int, device=None) -> torch.Tensor:
    ys = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) * patch_size
    xs = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) * patch_size
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)


def prediction_spread_patches(predicted_px: torch.Tensor, patch_size: int) -> torch.Tensor:
    """How far apart one sample's predictions are, in patches: the collapse metric.

    Near zero means every query of a sample landed on the same spot, which is
    what a flat softmax gives -- the prediction is the mean of the valid patch
    centres, so all queries share it. It starts near zero by construction and
    should rise as the features sharpen; falling back is the collapse.
    """
    return (predicted_px.std(dim=1).norm(dim=-1) / patch_size).mean()


def sample_features_at_pixels(feature_map: torch.Tensor, pixels: torch.Tensor,
                              image_hw) -> torch.Tensor:
    """Bilinear features at continuous pixel positions.

    `feature_map` is (B,C,h,w) for an image of `image_hw` pixels; `pixels` is
    (B,Q,2) as (u, v). `align_corners=False` makes normalised -1..1 span the
    image edges, consistent with pixel centres at half-integers. Re-normalised,
    because a blend of unit vectors is shorter than one.
    """
    height, width = image_hw
    grid = torch.empty_like(pixels)
    grid[..., 0] = pixels[..., 0] / width * 2 - 1
    grid[..., 1] = pixels[..., 1] / height * 2 - 1
    sampled = F.grid_sample(feature_map, grid.unsqueeze(2).to(feature_map.dtype),
                            mode="bilinear", padding_mode="border",
                            align_corners=False)                      # (B,C,Q,1)
    return F.normalize(sampled[..., 0].transpose(1, 2), dim=-1)       # (B,Q,C)


class DensityPriorMatcher(nn.Module):
    """One shared encoder; queries on the density map, targets on the prior BEV window."""

    def __init__(self, encoder: DinoPatchEncoder, temperature: float = 0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    @property
    def patch_size(self) -> int:
        return self.encoder.patch_size

    def encode_density_and_target(self, density: torch.Tensor, target: torch.Tensor):
        """One forward for both when they share a size (training), two otherwise."""
        if density.shape[-2:] == target.shape[-2:]:
            both = self.encoder(torch.cat([density, target], dim=0))
            return both[: density.shape[0]], both[density.shape[0]:]
        return self.encoder(density), self.encoder(target)

    def match_queries_to_patches(self, query_features: torch.Tensor,
                                 target_features: torch.Tensor,
                                 target_patch_valid: torch.Tensor | None = None):
        """(B,Q,C) query features vs (B,C,h,w) prior map -> logits,
        probabilities, predicted pixels, confidence and the patch centres."""
        batch, _, rows, cols = target_features.shape
        flat_features = target_features.flatten(2)                 # (B, C, hw)
        with torch.autocast(device_type=flat_features.device.type, enabled=False):
            logits = torch.bmm(query_features.float(),
                               flat_features.float()) / self.temperature   # (B, Q, hw)
            if target_patch_valid is not None:
                valid = target_patch_valid.reshape(batch, 1, rows * cols)
                logits = logits.masked_fill(~valid, float("-inf"))
            probabilities = logits.softmax(dim=-1)
            centres = patch_centre_pixels(rows, cols, self.patch_size,
                                          device=flat_features.device)
            predicted_px = probabilities @ centres                    # (B, Q, 2)
            confidence = probabilities.max(dim=-1).values
        return logits, probabilities, predicted_px, confidence, centres

    def forward(self, density: torch.Tensor, target: torch.Tensor,
                query_px: torch.Tensor,
                target_patch_valid: torch.Tensor | None = None) -> MatchOutput:
        density_features, target_features = self.encode_density_and_target(
            density, target)
        query_features = sample_features_at_pixels(density_features, query_px,
                                                   density.shape[-2:])
        logits, probabilities, predicted_px, confidence, centres = \
            self.match_queries_to_patches(query_features, target_features,
                                          target_patch_valid)
        return MatchOutput(density_features=density_features,
                           target_features=target_features,
                           query_features=query_features,
                           similarity_logits=logits,
                           patch_probabilities=probabilities,
                           predicted_px=predicted_px, confidence=confidence,
                           patch_centres_px=centres)
