"""The feature loss: symmetric InfoNCE over the ground-truth pairs (paper eq. 2).

With a thousand queries on a capture that spans a few dozen prior
patches, many pairs share a target patch. Treating those as negatives asks
the features to tell apart points that are the same to the network, and
pins the loss at the log of the duplication (measured: 2.8 nats on our data).
Pairs whose targets fall within `near_duplicate_radius` patches of the
anchor's are therefore excluded from the denominator, in both directions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _near_duplicate_mask(positions: torch.Tensor, radius: float) -> torch.Tensor:
    """(B,Q,Q) True where pair j is a near-duplicate of pair i (and j != i)."""
    distance = torch.cdist(positions, positions)
    self_pairs = torch.eye(positions.shape[1], device=positions.device, dtype=torch.bool)
    return (distance < radius) & ~self_pairs


def feature_infonce_loss(query_features: torch.Tensor, target_features: torch.Tensor,
                         temperature: float = 0.07,
                         target_patch_units: torch.Tensor | None = None,
                         query_patch_units: torch.Tensor | None = None,
                         near_duplicate_radius: float = 1.0) -> torch.Tensor:
    """`query_features`, `target_features`: (B,Q,C) unit features of the density
    queries and of the prior at their true locations. Pair i matches pair
    i; every other pair is a negative, in both directions, except
    near-duplicates when the (B,Q,2) patch-unit positions are given."""
    similarity = torch.bmm(query_features.float(),
                           target_features.float().transpose(1, 2)) / temperature   # (B,Q,Q)
    batch, num_queries, _ = similarity.shape
    target_index = torch.arange(num_queries, device=similarity.device).expand(batch, num_queries)
    density_to_target, target_to_density = similarity, similarity.transpose(1, 2)
    if target_patch_units is not None and near_duplicate_radius > 0:
        # density query i vs prior targets j: drop j whose target sits on i's
        density_to_target = density_to_target.masked_fill(
            _near_duplicate_mask(target_patch_units.float(), near_duplicate_radius),
            float("-inf"))
    if query_patch_units is not None and near_duplicate_radius > 0:
        # prior target i vs density queries j: drop j sitting on query i
        target_to_density = target_to_density.masked_fill(
            _near_duplicate_mask(query_patch_units.float(), near_duplicate_radius),
            float("-inf"))
    loss_density_to_target = F.cross_entropy(
        density_to_target.reshape(batch * num_queries, num_queries), target_index.reshape(-1))
    loss_target_to_density = F.cross_entropy(
        target_to_density.reshape(batch * num_queries, num_queries), target_index.reshape(-1))
    return 0.5 * (loss_density_to_target + loss_target_to_density)
