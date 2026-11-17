"""The geometry loss: pairwise distances must scale by one common factor
(paper eq. 5).

The log-ratio of a predicted to a query distance is the same for every pair
under a similarity. Its confidence-weighted mean is the scale the batch item
implies; deviations from that (stop-gradient on the mean) are penalised.
"""
from __future__ import annotations

import torch

from .regression import huber


def sample_index_pairs(batch: int, num_queries: int, num_pairs: int, device,
                       generator=None) -> torch.Tensor:
    index = torch.randint(0, num_queries, (batch, num_pairs, 2), device=device,
                          generator=generator)
    repeated = index[..., 0] == index[..., 1]
    index[..., 1] = torch.where(repeated, (index[..., 0] + 1) % num_queries, index[..., 1])
    return index


def scale_consistency_loss(predicted_px: torch.Tensor, query_px: torch.Tensor,
                           confidence: torch.Tensor, num_pairs: int = 2048,
                           delta: float = 1.0, eps: float = 1e-3,
                           detach_confidence: bool = False, generator=None) -> torch.Tensor:
    batch, num_queries, _ = predicted_px.shape
    if num_queries < 2:
        return predicted_px.sum() * 0
    index = sample_index_pairs(batch, num_queries, num_pairs, predicted_px.device, generator)

    def gather(values, corner):
        return torch.gather(values, 1, index[..., corner:corner + 1]
                            .expand(-1, -1, values.shape[-1]))

    predicted_distance = (gather(predicted_px, 0) - gather(predicted_px, 1)).norm(dim=-1)
    query_detached = query_px.detach()
    query_distance = (gather(query_detached, 0) - gather(query_detached, 1)).norm(dim=-1)
    usable = query_distance > eps
    log_ratio = torch.log(predicted_distance.clamp_min(eps)
                          / query_distance.clamp_min(eps))              # (B, num_pairs)
    weights = confidence.detach() if detach_confidence else confidence
    pair_weight = (torch.gather(weights, 1, index[..., 0])
                   * torch.gather(weights, 1, index[..., 1])
                   * usable.to(predicted_px.dtype))
    implied_log_scale = ((log_ratio * pair_weight).sum(dim=1, keepdim=True)
                         / pair_weight.sum(dim=1, keepdim=True).clamp_min(1e-9))
    per_pair = huber(log_ratio - implied_log_scale.detach(), delta)
    return (per_pair * pair_weight).sum() / pair_weight.sum().clamp_min(1e-9)
