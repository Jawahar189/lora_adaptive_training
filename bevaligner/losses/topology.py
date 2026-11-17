"""The topology loss: the angle at every sampled triangle vertex must survive
the mapping (paper eq. 4). A similarity preserves angles, and the *signed* sine
also carries chirality, so a mirrored prediction is punished, not tolerated."""
from __future__ import annotations

import torch

from .regression import huber, weighted_mean


def sample_index_triplets(batch: int, num_queries: int, num_triplets: int, device,
                          generator=None) -> torch.Tensor:
    """(batch, num_triplets, 3) indices with i, j, k distinct."""
    index = torch.randint(0, num_queries, (batch, num_triplets, 3), device=device,
                          generator=generator)
    repeated = ((index[..., 0] == index[..., 1]) | (index[..., 0] == index[..., 2])
                | (index[..., 1] == index[..., 2]))
    index[..., 1] = torch.where(repeated, (index[..., 0] + 1) % num_queries, index[..., 1])
    index[..., 2] = torch.where(repeated, (index[..., 0] + 2) % num_queries, index[..., 2])
    return index


def _vertex_angle_cos_sin(points: torch.Tensor, index: torch.Tensor, eps: float):
    """cos and signed sin of the angle at vertex i between (j - i) and (k - i)."""
    vertex_i = torch.gather(points, 1, index[..., 0:1].expand(-1, -1, 2))
    vertex_j = torch.gather(points, 1, index[..., 1:2].expand(-1, -1, 2))
    vertex_k = torch.gather(points, 1, index[..., 2:3].expand(-1, -1, 2))
    edge_ij, edge_ik = vertex_j - vertex_i, vertex_k - vertex_i
    length_ij, length_ik = edge_ij.norm(dim=-1), edge_ik.norm(dim=-1)
    denominator = (length_ij * length_ik).clamp_min(eps)
    cosine = (edge_ij * edge_ik).sum(-1) / denominator
    sine = (edge_ij[..., 0] * edge_ik[..., 1]
            - edge_ij[..., 1] * edge_ik[..., 0]) / denominator
    well_defined = (length_ij > eps) & (length_ik > eps)
    return cosine, sine, well_defined


def angle_topology_loss(predicted_px: torch.Tensor, query_px: torch.Tensor,
                        confidence: torch.Tensor, num_triplets: int = 2048,
                        delta: float = 1.0, eps: float = 1e-3,
                        detach_confidence: bool = False, generator=None) -> torch.Tensor:
    """`predicted_px`: (B,Q,2) predicted floor-plan coords; `query_px`: (B,Q,2)
    the queries' density-map coords; `confidence`: (B,Q). Units only matter
    for `eps`."""
    batch, num_queries, _ = predicted_px.shape
    if num_queries < 3:
        return predicted_px.sum() * 0
    index = sample_index_triplets(batch, num_queries, num_triplets,
                                  predicted_px.device, generator)
    cos_pred, sin_pred, ok_pred = _vertex_angle_cos_sin(predicted_px, index, eps)
    cos_query, sin_query, ok_query = _vertex_angle_cos_sin(query_px.detach(), index, eps)
    weights = confidence.detach() if detach_confidence else confidence
    triplet_weight = (torch.gather(weights, 1, index[..., 0])
                      * torch.gather(weights, 1, index[..., 1])
                      * torch.gather(weights, 1, index[..., 2]))
    per_triplet = huber(cos_pred - cos_query, delta) + huber(sin_pred - sin_query, delta)
    return weighted_mean(per_triplet, triplet_weight, ok_query & ok_pred)
