"""The regression loss: confidence-weighted Huber on the predicted coordinates
(paper eq. 3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
    return F.huber_loss(residual, torch.zeros_like(residual), reduction="none", delta=delta)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor,
                  mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        weights = weights * mask.to(weights.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-9)


def coordinate_regression_loss(predicted_px: torch.Tensor, target_px: torch.Tensor,
                               confidence: torch.Tensor, delta: float = 1.0,
                               detach_confidence: bool = False) -> torch.Tensor:
    """`predicted_px`, `target_px`: (B,Q,2) in patch units; `confidence`: (B,Q)."""
    weights = confidence.detach() if detach_confidence else confidence
    distance = (predicted_px - target_px).norm(dim=-1)
    return weighted_mean(huber(distance, delta), weights)
