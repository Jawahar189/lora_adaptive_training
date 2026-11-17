"""The paper's schedule, and one module that computes every term from a batch.

    0 - 10 %   feature alone:  the features first learn what matches what
   10 - 20 %   + regression:   then the soft-argmax is asked to land on the truth
   20 - 100 %  + topology, geometry: finally the field of predictions is made rigid

Each term after the first arrives over `gate_ramp` of training rather than at a
step: weight 50 landing whole on features that are not yet ready buries, under
the gradient clip, the InfoNCE term that has to make them ready.

Coordinates are converted to **patch units** before any Huber. One patch is
what the soft-argmax is expected to resolve, so `delta = 1` is quadratic within
it and linear beyond, and the loss means the same thing at every resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..models.matcher import MatchOutput, sample_features_at_pixels
from .feature import feature_infonce_loss
from .geometry import scale_consistency_loss
from .regression import coordinate_regression_loss
from .topology import angle_topology_loss

DEFAULT_LOSS_WEIGHTS = {"feature": 1.0, "regression": 50.0, "topology": 10.0, "geometry": 10.0}
# fraction of training after which each term switches on
DEFAULT_LOSS_GATES = {"feature": 0.0, "regression": 0.10, "topology": 0.20, "geometry": 0.20}


def active_loss_weights(progress: float, weights: dict[str, float] = DEFAULT_LOSS_WEIGHTS,
                        gates: dict[str, float] = DEFAULT_LOSS_GATES,
                        gate_ramp: float = 0.0) -> dict[str, float]:
    """Loss weights active at `progress` in [0, 1].

    A term reaches full weight `gate_ramp` of training after its gate, linearly.
    `gate_ramp = 0` is the step the paper describes; a ramp is the same schedule
    with the arrival spread out, so the term that switches on does not bury the
    one already training the features it needs. A gate at 0 -- the feature term
    -- is full from the first step, never ramped.
    """
    def factor(gate: float) -> float:
        if gate <= 0.0:
            return 1.0
        if gate_ramp <= 0.0:
            return 1.0 if progress >= gate else 0.0
        return min(max((progress - gate) / gate_ramp, 0.0), 1.0)

    return {name: weight * factor(gates[name]) for name, weight in weights.items()}


@dataclass
class LossSettings:
    temperature: float = 0.07
    huber_delta: float = 1.0
    num_triplets: int = 2048
    num_pairs: int = 2048
    detach_confidence: bool = True
    # patches; InfoNCE negatives closer than this are ignored
    near_duplicate_radius: float = 1.0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    gates: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_GATES))
    gate_ramp: float = 0.05


class BEVAlignerLoss(nn.Module):
    """All four terms, weighted by where training has got to."""

    def __init__(self, settings: LossSettings = LossSettings()):
        super().__init__()
        self.settings = settings

    def forward(self, output: MatchOutput, query_px: torch.Tensor, target_px: torch.Tensor,
                target_hw, patch_size: int, progress: float) -> dict[str, torch.Tensor]:
        """`query_px`: (B,Q,2) density queries, `target_px`: (B,Q,2) their true
        crop pixels. Returns every term (unweighted), the active weights and
        `total`."""
        settings = self.settings
        weights = active_loss_weights(progress, settings.weights, settings.gates,
                                      settings.gate_ramp)
        terms: dict[str, torch.Tensor] = {}

        predicted_units = output.predicted_px / patch_size
        target_units = target_px / patch_size
        query_units = query_px / patch_size
        target_features = sample_features_at_pixels(output.target_features, target_px,
                                                    target_hw)
        terms["feature"] = feature_infonce_loss(
            output.query_features, target_features, settings.temperature,
            target_units, query_units, settings.near_duplicate_radius)
        if weights["regression"] > 0:
            terms["regression"] = coordinate_regression_loss(
                predicted_units, target_units, output.confidence,
                settings.huber_delta, settings.detach_confidence)
        if weights["topology"] > 0:
            terms["topology"] = angle_topology_loss(
                predicted_units, query_units, output.confidence, settings.num_triplets,
                settings.huber_delta, detach_confidence=settings.detach_confidence)
        if weights["geometry"] > 0:
            terms["geometry"] = scale_consistency_loss(
                predicted_units, query_units, output.confidence, settings.num_pairs,
                settings.huber_delta, detach_confidence=settings.detach_confidence)

        total = sum(weights[name] * value for name, value in terms.items())
        logs = {f"loss/{name}": value.detach() for name, value in terms.items()}
        logs.update({f"weight/{name}": torch.tensor(value) for name, value in weights.items()})
        logs["total"] = total
        return logs
