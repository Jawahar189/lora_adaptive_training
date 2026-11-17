import math

import torch
import pytest

from bevaligner.losses import (active_loss_weights, angle_topology_loss,
                                 coordinate_regression_loss, feature_infonce_loss,
                                 scale_consistency_loss)
from bevaligner.losses.curriculum import DEFAULT_LOSS_WEIGHTS


def _apply_sim2(points, scale, theta, translation):
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    rotation = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return scale * points @ rotation.T + torch.tensor(translation)


@pytest.fixture
def pairs():
    generator = torch.Generator().manual_seed(0)
    queries = torch.rand(2, 256, 2, generator=generator) * 32
    targets = _apply_sim2(queries, 1.7, 0.6, [5.0, -3.0])
    confidence = torch.rand(2, 256, generator=generator) * 0.5 + 0.5
    return queries, targets, confidence


def test_regression_is_zero_at_truth_positive_otherwise(pairs):
    _, targets, confidence = pairs
    assert coordinate_regression_loss(targets, targets, confidence).item() == 0
    assert coordinate_regression_loss(targets + 0.5, targets, confidence).item() > 0
    # Huber: linear beyond delta, so a 10-patch error costs ~10x less than squared
    big = coordinate_regression_loss(targets + torch.tensor([10.0, 0.0]), targets,
                                     confidence, delta=1.0).item()
    assert 9 < big < 10


def test_topology_and_geometry_are_invariant_to_a_similarity(pairs):
    queries, targets, confidence = pairs
    generator = torch.Generator().manual_seed(1)
    assert angle_topology_loss(targets, queries, confidence,
                               generator=generator).item() < 1e-6
    generator = torch.Generator().manual_seed(1)
    assert scale_consistency_loss(targets, queries, confidence,
                                  generator=generator).item() < 1e-6


def test_topology_punishes_a_mirror_geometry_does_not(pairs):
    queries, targets, confidence = pairs
    mirrored = targets * torch.tensor([-1.0, 1.0])
    generator = torch.Generator().manual_seed(2)
    assert angle_topology_loss(mirrored, queries, confidence,
                               generator=generator).item() > 0.5
    generator = torch.Generator().manual_seed(2)
    assert scale_consistency_loss(mirrored, queries, confidence,
                                  generator=generator).item() < 1e-6


def test_geometry_punishes_anisotropic_scale(pairs):
    queries, targets, confidence = pairs
    stretched = targets * torch.tensor([2.0, 1.0])
    assert scale_consistency_loss(stretched, queries, confidence).item() > 0.01


def test_topology_and_geometry_have_gradients_through_prediction_and_confidence(pairs):
    queries, targets, confidence = pairs
    predicted = (targets + torch.randn_like(targets) * 0.3).requires_grad_()
    confidence = confidence.clone().requires_grad_()
    (angle_topology_loss(predicted, queries, confidence)
     + scale_consistency_loss(predicted, queries, confidence)).backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert confidence.grad is not None


def test_feature_loss_prefers_the_diagonal():
    generator = torch.Generator().manual_seed(3)
    query_features = torch.nn.functional.normalize(
        torch.randn(2, 64, 16, generator=generator), dim=-1)
    unrelated = torch.nn.functional.normalize(
        torch.randn(2, 64, 16, generator=generator), dim=-1)
    assert feature_infonce_loss(query_features, query_features) < \
        feature_infonce_loss(query_features, unrelated)
    # perfect one-hot matching drives it to ~0
    one_hot = torch.eye(64).expand(2, 64, 64)
    assert feature_infonce_loss(one_hot, one_hot, temperature=0.01).item() < 1e-3


def test_curriculum_gates_switch_terms_on_in_order():
    assert active_loss_weights(0.05) == {"feature": 1.0, "regression": 0.0,
                                         "topology": 0.0, "geometry": 0.0}
    assert active_loss_weights(0.15) == {"feature": 1.0, "regression": 50.0,
                                         "topology": 0.0, "geometry": 0.0}
    assert active_loss_weights(0.5) == DEFAULT_LOSS_WEIGHTS
    assert active_loss_weights(1.0) == DEFAULT_LOSS_WEIGHTS


def test_detached_confidence_keeps_the_flattening_gradient_out(pairs):
    """Attached, the normalised weighting pays for lowering confidence on an
    above-average-error query -- and confidence is the softmax peak, so the only
    way down is a flatter distribution, which is the collapse itself."""
    queries, targets, _ = pairs
    predicted = targets.clone()
    predicted[0, 0] += 20.0                      # one query far worse than the rest
    predicted = predicted.requires_grad_()       # so a detached loss still has a graph

    confidence = (torch.rand(*targets.shape[:2], generator=torch.Generator().manual_seed(1))
                  * 0.5 + 0.5).requires_grad_()
    coordinate_regression_loss(predicted, targets, confidence,
                               detach_confidence=False).backward()
    assert confidence.grad is not None
    # raising the bad query's confidence raises the loss: the model is paid to
    # lower it, and lowering it means flattening that query's softmax
    assert confidence.grad[0, 0] > 0

    detached = confidence.detach().clone().requires_grad_()
    coordinate_regression_loss(predicted, targets, detached,
                               detach_confidence=True).backward()
    assert detached.grad is None or torch.count_nonzero(detached.grad) == 0


def test_gate_ramp_brings_a_term_in_linearly():
    gates = {"feature": 0.0, "regression": 0.2, "topology": 0.4, "geometry": 0.4}
    ramped = lambda progress: active_loss_weights(progress, DEFAULT_LOSS_WEIGHTS,
                                                  gates, gate_ramp=0.1)
    # the feature term is never ramped: it is what the others wait for
    assert ramped(0.0)["feature"] == DEFAULT_LOSS_WEIGHTS["feature"]
    assert ramped(0.2)["regression"] == 0.0
    assert ramped(0.25)["regression"] == pytest.approx(DEFAULT_LOSS_WEIGHTS["regression"] / 2)
    assert ramped(0.3)["regression"] == pytest.approx(DEFAULT_LOSS_WEIGHTS["regression"])
    assert ramped(0.9)["regression"] == DEFAULT_LOSS_WEIGHTS["regression"]
    assert ramped(0.35)["topology"] == 0.0            # still waiting for its own gate
    assert ramped(0.45)["topology"] == pytest.approx(DEFAULT_LOSS_WEIGHTS["topology"] / 2)
    assert ramped(0.45)["geometry"] == pytest.approx(DEFAULT_LOSS_WEIGHTS["geometry"] / 2)


def test_gate_ramp_of_zero_is_the_paper_s_step():
    for progress in (0.05, 0.15, 0.5, 1.0):
        assert (active_loss_weights(progress, gate_ramp=0.0)
                == active_loss_weights(progress))
