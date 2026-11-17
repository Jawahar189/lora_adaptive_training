"""LoRA on a tiny random DINOv2 so nothing is downloaded."""
import torch
import torch.nn as nn
import pytest
from transformers import Dinov2Config, Dinov2Model

from bevaligner.models import (DensityPriorMatcher, DinoPatchEncoder, LoraSettings,
                                 inject_lora_adapters, iter_lora_parameters,
                                 load_lora_state_dict, lora_state_dict,
                                 sample_features_at_pixels)
from bevaligner.models.lora import LoRALinear, count_parameters


def tiny_dinov2(seed=0) -> Dinov2Model:
    torch.manual_seed(seed)
    config = Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                          intermediate_size=64, patch_size=14, image_size=56)
    return Dinov2Model(config).eval()


def test_inject_wraps_all_targets_and_starts_as_identity():
    pretrained = tiny_dinov2()
    images = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        before = pretrained(pixel_values=images).last_hidden_state
    names = inject_lora_adapters(pretrained,
                                 ["attention.attention.query", "attention.attention.key",
                                  "attention.attention.value", "attention.output.dense",
                                  "mlp.fc1", "mlp.fc2"], rank=4, alpha=4, dropout=0.0)
    assert len(names) == 2 * 6
    assert all(isinstance(pretrained.get_submodule(name), LoRALinear) for name in names)
    with torch.no_grad():
        after = pretrained(pixel_values=images).last_hidden_state
    assert torch.allclose(before, after, atol=1e-6)   # the B matrix is zero at init


def test_only_lora_gets_gradients():
    encoder = DinoPatchEncoder(family="dinov2", pretrained=tiny_dinov2(),
                               lora=LoraSettings(rank=4, alpha=4, dropout=0.1))
    encoder.train()
    images = torch.randn(2, 3, 56, 56)
    features = encoder(images)
    assert features.shape == (2, 32, 4, 4)
    assert torch.allclose(features.norm(dim=1), torch.ones(2, 4, 4), atol=1e-5)
    features.square().sum().backward()
    lora_params = list(iter_lora_parameters(encoder.pretrained))
    assert lora_params and all(p.grad is not None for p in lora_params)
    frozen = [p for name, p in encoder.pretrained.named_parameters() if ".lora_" not in name]
    assert frozen and all(p.grad is None and not p.requires_grad for p in frozen)
    assert 0 < count_parameters(lora_params) < count_parameters(frozen)


def test_encoder_stays_eval_but_lora_dropout_follows_mode():
    encoder = DinoPatchEncoder(family="dinov2", pretrained=tiny_dinov2(),
                               lora=LoraSettings(rank=4, dropout=0.5))
    encoder.train()
    assert encoder.training and not encoder.pretrained.training
    lora = next(m for m in encoder.pretrained.modules() if isinstance(m, LoRALinear))
    assert lora.training and lora.lora_dropout.training
    encoder.eval()
    assert not lora.training


def test_lora_state_dict_roundtrip():
    encoder = DinoPatchEncoder(family="dinov2", pretrained=tiny_dinov2(),
                               lora=LoraSettings(rank=4, dropout=0.0))
    for parameter in iter_lora_parameters(encoder.pretrained):
        parameter.data.normal_()
    state = lora_state_dict(encoder.pretrained)
    assert state and all(".lora_A." in k or ".lora_B." in k for k in state)
    restored = DinoPatchEncoder(family="dinov2", pretrained=tiny_dinov2(),
                                lora=LoraSettings(rank=4, dropout=0.0))
    load_lora_state_dict(restored.pretrained, state)
    images = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        assert torch.allclose(encoder(images), restored(images), atol=1e-6)


def test_resolution_must_be_a_patch_multiple():
    encoder = DinoPatchEncoder(family="dinov2", pretrained=tiny_dinov2(), lora=None)
    with pytest.raises(ValueError):
        encoder(torch.randn(1, 3, 60, 60))


def test_sample_features_at_patch_centres_recovers_the_patch():
    feature_map = torch.nn.functional.normalize(torch.randn(1, 8, 4, 4), dim=1)
    # centre of patch (row 2, col 1) for a 56 px image with 14 px patches
    pixels = torch.tensor([[[1.5 * 14, 2.5 * 14]]])
    sampled = sample_features_at_pixels(feature_map, pixels, (56, 56))
    assert torch.allclose(sampled[0, 0], feature_map[0, :, 2, 1], atol=1e-5)


def test_soft_argmax_lands_on_the_matching_patch():
    encoder = DinoPatchEncoder(family="dinov2", pretrained=tiny_dinov2(), lora=None)
    matcher = DensityPriorMatcher(encoder, temperature=0.07).eval()
    image = torch.randn(1, 3, 56, 56)
    # same image on both sides: a query at a patch centre should match itself
    query_px = torch.tensor([[[2.5 * 14, 1.5 * 14]]])
    with torch.no_grad():
        output = matcher(image, image, query_px)
    assert output.predicted_px.shape == (1, 1, 2) and output.confidence.shape == (1, 1)
    assert (output.predicted_px[0, 0] - query_px[0, 0]).norm() < 14
    # with padding masked out, no probability lands there
    patch_valid = torch.ones(1, 4, 4, dtype=torch.bool)
    patch_valid[0, :, 3] = False
    with torch.no_grad():
        masked = matcher(image, image, query_px, target_patch_valid=patch_valid)
    assert masked.patch_probabilities[0, 0].reshape(4, 4)[:, 3].sum() == 0


def test_prediction_spread_is_zero_when_every_query_lands_on_one_point():
    """The collapse metric: a flat softmax puts every prediction on the mean of
    the patch centres, and this is what says so."""
    from bevaligner.models.matcher import prediction_spread_patches

    collapsed = torch.full((2, 64, 2), 37.0)
    assert prediction_spread_patches(collapsed, patch_size=16).item() == 0.0

    spread = torch.stack(torch.meshgrid(torch.arange(8.0), torch.arange(8.0),
                                        indexing="ij"), dim=-1).reshape(1, 64, 2) * 16
    spread = spread.expand(2, 64, 2)
    assert prediction_spread_patches(spread, patch_size=16).item() > 1.0


def test_the_adapters_are_in_train_mode_during_training_and_not_during_validation():
    """The regression this pins is that `lora_dropout` was dead for the whole
    life of the package.

    `DinoPatchEncoder` puts itself in eval in its own __init__ and nothing
    calls `.train()` on the LightningModule at fit start; worse, after each
    validation Lightning restores the captured per-module flags by writing
    `module.training` directly, which reinstates eval on the adapters. So the
    validation in the middle of this fit is the load-bearing part -- without
    it the bug does not reproduce, and a test without it would have passed all
    along.
    """
    import lightning as L
    from bevaligner.lightning.module import BEVAlignerModule

    in_training, in_validation = [], []

    class Tiny(L.LightningModule):
        # the two hooks under test, taken from the real module
        on_validation_model_train = BEVAlignerModule.on_validation_model_train
        on_train_epoch_start = BEVAlignerModule.on_train_epoch_start

        def __init__(self):
            super().__init__()
            layer = nn.Linear(4, 4)
            self.model = nn.Sequential(LoRALinear(layer, rank=2, alpha=2.0, dropout=0.1))
            self.model.train(False)          # what DinoPatchEncoder.__init__ does

        def training_step(self, batch, _):
            in_training.append(self.model[0].lora_dropout.training)
            return self.model(batch).sum()

        def validation_step(self, batch, _):
            in_validation.append(self.model[0].lora_dropout.training)
            return self.model(batch).sum()

        def configure_optimizers(self):
            return torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=0.0)

    data = torch.utils.data.DataLoader(torch.randn(8, 4), batch_size=2)
    L.Trainer(max_steps=4, val_check_interval=2, limit_val_batches=1,
              num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
              enable_progress_bar=False, enable_model_summary=False,
              accelerator="cpu").fit(Tiny(), data, data)

    assert in_training and all(in_training), in_training
    assert in_validation and not any(in_validation), in_validation
