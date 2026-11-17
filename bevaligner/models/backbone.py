"""A frozen DINO encoder with LoRA, returning a normalised patch-feature map.

DINOv3 ViT-B/16 as the paper uses; DINOv2 ViT-B/14 as the fallback that is
already on disk. Both come from `transformers`. The HF module is held in eval
mode permanently: DINOv3 jitters its rotary position encoding in training
mode, and since the density map and the sheet share one encoder, they must
see the same, deterministic positions. Only the LoRA adapters follow the
training flag (their dropout).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LORA_TARGET_SUFFIXES, inject_lora_adapters, iter_lora_modules

PRETRAINED_IDS = {
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov2": "facebook/dinov2-base",
}


@dataclass
class LoraSettings:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.1


def _load_pretrained(pretrained_id: str, family: str, dtype: torch.dtype | None) -> nn.Module:
    import transformers
    kwargs = {} if dtype is None else {"torch_dtype": dtype}
    if family == "dinov3":
        return transformers.DINOv3ViTModel.from_pretrained(pretrained_id, **kwargs)
    if family == "dinov2":
        return transformers.Dinov2Model.from_pretrained(pretrained_id, **kwargs)
    raise ValueError(f"unknown backbone family {family!r}")


def _num_prefix_tokens(pretrained: nn.Module, family: str) -> int:
    """Tokens before the patch tokens: the class token, plus DINOv3 registers."""
    if family == "dinov3":
        return 1 + int(getattr(pretrained.config, "num_register_tokens", 0))
    return 1


class DinoPatchEncoder(nn.Module):
    """Image in, unit-norm patch features out. Only the LoRA adapters train."""

    def __init__(self, family: str = "dinov3", pretrained_id: str | None = None,
                 lora: LoraSettings | dict | None = LoraSettings(),
                 gradient_checkpointing: bool = False,
                 pretrained: nn.Module | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if family not in PRETRAINED_IDS:
            raise ValueError(f"unknown backbone family {family!r}")
        self.family = family
        self.pretrained_id = pretrained_id or PRETRAINED_IDS[family]
        self.pretrained = (pretrained if pretrained is not None
                           else _load_pretrained(self.pretrained_id, family, dtype))
        self.patch_size = int(self.pretrained.config.patch_size)
        self.feature_dim = int(self.pretrained.config.hidden_size)
        self.num_prefix_tokens = _num_prefix_tokens(self.pretrained, family)

        for parameter in self.pretrained.parameters():
            parameter.requires_grad_(False)
        self.lora_layer_names: list[str] = []
        if lora is not None:
            if isinstance(lora, dict):
                lora = LoraSettings(**lora)
            self.lora_layer_names = inject_lora_adapters(
                self.pretrained, LORA_TARGET_SUFFIXES[family],
                rank=lora.rank, alpha=lora.alpha, dropout=lora.dropout)
        if gradient_checkpointing:
            self.pretrained.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.train(False)

    # The pretrained model stays in eval; the adapters follow the flag.
    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        for module in self.pretrained.modules():
            module.training = False
        for module in iter_lora_modules(self.pretrained):
            module.train(mode)
        return self

    def check_resolution_is_patch_multiple(self, height_width) -> None:
        height, width = height_width
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"input {height}x{width} is not a multiple of "
                             f"patch {self.patch_size}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) normalised pixels -> (B,C,H/p,W/p) unit-norm features, fp32."""
        batch, _, height, width = images.shape
        self.check_resolution_is_patch_multiple((height, width))
        hidden_states = self.pretrained(pixel_values=images).last_hidden_state
        patch_tokens = hidden_states[:, self.num_prefix_tokens:, :]
        rows, cols = height // self.patch_size, width // self.patch_size
        feature_map = patch_tokens.reshape(batch, rows, cols, self.feature_dim).permute(0, 3, 1, 2)
        return F.normalize(feature_map.float(), dim=1)
