"""Low-rank adaptation of the linear layers in a frozen transformer.

`peft` is not installed and would be the only reason to chase transformers
version compatibility, so this is the fifty lines it would have provided:
`y = W x + (alpha / rank) * B A dropout(x)` with `B` zero at start, so the
adapted network begins exactly as the pretrained one.
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator

import torch
import torch.nn as nn

# Which linears take an adapter, as suffixes of the module's dotted name.
LORA_TARGET_SUFFIXES: dict[str, tuple[str, ...]] = {
    "dinov3": ("attention.q_proj", "attention.k_proj", "attention.v_proj",
               "attention.o_proj", "mlp.up_proj", "mlp.down_proj"),
    "dinov2": ("attention.attention.query", "attention.attention.key",
               "attention.attention.value", "attention.output.dense",
               "mlp.fc1", "mlp.fc2"),
}


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable low-rank update."""

    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 16.0,
                 dropout: float = 0.1):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_linear = base_linear
        for parameter in self.base_linear.parameters():
            parameter.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.update_scale = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_linear(x)
        update = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_output + update.to(base_output.dtype) * self.update_scale

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"


def inject_lora_adapters(root: nn.Module, target_suffixes: Iterable[str], rank: int = 16,
                         alpha: float = 16.0, dropout: float = 0.1) -> list[str]:
    """Replace every `nn.Linear` whose dotted name ends with one of the
    suffixes. Returns the names that were wrapped."""
    target_suffixes = tuple(target_suffixes)
    wrapped_names = []
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear) or not name.endswith(target_suffixes):
            continue
        parent_name, _, leaf_name = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, leaf_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        wrapped_names.append(name)
    if not wrapped_names:
        raise ValueError(f"no linear layer matched {target_suffixes}")
    return wrapped_names


def iter_lora_modules(root: nn.Module) -> Iterator[LoRALinear]:
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield module


def iter_lora_parameters(root: nn.Module) -> Iterator[nn.Parameter]:
    for module in iter_lora_modules(root):
        yield from module.lora_A.parameters()
        yield from module.lora_B.parameters()


def lora_state_dict(root: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor for name, tensor in root.state_dict().items()
            if ".lora_A." in name or ".lora_B." in name}


def load_lora_state_dict(root: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = root.load_state_dict(state, strict=False)
    missing_lora = [name for name in missing if ".lora_A." in name or ".lora_B." in name]
    if missing_lora or unexpected:
        raise KeyError(f"LoRA state mismatch: missing {missing_lora[:3]}..., "
                       f"unexpected {list(unexpected)[:3]}...")


def count_parameters(parameters: Iterable[nn.Parameter]) -> int:
    return sum(p.numel() for p in parameters)
