"""AdamW over the LoRA adapters, linear warm-up then cosine decay."""
from __future__ import annotations

import math
from typing import Iterable

import torch


def build_optimizer(parameters: Iterable[torch.nn.Parameter], learning_rate: float = 1e-4,
                    weight_decay: float = 0.01, betas=(0.9, 0.999)) -> torch.optim.Optimizer:
    trainable = [p for p in parameters if p.requires_grad]
    if not trainable:
        raise ValueError("nothing to optimise: no parameter requires grad")
    return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay, betas=betas)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int,
                       warmup_steps: int = 500, final_lr_fraction: float = 0.1):
    """Linear warm-up to the base rate, cosine down to `final_lr_fraction` of it."""
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(int(warmup_steps), total_steps)

    def lr_scale_at(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(total_steps - warmup_steps, 1)
        decay_progress = min((step - warmup_steps) / decay_steps, 1.0)
        return (final_lr_fraction + (1 - final_lr_fraction)
                * 0.5 * (1 + math.cos(math.pi * decay_progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale_at)
