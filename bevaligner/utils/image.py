"""Pixels in, tensors out. torchvision is not installed, so this is by hand."""
from __future__ import annotations

import numpy as np
import torch

from align.visual import MEAN, STD  # ImageNet statistics, as the backbone expects

IMAGENET_MEAN = torch.tensor(MEAN).view(3, 1, 1)
IMAGENET_STD = torch.tensor(STD).view(3, 1, 1)


def rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 or float in [0,1], or (H,W) grey -> float32 (3,H,W) in [0,1]."""
    array = np.asarray(rgb)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    """(3,H,W) or (B,3,H,W) in [0,1] -> ImageNet-normalised."""
    mean = IMAGENET_MEAN.to(image.device, image.dtype)
    std = IMAGENET_STD.to(image.device, image.dtype)
    return (image - mean) / std


def imagenet_denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(image.device, image.dtype)
    std = IMAGENET_STD.to(image.device, image.dtype)
    return (image * std + mean).clamp(0, 1)


def tensor_to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    """(3,H,W) in [0,1] -> (H,W,3) uint8, for dumps and viewers."""
    return (image.detach().cpu().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
