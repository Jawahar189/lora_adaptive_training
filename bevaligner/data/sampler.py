"""Batches that share one resolution, drawn per batch as the paper does.

Yields lists of `(index, resolution)` keys. The schedule for an epoch is
generated up front from (seed, epoch) so `__len__` is exact and resumable.
"""
from __future__ import annotations

import numpy as np
import torch


class ResolutionBatchSampler(torch.utils.data.Sampler):
    def __init__(self, num_samples: int, resolutions, resolution_weights=None,
                 batch_size_for=None, batch_size: int = 4, seed: int = 0,
                 shuffle: bool = True):
        self.num_samples = int(num_samples)
        self.resolutions = [int(r) for r in resolutions]
        weights = (np.ones(len(self.resolutions)) if resolution_weights is None
                   else np.asarray(resolution_weights, float))
        self.resolution_weights = weights / weights.sum()
        self.batch_size_for = batch_size_for or (lambda resolution: batch_size)
        self.seed, self.shuffle = seed, shuffle
        self.epoch = 0
        self.batches = self._build_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.batches = self._build_batches()

    def _build_batches(self) -> list[list[tuple[int, int]]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        order = (rng.permutation(self.num_samples) if self.shuffle
                 else np.arange(self.num_samples))
        batches, position = [], 0
        while position < self.num_samples:
            resolution = int(rng.choice(self.resolutions, p=self.resolution_weights))
            batch_size = self.batch_size_for(resolution)
            indices = order[position:position + batch_size]
            position += batch_size
            if len(indices) < batch_size:   # drop the ragged tail; the epoch is virtual
                break
            batches.append([(int(i), resolution) for i in indices])
        return batches

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)
