"""The LightningDataModule: CSV rows -> train and validation loaders.

`data.csv_files` is the dataset and `data.prior_csv_files` is what its rows are
aligned against; the two are joined by `align.io.cases.PairedIndex` on
`prior_capture_id -> capture_id`. The store fills in whatever the cache lacks
for both sides, and the split is by structure (or by `data.val_csv_files` when
given). The preprocessing script is only a way of doing the store's work ahead
of time.

Splitting by structure is what keeps a pair together: a current row and every
prior split it is aligned against share a structure, so neither can straddle
train and validation.
"""
from __future__ import annotations

import lightning as L
import numpy as np
import torch

from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex

from ..config import REPO_ROOT
from ..utils import io
from .dataset import DensityPriorPairDataset, collate_pairs
from .sampler import ResolutionBatchSampler
from .splits import save_splits, split_by_csv, split_by_structure
from .store import CaptureStore, summarise_capture_results


def build_capture_store(config) -> CaptureStore:
    data = config.data
    return CaptureStore(config.cache_dir, stride=data.cloud_stride,
                        min_source_frames=data.min_source_frames,
                        max_residual_m=data.max_gt_fit_residual_m,
                        data_config=data)


def load_paired_index(config) -> PairedIndex:
    """The current rows, the prior rows, and the edges between them."""
    data = config.data
    current = [str(REPO_ROOT / path) for path in data.csv_files]
    prior = [str(REPO_ROOT / path) for path in (data.prior_csv_files or [])]
    if not prior:
        raise ValueError(
            "data.prior_csv_files is empty: there is nothing to align against. "
            "The prior rows ARE the target on this branch.")
    return PairedIndex.load(datasets_mod.resolve(current, prior, name="bevaligner"))


def load_cases_from_csv(csv_files, role: str = "current"):
    return io.load_cases([REPO_ROOT / path for path in csv_files], role=role)


class BEVAlignerDataModule(L.LightningDataModule):
    def __init__(self, config, patch_size: int, overfit_reality_ids: list[str] | None = None):
        super().__init__()
        self.config, self.patch_size = config, patch_size
        self.overfit_reality_ids = set(overfit_reality_ids) if overfit_reality_ids else None
        self.store = build_capture_store(config)
        self.train_dataset = self.val_dataset = None
        self.batch_sampler = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        data, trainer = self.config.data, self.config.trainer
        index = load_paired_index(self.config)
        cases = list(index.current)
        if self.overfit_reality_ids:
            cases = [c for c in cases if c.reality_id in self.overfit_reality_ids]
        val_cases_from_csv = (load_cases_from_csv(data.val_csv_files, role="val")
                              if data.val_csv_files else [])
        results = self.store.ensure_prepared(cases + val_cases_from_csv, index,
                                             workers=data.preprocess_workers)
        print(summarise_capture_results(results, [c.reality_id for c in cases],
                                        "current"))
        print(summarise_capture_results(results,
                                        [c.reality_id for c in index.prior],
                                        "prior-side"))

        if self.overfit_reality_ids:
            train_cases = [c for c in cases if results[c.reality_id].ok]
            val_cases = train_cases
            split_source = (f"--overfit-realities: {len(self.overfit_reality_ids)} "
                            f"captures, val = train")
        elif val_cases_from_csv:
            train_cases, val_cases = split_by_csv(cases, val_cases_from_csv, results)
            split_source = f"data.val_csv_files: {', '.join(data.val_csv_files)}"
        else:
            train_cases, val_cases = split_by_structure(
                cases, results, data.num_val_structures, data.split_seed)
            split_source = (f"cut here by structure: {data.num_val_structures} val "
                            f"structures, seed {data.split_seed}")
            if not val_cases:                     # tiny runs: validate on train
                val_cases = train_cases
                split_source += " (no val structures: validating on train)"
        save_splits(self.store.paths.splits, train_cases, val_cases, data.split_seed)

        train_records = self.store.capture_records(train_cases, results)
        val_records = self.store.capture_records(val_cases, results)
        if not train_records:
            raise RuntimeError("no usable training captures in the CSV")
        self.train_dataset = DensityPriorPairDataset(
            train_records, self.store, data, self.patch_size, train=True,
            seed=trainer.seed, length=data.samples_per_epoch)
        self.val_dataset = DensityPriorPairDataset(
            val_records, self.store, data, self.patch_size, train=False,
            seed=trainer.seed + 1, fixed_resolution=data.eval_resolution)
        self.batch_sampler = ResolutionBatchSampler(
            len(self.train_dataset), data.resolutions, data.resolution_weights,
            batch_size_for=data.batch_size_for, seed=trainer.seed)
        # Name every file the run's composition came from. A resume re-runs
        # this, and which CSVs a preset resolved to is the first thing wanted
        # when two runs in the same directory disagree -- the log is the only
        # record of it, since `metrics.csv` is rewritten on each start.
        print(f"data: {len(train_records)} train captures, {len(val_records)} val captures, "
              f"{len(self.batch_sampler)} batches/epoch")
        print(f"  train csv : {', '.join(data.csv_files)}")
        print(f"  val split : {split_source}")
        print(f"  prior csv : {', '.join(data.prior_csv_files)}")
        print(f"  cache     : {self.config.cache_dir}")
        print(f"  splits    : {self.store.paths.splits}")

    def train_dataloader(self):
        data = self.config.data
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_sampler=self.batch_sampler, collate_fn=collate_pairs,
            num_workers=data.num_workers, pin_memory=True,
            persistent_workers=data.num_workers > 0,
            prefetch_factor=4 if data.num_workers else None)

    def val_dataloader(self):
        data = self.config.data
        # A fixed permutation, not CSV order and not a fresh shuffle each time.
        # `limit_val_batches` truncates this loader, so in CSV order validation
        # reads a prefix: at 20 batches of 2 that is 40 realities, which fell in
        # 13 of 89 val structures and left the rest unable to move the curve at
        # all. Permuted once from the seed, the same 40 are spread across the
        # split AND are the same 40 at every validation, so the curve compares
        # with itself.
        order = np.random.default_rng(self.config.trainer.seed + 2) \
                  .permutation(len(self.val_dataset)).tolist()
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=max(1, data.batch_size_for(data.eval_resolution) // 2),
            sampler=order, collate_fn=collate_pairs,
            num_workers=min(data.num_workers, 4), pin_memory=True,
            persistent_workers=data.num_workers > 0)
