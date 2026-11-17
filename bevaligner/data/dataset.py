"""Records -> one training pair, at the resolution the sampler asks for.

`DensityPriorPairDataset[(index, resolution)]` builds the capture's density map
and a window of its PRIOR capture's bird's-eye view at that side length, with
augmentation seeded by (seed, epoch, index) so every sample is reproducible and
every epoch differs. Its length is a virtual epoch, `samples_per_epoch`, not the
number of captures. Records are the CSV rows the store found usable, in CSV order.

Both sides are density maps of the same rooms a median of one day apart, so both
go through `density.render_density_map`; the only asymmetry is that the query is
rendered in its own square extent and the target in a window of paper inches.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


from ..utils import affine
from ..utils import image as image_utils
from . import correspondences, prior_bev, transforms


@dataclass
class CaptureRecord:
    """Everything the cache holds for one capture, ready to render."""

    reality_id: str
    structure_key: str
    role: str
    band_xy: np.ndarray                   # (N, 2) gravity-up cloud xy, wide wall band
    band_relative_height: np.ndarray      # (N,) height above the nearest camera, normalised
    band_camera_index: np.ndarray         # (N,) index of that camera
    camera_xy: np.ndarray                 # (M, 2) gravity-up cloud xy
    camera_z: np.ndarray                  # (M,)
    camera_paper_xy: np.ndarray           # (M, 2) ground-truth paper inches
    up_index: int
    sign_ambiguous: bool
    rotation_to_gravity_up: np.ndarray    # 3x3
    gt_sim2_params: np.ndarray            # [scale, theta, tx, ty], cloud xy -> paper inches
    camera_height_above_floor: float
    metres_per_inch: float
    # the survey this reality belongs to. Not in the npz -- the store attaches it
    # from the CSV row, because balancing needs the unit of independent
    # information, and that is the survey, not the reality: one capture here
    # holds a median of 3 realities and as many as 69.
    capture_id: str = ""
    # the prior splits this row may be aligned against, as
    # (reality_id, capture_id, capture_date). The store decides membership; the
    # pool itself is built lazily per worker, because it is the big object.
    pool_members: tuple = ()
    # which of this walk's cameras the prior actually walked near. The target
    # window is anchored on THESE, not on the whole footprint: the walk often
    # overlaps its prior only in part, and a window around the whole walk then
    # renders a target that is mostly empty.
    overlap_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    metres_per_inch_source: str = ""
    same_day: bool = False
    later_dated: bool = False

    @property
    def gt_sim2_affine(self) -> np.ndarray:
        scale, theta, tx, ty = self.gt_sim2_params
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        return np.array([[scale * cos_theta, -scale * sin_theta, tx],
                         [scale * sin_theta, scale * cos_theta, ty]])

    @property
    def anchor_paper_xy(self) -> np.ndarray:
        """The cameras the target window must hold: the overlapping ones."""
        if len(self.overlap_mask) == len(self.camera_paper_xy) and self.overlap_mask.any():
            return self.camera_paper_xy[self.overlap_mask]
        return self.camera_paper_xy

    @property
    def overlap(self) -> float:
        return float(self.overlap_mask.mean()) if len(self.overlap_mask) else 0.0

    @classmethod
    def load(cls, npz_path: Path) -> "CaptureRecord":
        stored = np.load(npz_path, allow_pickle=False)
        return cls(reality_id=str(stored["reality_id"]),
                   structure_key=str(stored["structure_key"]),
                   role=str(stored["role"]),
                   band_xy=stored["band_xy"],
                   band_relative_height=stored["band_relative_height"],
                   band_camera_index=stored["band_camera_index"],
                   camera_xy=stored["camera_xy"], camera_z=stored["camera_z"],
                   camera_paper_xy=stored["camera_paper_xy"],
                   up_index=int(stored["up_index"]),
                   sign_ambiguous=bool(stored["sign_ambiguous"]),
                   rotation_to_gravity_up=stored["rotation_to_gravity_up"],
                   gt_sim2_params=stored["gt_sim2_params"],
                   camera_height_above_floor=float(stored["camera_height_above_floor"]),
                   metres_per_inch=float(stored["metres_per_inch"]))


class DensityPriorPairDataset(torch.utils.data.Dataset):
    """Pairs of (current density map, prior BEV window) and their correspondences."""

    def __init__(self, records: list[CaptureRecord], store, data_config, patch_size: int,
                 train: bool, seed: int = 0, length: int | None = None,
                 fixed_resolution: int | None = None):
        if not records:
            raise ValueError("no records")
        self.records, self.store = records, store
        self.data_config, self.patch_size = data_config, patch_size
        self.train, self.seed = train, seed
        self.epoch = 0
        self.length = length if length is not None else len(records)
        self.fixed_resolution = fixed_resolution
        self.record_cdf = (self._structure_balanced_cdf(data_config.structure_balance)
                           if train else None)

    def _structure_balanced_cdf(self, power: float | None) -> np.ndarray | None:
        """Two-level sampling weights: structure by survey count, then survey.

        Drawn as if by three uniform choices -- a structure with probability
        proportional to `n_captures ** power`, then one of its captures, then
        one of that capture's realities:

            weight(record) = n_captures(structure) ** (power - 1)
                             / n_realities(capture)

        The capture, not the reality, is the unit of independent information.
        A reality is one walk of a survey, and a survey here holds a median of
        3 and as many as 69, so weighting realities lets a single long survey
        outvote every other survey of the same building by up to 23x -- skew
        that carries no new geometry. The inner two draws remove that at any
        `power`.

        `power` then sets only how much a building's *survey count* buys it:
        0 draws every structure equally, the shape the paper's per-scene
        augmentation has; 1 makes a building surveyed 24 times 24x as likely
        as one surveyed once, which is the honest reading of how much more is
        known about it. None is the plain walk over realities -- the default,
        allocating nothing, so runs that predate this cannot drift.
        """
        if power is None:
            return None
        surveys: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            surveys[record.structure_key].add(record.capture_id)
        captures_in_structure = {key: len(ids) for key, ids in surveys.items()}
        realities_in_capture = Counter(record.capture_id for record in self.records)
        weights = np.array(
            [captures_in_structure[record.structure_key] ** (float(power) - 1.0)
             / realities_in_capture[record.capture_id]
             for record in self.records], dtype=np.float64)
        return np.cumsum(weights / weights.sum())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def _rng_for(self, index: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self.epoch, index])

    def _record_for(self, index: int) -> int:
        """Which record a virtual-epoch index starts at.

        Drawn from the balanced CDF rather than taken modulo, on its own RNG
        stream so that turning balancing on does not change any augmentation
        the sample would otherwise have had. Stateless in (seed, epoch, index),
        so forked workers agree without `set_epoch` reaching them.
        """
        if self.record_cdf is None:
            return index % len(self.records)
        draw = np.random.default_rng([self.seed, self.epoch, index, 0x5EED]).random()
        return int(np.searchsorted(self.record_cdf, draw, side="right"))

    def pool_for(self, record: CaptureRecord):
        return self.store.pool_for(record)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index, resolution = item
        else:
            index = item
            resolution = self.fixed_resolution or self.data_config.eval_resolution
        rng = (self._rng_for(index) if self.train
               else np.random.default_rng([self.seed, index]))
        augment = self.train and self.data_config.augment
        # A record that cannot yield a sample (its truth misses every crop) is
        # skipped for a neighbour rather than killing the run; the store flags
        # such captures, this is the belt to its braces.
        base = self._record_for(index)
        for hop in range(5):
            record = self.records[(base + hop) % len(self.records)]
            for attempt in range(4):
                augment_rng = rng if augment and attempt < 3 else None
                sample = self._build_sample(record, resolution, augment_rng, rng)
                if sample is not None:
                    sample["meta"]["attempt"] = attempt
                    sample["meta"]["hop"] = hop
                    return sample
            if hop == 0:
                print(f"[dataset] {record.reality_id}: no usable sample, "
                      f"skipping to a neighbour")
        raise RuntimeError(f"{record.reality_id}: could not build a sample")

    def _build_sample(self, record: CaptureRecord, resolution: int, augment_rng, query_rng):
        data_config = self.data_config
        pool = self.pool_for(record)
        if pool is None:
            return None
        density_map = transforms.render_record_density(record, resolution, augment_rng,
                                                       data_config)
        if density_map is None:
            return None
        # The window is anchored on the cameras the prior actually covered, so a
        # walk that overlaps its prior in part still yields a target with
        # something in it. `sample_ground_truth_pairs` then drops the queries
        # whose truth lands outside the prior's coverage.
        # Continuous yaw on the prior is drawn BEFORE the window, and the anchors
        # turn first: `choose_window` then picks its square in the frame the
        # target is actually rendered in, so its two guarantees -- the window
        # holds the anchors, and the capture never spans fewer than
        # `min_footprint_patches` -- hold verbatim without the function
        # changing at all.
        target_rotation = prior_bev.draw_target_rotation(augment_rng, data_config,
                                                         record.anchor_paper_xy)
        anchors = (record.anchor_paper_xy if target_rotation is None
                   else affine.apply(target_rotation, record.anchor_paper_xy))
        window = prior_bev.choose_window(pool, anchors, augment_rng,
                                         data_config, resolution, self.patch_size)
        crop = prior_bev.render_target(pool, window, resolution, augment_rng, data_config,
                                       paper_rotation=target_rotation)
        if crop is None:
            return None

        density_to_crop = correspondences.density_to_crop_affine(
            density_map.density_px_to_world_xy, record.gt_sim2_affine, crop.paper_to_crop)
        pairs = correspondences.sample_ground_truth_pairs(
            density_map.image, density_to_crop, crop.valid_mask,
            data_config.queries_per_sample, query_rng)
        if pairs is None:
            return None
        query_px, target_px = pairs

        density_image = image_utils.rgb_to_tensor(density_map.image)
        target_image = image_utils.rgb_to_tensor(crop.image)
        if augment_rng is not None:
            # one jitter for both sides now: the target is a density map too, so
            # the sheet's colour jitter has nothing left to act on
            density_image = transforms.jitter_brightness_contrast(
                density_image, augment_rng, data_config.density_jitter,
                data_config.density_jitter)
            target_image = transforms.jitter_brightness_contrast(
                target_image, augment_rng, data_config.density_jitter,
                data_config.density_jitter)
        return dict(
            density=image_utils.imagenet_normalize(density_image),
            target=image_utils.imagenet_normalize(target_image),
            target_patch_valid=torch.from_numpy(
                prior_bev.patch_valid_mask(crop.valid_mask, self.patch_size)),
            query_px=torch.from_numpy(query_px), target_px=torch.from_numpy(target_px),
            density_px_to_world_xy=torch.from_numpy(
                density_map.density_px_to_world_xy.astype(np.float32)),
            paper_to_crop=torch.from_numpy(crop.paper_to_crop.astype(np.float32)),
            density_to_crop=torch.from_numpy(density_to_crop.astype(np.float32)),
            meta=dict(reality_id=record.reality_id, structure_key=record.structure_key,
                      up_index=record.up_index, resolution=resolution,
                      prior_key=pool.key, prior_splits=len(pool.splits),
                      overlap=record.overlap, same_day=record.same_day),
        )


def collate_pairs(samples: list[dict]) -> dict:
    batch = {}
    for key in samples[0]:
        if key == "meta":
            batch[key] = [sample[key] for sample in samples]
        else:
            batch[key] = torch.stack([sample[key] for sample in samples])
    return batch
