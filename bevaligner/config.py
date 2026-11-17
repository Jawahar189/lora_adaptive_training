"""Configuration: dataclasses with a YAML face and dotted command-line overrides.

    config = load_config("bevaligner/configs/gpu24.yaml",
                         overrides=["optimizer.learning_rate=3e-5",
                                    "data.resolutions=[512,768]"])

A config may build on another with `extends:`, so a run's file holds only what
makes it that run:

    extends: bevaligner/configs/defaults.yaml
    data:
      batch_size: 4          # merges into the parent's `data`
      resolutions: [512]     # replaces the parent's list outright

`defaults.yaml` is the one place a shared default changes; the field defaults
below are the floor under it, for a config built in Python rather than read from
disk. The loader is generic, so `pipeline.py` reads its own dataclasses through
the same `extends` and override machinery.
"""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelConfig:
    backbone_family: str = "dinov3"            # "dinov3" | "dinov2"
    pretrained_id: str | None = None           # Hugging Face id; default per family
    lora_rank: int = 16
    lora_alpha: float = 16.0
    # 0.0, because until the training-flag fix in `lightning/module.py` this
    # knob was inert: the adapters were never in train mode, so the 0.1 every
    # run nominally carried never dropped anything. Repairing the mechanism
    # without changing the value would have switched dropout on everywhere at
    # once, and the measured train/val gap is zero -- this model underfits.
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = False
    temperature: float = 0.07                  # softmax temperature of the soft-argmax


@dataclass
class DataConfig:
    # The dataset IS these CSV rows (the repo's 24-column capture index): one
    # sample per CURRENT row. More than one file may be listed.
    csv_files: list[str] = field(default_factory=lambda: ["datasets/100_prod_current.csv"])
    # The prior-side rows. A current row is joined to them on
    # `prior_capture_id -> capture_id`, and every reality of every named prior
    # capture, merged in paper inches, is that row's TARGET. Without these there
    # is nothing to align against and every row is skipped `no_prior`.
    prior_csv_files: list[str] = field(
        default_factory=lambda: ["datasets/100_prod_prior.csv"])
    # Optional: validation rows from their own CSV instead of a structure split.
    val_csv_files: list[str] | None = None
    cache_dir: str = "cache/bevaligner"      # transparent memo of the slow per-row work
    preprocess_workers: int = 8                # processes used to fill the memo on demand
    num_val_structures: int = 10
    split_seed: int = 0
    resolutions: list[int] = field(default_factory=lambda: [512, 768, 1024])
    resolution_weights: list[float] | None = None
    batch_size: int = 4
    batch_size_per_resolution: dict[int, int] = field(default_factory=dict)   # override
    eval_resolution: int = 1024
    samples_per_epoch: int = 100_000
    queries_per_sample: int = 1024
    # Two-level sampling: a structure is drawn with probability proportional to
    # `n_captures ** structure_balance`, then one of its captures uniformly,
    # then one of that capture's realities. The inner draws are the point --
    # they stop one long survey outvoting its siblings by up to 23x at any
    # setting. The number then buys a building only what its SURVEY count
    # earns: 0 draws every structure equally (the paper's per-scene shape),
    # 1 makes 24 surveys 24x as likely as one.
    # None walks realities in order, which weights a building by how often it
    # was surveyed AND how long each survey ran -- 532x end to end. It is the
    # default only because it is what every run before this did.
    structure_balance: float | None = None
    num_workers: int = 12
    augment: bool = True
    # preprocessing
    cloud_stride: int = 1
    min_source_frames: int = 2
    max_gt_fit_residual_m: float = 0.5
    # --- the prior target ---
    # A prior split joins a row's target when its camera bounding box meets the
    # row's. That filter is an OPTIMISATION, not a criterion -- content can only
    # overlap where boxes do -- and it keeps 17% of the splits a pool would
    # otherwise merge, so the bands of the other 83% are never read.
    #
    # The criterion is how much of the walk's wall the prior also has. A box
    # around a diagonal walk is mostly empty and scores 1.00 on a walk that
    # shares nothing; camera proximity scores 0.08 on a pair whose walls line up
    # exactly. Occupancy got all six measured rows right where each of those got
    # two wrong. See `prior_bev.band_occupancy`.
    min_occupancy: float = 0.2
    #
    # how often the partial-capture camera window is centred on the part of the
    # walk the prior's box covers. A walk that leaves that area part-way through
    # is still trainable on the part that did not, which is why this exists.
    camera_window_overlap_bias: float = 0.8
    # merged prior pools held in memory per worker. A median pool is 4 splits and
    # ~170k points (~1.4 MB) and serves a median of 4 current rows.
    prior_pool_cache: int = 24
    # a prior drawn from several dates: keep each 5 m cell's walls from the
    # newest date only (`prior_bev.mosaic_latest`). Off: every split stacked.
    prior_mosaic: bool = False
    prior_mosaic_cell_m: float = 5.0
    # augmentation knobs
    random_crop_scale: tuple[float, float] = (0.7, 1.0)
    # Continuous yaw, drawn uniformly in +-this many degrees and applied to the
    # POINTS before they are counted -- never to a rendered map, which would
    # interpolate a count and pad the corners with a zero indistinguishable from
    # "no wall here". It composes with the quarter turn, which stays hard-coded
    # and lossless and so keeps three quarters of the angular range on the exact
    # path; this only has to fill the 90 degrees between the cardinals.
    #
    # 45.0 is the only value that buys anything. Measured over 4M draws, 45 on
    # EITHER side makes the relative yaw the matcher sees uniform on the circle
    # to about 1% across 36 bins -- a triangular kernel of half-width 90 deg
    # summed on a 90 deg lattice is flat -- while 20 leaves holes, the emptiest
    # bin at 1% of the fullest. Anything between 0 and 45 is a curriculum.
    #
    # 0.0, the default, draws NOTHING from the generator, so every run before
    # this reproduces: one extra draw would shift every later one.
    density_rotation_degrees: float = 0.0
    # The same on the prior side, drawn INDEPENDENTLY -- a shared angle is an
    # exact no-op, since the matcher only ever sees the difference of the two.
    # It costs what the query side does not: the target window is anchored on
    # the walk's cameras, so a rotated window must be |cos| + |sin| larger to
    # still hold them (mean 1.27, worst 1.41), which spends zoom range.
    target_rotation_degrees: float = 0.0
    # Target windows zoom between the covered part of the walk (x
    # zoom_footprint_margin) and the whole prior, log-uniformly. Our captures are
    # small walks through large sites -- the merged prior is a median 352 m across
    # and a quarter of them exceed a kilometre -- so without the
    # min_footprint_patches cap below a fixed window leaves the walk a few pixels
    # wide. That cap is why no separate limit in metres is needed.
    zoom_crops: bool = True
    zoom_footprint_margin: float = 1.3
    # never zoom out so far that the capture spans fewer patches than this
    min_footprint_patches: float = 4.0
    # inference: re-match on a crop around the first pose's footprint
    refine_on_footprint: bool = True
    refine_footprint_margin: float = 1.6
    density_gammas: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    density_min_counts: list[int] = field(default_factory=lambda: [1, 1, 2])
    band_low_range: tuple[float, float] = (-0.95, -0.75)    # wall band, x camera height
    band_high_range: tuple[float, float] = (0.15, 0.35)
    # the prior side's own point dropout, mirroring point_keep_fraction below.
    # (1.0, 1.0) is off: the prior has never been thinned, though it is as
    # partial a capture as the current one.
    target_point_keep_fraction: tuple[float, float] = (1.0, 1.0)
    camera_window_probability: float = 0.5
    camera_window_min_fraction: float = 0.4
    point_keep_fraction: tuple[float, float] = (0.6, 1.0)
    # applied to BOTH sides now: the target is a density map too, so the sheet's
    # brightness/contrast/saturation/hue jitter has nothing left to act on
    density_jitter: float = 0.2

    def batch_size_for(self, resolution: int) -> int:
        return int(self.batch_size_per_resolution.get(resolution, self.batch_size))


@dataclass
class LossConfig:
    huber_delta: float = 1.0                   # in patches
    num_triplets: int = 2048                   # topology loss samples per item
    num_pairs: int = 2048                      # geometry loss samples per item
    detach_confidence: bool = True
    # InfoNCE negatives whose target lies within this many patches of the
    # anchor's are near-duplicates and are ignored
    near_duplicate_radius: float = 1.0
    weights: dict[str, float] = field(default_factory=lambda: {
        "feature": 1.0, "regression": 50.0, "topology": 10.0, "geometry": 10.0})
    # fraction of training after which each term switches on
    gates: dict[str, float] = field(default_factory=lambda: {
        "feature": 0.0, "regression": 0.10, "topology": 0.20, "geometry": 0.20})
    # fraction of training a term takes to reach full weight after its gate
    gate_ramp: float = 0.05


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    final_lr_fraction: float = 0.1             # cosine floor, as a fraction of learning_rate
    gradient_clip_norm: float = 1.0
    accumulate_grad_batches: int = 1


@dataclass
class TrainerConfig:
    precision: str = "bf16-mixed"
    max_epochs: int = 1
    # -1 = run to max_epochs. `--max-steps` overrides it, but the curriculum gates
    # are fractions of `estimated_stepping_batches`, which this caps -- so a run
    # whose gates mean particular STEPS has to be able to say so in its YAML, or a
    # restart that forgets the flag moves every gate.
    max_steps: int = -1
    val_check_interval: int = 2000
    val_pose_eval_captures: int = 20           # captures scored by the pose callback
    val_batches: int = 20                      # correspondence-loss batches per validation
    # the metric `save_top_k` ranks checkpoints by. "" picks the historical
    # default: val/pose/cameras_within_1.0m when the pose callback runs,
    # val/match/pck_within_2_patches when it does not.
    #
    # tpr@1m is the metric that matters, and the worst one to SELECT on: it
    # averages `val_pose_eval_captures` captures, each near 0 or 1 because a
    # capture either registers or does not, so at 20 captures its standard
    # error is about 0.11. Taking the best of ~17 validations then inflates the
    # winner by roughly 0.2 of noise. pck@2p is the same quantity RANSAC
    # actually consumes -- the inlier rate among correspondences -- measured
    # over ~800 realities instead of 20.
    monitor: str = ""
    save_top_k: int = 2
    log_every_n_steps: int = 20
    # validation samples drawn as predicted-match pictures each validation pass
    # (0 turns the pictures off); a match counts as correct within this many patches
    num_match_images: int = 4
    match_correct_within_patches: float = 2.0
    out_dir: str = "out/bevaligner"
    seed: int = 0
    deterministic: bool = False


@dataclass
class Config:
    run_name: str = "bevaligner"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @property
    def cache_dir(self) -> Path:
        return REPO_ROOT / self.data.cache_dir

    @property
    def run_dir(self) -> Path:
        return REPO_ROOT / self.trainer.out_dir / self.run_name


def _dataclass_from_dict(cls, values: dict):
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for key, value in values.items():
        if key not in hints:
            raise KeyError(f"unknown config key {key!r} for {cls.__name__}")
        hint = hints[key]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = _dataclass_from_dict(hint, value)
        elif typing.get_origin(hint) is tuple and isinstance(value, (list, tuple)):
            value = tuple(value)
        elif (typing.get_origin(hint) is dict and isinstance(value, dict)
              and typing.get_args(hint)[:1] == (int,)):
            value = {int(k): v for k, v in value.items()}
        kwargs[key] = value
    return cls(**kwargs)


# Configs written before the fields were given their present names: checkpoints
# and saved run configs from those runs still have to load.
_LEGACY_SECTIONS = {"optim": "optimizer"}
_LEGACY_KEYS = {
    "model": {"family": "backbone_family", "model_id": "pretrained_id",
              "lora_r": "lora_rank", "grad_ckpt": "gradient_checkpointing",
              "tau": "temperature"},
    "data": {"csv": "csv_files", "val_csv": "val_csv_files",
             "n_val_structures": "num_val_structures",
             "batch_sizes": "batch_size_per_resolution",
             "queries": "queries_per_sample", "sheet_cache_res": "sheet_cache_long_side",
             "max_fit_resid_m": "max_gt_fit_residual_m", "crop_scale": "random_crop_scale",
             "zoom_margin": "zoom_footprint_margin", "refine": "refine_on_footprint",
             "refine_margin": "refine_footprint_margin", "gammas": "density_gammas",
             "min_counts": "density_min_counts", "band_lo": "band_low_range",
             "band_hi": "band_high_range",
             "camera_window_prob": "camera_window_probability",
             "camera_window_min_frac": "camera_window_min_fraction",
             "point_keep": "point_keep_fraction", "sheet_jitter": "sheet_colour_jitter"},
    "loss": {"n_triplets": "num_triplets", "n_pairs": "num_pairs",
             "detach_conf": "detach_confidence",
             "neighbour_radius": "near_duplicate_radius"},
    "optimizer": {"lr": "learning_rate", "final_lr_frac": "final_lr_fraction",
                  "grad_clip": "gradient_clip_norm",
                  "accumulate": "accumulate_grad_batches"},
    "trainer": {"val_pose_captures": "val_pose_eval_captures",
                "val_pairs_batches": "val_batches"},
}
_LEGACY_LOSS_TERMS = {"feat": "feature", "regr": "regression",
                      "topo": "topology", "geo": "geometry"}


def _migrate_legacy_keys(values: dict) -> dict:
    migrated = {_LEGACY_SECTIONS.get(name, name): section
                for name, section in values.items()}
    for section_name, renames in _LEGACY_KEYS.items():
        section = migrated.get(section_name)
        if not isinstance(section, dict):
            continue
        migrated[section_name] = {renames.get(key, key): value
                                  for key, value in section.items()}
    data = migrated.get("data")
    if isinstance(data, dict) and ("current_csv" in data or "prior_csv" in data):
        # older still: the dataset was two named CSV roles, now one ordered list
        csv_files = list(data.pop("current_csv", [])) + list(data.pop("prior_csv", []))
        data.setdefault("csv_files", csv_files)
    loss = migrated.get("loss")
    if isinstance(loss, dict):
        for field_name in ("weights", "gates"):
            terms = loss.get(field_name)
            if isinstance(terms, dict):
                loss[field_name] = {_LEGACY_LOSS_TERMS.get(term, term): weight
                                    for term, weight in terms.items()}
    return migrated


def config_from_dict(values: dict) -> Config:
    """A config from a plain dict, accepting the pre-rename key names."""
    try:
        return _dataclass_from_dict(Config, values)
    except KeyError:
        return _dataclass_from_dict(Config, _migrate_legacy_keys(values))


def _parse_override_value(raw_value: str) -> Any:
    """YAML, with one repair: PyYAML reads `5e-5` as a string (YAML 1.1 wants
    `5.0e-5`), and a learning rate that arrives as text fails far from here."""
    value = yaml.safe_load(raw_value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _set_nested_key(values: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        values = values.setdefault(part, {})
    values[parts[-1]] = value


EXTENDS_KEY = "extends"


def _deep_merge(base: dict, over: dict) -> dict:
    """`over` wins. Two dicts at one key merge; anything else replaces.

    A list replaces rather than concatenates: `data.resolutions` in a run's
    config is the whole schedule, not an addition to the default one.
    """
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml_with_extends(path: str | Path, seen: tuple[Path, ...] = ()) -> dict:
    """One config's values, its `extends:` chain already merged underneath.

    `extends` is repo-relative, like every other path a config names. The chain
    is walked parent-first so the nearest file wins, and a cycle is an error
    rather than a hang.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ValueError(f"`{EXTENDS_KEY}` cycle: {chain}")
    values = yaml.safe_load(path.read_text()) or {}
    parent = values.pop(EXTENDS_KEY, None)
    if parent is None:
        return values
    return _deep_merge(_read_yaml_with_extends(parent, (*seen, path)), values)


def load_dataclass_config(cls, path: str | Path | None = None,
                          overrides: list[str] | None = None, from_dict=None):
    """A dataclass from a YAML file (with `extends:`) plus dotted overrides.

    `from_dict` is the constructor to use instead of `_dataclass_from_dict`,
    which is how `Config` keeps its legacy-key migration.
    """
    values = _read_yaml_with_extends(path) if path is not None else {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override {override!r} is not key=value")
        key, raw_value = override.split("=", 1)
        _set_nested_key(values, key.strip(), _parse_override_value(raw_value))
    return (from_dict or (lambda v: _dataclass_from_dict(cls, v)))(values)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    return load_dataclass_config(Config, path, overrides, from_dict=config_from_dict)
