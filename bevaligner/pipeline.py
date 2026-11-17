"""A run on the structure-stratified splits, start to finish, in one process.

    python -m bevaligner.pipeline all
    python -m bevaligner.pipeline train --config bevaligner/configs/pipeline.yaml
    python -m bevaligner.pipeline train run_name=gpu48_stratified preprocess.workers=32

    from bevaligner.pipeline import Pipeline
    Pipeline.from_yaml("bevaligner/configs/pipeline.yaml").run_all()

The YAML is the interface -- `bevaligner/configs/pipeline_defaults.yaml` holds
every setting and a run's config `extends:` it -- and dotted `key=value`
overrides work exactly as they do for `bevaligner.train`, because both go
through `config.load_dataclass_config`.

Stages are imported and called, not spawned. `preprocess` therefore builds its
`CaptureStore` from the very `Config` that training is about to be given, which
is the one thing here that has to be right: `CaptureStore.ensure_prepared`
decides what to rebuild from whether a capture's json exists and never checks
what parameters built it, so a cache filled under a different `data.cloud_stride`
or `data.sheet_cache_long_side` is silently reused. Handing both stages the same
config is what makes that impossible rather than merely unlikely.

`train` and `eval` go through their own `main(argv)` so that the Trainer
construction and the checkpoint loading stay in one place; this module only
decides the arguments.
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, load_config, load_dataclass_config

DEFAULT_PIPELINE_CONFIG = "bevaligner/configs/pipeline.yaml"
STAGES = ("splits", "preprocess", "smoke", "train", "resume", "eval")


@dataclass
class SplitsConfig:
    dir: str = "datasets/splits"
    prefix: str = "100_prod"
    source_csv: str = "datasets/100_prod_merged.csv"
    fractions: dict[str, float] = field(
        default_factory=lambda: {"train": 0.75, "val": 0.15, "test": 0.10})
    seed: int = 0


@dataclass
class PreprocessConfig:
    workers: int = 16


@dataclass
class EvalConfig:
    qualitative: int = 5                       # captures drawn; 0 turns pictures off
    images: bool = True                        # per-photograph poses, per reality
    images_plot: bool = False                  # and floormap.png beside each one
    checkpoint: str | None = None              # default: <run_dir>/checkpoints/last.ckpt


@dataclass
class PipelineConfig:
    run_name: str = "bevaligner"
    train_config: str = "bevaligner/configs/gpu24.yaml"
    smoke_config: str = "bevaligner/configs/t4_smoke.yaml"
    train_overrides: list[str] = field(default_factory=list)
    eval_overrides: list[str] = field(default_factory=list)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    stages: list[str] = field(default_factory=lambda: ["preprocess", "smoke",
                                                       "train", "eval"])

    def __post_init__(self) -> None:
        total = sum(self.splits.fractions.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"splits.fractions sum to {total:g}, not 1")
        unknown = [stage for stage in self.stages if stage not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; expected {list(STAGES)}")


def load_pipeline_config(path=DEFAULT_PIPELINE_CONFIG,
                         overrides: list[str] | None = None) -> PipelineConfig:
    return load_dataclass_config(PipelineConfig, path, overrides)


class Pipeline:
    """The stages of a run, sharing one loaded training config."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._train_config: Config | None = None

    @classmethod
    def from_yaml(cls, path=DEFAULT_PIPELINE_CONFIG,
                  overrides: list[str] | None = None) -> "Pipeline":
        return cls(load_pipeline_config(path, overrides))

    # ------------------------------------------------------------ the splits
    def split_csv(self, name: str) -> str:
        return f"{self.config.splits.dir}/{self.config.splits.prefix}_{name}.csv"

    @property
    def split_csvs(self) -> list[str]:
        return [self.split_csv(name) for name in ("train", "val", "test")]

    def require_splits(self) -> None:
        from .config import REPO_ROOT
        missing = [csv for csv in self.split_csvs if not (REPO_ROOT / csv).is_file()]
        if missing:
            raise SystemExit("\n".join(
                [f"missing {csv}" for csv in missing]
                + ["datasets/ is gitignored; run "
                   "`python -m bevaligner.pipeline splits` first"]))

    # -------------------------------------------------------- the train config
    @property
    def train_config(self) -> Config:
        """The training config, loaded once.

        `run_name` is forced onto it so that `train`, `eval` and `run_dir` agree
        on where a run lives, whatever the training YAML calls itself.
        """
        if self._train_config is None:
            self._train_config = load_config(
                self.config.train_config,
                [f"run_name={self.config.run_name}", *self._config_overrides()])
        return self._train_config

    def _config_overrides(self) -> list[str]:
        """The `key=value` part of `train_overrides`.

        That list is appended to the train argv verbatim, so it may also hold
        flags such as `--max-steps 5000`, which are not config keys. Only the
        assignments are replayed onto the loaded config -- enough for `run_dir`
        to agree with where training will actually write.
        """
        return [item for item in self.config.train_overrides
                if "=" in item and not item.startswith("-")]

    @property
    def run_dir(self) -> Path:
        return self.train_config.run_dir

    def resolve_checkpoint(self, must_exist: bool = True) -> Path:
        """`--ckpt`-style path, or the run's own `last.ckpt`.

        A relative path is the caller's, resolved against the working directory
        before the repo root; `must_exist` is off for a dry run, whose point is
        to show the command a later stage will build.
        """
        from .config import REPO_ROOT
        named = self.config.eval.checkpoint
        if named is None:
            path = self.run_dir / "checkpoints" / "last.ckpt"
        else:
            path = Path(named)
            if not path.is_absolute():
                path = path if path.is_file() else REPO_ROOT / path
        if must_exist and not path.is_file():
            raise SystemExit(f"no checkpoint at {path}")
        return path

    # ------------------------------------------------------------ the argv
    def train_argv(self, resume: bool = False, smoke: bool = False,
                   must_exist: bool = True) -> list[str]:
        config = self.config
        argv = ["--config", config.smoke_config if smoke else config.train_config,
                f"data.csv_files=[{self.split_csv('train')}]",
                f"data.val_csv_files=[{self.split_csv('val')}]"]
        if smoke:
            # the smoke run keeps the smoke config's own run_name, so a real
            # run's directory is never written into by a one-batch check
            return argv + ["--fast-dev-run"]
        argv.append(f"run_name={config.run_name}")
        if resume:
            argv += ["--resume", str(self.resolve_checkpoint(must_exist))]
        return argv + config.train_overrides

    def eval_argv(self, must_exist: bool = True) -> list[str]:
        # No --split: the test CSV already restricts the structures, and
        # --split reads the cache's splits.json, which knows only train and val.
        argv = ["--ckpt", str(self.resolve_checkpoint(must_exist)),
                "--csv", self.split_csv("test"),
                "--stem", str(self.run_dir / "eval" / "test"),
                "--qualitative", str(self.config.eval.qualitative)]
        if self.config.eval.images:
            argv.append("--images")
            if self.config.eval.images_plot:
                argv.append("--images-plot")
        return argv + self.config.eval_overrides

    # ------------------------------------------------------------- stages
    def stage_splits(self) -> None:
        from .data import stratify
        splits = self.config.splits
        stratify.write_splits(splits.source_csv, splits.dir, splits.prefix,
                              splits.fractions, seed=splits.seed)

    def stage_preprocess(self) -> None:
        """Fill the cache for every split, with the config training will use.

        This is what `BEVAlignerDataModule.setup` does, through the same two
        helpers, so the cache cannot be built to a different recipe than the one
        the run then reads it with. It deliberately does not go through
        `data.preprocess.prepare_from_csv`, which would also write a
        `splits.json` describing a random structure split over all three CSVs --
        held-out test structures included, under `train`.
        """
        self.require_splits()
        from .data.datamodule import (build_capture_store, load_cases_from_csv,
                                      load_paired_index)
        from .data.store import summarise_capture_results
        store = build_capture_store(self.train_config)
        # the split CSVs are the current side; the prior side comes from the
        # training config, because the targets are built from it
        index = load_paired_index(self.train_config)
        cases = load_cases_from_csv(self.split_csvs)
        results = store.ensure_prepared(cases, index,
                                        workers=self.config.preprocess.workers)
        print(summarise_capture_results(results))

    def stage_train(self, resume: bool = False, smoke: bool = False) -> None:
        self.require_splits()
        from . import train as train_module
        train_module.main(self.train_argv(resume=resume, smoke=smoke))

    def stage_eval(self) -> None:
        self.require_splits()
        from . import evaluate as evaluate_module
        evaluate_module.main(self.eval_argv())

    def describe(self, stage: str) -> str:
        """What `--dry-run` prints: the call a stage would make."""
        if stage == "splits":
            splits = self.config.splits
            return (f"stratify.write_splits({splits.source_csv!r}, {splits.dir!r}, "
                    f"{splits.prefix!r}, {splits.fractions}, seed={splits.seed})")
        if stage == "preprocess":
            data = self.train_config.data
            return (f"CaptureStore({self.train_config.cache_dir}, "
                    f"stride={data.cloud_stride}, "
                    f"sheet_long_side={data.sheet_cache_long_side})"
                    f".ensure_prepared({len(self.split_csvs)} CSVs, "
                    f"workers={self.config.preprocess.workers})")
        if stage in ("smoke", "train", "resume"):
            argv = self.train_argv(resume=stage == "resume", smoke=stage == "smoke",
                                   must_exist=False)
            return f"bevaligner.train.main({argv})"
        if stage == "eval":
            return f"bevaligner.evaluate.main({self.eval_argv(must_exist=False)})"
        raise SystemExit(f"unknown stage {stage!r}")

    # ------------------------------------------------------------- running
    def run(self, stage: str, dry_run: bool = False) -> None:
        print(f"\n=== {stage} ===\n{self.describe(stage)}\n", flush=True)
        if dry_run:
            return
        runners = {
            "splits": self.stage_splits,
            "preprocess": self.stage_preprocess,
            "smoke": lambda: self.stage_train(smoke=True),
            "train": self.stage_train,
            "resume": lambda: self.stage_train(resume=True),
            "eval": self.stage_eval,
        }
        if stage not in runners:
            raise SystemExit(f"unknown stage {stage!r}")
        try:
            runners[stage]()
        except KeyboardInterrupt:
            raise SystemExit(f"{stage} interrupted")
        except SystemExit as error:
            if error.code:
                raise SystemExit(f"{stage} failed: {error.code}") from error
        finally:
            _release_gpu()

    def run_all(self, dry_run: bool = False) -> None:
        for stage in self.config.stages:
            self.run(stage, dry_run=dry_run)


def _release_gpu() -> None:
    """Hand back what a stage allocated, so the next one starts on an empty card.

    In one process the smoke run's model and Trainer would otherwise still be
    holding the cache when training tries to allocate.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # One positional, like `bevaligner.train`: two of them (a stage and a
    # variable-length override list) cannot be interleaved with optionals, so
    # the stage is picked out of the same list by not being a `key=value`.
    parser.add_argument("args", nargs="*", metavar="[STAGE] [key=value ...]",
                        help=f"stage ({', '.join((*STAGES, 'all'))}; default all) "
                             "and dotted overrides, in any order")
    parser.add_argument("--config", default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--dry-run", action="store_true",
                        help="print what each stage would call, and do nothing")
    # argparse stops filling a `nargs="*"` positional at the first optional, so
    # `train --dry-run key=value` would otherwise be rejected. Leftovers are
    # taken as positionals -- but anything flag-shaped is still a typo, not an
    # override, and is reported as one.
    args, leftover = parser.parse_known_args(argv)
    misspelt = [item for item in leftover if item.startswith("-")]
    if misspelt:
        parser.error(f"unrecognized arguments: {' '.join(misspelt)}")

    stage, overrides = split_stage(args.args + leftover)
    pipeline = Pipeline.from_yaml(args.config, overrides)
    if stage == "all":
        pipeline.run_all(dry_run=args.dry_run)
    else:
        pipeline.run(stage, dry_run=args.dry_run)


def split_stage(items: list[str]) -> tuple[str, list[str]]:
    """`["train", "preprocess.workers=32"]` -> `("train", [...])`."""
    stages = [item for item in items if "=" not in item]
    if len(stages) > 1:
        raise SystemExit(f"more than one stage given: {stages}")
    stage = stages[0] if stages else "all"
    if stage not in (*STAGES, "all"):
        raise SystemExit(f"unknown stage {stage!r}; "
                         f"expected one of {', '.join((*STAGES, 'all'))}")
    return stage, [item for item in items if "=" in item]


if __name__ == "__main__":
    main()
