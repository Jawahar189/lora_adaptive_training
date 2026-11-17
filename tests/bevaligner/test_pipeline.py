"""The stages `bevaligner.pipeline` runs, and the arguments it gives them.

None of this is exercised by a training run until far too late, and the failure
modes are quiet: a split override that does not reach `bevaligner.train` means
the datamodule falls back to `split_by_structure` and silently validates on a
different set of buildings than the one that was stratified, and a cache built
to a different recipe than the run that reads it is never detected at all
(`CaptureStore.ensure_prepared` keys only on the capture id). So both are
asserted here.
"""
import pytest
import yaml

from bevaligner.pipeline import (Pipeline, PipelineConfig, SplitsConfig,
                                   load_pipeline_config, main, split_stage)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo root holding the split CSVs and a training config.

    `config.py` keeps its own REPO_ROOT and `Config.run_dir` reads it at call
    time, so both module globals move or `run_dir` walks out of the sandbox and
    into the real checkout.
    """
    for name in ("train", "val", "test"):
        path = tmp_path / "splits" / f"100_prod_{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reality_id\n")
    monkeypatch.setattr("bevaligner.config.REPO_ROOT", tmp_path)
    monkeypatch.setattr("bevaligner.data.stratify.REPO_ROOT", tmp_path, raising=False)
    return tmp_path


def make_pipeline(repo, train_config: dict | None = None, **overrides) -> Pipeline:
    path = repo / "train.yaml"
    path.write_text(yaml.safe_dump(train_config or {"run_name": "unused"}))
    config = PipelineConfig(run_name="a_run", train_config="train.yaml",
                            smoke_config="train.yaml", **overrides)
    config.splits.dir = "splits"
    return Pipeline(config)


# ------------------------------------------------------------------ the split

def test_training_is_given_the_validation_csv(repo):
    """The whole point: `data.val_csv_files` is what makes the datamodule use
    the stratified split verbatim instead of re-splitting by structure."""
    argv = make_pipeline(repo).train_argv()
    assert "data.csv_files=[splits/100_prod_train.csv]" in argv
    assert "data.val_csv_files=[splits/100_prod_val.csv]" in argv
    assert "run_name=a_run" in argv


def test_eval_uses_the_test_csv_and_no_split_flag(repo):
    """`--split` reads the cache's splits.json, which knows only train and val;
    the test CSV restricts the structures by itself."""
    argv = make_pipeline(repo).eval_argv(must_exist=False)
    assert "--split" not in argv
    assert "splits/100_prod_test.csv" in argv


def test_missing_splits_name_every_missing_file(repo):
    pipeline = make_pipeline(repo)
    pipeline.config.splits.dir = "nosuch"
    with pytest.raises(SystemExit, match="100_prod_train.csv"):
        pipeline.require_splits()


# ------------------------------------------------- the cache matches the run

def test_preprocess_builds_its_store_from_the_training_config(repo, monkeypatch):
    """The regression this module exists to prevent: `ensure_prepared` reuses a
    cached capture without checking what parameters built it, so preprocessing
    must be handed the very config training will read the cache with."""
    pipeline = make_pipeline(repo, train_config={"data": {
        "cloud_stride": 3, "camera_window_overlap_bias": 0.4, "min_source_frames": 5,
        "max_gt_fit_residual_m": 0.25, "cache_dir": "cache/elsewhere"}})
    seen = {}
    monkeypatch.setattr("bevaligner.data.datamodule.load_cases_from_csv",
                        lambda *a, **k: [])
    monkeypatch.setattr("bevaligner.data.store.CaptureStore.ensure_prepared",
                        lambda self, cases, index=None, workers=8, **kw: seen.update(
                            store=self, workers=workers) or {})
    monkeypatch.setattr("bevaligner.data.store.summarise_capture_results",
                        lambda results: "")
    pipeline.stage_preprocess()
    store = seen["store"]
    assert store.stride == 3
    assert store.data.camera_window_overlap_bias == 0.4
    assert store.min_source_frames == 5
    assert store.max_residual_m == 0.25
    assert store.paths.root.name == "elsewhere"
    assert seen["workers"] == pipeline.config.preprocess.workers


def test_preprocess_writes_no_splits_json(repo, monkeypatch):
    """`prepare_from_csv` would write one describing a random structure split
    over all three CSVs -- test structures included, under `train`."""
    pipeline = make_pipeline(repo)
    monkeypatch.setattr("bevaligner.data.datamodule.load_cases_from_csv",
                        lambda *a, **k: [])
    monkeypatch.setattr("bevaligner.data.store.CaptureStore.ensure_prepared",
                        lambda self, cases, index=None, workers=8, **kw: {})
    monkeypatch.setattr("bevaligner.data.store.summarise_capture_results",
                        lambda results: "")
    pipeline.stage_preprocess()
    assert not list(repo.rglob("splits.json"))


# --------------------------------------------------------------- the stages

def test_stages_call_the_modules_entry_points(repo, monkeypatch):
    calls = {}
    monkeypatch.setattr("bevaligner.train.main", lambda argv: calls.update(train=argv))
    monkeypatch.setattr("bevaligner.evaluate.main", lambda argv: calls.update(eval=argv))
    pipeline = make_pipeline(repo)
    checkpoint = pipeline.run_dir / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    pipeline.stage_train()
    pipeline.stage_eval()
    assert "run_name=a_run" in calls["train"]
    assert str(checkpoint) in calls["eval"]


def test_smoke_keeps_the_smoke_configs_own_run_name(repo):
    """A one-batch check must not write into a real run's directory."""
    argv = make_pipeline(repo).train_argv(smoke=True)
    assert "--fast-dev-run" in argv
    assert not any(item.startswith("run_name=") for item in argv)


def test_train_overrides_land_at_the_end_of_the_argv(repo):
    pipeline = make_pipeline(repo, train_overrides=["optimizer.learning_rate=5e-5",
                                                    "--max-steps", "10"])
    assert pipeline.train_argv()[-3:] == ["optimizer.learning_rate=5e-5",
                                          "--max-steps", "10"]


def test_only_assignments_from_train_overrides_reach_the_config(repo):
    """`train_overrides` may hold argv flags, which are not config keys."""
    pipeline = make_pipeline(repo, train_overrides=["--max-steps", "10",
                                                   "trainer.out_dir=out/elsewhere"])
    assert pipeline.run_dir.parent.name == "elsewhere"


def test_eval_without_a_checkpoint_is_an_error(repo):
    with pytest.raises(SystemExit, match="no checkpoint"):
        make_pipeline(repo).eval_argv()


def test_a_relative_checkpoint_resolves_against_the_callers_cwd(repo, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "hand.ckpt").touch()
    monkeypatch.chdir(elsewhere)
    pipeline = make_pipeline(repo)
    pipeline.config.eval.checkpoint = "hand.ckpt"
    assert pipeline.resolve_checkpoint().name == "hand.ckpt"


# ---------------------------------------------------------------- the config

def test_fractions_that_do_not_sum_to_one_are_rejected(tmp_path):
    """85 / 15 / 5 sums to 105 %, which is easy to write and impossible to mean."""
    with pytest.raises(ValueError, match="sum to"):
        PipelineConfig(splits=SplitsConfig(
            fractions={"train": 0.85, "val": 0.15, "test": 0.05}))
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(
        {"splits": {"fractions": {"train": 0.85, "val": 0.15, "test": 0.05}}}))
    with pytest.raises(ValueError, match="sum to"):
        load_pipeline_config(path)


def test_an_unknown_stage_in_the_config_is_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        PipelineConfig(stages=["preprocess", "trian"])


def test_the_shipped_pipeline_config_loads():
    config = load_pipeline_config()
    assert config.stages and config.train_config.endswith(".yaml")


@pytest.mark.parametrize("items, expected", [
    ([], ("all", [])),
    (["train"], ("train", [])),
    (["train", "a.b=1"], ("train", ["a.b=1"])),
    (["a.b=1", "eval"], ("eval", ["a.b=1"])),
    (["a.b=1"], ("all", ["a.b=1"])),
])
def test_the_stage_is_the_argument_without_an_equals(items, expected):
    assert split_stage(items) == expected


def test_two_stages_is_an_error():
    with pytest.raises(SystemExit, match="more than one stage"):
        split_stage(["train", "eval"])


def test_an_unknown_stage_on_the_command_line_is_named():
    with pytest.raises(SystemExit, match="trian"):
        main(["trian", "--dry-run"])
