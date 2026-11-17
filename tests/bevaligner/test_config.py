"""Config layering: `extends:`, deep merge, and the defaults file.

A merge that silently drops a key is the kind of mistake that shows up as a
training run with the wrong batch size and no error, so the merge rules are
pinned here rather than inferred from the configs that happen to ship.
"""
from types import SimpleNamespace

import pytest
import yaml

from bevaligner.config import (Config, DataConfig, LossConfig, ModelConfig,
                                 OptimizerConfig, REPO_ROOT, TrainerConfig,
                                 load_config)

DEFAULTS = REPO_ROOT / "bevaligner/configs/defaults.yaml"


def write(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values))
    return path


def test_child_merges_into_the_parents_block(tmp_path):
    parent = write(tmp_path / "parent.yaml",
                   {"data": {"batch_size": 4, "num_workers": 12}})
    child = write(tmp_path / "child.yaml",
                  {"extends": str(parent), "data": {"batch_size": 2}})
    config = load_config(child)
    assert config.data.batch_size == 2          # the child wins
    assert config.data.num_workers == 12        # and the parent survives


def test_a_list_replaces_rather_than_extends(tmp_path):
    """`data.resolutions` is a whole schedule, not an addition to one."""
    parent = write(tmp_path / "parent.yaml", {"data": {"resolutions": [512, 768, 1024]}})
    child = write(tmp_path / "child.yaml",
                  {"extends": str(parent), "data": {"resolutions": [512]}})
    assert load_config(child).data.resolutions == [512]


def test_a_chain_resolves_nearest_first(tmp_path):
    base = write(tmp_path / "base.yaml", {"run_name": "base", "data": {"batch_size": 1}})
    middle = write(tmp_path / "middle.yaml",
                   {"extends": str(base), "run_name": "middle", "data": {"num_workers": 3}})
    leaf = write(tmp_path / "leaf.yaml", {"extends": str(middle), "run_name": "leaf"})
    config = load_config(leaf)
    assert config.run_name == "leaf"
    assert config.data.batch_size == 1 and config.data.num_workers == 3


def test_a_cycle_is_an_error_not_a_hang(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    write(first, {"extends": str(second)})
    write(second, {"extends": str(first)})
    with pytest.raises(ValueError, match="cycle"):
        load_config(first)


def test_extends_never_reaches_the_dataclass(tmp_path):
    """`_dataclass_from_dict` rejects unknown keys, so it must be popped."""
    parent = write(tmp_path / "parent.yaml", {"run_name": "parent"})
    child = write(tmp_path / "child.yaml", {"extends": str(parent)})
    assert load_config(child).run_name == "parent"


def test_overrides_beat_both_files(tmp_path):
    parent = write(tmp_path / "parent.yaml", {"data": {"batch_size": 4}})
    child = write(tmp_path / "child.yaml", {"extends": str(parent), "data": {"batch_size": 2}})
    assert load_config(child, ["data.batch_size=8"]).data.batch_size == 8


# ------------------------------------------------------------- defaults.yaml

def test_defaults_yaml_loads():
    assert isinstance(load_config(DEFAULTS), Config)


@pytest.mark.parametrize("section, cls", [
    (None, Config), ("model", ModelConfig), ("data", DataConfig),
    ("loss", LossConfig), ("optimizer", OptimizerConfig), ("trainer", TrainerConfig)])
def test_defaults_yaml_names_every_field_and_invents_none(section, cls):
    """It is the place a shared default is changed, so a field missing from it
    is a setting nobody can find."""
    values = yaml.safe_load(DEFAULTS.read_text())
    present = set(values if section is None else values[section])
    expected = set(cls.__dataclass_fields__)
    if section is None:
        expected -= {"model", "data", "loss", "optimizer", "trainer"}
        present -= {"model", "data", "loss", "optimizer", "trainer"}
    assert not expected - present, f"missing from defaults.yaml: {sorted(expected - present)}"
    assert not present - expected, f"unknown in defaults.yaml: {sorted(present - expected)}"


@pytest.mark.parametrize("name", ["gpu24", "gpu48", "t4_smoke", "t4_smoke_dinov3",
                                  "pipeline", "pipeline_gpu48", "pipeline_t4"])
def test_every_shipped_config_extends_and_loads(name):
    path = REPO_ROOT / f"bevaligner/configs/{name}.yaml"
    assert "extends:" in path.read_text(), f"{name}.yaml should build on a defaults file"
    if name.startswith("pipeline"):
        from bevaligner.pipeline import load_pipeline_config
        load_pipeline_config(path)
    else:
        load_config(path)


def test_run_summary_reports_where_an_interrupted_run_got_to(tmp_path):
    """The summary prints from a `finally`, so it has to render for a run that
    was killed: no best checkpoint, no validation, and it must still say the
    step and the CSVs rather than raise."""
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint

    from bevaligner.train import _run_summary

    config = load_config(None, ["run_name=summary_test"])
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val/pose/cameras_within_1.0m",
                                 mode="max")
    trainer = SimpleNamespace(global_step=13000, current_epoch=0, callback_metrics={})
    text = _run_summary(trainer, config, [checkpoint], elapsed_s=3 * 3600 + 25 * 60)
    assert "step 13000" in text and "3h 25m" in text
    assert "val/pose/cameras_within_1.0m (max)" in text
    assert "(none written)" in text and "last validation: none ran" in text
    assert ", ".join(config.data.csv_files) in text

    # and with a selected checkpoint and metrics in hand, it names both
    checkpoint.best_model_path = str(tmp_path / "best-step=10000.ckpt")
    checkpoint.best_model_score = torch.tensor(0.4231)
    trainer.callback_metrics = {"val/pck@2p": torch.tensor(0.83),
                                "train/loss/total": torch.tensor(64.5)}
    text = _run_summary(trainer, config, [checkpoint], elapsed_s=60,
                        resumed_from="checkpoints/last1.ckpt")
    assert "best-step=10000.ckpt   0.4231" in text
    assert "resumed from : checkpoints/last1.ckpt" in text
    assert "val/pck@2p" in text and "train/loss/total" not in text  # val/* only


def test_init_from_and_resume_are_refused_together():
    """They mean opposite things -- `--resume` restores the optimiser, schedule,
    curriculum position and data order, `--init-from` deliberately restores none
    of them -- and taking both silently would resume while looking like a new
    experiment. The guard runs before the config and the cache, which take
    minutes, so the mistake is cheap."""
    from bevaligner.train import main

    with pytest.raises(SystemExit) as caught:
        main(["--init-from", "a.ckpt", "--resume", "b.ckpt"])
    assert caught.value.code == 2      # argparse usage error, not a crash


def test_max_steps_can_be_pinned_in_the_config():
    """The loss gates are fractions of `estimated_stepping_batches`, which
    max_steps caps, so a run whose gates mean particular STEPS has to be able to
    say so in its YAML -- on the command line a restart can lose it."""
    config = load_config(None, ["trainer.max_steps=280000"])
    assert config.trainer.max_steps == 280000
    assert load_config(None).trainer.max_steps == -1
