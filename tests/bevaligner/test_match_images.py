"""The match pictures: what is drawn, and that it reaches TensorBoard.

The pictures are the only way a collapsed or mirrored match field shows up, so
the drawing, the Lightning hook that feeds it and the logger write are all
pinned here. Nothing downloads weights: a stub module stands in for the matcher.
"""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from bevaligner.data.dataset import collate_pairs
from bevaligner.lightning.callbacks import (CORRECT_COLOUR, TRUTH_COLOUR, WRONG_COLOUR,
                                              LogPredictedMatchImages, log_image_to_loggers,
                                              render_match_image)

PATCH = 14
SIDE = 56
QUERIES = 12


def _sample(reality_id="RLT0001"):
    rng = np.random.default_rng(0)
    return dict(
        density=torch.rand(3, SIDE, SIDE), target=torch.rand(3, SIDE, SIDE),
        target_patch_valid=torch.ones(SIDE // PATCH, SIDE // PATCH, dtype=torch.bool),
        query_px=torch.from_numpy(rng.uniform(2, SIDE - 2, (QUERIES, 2)).astype(np.float32)),
        target_px=torch.from_numpy(rng.uniform(2, SIDE - 2, (QUERIES, 2)).astype(np.float32)),
        density_px_to_world_xy=torch.eye(3)[:2].float(),
        sheet_to_crop=torch.eye(3)[:2].float(),
        density_to_crop=torch.eye(3)[:2].float(),
        meta=dict(reality_id=reality_id, structure_key="STR1", up_index=0,
                  resolution=SIDE, viewport_index=0),
    )


def _batch(size=2):
    return collate_pairs([_sample(f"RLT000{i}") for i in range(size)])


def _colours(image):
    return {tuple(c) for c in np.asarray(image).reshape(-1, 3)}


def test_truth_only_picture_has_both_panels():
    batch = _batch(1)
    image = render_match_image(np.zeros((SIDE, SIDE, 3), np.uint8),
                               np.zeros((SIDE, SIDE, 3), np.uint8),
                               batch["query_px"][0].numpy(), batch["target_px"][0].numpy(),
                               caption="truth")
    assert image.size == (2 * SIDE, SIDE + 14)   # two panels plus the caption strip


def test_a_correct_prediction_is_green_and_a_wrong_one_is_red():
    query_px = np.array([[10.0, 10.0]] * 4)
    target_px = np.array([[20.0, 20.0]] * 4)
    panel = np.zeros((SIDE, SIDE, 3), np.uint8)

    exact = render_match_image(panel, panel, query_px, target_px, predicted_px=target_px,
                               correct_within_px=2.0)
    assert CORRECT_COLOUR in _colours(exact) and WRONG_COLOUR not in _colours(exact)
    assert TRUTH_COLOUR in _colours(exact)          # the truth is always ringed

    far = render_match_image(panel, panel, query_px, target_px,
                             predicted_px=target_px + 20, correct_within_px=2.0)
    assert WRONG_COLOUR in _colours(far) and CORRECT_COLOUR not in _colours(far)


def test_log_image_to_loggers_skips_loggers_without_images(tmp_path):
    recorded = []
    image_logger = SimpleNamespace(experiment=SimpleNamespace(
        add_image=lambda tag, array, global_step, dataformats: recorded.append(
            (tag, array.shape, global_step))))
    trainer = SimpleNamespace(loggers=[CSVLogger(tmp_path), image_logger])
    panel = np.zeros((8, 8, 3), np.uint8)
    image = render_match_image(panel, panel, np.zeros((1, 2)), np.zeros((1, 2)))
    assert log_image_to_loggers(trainer, "val/matches/0", image, 7) is True
    assert len(recorded) == 1 and recorded[0][0] == "val/matches/0" and recorded[0][2] == 7


def test_callback_writes_pngs_when_no_logger_takes_images(tmp_path):
    callback = LogPredictedMatchImages(num_samples=2, fallback_dir=tmp_path / "matches")
    batch = _batch(2)
    outputs = {"predicted_px": batch["target_px"] + 1.0,
               "confidence": torch.rand(2, QUERIES)}
    trainer = SimpleNamespace(loggers=[], global_step=42, sanity_checking=False)
    pl_module = SimpleNamespace(patch_size=PATCH)
    callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)
    written = sorted((tmp_path / "matches").glob("*.png"))
    assert len(written) == 2 and all("step0000042" in p.name for p in written)


def test_callback_spans_batches_up_to_the_quota(tmp_path):
    """The val loader batches at half the training size, so a `num_samples`
    larger than one batch keeps drawing from the next ones -- and stops dead
    once the quota is met, however many batches follow."""
    callback = LogPredictedMatchImages(num_samples=3, fallback_dir=tmp_path / "matches")
    batch = _batch(2)
    outputs = {"predicted_px": batch["target_px"], "confidence": torch.rand(2, QUERIES)}
    pl_module = SimpleNamespace(patch_size=PATCH)
    trainer = SimpleNamespace(loggers=[], global_step=42, sanity_checking=False)
    callback.on_validation_epoch_start(trainer, pl_module)
    for batch_idx in range(4):
        callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    written = sorted(p.name for p in (tmp_path / "matches").glob("*.png"))
    assert len(written) == 3, written
    # one slot per index, so val/matches/<n> is the same slot at every pass
    assert [n.split("_")[1] for n in written] == ["0", "1", "2"]

    # and the count resets for the next validation pass rather than staying full
    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)
    assert callback._drawn == 2


def test_callback_ignores_sanity_checking(tmp_path):
    callback = LogPredictedMatchImages(num_samples=2, fallback_dir=tmp_path / "matches")
    batch = _batch(1)
    outputs = {"predicted_px": batch["target_px"], "confidence": torch.rand(1, QUERIES)}
    pl_module = SimpleNamespace(patch_size=PATCH)
    callback.on_validation_batch_end(
        SimpleNamespace(loggers=[], global_step=1, sanity_checking=True),
        pl_module, outputs, batch, batch_idx=0)
    assert not (tmp_path / "matches").exists()


class _StubMatcher(L.LightningModule):
    """Returns what `BEVAlignerModule.validation_step` returns, nothing more."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.patch_size = PATCH

    def training_step(self, batch, batch_idx):
        return self.layer(batch["query_px"]).mean()

    def validation_step(self, batch, batch_idx):
        predicted = batch["target_px"] + 3.0
        return {"predicted_px": predicted.detach(),
                "confidence": torch.rand(predicted.shape[:2])}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def test_pictures_reach_a_real_tensorboard_event_file(tmp_path):
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator")
    dataset = [_sample() for _ in range(2)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_pairs)
    logger = TensorBoardLogger(save_dir=str(tmp_path), name="run", version="")
    trainer = L.Trainer(accelerator="cpu", devices=1, max_steps=1, limit_train_batches=1,
                        limit_val_batches=1, num_sanity_val_steps=0, logger=logger,
                        callbacks=[LogPredictedMatchImages(num_samples=2)],
                        enable_progress_bar=False, enable_model_summary=False,
                        enable_checkpointing=False, default_root_dir=str(tmp_path))
    trainer.fit(_StubMatcher(), train_dataloaders=loader, val_dataloaders=loader)
    logger.experiment.flush()

    accumulator = event_accumulator.EventAccumulator(
        logger.log_dir, size_guidance={event_accumulator.IMAGES: 10})
    accumulator.Reload()
    tags = accumulator.Tags()["images"]
    assert any(tag.startswith("val/matches/") for tag in tags), tags
    assert accumulator.Images(tags[0])[0].step == trainer.global_step


def test_validation_logs_the_metric_names_the_checkpoint_monitors():
    """A ModelCheckpoint whose `monitor` never gets logged silently saves
    nothing, so the names `trainer.monitor` may hold are pinned here."""
    from bevaligner.lightning.module import BEVAlignerModule

    module = BEVAlignerModule.__new__(BEVAlignerModule)
    L.LightningModule.__init__(module)
    logged: dict[str, float] = {}
    module.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    module._log_health = lambda *a, **k: None
    module._forward_and_loss = lambda batch, progress: (
        SimpleNamespace(predicted_px=batch["target_px"] + 3.0,
                        confidence=torch.full(batch["target_px"].shape[:2], 0.5)),
        {"total": torch.tensor(0.0), "loss/feature": torch.tensor(1.0)})
    module.model = SimpleNamespace(patch_size=PATCH)     # patch_size delegates to it

    module.validation_step(_batch(), 0)

    assert "val/match/pck_within_2_patches" in logged   # the historical default
    for percent in (1, 3, 5, 10, 15, 30):               # the paper's thresholds
        assert f"val/match/pck_within_{percent}pct_of_crop" in logged
    # the offset is +3 on both axes, so the error is 3*sqrt(2) = 4.24 px.
    # Of a 56 px crop that is 7.6%: inside the paper's 10%, outside its 5% --
    # the boundary the new metric has to land on the right side of.
    assert logged["val/match/pck_within_5pct_of_crop"] == 0.0
    assert logged["val/match/pck_within_10pct_of_crop"] == 1.0
    # and 0.30 of a 14 px patch, so every patch threshold is satisfied
    assert logged["val/match/pck_within_1_patches"] == 1.0


def _pose_row(solved=True, errors_m=(0.1,), errors_diagonal=(0.01,), yaw_error_deg=1.0):
    return dict(solved=solved, errors_m=np.array(errors_m),
                errors_diagonal=np.array(errors_diagonal), yaw_error_deg=yaw_error_deg,
                num_inliers=100, refined=True)


def test_pose_metrics_report_the_paper_s_units():
    from bevaligner.lightning.callbacks import pose_metrics
    rows = [_pose_row(errors_diagonal=(0.01, 0.30), yaw_error_deg=3.0),    # half inside 20%
            _pose_row(errors_diagonal=(0.12, 0.18), yaw_error_deg=25.0),   # both inside
            _pose_row(solved=False, errors_m=None, errors_diagonal=None,
                      yaw_error_deg=None)]
    m = pose_metrics(rows)
    assert m["solved_fraction"] == pytest.approx(2 / 3)
    # 5%: row0 has 1 of 2 cameras inside, row1 none, row2 unsolved -> (0.5+0+0)/3
    assert m["cameras_within_5pct_prior_diagonal"] == pytest.approx(0.5 / 3)
    # 20%: row0 1 of 2, row1 2 of 2, row2 0 -> (0.5+1+0)/3
    assert m["cameras_within_20pct_prior_diagonal"] == pytest.approx(1.5 / 3)
    # angular recall counts CAPTURES, one yaw each; unsolved never counts
    assert m["rotation_within_5deg"] == pytest.approx(1 / 3)
    assert m["rotation_within_30deg"] == pytest.approx(2 / 3)
    # combined: row0 yaw ok but only half its cameras inside 20% (>= 0.5, counts);
    # row1 yaw 25 < 30 and both inside -> counts; row2 unsolved
    assert m["pose_within_30deg_20pct_prior_diagonal"] == pytest.approx(2 / 3)


def test_pose_metrics_survive_every_capture_failing():
    """An early checkpoint solves nothing; the callback must still log numbers
    rather than raise, or validation dies before the model can recover."""
    from bevaligner.lightning.callbacks import pose_metrics
    m = pose_metrics([_pose_row(solved=False, errors_m=None, errors_diagonal=None,
                                yaw_error_deg=None)] * 3)
    assert m["solved_fraction"] == 0.0
    assert m["cameras_within_1.0m"] == 0.0 and m["rotation_within_30deg"] == 0.0
    assert m["pose_within_30deg_20pct_prior_diagonal"] == 0.0
    assert np.isnan(m["rmse_xy_median_m"]) and np.isnan(m["yaw_error_median_deg"])


def test_input_health_counts_the_dataset_fallbacks():
    """A broken augmentation never raises -- the sample comes back None, the
    dataset retries, drops the augmentation, then hops to a neighbour. These
    three fractions are the only trace it leaves, so they have to be right."""
    from bevaligner.lightning.module import BEVAlignerModule

    logged = {}
    module = SimpleNamespace(
        log=lambda name, value, **kw: logged.__setitem__(name, float(value)),
        _log_input_health=BEVAlignerModule._log_input_health)
    batch = {"meta": [{"attempt": 0, "hop": 0}, {"attempt": 1, "hop": 0},
                      {"attempt": 3, "hop": 0}, {"attempt": 0, "hop": 2}],
             "target_patch_valid": torch.tensor([[True, True], [True, False]])}
    module._log_input_health(module, batch, batch_size=4)

    assert logged["train/input/retry_fraction"] == 0.5          # attempts 1 and 3
    assert logged["train/input/unaugmented_fallback_fraction"] == 0.25   # attempt 3 only
    assert logged["train/input/neighbour_hop_fraction"] == 0.25
    assert logged["train/input/valid_patch_fraction"] == 0.75


def test_input_health_is_silent_without_meta():
    """`capture_records`-free callers and any collate that drops meta must not
    take the training step down with them."""
    from bevaligner.lightning.module import BEVAlignerModule

    calls = []
    module = SimpleNamespace(log=lambda *a, **k: calls.append(a))
    BEVAlignerModule._log_input_health(module, {}, batch_size=1)
    assert calls == []
