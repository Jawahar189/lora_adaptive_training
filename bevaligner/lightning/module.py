"""The LightningModule: matcher + losses + curriculum + optimiser, LoRA-only checkpoints."""
from __future__ import annotations

import math

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm

from ..config import Config, config_from_dict
from ..losses.curriculum import LossSettings, BEVAlignerLoss
from ..models import DinoPatchEncoder, DensityPriorMatcher, iter_lora_parameters
from ..models.backbone import LoraSettings
from ..models.matcher import prediction_spread_patches
from ..models.lora import count_parameters, lora_state_dict
from ..optim import build_lr_scheduler, build_optimizer


class BEVAlignerModule(L.LightningModule):
    def __init__(self, config: Config | dict):
        super().__init__()
        if isinstance(config, dict):
            config = config_from_dict(config)
        self.config = config
        self.save_hyperparameters({"config": config.to_dict()})
        model_config = config.model
        encoder = DinoPatchEncoder(
            family=model_config.backbone_family, pretrained_id=model_config.pretrained_id,
            lora=LoraSettings(model_config.lora_rank, model_config.lora_alpha,
                              model_config.lora_dropout),
            gradient_checkpointing=model_config.gradient_checkpointing)
        self.model = DensityPriorMatcher(encoder, temperature=model_config.temperature)
        loss_config = config.loss
        self.loss_fn = BEVAlignerLoss(LossSettings(
            temperature=model_config.temperature, huber_delta=loss_config.huber_delta,
            num_triplets=loss_config.num_triplets, num_pairs=loss_config.num_pairs,
            detach_confidence=loss_config.detach_confidence,
            near_duplicate_radius=loss_config.near_duplicate_radius,
            weights=dict(loss_config.weights), gates=dict(loss_config.gates),
            gate_ramp=loss_config.gate_ramp))
        self.strict_loading = False        # checkpoints hold LoRA weights only
        num_lora_params = count_parameters(iter_lora_parameters(self.model))
        print(f"model: {model_config.backbone_family} ({encoder.pretrained_id}), "
              f"patch {encoder.patch_size}, {len(encoder.lora_layer_names)} LoRA layers, "
              f"{num_lora_params / 1e6:.2f} M trainable params")

    @property
    def patch_size(self) -> int:
        return self.model.patch_size

    # ---------------------------------------------------------------- steps
    def training_progress(self) -> float:
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        return min(self.global_step / total_steps, 1.0)

    def _forward_and_loss(self, batch, progress: float):
        output = self.model(batch["density"], batch["target"], batch["query_px"],
                            target_patch_valid=batch["target_patch_valid"])
        losses = self.loss_fn(output, batch["query_px"], batch["target_px"],
                              batch["target"].shape[-2:], self.patch_size, progress)
        return output, losses

    def training_step(self, batch, batch_idx):
        output, losses = self._forward_and_loss(batch, self.training_progress())
        batch_size = batch["density"].shape[0]
        self.log("train/loss/total", losses["total"], prog_bar=True, batch_size=batch_size)
        for name, value in losses.items():
            if name == "total":
                continue
            # "loss/feature" stays under loss/; "weight/feature" moves to the
            # curriculum group, which is what it describes
            group, _, term = name.partition("/")
            self.log(f"train/{'curriculum/weight_' + term if group == 'weight' else name}",
                     value, batch_size=batch_size)
        self.log("train/curriculum/progress_fraction", self.training_progress(),
                 batch_size=batch_size)
        self.log("train/input/resolution_px", float(batch["density"].shape[-1]),
                 batch_size=batch_size)
        # LearningRateMonitor logs this as a bare `lr-AdamW`, which sorts on its
        # own outside every other card. Same number, in the group it belongs to.
        self.log("train/optimizer/learning_rate",
                 self.optimizers().param_groups[0]["lr"], batch_size=batch_size)
        self.log("train/match/mean_confidence", output.confidence.mean(),
                 batch_size=batch_size)
        self._log_health("train", output, batch, losses, batch_size)
        self._log_input_health(batch, batch_size)
        return losses["total"]

    def _log_input_health(self, batch, batch_size: int) -> None:
        """How often the data path had to fall back, which is the only way an
        augmentation failure ever shows.

        A broken geometric augmentation does not raise: the ground-truth pairs
        come back empty, the sample comes back None, and the dataset retries,
        then drops the augmentation, then moves to a neighbouring record. So a
        rotation that clips every target reads as a slightly worse loss curve
        and nothing else.

        `unaugmented_fallback` is the sharp one: the last attempt is the one
        made with no augmentation generator, so a sample that only built there
        is a sample the augmentation broke. It should sit near zero, and a new
        knob that moves it is the cause.
        """
        meta = batch.get("meta") or []
        if not meta:
            return
        attempt = torch.tensor([float(item.get("attempt", 0)) for item in meta])
        hop = torch.tensor([float(item.get("hop", 0)) for item in meta])
        for name, value in (("retry_fraction", (attempt > 0).float().mean()),
                            ("unaugmented_fallback_fraction", (attempt >= 3).float().mean()),
                            ("neighbour_hop_fraction", (hop > 0).float().mean())):
            # at batch 4 a step reads one of {0, .25, .5, .75, 1}, which is
            # unreadable raw; the _epoch series Lightning also emits is the
            # running mean and is the one to look at
            self.log(f"train/input/{name}", value, batch_size=batch_size,
                     on_step=True, on_epoch=True)
        self.log("train/input/valid_patch_fraction",
                 batch["target_patch_valid"].float().mean(), batch_size=batch_size)

    def _log_health(self, stage: str, output, batch, losses, batch_size: int) -> None:
        """The two numbers that name a collapse while it is happening.

        `prediction_spread_patches` falling towards zero is the match field
        closing to a point; `feature_margin_nats` is how far InfoNCE is below
        the ln(Q) it would score by guessing, so a negative margin is worse
        than chance.
        """
        self.log(f"{stage}/match/prediction_spread_patches",
                 prediction_spread_patches(output.predicted_px, self.patch_size),
                 batch_size=batch_size, sync_dist=stage == "val")
        chance = math.log(batch["query_px"].shape[1])
        self.log(f"{stage}/match/feature_margin_nats", chance - losses["loss/feature"],
                 batch_size=batch_size, sync_dist=stage == "val")
        # PCK on BOTH sides, which it was not: val alone cannot show a
        # generalisation gap, and the gap is the only thing that says whether
        # regularisation is called for. The val path computes its own, with the
        # paper's crop-fraction thresholds beside it; this is the patch
        # threshold only, which is the unit that means the same at every
        # resolution.
        error_patches = ((output.predicted_px - batch["target_px"]).norm(dim=-1)
                         / self.patch_size)
        for threshold in (1, 2, 4):
            self.log(f"{stage}/match/pck_within_{threshold}_patches",
                     (error_patches <= threshold).float().mean(),
                     batch_size=batch_size, sync_dist=stage == "val")

    def validation_step(self, batch, batch_idx):
        output, losses = self._forward_and_loss(batch, 1.0)
        batch_size = batch["density"].shape[0]
        self.log("val/loss/total", losses["total"], batch_size=batch_size, sync_dist=True)
        for name, value in losses.items():
            if name.startswith("loss/"):
                self.log(f"val/{name}", value, batch_size=batch_size, sync_dist=True)
        error_px = (output.predicted_px - batch["target_px"]).norm(dim=-1)
        error_patches = error_px / self.patch_size
        for threshold in (1, 2, 4):
            self.log(f"val/match/pck_within_{threshold}_patches",
                     (error_patches <= threshold).float().mean(),
                     batch_size=batch_size, sync_dist=True)
        # The paper's PCK, to compare with its tables: "distances are normalised
        # by the target resolution", reported at {1, 3, 5, 10, 15, 30}%. The
        # patch thresholds above are this repo's own unit and are kept because
        # they mean the same thing at every training resolution, which a
        # fraction of the crop does not. Normalised by the long side of the crop
        # the network saw -- square at every resolution here, so side and
        # diagonal differ only by root 2, but the paper does not say which.
        long_side = max(batch["target"].shape[-2:])
        error_fraction = error_px / long_side
        for percent in (1, 3, 5, 10, 15, 30):
            self.log(f"val/match/pck_within_{percent}pct_of_crop",
                     (error_fraction <= percent / 100).float().mean(),
                     batch_size=batch_size, sync_dist=True)
        self.log("val/match/mean_confidence", output.confidence.mean(),
                 batch_size=batch_size, sync_dist=True)
        self.log("val/match/median_error_patches", error_patches.median(),
                 batch_size=batch_size, sync_dist=True)
        self._log_health("val", output, batch, losses, batch_size)
        # handed to `LogPredictedMatchImages`, which draws them; detached and
        # small (B,Q,2) and (B,Q), so nothing of the graph is kept alive
        return {"predicted_px": output.predicted_px.detach(),
                "confidence": output.confidence.detach()}

    # ---------------------------------------------------- the training flag
    # `DinoPatchEncoder.train()` is the thing that keeps the frozen backbone in
    # eval while letting the adapters follow the flag. Two Lightning behaviours
    # bypass it, and between them they left `model.lora_dropout` switched off
    # for the entire life of this package:
    #
    #   * nothing calls `.train()` on the LightningModule at fit start, and the
    #     encoder puts itself in eval in its own __init__;
    #   * after each validation Lightning restores the *captured* per-module
    #     flags by writing `module.training` directly, which reinstates eval on
    #     the adapters and undoes the `model.train()` in the pose callback.
    #
    # Overriding this hook is what stops Lightning taking that restore branch.
    def on_validation_model_train(self) -> None:
        self.model.train()

    def on_train_epoch_start(self) -> None:
        # and the same for the path with no validation before step 0
        self.model.train()

    def on_before_optimizer_step(self, optimizer) -> None:
        """The PRE-clip gradient norm.

        `optimizer.gradient_clip_norm` only changes the step when it binds, and
        how often it binds is otherwise invisible -- which matters most when the
        loss weights change, since that moves the norm without moving anything
        the other scalars show. Lightning calls this before `_clip_gradients`,
        and bf16-mixed uses no GradScaler, so this is the unscaled, unclipped
        norm over the LoRA parameters (the frozen ones have no grad).
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log("train/optimizer/grad_norm", norms["grad_2.0_norm_total"],
                 batch_size=1)

    # ------------------------------------------------------------ optimiser
    def configure_optimizers(self):
        optimizer_config = self.config.optimizer
        optimizer = build_optimizer(iter_lora_parameters(self.model),
                                    learning_rate=optimizer_config.learning_rate,
                                    weight_decay=optimizer_config.weight_decay)
        scheduler = build_lr_scheduler(optimizer, self.trainer.estimated_stepping_batches,
                                       optimizer_config.warmup_steps,
                                       optimizer_config.final_lr_fraction)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # ---------------------------------------------------------- checkpoint
    def on_save_checkpoint(self, checkpoint) -> None:
        lora_weights = lora_state_dict(self.model)
        checkpoint["state_dict"] = {
            name if name.startswith("model.") else f"model.{name}": tensor
            for name, tensor in lora_weights.items()}
        checkpoint["bevaligner_config"] = self.config.to_dict()

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu", overrides=None) -> "BEVAlignerModule":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        hyper_parameters = checkpoint.get("hyper_parameters", {})
        stored = (checkpoint.get("bevaligner_config")
                  or checkpoint.get("bevaligner_cfg")          # pre-rename checkpoints
                  or hyper_parameters.get("config")
                  or hyper_parameters["cfg"])
        config = config_from_dict(stored)
        if overrides:
            for dotted_key, value in overrides.items():
                section, key = dotted_key.split(".")
                setattr(getattr(config, section), key, value)
        module = cls(config)
        missing, unexpected = module.load_state_dict(checkpoint["state_dict"], strict=False)
        missing_lora = [name for name in missing if ".lora_" in name]
        if missing_lora or unexpected:
            raise KeyError(f"checkpoint mismatch: {missing_lora[:3]} / {list(unexpected)[:3]}")
        return module
