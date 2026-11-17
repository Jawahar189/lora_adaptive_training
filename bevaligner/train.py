"""Train BEVAligner.

    python -m bevaligner.train --config bevaligner/configs/t4_smoke.yaml
    python -m bevaligner.train --config bevaligner/configs/gpu24.yaml \
        optimizer.learning_rate=5e-5
    python -m bevaligner.train --config ... --overfit-realities RLT... RLT... --no-augment

Positional `key=value` arguments override the YAML (`data.resolutions=[512]`).
"""
from __future__ import annotations

import argparse
import os
import time

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from .config import REPO_ROOT, load_config
from .data.datamodule import BEVAlignerDataModule
from .lightning.callbacks import (LogPredictedMatchImages, PoseEvaluation,
                                  SaveTrainingPairImages)
from .lightning.module import BEVAlignerModule
from .models.lora import load_lora_state_dict
from .utils.logging import build_loggers


def build_training_objects(config, overfit_reality_ids=None, save_pair_images: int = 0,
                           skip_pose_eval: bool = False):
    module = BEVAlignerModule(config)
    datamodule = BEVAlignerDataModule(config, patch_size=module.patch_size,
                                        overfit_reality_ids=overfit_reality_ids)
    run_dir = config.run_dir
    monitored_metric = (config.trainer.monitor
                        or ("val/match/pck_within_2_patches" if skip_pose_eval
                            else "val/pose/cameras_within_1.0m"))
    callbacks = [
        ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="best-{step}",
                        monitor=monitored_metric, mode="max",
                        save_top_k=config.trainer.save_top_k, save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    if not skip_pose_eval:
        callbacks.append(PoseEvaluation(datamodule, config.trainer.val_pose_eval_captures,
                                        config.data.eval_resolution))
    if save_pair_images:
        callbacks.append(SaveTrainingPairImages(run_dir / "batches",
                                                num_batches=save_pair_images))
    if config.trainer.num_match_images:
        callbacks.append(LogPredictedMatchImages(
            num_samples=config.trainer.num_match_images,
            correct_within_patches=config.trainer.match_correct_within_patches,
            fallback_dir=run_dir / "matches"))
    return module, datamodule, callbacks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("overrides", nargs="*", help="key=value YAML overrides")
    parser.add_argument("--overfit-realities", nargs="*", default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--save-pair-images", type=int, default=0,
                        help="write this many training batches as annotated PNGs")
    parser.add_argument("--no-pose-eval", action="store_true")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--init-from", default=None,
                        help="checkpoint to take LoRA weights from. A NEW run: the "
                             "optimiser, the LR schedule, the curriculum and the data "
                             "order all start fresh, unlike --resume")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(argv)
    # Before the config and the cache, which take minutes.
    if args.init_from and args.resume:
        parser.error("--init-from and --resume are mutually exclusive: --resume "
                     "continues a run, restoring its optimiser, schedule, curriculum "
                     "position and data order; --init-from starts a new one from the "
                     "same weights and nothing else")

    config = load_config(args.config, args.overrides)
    if args.no_augment:
        config.data.augment = False
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "cache" / "hf"))
    L.seed_everything(config.trainer.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    module, datamodule, callbacks = build_training_objects(
        config, args.overfit_realities, args.save_pair_images, args.no_pose_eval)
    if args.init_from:
        # Deliberately NOT BEVAlignerModule.from_checkpoint, which rebuilds the
        # module from the checkpoint's OWN config and would silently restore the
        # loss weights a new experiment exists to change. Load into the module
        # already built from this run's YAML.
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        load_lora_state_dict(module, state)
        source = (checkpoint.get("bevaligner_config") or {}).get("run_name", "?")
        print(f"init-from   : {args.init_from}\n"
              f"              {len(state)} LoRA tensors from run '{source}' at step "
              f"{checkpoint.get('global_step', '?')}\n"
              f"              weights only: optimiser, schedule, curriculum and data "
              f"order all start at zero")

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")

    trainer_config = config.trainer
    # The command line wins when given; otherwise the config carries it. It has to
    # be expressible in the YAML: the loss gates are fractions of
    # `estimated_stepping_batches`, which max_steps caps, so a restart that forgot
    # the flag would silently move every gate.
    max_steps = args.max_steps if args.max_steps != -1 else trainer_config.max_steps
    trainer = L.Trainer(
        accelerator="auto", devices=1, precision=trainer_config.precision,
        max_epochs=trainer_config.max_epochs, max_steps=max_steps,
        accumulate_grad_batches=config.optimizer.accumulate_grad_batches,
        gradient_clip_val=config.optimizer.gradient_clip_norm,
        val_check_interval=trainer_config.val_check_interval,
        limit_val_batches=trainer_config.val_batches,
        num_sanity_val_steps=1, log_every_n_steps=trainer_config.log_every_n_steps,
        callbacks=callbacks,
        logger=build_loggers(trainer_config.out_dir, config.run_name),
        default_root_dir=str(run_dir), deterministic=trainer_config.deterministic,
        fast_dev_run=args.fast_dev_run, enable_model_summary=False,
    )
    started = time.time()
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)
    finally:
        # In a `finally`, because the summary is worth most exactly when the run
        # did NOT finish: a SIGTERM'd job otherwise leaves nothing in the log
        # but a truncated progress bar, and `metrics.csv` is rewritten by the
        # next start, so this is the only durable record of where it got to.
        print(_run_summary(trainer, config, callbacks, time.time() - started,
                           resumed_from=args.resume,
                           init_from=args.init_from))


def _run_summary(trainer, config, callbacks, elapsed_s: float, resumed_from=None,
                 init_from=None) -> str:
    """Where the run got to, what it selected, and on what.

    Read alongside TensorBoard rather than instead of it: these are the last
    values, not the curves, and a checkpoint is selected on the monitored
    metric over the whole run, not on where it happened to end.
    """
    checkpoint = next((c for c in callbacks if isinstance(c, ModelCheckpoint)), None)
    hours, remainder = divmod(int(elapsed_s), 3600)
    lines = [
        "",
        f"=== {config.run_name} ===",
        f"  stopped at   : step {trainer.global_step}, epoch {trainer.current_epoch}"
        f"  ({hours}h {remainder // 60:02d}m this process)",
    ]
    if resumed_from:
        lines.append(f"  resumed from : {resumed_from}")
    if init_from:
        lines.append(f"  initialised  : {init_from}  (weights only)")
    if checkpoint is not None:
        score = checkpoint.best_model_score
        best = checkpoint.best_model_path or "(none written)"
        lines += [
            f"  selected on  : {checkpoint.monitor} (max)",
            f"  best         : {best}"
            + (f"   {float(score):.4f}" if score is not None else ""),
            f"  last         : {checkpoint.last_model_path or '(none written)'}",
        ]
    metrics = {k: v for k, v in trainer.callback_metrics.items() if k.startswith("val/")}
    if metrics:
        lines.append("  last validation:")
        for key in sorted(metrics):
            lines.append(f"    {key:<44} {float(metrics[key]):.4f}")
    else:
        lines.append("  last validation: none ran")
    lines += [
        f"  train csv    : {', '.join(config.data.csv_files)}",
        "  val csv      : "
        + (', '.join(config.data.val_csv_files) if config.data.val_csv_files
           else f"cut by structure, seed {config.data.split_seed}"),
        f"  prior csv    : {', '.join(config.data.prior_csv_files)}",
        f"  run dir      : {config.run_dir}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
