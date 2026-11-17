"""Callbacks: pictures of what the network is asked to learn and of what it
predicts, and the metric that actually matters -- where the cameras land, in
metres.

The two picture callbacks draw the same two panels, density map on the left and
prior crop on the right, and send them to every logger that takes images
(TensorBoard), falling back to PNGs on disk when none does. A mirror, a shift
or a collapsed match field is obvious there and invisible in a loss curve.
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
from PIL import Image

from ..inference import PoseProposalOptions, propose_pose
from ..viz import (CORRECT_COLOUR, TRUTH_COLOUR, WRONG_COLOUR,  # noqa: F401
                   render_match_image)
from ..utils import image as image_utils
from ..utils.groundtruth import sim2_from_params
from ..utils.gravity import GravityView


def log_image_to_loggers(trainer, tag: str, image: Image.Image, step: int) -> bool:
    """Send one picture to every logger that accepts images. True if any did."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    logged = False
    for logger in getattr(trainer, "loggers", None) or []:
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_image"):
            continue                                  # e.g. the CSV logger
        experiment.add_image(tag, array, global_step=step, dataformats="HWC")
        logged = True
    return logged


def _panels(batch, item: int) -> tuple[np.ndarray, np.ndarray]:
    density = image_utils.tensor_to_rgb_uint8(
        image_utils.imagenet_denormalize(batch["density"][item]))
    target = image_utils.tensor_to_rgb_uint8(
        image_utils.imagenet_denormalize(batch["target"][item]))
    return density, target


class SaveTrainingPairImages(L.Callback):
    """The ground-truth pairs of the first few training batches: what the
    network is being asked to learn, before it has learned anything."""

    def __init__(self, out_dir: Path, num_batches: int = 4, num_lines: int = 40):
        self.out_dir = Path(out_dir)
        self.num_batches, self.num_lines = num_batches, num_lines
        self.saved_batches = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.saved_batches >= self.num_batches:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for item in range(batch["density"].shape[0]):
            meta = batch["meta"][item]
            density, target = _panels(batch, item)
            image = render_match_image(
                density, target, batch["query_px"][item].cpu().numpy(),
                batch["target_px"][item].cpu().numpy(), num_lines=self.num_lines,
                caption=f"{meta['reality_id']}  truth  R{meta['resolution']}")
            image.save(self.out_dir / f"batch{batch_idx:03d}_{item}_{meta['reality_id']}_"
                                      f"R{meta['resolution']}.png")
            log_image_to_loggers(trainer, f"train/pairs/{self.saved_batches}_{item}",
                                 image, trainer.global_step)
        self.saved_batches += 1


class LogPredictedMatchImages(L.Callback):
    """Every validation pass, draw what the matcher actually predicted for a few
    validation samples and log it as an image.

    Green means the prediction landed within `correct_within_patches` of the
    truth, red means it did not, and the amber ring with its stub shows where
    the truth was. Line width grows with the match's confidence.
    """

    def __init__(self, num_samples: int = 4, num_lines: int = 40,
                 correct_within_patches: float = 2.0, fallback_dir: Path | None = None):
        self.num_samples, self.num_lines = num_samples, num_lines
        self.correct_within_patches = correct_within_patches
        self.fallback_dir = Path(fallback_dir) if fallback_dir else None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._drawn = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx: int = 0):
        # The val loader batches at half the training size, so `num_samples`
        # larger than that spans several batches rather than being silently
        # capped at the first one's length. The tag index runs across them, so
        # `val/matches/<n>` stays the same slot at every validation pass.
        if trainer.sanity_checking or not self.num_samples:
            return
        drawn = getattr(self, "_drawn", 0)
        if drawn >= self.num_samples:
            return
        if not isinstance(outputs, dict) or "predicted_px" not in outputs:
            return
        patch_size = pl_module.patch_size
        tolerance_px = self.correct_within_patches * patch_size
        predicted = outputs["predicted_px"].float().cpu().numpy()
        confidence = outputs["confidence"].float().cpu().numpy()
        for item in range(min(self.num_samples - drawn, batch["density"].shape[0])):
            meta = batch["meta"][item]
            query_px = batch["query_px"][item].cpu().numpy()
            target_px = batch["target_px"][item].cpu().numpy()
            error_patches = np.linalg.norm(predicted[item] - target_px, axis=1) / patch_size
            caption = (f"{meta['reality_id']}  R{meta['resolution']}  "
                       f"pck@{self.correct_within_patches:g}p "
                       f"{float((error_patches <= self.correct_within_patches).mean()):.2f}  "
                       f"median err {float(np.median(error_patches)):.1f}p  "
                       f"conf {float(confidence[item].mean()):.3f}")
            density, target = _panels(batch, item)
            image = render_match_image(density, target, query_px, target_px,
                                       predicted_px=predicted[item],
                                       confidence=confidence[item],
                                       num_lines=self.num_lines,
                                       correct_within_px=tolerance_px, caption=caption)
            index = drawn + item
            logged = log_image_to_loggers(trainer, f"val/matches/{index}", image,
                                          trainer.global_step)
            if not logged and self.fallback_dir is not None:
                self.fallback_dir.mkdir(parents=True, exist_ok=True)
                image.save(self.fallback_dir
                           / f"step{trainer.global_step:07d}_{index}_{meta['reality_id']}.png")
        self._drawn = drawn + min(self.num_samples - drawn, batch["density"].shape[0])


def pose_metrics(rows: list[dict], thresholds_m=(0.5, 1.0, 2.0),
                 thresholds_pct_diagonal=(5, 10, 20),
                 thresholds_deg=(5, 10, 20, 30)) -> dict[str, float]:
    """Score proposed poses against the truth, in metres and in the paper's units.

    One `row` per capture: `solved`, and when solved `errors_m` and
    `errors_diagonal` (per camera) plus one `yaw_error_deg` for the capture,
    since a capture gets a single similarity.

    `errors_diagonal` is a fraction of the PRIOR reference's diagonal, not a
    floorplan's. Metres are the comparable number, and on structures whose
    `scale_source` is not `design_tm` even those rest on a rig-height estimate
    -- rows carry `scale_source` so the two populations can be told apart.

    Camera recalls are the mean over CAPTURES of the fraction of that capture's
    cameras inside the threshold, so a 500-camera capture counts the same as a
    5-camera one, and a capture that did not solve scores 0. That is this
    repo's existing convention for tpr; the paper may instead pool all cameras,
    which with captures this uneven is a different number. Its tables should be
    read knowing which.
    """
    if not rows:
        return {}
    solved = [row for row in rows if row["solved"]]
    logs = {"solved_fraction": len(solved) / len(rows)}

    def camera_recall(key: str, threshold: float) -> float:
        return float(np.mean([float((row[key] <= threshold).mean()) if row["solved"] else 0.0
                              for row in rows]))

    for threshold in thresholds_m:
        logs[f"cameras_within_{threshold}m"] = camera_recall("errors_m", threshold)
    for percent in thresholds_pct_diagonal:
        logs[f"cameras_within_{percent}pct_prior_diagonal"] = camera_recall(
            "errors_diagonal", percent / 100)
    for degrees in thresholds_deg:
        logs[f"rotation_within_{degrees}deg"] = float(np.mean(
            [bool(row["solved"] and row["yaw_error_deg"] <= degrees) for row in rows]))
    # the paper's combined criterion in shape: one yaw inside 30 degrees AND
    # most of the capture's cameras inside 20% of the reference diagonal
    logs["pose_within_30deg_20pct_prior_diagonal"] = float(np.mean(
        [bool(row["solved"] and row["yaw_error_deg"] <= 30
              and (row["errors_diagonal"] <= 0.20).mean() >= 0.5) for row in rows]))
    if solved:
        logs["rmse_xy_median_m"] = float(np.median(
            [np.sqrt((row["errors_m"] ** 2).mean()) for row in solved]))
        logs["yaw_error_median_deg"] = float(np.median(
            [row["yaw_error_deg"] for row in solved]))
        logs["inliers_median"] = float(np.median([row["num_inliers"] for row in solved]))
        logs["refined_fraction"] = float(np.mean([row["refined"] for row in solved]))
        # the metre numbers above mix two populations: structures whose scale
        # comes from a design transform, and structures where it is a
        # rig-height estimate. Logged so the mixture is visible rather than
        # quoted as one number.
        exact = [row for row in solved if row.get("scale_source") == "design_tm"]
        logs["scale_from_design_tm_fraction"] = len(exact) / len(solved)
        if exact:
            logs["rmse_xy_median_m_design_tm"] = float(np.median(
                [np.sqrt((row["errors_m"] ** 2).mean()) for row in exact]))
        logs["overlap_median"] = float(np.median(
            [row.get("overlap", float("nan")) for row in solved]))
    else:
        logs["rmse_xy_median_m"] = float("nan")
        logs["yaw_error_median_deg"] = float("nan")
    return logs


class PoseEvaluation(L.Callback):
    """After each validation pass: propose a pose for some validation captures
    from the cache (the oracle's gravity frame, searched over windows of the
    prior capture) and score the cameras against the ground-truth similarity.

    The %-of-diagonal recalls are the paper's unit but no longer the paper's
    number: the denominator was the floorplan's diagonal and there is no
    floorplan. It is the prior reference's diagonal here, which varies far more
    across the dataset -- a median 352 m, a quarter over a kilometre -- so these
    are logged as `..._pct_prior_diagonal` and are NOT comparable to the paper's
    tables. Metres are, and they are what the product is judged on.
    """

    def __init__(self, datamodule, num_captures: int = 20, resolution: int = 1024,
                 thresholds_m=(0.5, 1.0, 2.0),
                 thresholds_deg=(5, 10, 20, 30), thresholds_pct_diagonal=(5, 10, 20)):
        self.datamodule = datamodule
        self.num_captures, self.resolution = num_captures, resolution
        self.thresholds_m = thresholds_m
        # the paper's own pose metrics, so its tables can be read against these:
        # angular recall at {5,10,20,30} degrees, positional recall at {5,10,20}%
        # of the target diagonal, and the pair at (30 deg, 20%). Metres are
        # kept alongside because they are what the product is judged on.
        self.thresholds_deg = thresholds_deg
        self.thresholds_pct_diagonal = thresholds_pct_diagonal

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.datamodule.val_dataset is None:
            return
        dataset = self.datamodule.val_dataset
        # Spread over the split, not the front of it. In CSV order the first 20
        # records fell in 7 of 89 val structures -- the pose numbers would
        # describe those 7 and move when their sheets got easier. Permuted from
        # the seed, so it is a different 20 but the SAME 20 at every validation.
        order = np.random.default_rng(dataset.seed + 3).permutation(len(dataset.records))
        records = [dataset.records[i] for i in order[: self.num_captures]]
        model = pl_module.model.eval()
        options = PoseProposalOptions(
            resolution=self.resolution,
            num_queries=dataset.data_config.queries_per_sample,
            refine_on_footprint=dataset.data_config.refine_on_footprint,
            refine_footprint_margin=dataset.data_config.refine_footprint_margin)
        rows = []
        for record in records:
            view = GravityView(
                up_index=record.up_index,
                rotation_to_gravity_up=record.rotation_to_gravity_up,
                band_xy=record.band_xy, band_relative_height=record.band_relative_height,
                band_camera_index=record.band_camera_index, camera_xy=record.camera_xy,
                camera_z=record.camera_z,
                camera_height_above_floor=record.camera_height_above_floor,
                floor_relative_mad=0.0)
            pool = self.datamodule.store.pool_for(record)
            proposal = (propose_pose(model, {record.up_index: view}, pool,
                                     dataset.data_config, record.metres_per_inch,
                                     options, device=pl_module.device,
                                     patch_size=pl_module.patch_size)
                        if pool is not None else None)
            gt_sim2 = sim2_from_params(record.gt_sim2_params)
            gt_camera_paper = gt_sim2.apply(record.camera_xy)
            if proposal is None:
                rows.append(dict(solved=False, errors_m=None, errors_diagonal=None,
                                 yaw_error_deg=None))
                continue
            predicted_paper = proposal.sim2.apply(record.camera_xy)
            errors_inches = np.linalg.norm(predicted_paper - gt_camera_paper, axis=1)
            # as a fraction of the PRIOR reference's diagonal. Error and
            # diagonal are both in paper inches, so they share a scale whatever
            # that turns out to be in metres.
            lo_x, lo_y, hi_x, hi_y = pool.extent
            diagonal_in = float(np.hypot(hi_x - lo_x, hi_y - lo_y))
            # one similarity per capture, so one yaw error for all its cameras
            yaw_error = abs((proposal.sim2.theta - gt_sim2.theta + np.pi) % (2 * np.pi) - np.pi)
            rows.append(dict(solved=True, errors_m=errors_inches * record.metres_per_inch,
                             errors_diagonal=errors_inches / max(diagonal_in, 1e-9),
                             yaw_error_deg=float(np.degrees(yaw_error)),
                             scale_source=record.metres_per_inch_source,
                             overlap=record.overlap, same_day=record.same_day,
                             num_inliers=proposal.num_inliers, refined=proposal.refined))
        num_rows = len(rows)
        logs = pose_metrics(rows, self.thresholds_m, self.thresholds_pct_diagonal,
                            self.thresholds_deg)
        for name, value in logs.items():
            pl_module.log(f"val/pose/{name}", value, rank_zero_only=True)
        print(f"\n[pose eval] n={num_rows} "
              + "  ".join(f"{name}={value:.3f}" for name, value in logs.items()))
        pl_module.model.train()
