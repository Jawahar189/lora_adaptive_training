# BEVAligner: a learned density-map to PRIOR-CAPTURE matcher

An implementation of *BEVAligner: 3D-Grounded Floorplan Localization in the
Wild* (Cho, Cai, Averbuch-Elor; arXiv 2605.22581) on this repo's captures, in
the `bevaligner/` package. `README.md` is how to run it, this
file is what it does and why, and `docs/BEVALIGNER_METHOD.md` is the method in
detail: the maths, the units and the thresholds. The paper reconstructs a scene from photographs;
here the point clouds already exist, so the pipeline starts at the cloud and
the GPU is only needed for the matcher.

**On this branch the target is not the drawing.** A capture is matched against
the bird's-eye view of the previous capture of the same building, merged from
that capture's splits in paper inches. `docs/PRIOR_CAPTURE.md` is the method and
the measurements behind the substitution; it was made in the classical solver
first, and `bevaligner/data/prior_bev.py` brings it to the learned matcher.
Everything below that does not mention the sheet is unchanged by it: the frames,
the gravity, the wall band, the head, the losses and the schedule.

```
current cloud (PLY) --gravity--> top-down density map ─┐
                                                       ├─ DINOv3 ViT-B/16 + LoRA ─ soft-argmax ─ RANSAC ─ Sim(2) ─ tm
prior capture's splits --their own tm--> paper inches ─┘
                        --> merged BEV --> window
```

The chain the ground truth travels lost a link with the sheet:

```
before   density px -> cloud xy -[gt_sim2]-> paper inches -> sheet px -> crop px
after    density px -> cloud xy -[gt_sim2]-> paper inches ------------> crop px
```

`align/reference.py` already put the merged prior in paper inches, which is why
the substitution touches only the last links. `gt_sim2`, the Umeyama fit, the
correspondence sampler and all four loss terms are untouched.

## Layout

```
bevaligner/
  configs/     t4_smoke.yaml (DINOv2 fallback)  t4_smoke_dinov3.yaml  gpu24.yaml  gpu48.yaml
  data/        store.py (CaptureStore: CSV rows -> cached records), preprocess.py (the slow
               per-row work + CLI), density.py, floorplan.py, transforms.py,
               correspondences.py, dataset.py (DensityFloorplanPairDataset), sampler.py,
               datamodule.py, splits.py
  models/      lora.py, backbone.py (DinoPatchEncoder over DINOv3 / DINOv2),
               matcher.py (DensityFloorplanMatcher)
  losses/      feature.py (feature_infonce_loss), regression.py (coordinate_regression_loss),
               topology.py (angle_topology_loss), geometry.py (scale_consistency_loss),
               curriculum.py (BEVAlignerLoss + active_loss_weights)
  optim/       AdamW + warm-up/cosine
  lightning/   module.py, callbacks.py (SaveTrainingPairImages,
               LogPredictedMatchImages, PoseEvaluation)
  inference/   matching.py (match_density_to_crop), ransac.py (fit_planar_pose),
               propose.py (propose_pose)
  utils/       everything borrowed from `align`, wrapped once; groundtruth.py is the only GT reader
  train.py     evaluate.py
tests/bevaligner/
```

`align/` is untouched. Ground truth (`tm.json`, `images_full.json`) is read
only through `bevaligner/utils/groundtruth.py`, which wraps
`align.groundtruth` and the oracle in `align.eval.oracle`; `models`, `losses`
and `inference` never import it, so they can be used inside the solver.

## Setup

```bash
.venv/bin/pip install -r requirements.txt          # adds lightning, tensorboard, pyyaml
export HF_HOME=$PWD/cache/hf                       # where the DINO weights live
.venv/bin/huggingface-cli login                    # DINOv3 weights are gated; accept the licence first
```

Without a token, use the `dinov2` configs: `facebook/dinov2-base` is already
cached. Its patch is 14, so resolutions are 504 / 756 / 1008 instead of 512 /
768 / 1024. `Backbone` refuses a resolution that is not a patch multiple.

## 1. Data: the CSV is the dataset

`data.csv_files` in the config (default `datasets/100_prod_current.csv`) is the
dataset: its rows, in the repo's 24-column capture index format, are loaded
with `align.io.cases.CaseIndex`, split by structure (10 validation
structures, or the rows of `data.val_csv_files` when given), and fed to the
loaders. Listing more than one file adds rows, e.g. the prior captures.

Each row needs slow work once -- gravity, the ground-truth planar fit, the
wall band, the structure's viewport crops -- and `CaptureStore`
(`bevaligner/data/store.py`) keeps that under `cache/bevaligner/` keyed by
reality and structure id. The DataModule fills whatever is missing before
training starts; to do it ahead of time:

```bash
.venv/bin/python scripts/bevaligner_preprocess.py --csv datasets/100_prod_current.csv --workers 16
```

Per capture: cloud, gravity (`align.gravity`), the ground-truth planar fit
over both up hypotheses (`align.eval.oracle.best_planar_fit`, which also
settles the up sign), a wide wall band with per-point relative height and
nearest camera. Per structure: wall extraction (`align.sheet.prepare`),
viewport boxes, every box cropped to a 1280-px square with its exact affine.
Then each capture is assigned the viewport that holds its cameras.

Output under `cache/bevaligner/`: `captures/<rlt>.npz|json`,
`sheets/<structure>/viewport*.png|index.json`, `manifest.csv` (one row per capture
with its status or skip reason), `splits.json` (10 validation structures).

On `100_prod_current.csv` + `100_prod_prior.csv` (2026-09-24): 412 of 470 rows usable;
36 `bad_fit` (the ground-truth `tm` is not a planar similarity of the camera
walk to within 0.5 m), 11 `no_floor`, 10 `gt_outside_sheet` (the truth puts
the cameras off every drawing region of the sheet, half of them on one
structure without a design transform), 1 `thin_band`.

## 2. Train

```bash
# sanity: two captures, no augmentation; loss must collapse and PCK@2 patches -> ~1
# measured 2026-09-25 (DINOv2, 1008 px, batch 2, 200 steps, 8 min 16 s on the T4):
#   val/pck@2p        0.09 -> 0.99 -> 1.00 -> 1.00   at steps 50/100/150/200
#   val/rmse_xy_median 23.6 m -> 0.32 -> 0.28 -> 0.10 m, tpr@1.0 0.00 -> 1.00
.venv/bin/python -m bevaligner.train --config bevaligner/configs/t4_smoke.yaml \
    --overfit-realities RLT1789764521954 RLT1789764415209 --no-augment --max-steps 200 \
    run_name=overfit2 data.resolutions=[1008] data.samples_per_epoch=400

# smoke on the T4
.venv/bin/python -m bevaligner.train --config bevaligner/configs/t4_smoke.yaml --save-pair-images 2

# the real run, 24 GB card
.venv/bin/python -m bevaligner.train --config bevaligner/configs/gpu24.yaml
```

Positional `key=value` arguments override the YAML. Runs land in
`out/bevaligner/<run_name>/` with `config.yaml`, CSV + TensorBoard logs,
`checkpoints/` (LoRA weights only: a 10 MB state dict, 31 MB on disk once
AdamW's moments are in the file) and, with `--save-pair-images`,
`batches/*.png`: density | crop with the ground-truth pairs drawn. Look at
those first; a mirror or shift is obvious there and invisible in a loss curve.

### Pictures in TensorBoard

Every validation pass draws what the matcher actually predicted for the first
few validation samples and logs it under the **Images** tab as
`val/matches/<n>`, one picture per sample and one step per validation, so the
slider walks through training. Each shows the density map beside the floor-plan
crop, a line from every drawn query to the patch the soft-argmax chose, green
when that landed within `trainer.match_correct_within_patches` (2) of the truth
and red when it did not, an amber ring on the truth with a stub to the
prediction, and line width growing with the match's confidence. The caption
carries the reality id, the resolution, PCK, the median error in patches and
the mean confidence.

`trainer.num_match_images` (4) sets how many samples are drawn; 0 turns them
off. The training-pair dumps from `--save-pair-images` go to the same tab under
`train/pairs/<batch>_<item>` as well as to `batches/*.png`. If TensorBoard is
not installed, the validation pictures fall back to
`out/bevaligner/<run_name>/matches/*.png`.

```bash
.venv/bin/python -m tensorboard.main --logdir out/bevaligner --port 6006
```

Checkpoints are selected on `val/tpr@1.0_mean`, the fraction of validation
cameras placed within 1 m by the full propose-and-RANSAC path (callback
`PoseEvaluation`), not on the correspondence loss.

## 3. Evaluate

```bash
.venv/bin/python -m bevaligner.evaluate --ckpt out/bevaligner/gpu24/checkpoints/last.ckpt \
    --csv datasets/100_prod_current.csv --split val --stem out/bevaligner/eval/gpu24_val
```

Everything is computed live from the CSV (cloud, gravity, both up signs), so
unseen captures work; sheet crops come from the cache when the structure is
there. Output through `align.eval.reality.ResultWriter`, the same three files
`scripts/run_alignment.py` writes, so `summarise` numbers are comparable.

## What differs from the paper, and why

- **No reconstruction.** π³/VGGT and GeoCalib are replaced by the PLYs and
  `align.gravity`. Their confidence filter has no analogue; `source_frame_count >= 2` stands in.
- **Density is rendered y-down.** `align.segments.rasterize` is y-up; a y-up
  map against a y-down sheet differs by a mirror, which a proper-rotation
  `Sim2` cannot represent. `tests/bevaligner/test_affine.py` pins `det > 0`.
- **Zoomed sheet crops.** The paper's scenes cover most of their floor plan;
  ours are short walks through large buildings, a few patches wide on a whole
  viewport at 1024 px. Training crops zoom log-uniformly between the capture's
  footprint (x1.3) and the whole viewport; inference runs the whole viewport
  first, then re-matches on a crop around the winning footprint (`--no-refine` disables).
- **Up sign.** Roughly one capture in six has an ambiguous floor/ceiling. The
  cache stores the oracle's choice; no flip augmentation is used; inference
  tries both and RANSAC's inlier count decides.
- **Scale of the data.** ~420 captures vs the paper's ~280K augmented samples,
  hence heavy on-the-fly augmentation (camera-window subsets, band jitter,
  gamma, re-rendered crops, rot90, photometric) and a structure-level split.
- **Huber δ = 1 patch**, coordinates in patch units, so the loss is the same
  quantity at every resolution. The paper does not state its δ.
- **InfoNCE ignores near-duplicate negatives.** With 1024 queries on a capture
  that spans a median of 108 floor-plan patches, many pairs share a target;
  as plain negatives they pin the loss at about 2.8 nats and pull the features
  in contradictory directions (the first smoke run's InfoNCE *rose* above
  log 1024). Pairs whose targets lie within one patch of the anchor's are
  masked out of the denominator on both sides (`loss.near_duplicate_radius`).
- **Zoom-out is capped** so the capture spans at least
  `data.min_footprint_patches` (4) patches of the crop; a footprint of one
  patch carries no localisation signal.
- **Curriculum gates are configurable.** The paper's 10 % / 20 % assume a
  long run; a 1000-step smoke switches the weight-50 regression on after 100
  steps, before the features mean anything, and the smoke config moves the
  gates to 30 % / 50 %. The gpu24/gpu48 presets keep the paper's.

## Wiring into the solver (not done)

`bevaligner.inference.as_solver_fit_dict(proposal)` returns the dictionary
`align.runner.visual_local_search` already consumes from the panorama prior,
so a `stage_bevaligner` proposing a candidate for the correlation search is
one call away once a checkpoint is worth it.
