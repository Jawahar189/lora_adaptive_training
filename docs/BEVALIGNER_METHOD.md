# BEVAligner: the method in detail

How the learned floor-plan matcher works, end to end: what is computed once per
capture, how ground truth is prepared, what the network is, what it is trained
on, and how a pose comes out the other side.

`README.md` is how to run it. `docs/BEVALIGNER.md` is the shorter account and
lists every departure from the paper. **This** file is the method: the maths,
the units, the thresholds, and the reason each one is where it is.

Everything here refers to the `bevaligner/` package. Code references are
`file.py:symbol`.

---

## 0. The problem, and the shape of the answer

A capture is a point cloud in an arbitrary frame at an arbitrary scale, with one
camera pose per photograph in that same frame. A structure is an architectural
sheet: a drawing several thousand pixels wide, with a title block, notes, and
often more than one plan viewport on the page.

The answer is `tm`, the 4x4 that carries the point cloud into the building's
coordinates. Under the planar assumption it has four degrees of freedom — a
scale, a yaw, and two translations — plus a height that is set afterwards, so
the thing actually solved for is a similarity of the plane:

```
Sim2 = (s, θ, tx, ty)      gravity-up cloud xy  ->  paper inches
```

Everything in this document exists to produce that `Sim2` and to measure it.

### Frames, and the units that separate them

Five frames, four conversions. Confusing any two of them is the failure mode
this codebase is most careful about, because a wrong frame trains happily and
only the final accuracy tells.

| Frame | Units | Origin / orientation |
|---|---|---|
| cloud | arbitrary | as reconstructed |
| gravity-up cloud | same arbitrary units | rotated so gravity is `-z` |
| paper | inches | sheet origin bottom-left, **y up** |
| sheet pixels | pixels | image top-left, **y down** |
| crop pixels | pixels | the square crop fed to the network, y down |

Every conversion in the data path is a 2x3 affine, named for the two frames it
connects, and they compose exactly (`utils/affine.py`):

```
density px --density_px_to_world_xy--> gravity-up cloud xy
           --gt_sim2_affine----------> paper inches
           --paper_to_sheet_px-------> sheet pixels
           --sheet_to_crop-----------> crop pixels
                     = density_to_crop, one 2x3 matrix
```

Two of those flip the y axis: the density map is rendered y-down like an image,
and paper-to-pixels flips y-up to y-down. **Their determinants multiply to a
positive number**, which is the whole point: the density map and the sheet crop
must differ by a rotation and never by a mirror, because a proper `Sim2` cannot
represent a reflection. `tests/bevaligner/test_affine.py` asserts
`determinant(density_to_crop) > 0` on a synthetic chain, and the augmentation
composes each op's inverse into the stored affine rather than re-deriving it, so
the chain stays exact to floating point.

---

## 1. Preprocessing, per capture

Run by `data/preprocess.py:prepare_capture`, on demand by `data/store.py` or
ahead of time by the CLI. One `.npz` of arrays and one `.json` of status per
reality. Everything slow and everything that touches ground truth happens here,
exactly once.

### 1.1 The cloud

`utils/io.py:load_point_cloud_xyz` reads the PLY through `align.cloud.load` with
two filters:

- `stride` (default 1) keeps every Nth point.
- `min_source_frames` (default 2) drops points seen by fewer source frames,
  which stands in for the paper's reconstruction-confidence filter.

A capture with fewer than 3 cameras or fewer than 1000 points is rejected as
`too_small`.

### 1.2 Gravity, and why there are two answers

`utils/gravity.py:gravity_views` estimates the up vector through
`align.gravity.estimate_gravity`. The first estimate is the **normal of the
camera trajectory's plane** (an SVD of the centred camera positions). When the
walk is too far from planar — a planarity measure, the third singular value as a
fraction of the walk's horizontal extent, above 0.02 — it falls back to
**surface normals**: local covariance eigenvectors over a subsample of the
cloud, keeping only well-defined surfaces, then the direction most orthogonal to
all of those wall normals.

That gives an axis, not a direction. The sign is resolved by fitting a floor for
`+up` and for `−up` and comparing how much of the cloud lies **past** each
candidate surface; if those tie within a small tolerance, by which fit is more
internally consistent across cameras. **When both tests tie, the capture is
flagged `sign_ambiguous` and both hypotheses are kept**, one `GravityView` each.
Roughly one capture in six lands there.

Getting this wrong mirrors the map. That is precisely why no flip augmentation
is ever used, and why RANSAC with a proper-rotation model is the thing that
settles the sign at inference: the wrong sign cannot be explained by any
similarity hypothesis and scores few inliers.

For each hypothesis, `make_gravity_view` rotates cloud and cameras so gravity is
`-z`, fits the floor beneath the trajectory, and keeps a **wide wall band**:

```
relative height = (z_point − z_nearest_camera) / camera_height_above_floor
cached band     = (−0.95, +0.35)      what preprocessing keeps
default band    = (−0.85, +0.25)      align.cloud's own band, used at inference
```

The floor sits at `−1.0` on this scale, so the default band runs from about
0.15 to 1.25 camera-heights above the floor: above the floor itself, below the
ceiling and the roof structure. The cache keeps a wider slice so training can
jitter the edges without touching the cloud again.

The nearest camera is found with a 2-D KD-tree over camera xy, so the band
follows the walk rather than a single global plane — a capture that climbs half
a storey still bands correctly. Each kept point stores its relative height and
its nearest-camera index, which is what makes two later augmentations possible
without touching the cloud again: narrowing the band, and keeping only the
points seen from a contiguous run of cameras.

A hypothesis with no usable floor is `no_floor`; one whose default band holds
too few points is `thin_band`.

### 1.3 Scale: metres per paper inch

Two sources, in order:

1. A georeferenced structure has a `design_tm`, and `align.frames.design_scale`
   reads metres per paper inch straight out of it.
2. Otherwise `align.reference.metres_per_inch_for` inverts the known rig: the
   360 camera sits on a monopod at a measured **2.11 m**, so metres per inch is
   `2.11 / camera_height_in_inches`, accepted only if it lands inside
   `[0.5, 80]` m/inch. Failing that, a flat 3.0 fallback, recorded as such.

This number is what lets a tolerance be stated in metres and applied in inches,
and it is recorded per capture (`metres_per_inch`, `metres_per_inch_source`) so
no later number silently mixes the two.

### 1.4 Rig plausibility

```
rig_height_m = camera_height_above_floor × sim2.s × metres_per_inch
             = cloud units × (inches per cloud unit) × (metres per inch)
```

It should come out near 2.11 m. Outside `RIG_PLAUSIBLE_M = (1.2, 3.5)` the
capture is flagged `rig_height_implausible` — marked, not rejected, because an
implausible rig height means the floor fit found something that is not the floor
(a stairwell, an open edge, the ground outside), which corrupts the wall band
and the scale as well as the height.

### 1.5 What is cached

Per capture, `captures/<reality>.npz`:

| Array | Meaning |
|---|---|
| `band_xy`, `band_relative_height`, `band_camera_index` | the wide wall band, and what each point is relative to |
| `camera_xy`, `camera_z` | camera positions in the gravity-up frame |
| `camera_paper_xy` | the same cameras in paper inches, through the ground-truth `Sim2` |
| `rotation_to_gravity_up`, `up_index`, `sign_ambiguous` | which gravity hypothesis won, and whether it was contested |
| `gt_sim2_params` | `[s, θ, tx, ty]`, the ground-truth similarity |
| `camera_height_above_floor`, `floor_relative_mad`, `metres_per_inch` | the floor fit and the scale |

Per capture, `captures/<reality>.json`: the `CaptureResult` — status, counts,
residuals, timings. Those rows become `manifest.csv`, which is the denominator
for every number quoted anywhere.

---

## 2. Ground truth, prepared once

Read in exactly one module, `utils/groundtruth.py`, which wraps
`align.groundtruth` and `align.eval.oracle`. `models/`, `losses/` and
`inference/` never import it, which is what lets the matcher be used inside the
solver later.

### 2.1 What the truth is, on disk

`gt_trajectory(rlt_dir)` reads one file, `images_full.json`, and takes only the
**positions** — rotations are deliberately not surfaced, because gravity, scale
and the planar fit all work from positions alone, while the euler convention in
the pose files fails silently if guessed wrong.

Those positions are already in the **target frame**: UTM metres where the
structure has a `design_tm`, paper inches where it does not.
`target_to_paper(positions, design_tm)` brings both cases to paper inches, which
is the frame the fit happens in.

Six splits in the wider dataset carry `"reg_tool": true` in their `tm.json` —
registered by hand, so they would score `tpr = 1.0` against their own truth
tautologically. `is_manually_registered` exists so they stay identifiable.

### 2.2 Pairing

`matched_camera_positions` intersects the solver's trajectory (from
`shots.geojson`, which is production's own output and not derived from the
answer) with the ground-truth trajectory **by image name**, and returns both
arrays in the same order. Images present in only one are dropped. Fewer than
three shared images and the capture has no ground truth to fit.

### 2.3 The fit

`ground_truth_planar_fit` hands the pairs to `align.eval.oracle.best_planar_fit`.
For each gravity hypothesis `R`:

1. rotate the SfM camera positions into that frame, `C = traj @ Rᵀ`;
2. fit a **4-DOF planar similarity in closed form** — Umeyama on `C[:, :2]`
   against the truth in paper inches. Least squares over every shared camera; no
   RANSAC, no iteration, no robustness;
3. score it by the **mean Euclidean per-camera residual, in paper inches**.

The hypothesis with the smaller residual wins, strictly, so `+up` takes an exact
tie. This is where the up-sign ambiguity is **resolved for training**: the cache
stores the winner's `up_index` and its rotation. Inference gets no such help and
tries both.

Umeyama's determinant correction forces `det R = +1`. The fit is therefore a
proper rotation, and a capture whose truth is a mirror of its cloud cannot be
fitted well — it surfaces as a large residual rather than as a silent success.

Why four degrees of freedom rather than the ground-truth 4x4: the target is what
the method can actually emit, a `Sim2` applied after a gravity tilt. Scoring
against the full transform would charge the method for a tilt it never had the
freedom to produce.

### 2.4 The rejection that keeps the dataset honest

```
residual_m = residual_inches × metres_per_inch
residual_m > max_gt_fit_residual_m (0.5)   ->   status "bad_fit", capture dropped
```

The whole method assumes a capture can be placed by a planar similarity. Where
the recorded truth is not one to within half a metre — a drifting reconstruction,
a capture spanning two levels, a bad registration — the sample would teach the
network a correspondence field that no `Sim2` can reproduce. Measured: **36 of
470 rows**, the largest single rejection reason.

### 2.5 What the network actually sees

The fitted `Sim2` is stored as `gt_sim2_params = [s, θ, tx, ty]`, and the camera
positions are stored twice: in gravity-up cloud xy, and in paper inches through
that same `Sim2`. After preprocessing, **training never opens a ground-truth
file again** — the per-sample targets come out of the affine chain, and the
chain's only ground-truth-derived link is that one similarity.

Ground truth is used exactly three times in the whole method:

| Where | What for |
|---|---|
| preprocessing | the `Sim2` that makes correspondences, and the up-sign |
| preprocessing | which viewport of the sheet holds the cameras |
| evaluation | scoring the predicted pose in metres |

Nothing in the model, the losses or the inference path can reach it.

---

## 3. Sheet preparation, per structure

`data/preprocess.py:prepare_structure_sheet`, once per structure, shared by all
its captures.

### 3.1 Which part of the sheet

A 9600x7200 sheet resized to 1024 px thins its walls to nothing, so the network
never sees the whole page. `align.sheet.prepare` extracts wall features and
viewport rectangles; `data/floorplan.py:viewport_boxes` turns them into
candidate regions:

- each detected viewport, expanded by 5% of its long side;
- viewports whose short side is under 15% of the sheet's long side are dropped
  as details and legends, not plans;
- the border box (or the whole sheet when no border was found) is always
  appended as the catch-all, expanded by 1%, so the list is never empty;
- boxes overlapping an earlier one by IoU ≥ 0.95 are dropped.

Viewports come first and the catch-all last, which matters in the next step.

### 3.2 Crop, resize, pad

`crop_and_resize` cuts a box, resizes so its long side is `sheet_cache_long_side`
(1280 px in the cache, the working resolution at train time), and pads to a
square that is a multiple of the patch size, filling with the sheet's own white.
The content sits top-left; `valid_mask` marks it.

Two details that keep the chain exact:

- The resize is anisotropic by up to one pixel, because output sizes are
  integers. `sheet_to_crop` records the **actual** per-axis factors rather than
  assuming one.
- Downsampling uses Lanczos and upsampling bicubic, chosen by the factor.

Padding is not content, and the network is told so: `patch_valid_mask` reduces
pixel validity to patch validity, a patch counting as content when at least half
its pixels are, and the matcher masks invalid patches to `−inf` before the
softmax so no probability mass can land on white padding.

### 3.3 Which viewport holds the capture

`assign_capture_viewport` pushes the ground-truth camera positions into sheet
pixels and takes the first box containing at least 80% of them. Because
viewports come before the catch-all, a plan viewport wins over the border box
that also contains everything. If fewer than half the cameras land inside even
the catch-all, the capture is flagged `gt_outside_sheet`: either its ground
truth or the sheet scale is wrong.

---

## 4. From a cached capture to one training sample

`data/dataset.py:DensityFloorplanPairDataset.__getitem__` builds one (density
map, sheet crop, correspondences) triple at the resolution the sampler asks for.
Augmentation is seeded by `(seed, epoch, index)`, so every sample is
reproducible and every epoch differs.

### 4.1 The density map

`data/density.py:render_density_map`. Band points are counted into a square
grid, gamma-compressed and normalised — the paper's floor-plan proxy.

```
extent  = square_extent(points)            centred, outliers trimmed
res     = span / size                      cloud units per pixel
col     = floor((x − lo_x) / res)
row     = size − 1 − floor((y − lo_y) / res)      y-down image
image   = counts ** gamma, divided by its max
density_px_to_world_xy = [[res, 0, lo_x], [0, −res, hi_y]]
```

`square_extent` trims 2.5% at each end of each axis (the paper's `ρ_xz`), pads
5%, and **centres the square on the trimmed box**, so a long thin walk sits in
the middle of the map with blank space either side rather than pressed against
an edge. A point exactly on the upper edge is pulled into the last cell rather
than dropped.

The map is rendered y-down deliberately. `align.segments.rasterize` is y-up; a
y-up map against a y-down sheet differs by a mirror, which a proper `Sim2`
cannot express.

**Augmentation, all of it before rendering** (`data/transforms.py`), so the map
is re-counted rather than resampled:

| Knob | Default | What it does |
|---|---|---|
| band jitter | low ∈ (−0.95, −0.75), high ∈ (0.15, 0.35) | narrows the wide band at random, varying what counts as wall |
| camera window | p = 0.5, ≥ 40% of cameras | keeps only points seen from a contiguous run of cameras: a partial capture |
| point keep | 60–100% | otherwise, a random subset of points |
| extent crop | side 70–100% of the span | a random sub-square of the extent: the capture seen partly |
| gamma | {0.25, 0.5, 0.75, 1.0} | how hard the counts are compressed |
| min count | {1, 1, 2} | cells below this are zeroed, thinning speckle |
| quarter turn | uniform over 4 | the one pixel op, lossless, affine composed exactly |
| continuous yaw | ±`density_rotation_degrees` / ±`target_rotation_degrees`, both 0 by default | a POINT op, before the count, with the extent re-squared around the rotated cloud, so nothing is interpolated and nothing is padded. Composes with the quarter turn: at 45° the relative yaw the matcher sees is uniform on the circle, where the quarter turns alone give four cardinals and nothing between. The two sides draw independently — a shared angle is an exact no-op, since only the difference is visible |
| prior point keep | `target_point_keep_fraction`, off by default | the same dropout the query side has always had, on the prior |
| brightness / contrast | ±0.2 | after normalisation, on the tensor |

No flips. A mirror is the failure mode the up-sign ambiguity produces, and the
network must learn to reject it rather than shrug at it. This is also why the
continuous yaw is a proper rotation about a pivot: its determinant is +1 by
construction, so no setting of it can turn the pair into a mirror.

### 4.2 The sheet crop

The cached 1280-px viewport crop is re-cut, not re-read from the sheet, so a
data worker stays cheap (`floorplan.crop_within`, which chains the affines).

At evaluation the crop is the whole viewport content resized to the working
resolution. In training it is a **zoom crop** (`transforms.random_zoom_box`):

- the density map's four extent corners and the true camera pixels are pushed
  onto the parent crop; their bounding box is the capture's footprint;
- a square side is drawn **log-uniformly** between `1.3 ×` the footprint and the
  whole content;
- the box is placed at random subject to containing the footprint;
- the side is capped so the footprint never spans fewer than
  `min_footprint_patches` (4) patches of the crop:
  `max_zoom_out = resolution / (4 × patch)`.

The cap is the fix for this dataset's mismatch with the paper's: the paper's
scenes cover most of their floor plan, while these captures are short walks
through large buildings. Without a cap, the target is a couple of patches wide
and carries no localisation signal.

Then a random quarter turn, and colour jitter on the tensor: brightness,
contrast and saturation ±0.5, hue ±0.1 turns, written by hand in YIQ because
torchvision is absent.

### 4.3 The correspondences

`data/correspondences.py`.

**What is sampled, and what is not.** The correspondence itself is neither
sampled nor searched. Once a capture has its ground-truth similarity, the map
from a density pixel to its true crop pixel is one 2x3 matrix,
`density_to_crop`, composed from the four affines of §0. Any pixel you pick
already has an exact answer. The only question this code decides is **which**
pixels to ask about.

**The draw.** `density.sample_query_pixels` flattens the density image,
normalises it into a probability vector, and draws from it with replacement, so
a cell's chance of being picked is exactly its share of the total mass. It then
adds a uniform offset in `[0, 1)` to the integer row and column, turning a cell
index into a continuous position somewhere inside that cell.

Two things follow from that, and both matter:

- Queries concentrate where the walk actually saw structure. An empty cell has
  probability zero and can never be drawn, so no supervision is spent asking
  what a blank patch of the density map corresponds to.
- Because positions are continuous, the query feature is a **bilinear blend** of
  neighbouring patch features rather than a patch centre. The soft-argmax is
  therefore trained to resolve *inside* a patch. Without the jitter every query
  would sit at a cell centre, and the head would only ever see quantised
  positions.

**The filter, and the constants in it.**

| Constant | Value | Why |
|---|---|---|
| `oversample` | 2 | draw 2048 to end up with 1024, because some will miss |
| `min_inside` | 64 | below this the crop genuinely missed the cloud: rebuild |
| `queries_per_sample` | 1024 | the fixed output size |

A candidate survives two tests. Its target must land inside the crop's
half-open bounds, and it must land on **content** rather than on the white
padding, checked against the crop's own validity mask at the floored pixel.
Padding fails because the matcher masks padding patches to `−inf` before the
softmax: a target there would be unreachable by construction, and the feature
loss would be asking for something impossible.

Survivors are then forced to exactly 1024 — truncated if there are more, topped
up by resampling with replacement if there are fewer. The fixed size is not
cosmetic. `collate_pairs` stacks these into a batch tensor, and the feature loss
builds a `Q x Q` similarity matrix per item, so a variable `Q` would break
batching and quietly change the contrastive difficulty from sample to sample.

**Three consequences worth holding onto.**

- **Duplicates are expected, twice over.** The multinomial draw repeats cells,
  and the top-up repeats whole pairs. In the feature loss those duplicates are
  positives sitting among the negatives, which is exactly what the
  near-duplicate mask of §6.1 handles: their separation is zero, well inside the
  one-patch radius.
- **The labels carry the fit's error and nothing else.** There is no matching
  heuristic and no nearest-neighbour step anywhere in the target, so the only
  label noise is the ground-truth similarity's own residual. That is why §2.4
  rejects a capture whose residual exceeds half a metre: it is bounding label
  noise at the source.
- **The cost is in the pixels, not the queries.** Building the probability
  vector is linear in the number of cells — about a million at 1008 px square —
  while drawing 2048 from it is cheap by comparison. Asking for more queries is
  nearly free; rendering at a higher resolution is not.

**Sampling proportional to density is also what inference does.**
`inference/matching.py` calls the same function before it filters by confidence,
so the network is trained on the query distribution it will actually be handed,
not on a grid it will never see.

### 4.4 When a sample cannot be built

A sample fails for one of three reasons: too few band points survive the
subsetting, the rendered map is empty, or fewer than 64 targets land on content.
`DensityFloorplanPairDataset.__getitem__` then works through a fixed ladder:

| Attempt | What changes |
|---|---|
| 0, 1, 2 | the same record, fresh augmentation — the generator is consumed as it goes, so each attempt draws a different band, subset, extent and crop |
| 3 | the same record with **no augmentation generator**: the evaluation path — default band, every point, no extent crop, no quarter turn, no continuous yaw, and the whole viewport content as the crop. A sample that only builds here is a sample the augmentation broke, which is what `train/input/unaugmented_fallback_fraction` counts |
| then | the next record, up to five, printing once so the skip is visible |

The fourth attempt almost always succeeds, because the whole viewport contains
the capture's footprint by construction. Moving to a neighbour is the belt to
that brace: a record whose truth misses every crop is skipped rather than
allowed to kill a run several hours in. The attempt and hop that produced a
sample are recorded in its metadata, so a dumped training pair says which path
it came from.

### 4.5 Batching

`data/sampler.py:ResolutionBatchSampler` draws **one resolution per batch**, as
the paper does, from `resolutions` with optional weights, and yields
`(index, resolution)` keys. The schedule for an epoch is generated up front from
`(seed, epoch)` so `__len__` is exact and the run is resumable. An epoch is
virtual: `samples_per_epoch` (100k) samples over however many captures exist.

The train/validation split is **by structure, never by capture**
(`data/splits.py`): captures of one building share its floor plan, so a split by
capture would validate on sheets the network has memorised. Validation
structures are drawn from those with at least 3 usable captures.

---

## 5. Architecture

`models/`. Two images in, dense correspondences out, with almost no parameters
of its own.

### 5.1 The encoder

`models/backbone.py:DinoPatchEncoder` wraps a Hugging Face DINO model:

- **DINOv3 ViT-B/16** as the paper uses, `facebook/dinov3-vitb16-pretrain-lvd1689m`;
- **DINOv2 ViT-B/14** (`facebook/dinov2-base`) as the fallback that needs no
  gated licence, patch 14, hence resolutions 504/756/1008.

The forward takes `(B,3,H,W)` ImageNet-normalised pixels, drops the prefix
tokens (class token, plus DINOv3's register tokens), reshapes the patch tokens
to `(B, C, H/p, W/p)` and **L2-normalises along C**, in fp32. Cosine similarity
is then a dot product.

Two deliberate choices:

- **The pretrained module is held in eval mode permanently.** DINOv3 jitters its
  rotary position encoding in training mode; since the density map and the sheet
  share one encoder, they must see identical, deterministic positions. Only the
  LoRA adapters follow the training flag, for their dropout.
- **A resolution that is not a multiple of the patch size raises**, rather than
  being silently resampled.

The density map, a single-channel image, is repeated to three channels and
normalised with the same ImageNet statistics as the sheet. Both sides go through
the same weights: the model has to make a point-density picture and an
architectural drawing land in one feature space.

### 5.2 LoRA

`models/lora.py`. `peft` is not installed and would drag in a transformers
version constraint, so the fifty lines it would have provided are written out:

```
y = W x + (alpha / rank) · B A dropout(x),     A ~ Kaiming uniform,  B = 0
```

`B = 0` at initialisation, so the adapted network starts **exactly** as the
pretrained one. Every `nn.Linear` whose dotted name ends with one of the
family's target suffixes is wrapped: for DINOv3 the four attention projections
and the two MLP projections; for DINOv2 the equivalents under its older naming.
Base weights have `requires_grad_(False)`.

With rank 16 and alpha 16 on DINOv2 ViT-B: **72 adapted layers, 2.65 M trainable
parameters against 89 M frozen**. Checkpoints store only the adapters.

### 5.3 The matcher

`models/matcher.py:DensityFloorplanMatcher`. Given the density map `D`, the
sheet crop `F`, and `Q` query pixels on `D`:

1. **One forward for both** when they share a size, which is the training case:
   they are concatenated on the batch axis, so the encoder runs once.
2. Query features are sampled from the density feature map **bilinearly** at the
   continuous query positions (`grid_sample`, `align_corners=False`, border
   padding) and re-normalised, because a blend of unit vectors is shorter than
   one.
3. Logits are cosine similarities over temperature:
   `logits = (f_q · f_patch) / τ`, `τ = 0.07`.
4. Invalid patches (padding) are masked to `−inf`.
5. `p = softmax(logits)` over patches, and the prediction is the **expectation
   over patch centres**:

```
pred_px = Σ_k p_k · centre_k          (soft-argmax, differentiable)
conf    = max_k p_k
```

Patch centres are at `(col + 0.5) · p, (row + 0.5) · p` in crop pixels. The head
has **no learned parameters**: everything trainable is in the adapters. The
softmax and the expectation run in fp32 with autocast disabled, because an
`−inf` mask and a mean of coordinates are both easy to ruin in fp16.

---

## 6. The losses

`losses/`. Four terms, all computed in **patch units** — coordinates divided by
the patch size — so `δ = 1` in the Huber means "one patch", the resolution the
soft-argmax is expected to resolve, and the loss means the same thing at 512 px
and at 1280 px.

Notation: `q_i` the query on the density map, `t_i` its true crop pixel, `p_i`
the prediction, `c_i` the confidence, all for item `i` of the batch.

### 6.1 Feature — symmetric InfoNCE (paper eq. 2)

Positives are the ground-truth pairs: the query feature `f^D_i` against the
floor-plan feature sampled at the true location `f^F_i`.

```
S = (f^D · f^Fᵀ) / τ                        (B, Q, Q)
L_feat = ½ [ CE(S, diag) + CE(Sᵀ, diag) ]
```

**With one correction that matters here.** A capture spans a median of ~108
floor-plan patches while 1024 queries are drawn, so many pairs share a target
patch. Treating those as negatives asks the features to separate points that are
the same to the network, which pins the loss at the log of the duplication —
measured at about 2.8 nats — and pulls features in contradictory directions. In
the first smoke run the InfoNCE *rose* above log 1024. So pairs whose targets lie
within `near_duplicate_radius` (1 patch) of the anchor's are masked out of the
denominator, in **both** directions: by target position for the density-to-plan
direction, by query position for the reverse.

### 6.2 Regression — confidence-weighted Huber (paper eq. 3)

```
L_regr = Σ_i c_i · huber(‖p_i − t_i‖, δ) / Σ_i c_i
```

Quadratic within a patch, linear beyond, so a single wild prediction cannot
dominate. Weighting by the model's own confidence lets it decline to answer
where the evidence is thin.

**`detach_confidence` is on by default, and the sum is why.** The weights are
normalised, so ∂L/∂c_i = (e_i − L)/Σc: the loss falls when confidence falls on
any query whose error is above average, which early in training is most of
them. Confidence is not a head — it is the peak of the query's softmax over the
plan patches (§5.3) — so the only way to lower it is to flatten that softmax,
and a flat softmax puts the prediction on the mean of the patch centres. Every
query then lands on the same point. The first full run collapsed exactly so:
matches piled on one spot, confidence at 1/N, and the InfoNCE term rising
through ln Q while the three weighted terms fell. Detached, confidence still
downweights an uncertain query; it just no longer pays to become uncertain.

### 6.3 Topology — angles at sampled triangles (paper eq. 4)

Sample 2048 index triplets `(i, j, k)` per item, distinct. For the query
positions and for the predictions, take the cosine **and the signed sine** of
the angle at vertex `i`:

```
cos = (a·b)/(‖a‖‖b‖),   sin = (a×b)/(‖a‖‖b‖),   a = j−i, b = k−i
L_topo = Σ w_ijk [ huber(Δcos) + huber(Δsin) ] / Σ w_ijk,   w_ijk = c_i c_j c_k
```

A similarity preserves angles, so a correct prediction field matches the query
field triangle by triangle. The **signed** sine is what makes this term
chirality-aware: a mirrored prediction keeps every cosine and flips every sine,
so it is punished rather than tolerated. `tests/bevaligner/test_losses.py`
pins exactly that, and pins that the geometry term below does *not* catch it.

### 6.4 Geometry — one common scale (paper eq. 5)

Sample 2048 index pairs. Under a similarity, the log-ratio of predicted to query
distance is the same constant for every pair:

```
r_ij  = log( ‖p_i − p_j‖ / ‖q_i − q_j‖ )
r̄     = confidence-weighted mean of r_ij          (stop-gradient)
L_geo = Σ w_ij huber(r_ij − r̄) / Σ w_ij,          w_ij = c_i c_j
```

The mean is detached, so the term penalises *disagreement about scale* rather
than pushing the scale itself anywhere. It catches an anisotropic stretch, which
the angle term alone would not.

### 6.5 Weights and the curriculum

```
total = 1.0·L_feat + 50·L_regr + 10·L_topo + 10·L_geo
```

switched on by training progress (`losses/curriculum.py:active_loss_weights`):

| Progress | Active | Why |
|---|---|---|
| 0–10% | feature | the features first learn what matches what |
| 10–20% | + regression | then the soft-argmax is asked to land on the truth |
| 20–100% | + topology, geometry | finally the whole field of predictions is made rigid |

The gates are configurable because they assume a long run. In a 1000-step smoke
the weight-50 regression arrives after 100 steps, before the features mean
anything, so the smoke preset moves the gates to 30% / 50%.

---

## 7. Training: the loop, its dynamics, and every knob

### 7.1 What one optimisation step is

```
sampler        draws ONE resolution R for the batch, from `resolutions`
dataset        builds batch_size_for(R) samples at R, each: density map,
               sheet crop, 1024 correspondences
forward        density and crop concatenated on the batch axis -> one encoder pass
head           bilinear query features, cosine / τ, mask padding, softmax,
               expectation over patch centres
loss           the terms whose gate has opened, weighted, summed
backward       gradients reach the LoRA adapters only
clip           global norm 1.0
step           AdamW, then the LR schedule advances
```

Two properties of that loop matter more than they look:

- **One resolution per batch**, as the paper does, rather than one per sample.
  Mixing resolutions inside a batch would force padding or per-sample forwards;
  drawing per batch keeps the encoder pass dense. Because every loss is computed
  in patch units, the terms mean the same thing whichever resolution came up, so
  the curriculum and the weights need no per-resolution correction. Memory does
  not scale that way, which is what `batch_size_per_resolution` is for: the
  24 GB preset runs batch 4 at 512–1024 and batch 2 at 1280.
- **The density map and the sheet crop share the forward pass** when they share
  a size, which is always true in training. One encoder call per batch, not two.

### 7.2 Epochs are virtual

`samples_per_epoch` (100k by default) is the dataset's length, not the number of
captures. An "epoch" is a fixed number of samples drawn from however many
captures survived preprocessing, and `max_epochs: 1` with `max_steps` is the
usual way to run. Each index maps to `records[index % len(records)]`, so with
420 captures and 100k samples every capture is drawn about 240 times per epoch —
each time with a different band, subset, extent, gamma, quarter turn, zoom crop
and colour jitter, because the augmentation RNG is seeded by
`(seed, epoch, index)`.

That seeding is the reproducibility contract: **the same seed and epoch give the
same sample**, and a different epoch gives a different one. The batch schedule
itself is generated up front from `(seed, epoch)`, so the number of steps in an
epoch is known before it starts and a resumed run sees the same batches.

### 7.3 How the run is expected to behave

The curriculum makes the training curve discontinuous **by construction**, and
knowing where the steps are is the difference between reading the curve and
being alarmed by it.

| Phase | What is being optimised | What to watch |
|---|---|---|
| before the first gate | the feature loss alone | InfoNCE falling from roughly `log Q`; confidence still near `1/patches`; PCK at chance. Predictions are meaningless here — the soft-argmax is averaging a nearly uniform distribution, so it lands near the centre of the crop |
| the regression gate | a weight-50 term arrives | **the total jumps**, by about 50 × the current mean Huber error in patches. That is arithmetic, not divergence. It should decay over the following few hundred steps as the predictions move onto the truth |
| the topology and geometry gates | two weight-10 terms arrive | a smaller step, then a slow tightening. These terms constrain the *shape* of the prediction field and can only help once individual predictions are roughly right |
| after the last gate | everything | the pose metric is the one that matters; the loss can keep falling while it stalls |

**Confidence is a second, softer curriculum.** Three of the four terms are
weighted by the model's own peak probability, which starts near `1/patches`
(about 7.7e-4 at 504 px with a patch of 14) and grows as the features sharpen.
Early on, that weighting quietly suppresses the coordinate terms even after
their gate opens; the hard gates and this soft one work in the same direction.

Two failure signatures worth naming, because both look like ordinary
underperformance:

- **The InfoNCE sits near `log Q` and will not fall, or rises.** That is the
  duplicate-target problem of §6.1 — many queries share a target patch, and as
  plain negatives they make the objective unsatisfiable. The near-duplicate mask
  exists for this; if the loss still behaves this way, the radius is too small
  for the zoom level the crops are landing at.
- **The loss falls and the pose metric does not move.** Correspondences can be
  locally right and globally inconsistent, which RANSAC cannot turn into a pose.
  This is exactly why checkpoints are selected on the pose metric.

### 7.4 The optimiser

`optim/build.py`. **AdamW over the adapters only** — the parameter list is
`iter_lora_parameters(model)`, and `build_optimizer` raises if nothing in it
requires a gradient, which is the guard against silently training zero
parameters while the loss still falls to nothing.

| | Default | Notes |
|---|---|---|
| optimiser | AdamW, β = (0.9, 0.999) | on 2.65 M parameters, so its moments cost little |
| learning rate | `1e-4` | for adapters starting at zero, not for a fine-tune of the whole encoder |
| weight decay | `0.01` | applied to the adapters; the frozen base cannot decay |
| warm-up | 500 steps, linear | `B = 0` at init means the first steps decide the direction of every adapter |
| decay | cosine to `0.1 ×` the base rate | over the run's own length |
| gradient clipping | global norm `1.0` | a single bad batch (a crop that missed, a degenerate triangle sample) cannot take the step |
| accumulation | 1 | effective batch is `batch_size × accumulate_grad_batches` |

The schedule is a `LambdaLR` over Lightning's own `estimated_stepping_batches`,
which already accounts for accumulation and for `max_steps`, so it spans the
actual run rather than a guessed one. **The curriculum reads the same quantity**
as `progress = global_step / estimated_stepping_batches`, so gates and the decay
stay in step with each other.

### 7.5 Precision, and the parts held in fp32

`bf16-mixed` where the card supports it, `16-mixed` on the T4. Three islands are
kept in fp32 on purpose:

- the encoder's output features are cast and L2-normalised in fp32, so cosine
  similarity is not quantised;
- the logits, the `−inf` padding mask, the softmax and the expectation over
  patch centres run with autocast disabled — an `−inf` mask and a mean of
  coordinates are both easy to ruin in fp16;
- the InfoNCE similarity matrix is computed in fp32 for the same reason.

`gradient_checkpointing` trades throughput for the activation memory of the
frozen trunk, which is what lets 1280 px fit on a 24 GB card. It is on in the
presets that need it and off in the 48 GB one.

### 7.6 Validation, and what a checkpoint is selected on

| Knob | Default | What it does |
|---|---|---|
| `val_check_interval` | 2000 steps | how often validation runs |
| `val_batches` | 20 | correspondence-loss batches per validation |
| `val_pose_eval_captures` | 20 | captures put through the full propose-and-RANSAC path |
| `num_match_images` | 4 | validation samples drawn as pictures |
| `save_top_k` | 2 | plus `last.ckpt`, always |

Validation is deliberately two measurements at two costs. The cheap one is the
correspondence loss and PCK over a few batches. The expensive one runs inference
proper on whole captures — every gravity hypothesis, every viewport, RANSAC, the
refinement pass — and reports metres. **Checkpoints are selected on
`val/tpr@1.0_mean`** from that second measurement. With `--no-pose-eval` the
monitor falls back to `val/pck@2p`, which is the right thing for a wiring test
and the wrong thing for a real run.

The validation loader uses half the training batch size at the evaluation
resolution, and the validation dataset is built with `seed + 1` and a fixed
resolution, so the validation samples are stable across the run and comparable
between checkpoints.

### 7.7 The hyperparameters, in one place

| Knob | Default | t4_smoke | gpu24 | gpu48 | What moving it does |
|---|---|---|---|---|---|
| `model.backbone_family` | dinov3 | dinov2 | dinov3 | dinov3 | patch 16 vs 14, which sets the legal resolutions |
| `model.lora_rank` / `lora_alpha` | 16 / 16 | — | — | — | capacity of the adaptation; scaling is `alpha/rank`, so moving both together keeps the update size |
| `model.lora_dropout` | 0.0 | — | — | 0.1 | dropout on the input of each `lora_A`. It was 0.1 everywhere and **inert**: the encoder puts itself in eval in its own `__init__`, nothing called `.train()` at fit start, and Lightning restored the captured per-module flags after every validation, so the adapters were never in train mode and nothing was ever dropped. `lightning/module.py` now overrides `on_validation_model_train`, which makes the knob live — and the default drops to 0.0, because the measured train/val gap is zero. `paper_faithful` keeps 0.1 deliberately |
| `model.temperature` | 0.07 | — | — | — | how peaked the patch distribution is; lower sharpens the soft-argmax and the InfoNCE alike |
| `model.gradient_checkpointing` | false | true | true | false | memory against throughput |
| `data.resolutions` | 512/768/1024 | 504/756/1008 | + 1280 | + 1280 | must be multiples of the patch size |
| `data.resolution_weights` | uniform | 0.4/0.35/0.25 | 0.35/0.30/0.20/0.15 | uniform | how often the expensive resolutions come up |
| `data.batch_size` | 4 | 2 | 4 | 4 | with accumulation, the effective batch |
| `data.batch_size_per_resolution` | — | — | {1280: 2} | — | per-resolution override, for the one that does not fit |
| `data.samples_per_epoch` | 100k | 4k | 100k | 280k | the virtual epoch |
| `data.queries_per_sample` | 1024 | — | — | — | correspondences per sample; the InfoNCE is `Q × Q` per item |
| `data.eval_resolution` | 1024 | 1008 | 1024 | 1024 | validation and the pose path |
| `loss.weights` | 1 / 50 / 10 / 10 | — | — | — | feature, regression, topology, geometry |
| `loss.gates` | 0 / .10 / .20 / .20 | 0 / .3 / .5 / .5 | default | default | as a fraction of the run, so a short run needs later gates |
| `loss.huber_delta` | 1.0 | — | — | — | in patches: the error at which the penalty stops being quadratic |
| `loss.near_duplicate_radius` | 1.0 | — | — | — | patches; negatives closer than this to the anchor's target are ignored |
| `loss.num_triplets` / `num_pairs` | 2048 | — | — | — | Monte-Carlo samples for the topology and geometry terms |
| `optimizer.learning_rate` | 1e-4 | — | — | — | |
| `optimizer.warmup_steps` | 500 | 100 | 500 | 500 | shorter runs need shorter warm-up or they never leave it |
| `optimizer.accumulate_grad_batches` | 1 | 2 | 1 | 1 | effective batch without the memory |
| `trainer.precision` | bf16-mixed | 16-mixed | bf16-mixed | bf16-mixed | the T4 has no bf16 |
| `trainer.val_check_interval` | 2000 | 500 | 2000 | 4000 | |
| `data.num_workers` | 12 | 8 | 12 | 16 | rendering a density map is CPU work |

Anything not in a preset's YAML takes the default, and any of them can be moved
on the command line: `optimizer.learning_rate=5e-5`, `data.resolutions=[512]`,
`loss.gates={feature: 0, regression: 0.4, topology: 0.6, geometry: 0.6}`.

---

## 8. Inference: from a matcher to a pose

`inference/`. Ground-truth free, so the same code can be used inside the solver.

### 8.1 Stage one — every hypothesis against every viewport

For each gravity hypothesis and each viewport crop:

1. render the density map at the working resolution;
2. `match_density_to_crop` draws 1024 queries proportional to density, runs the
   matcher, and keeps the top half by confidence;
3. correspondences become `(gravity-up cloud xy → paper inches)` pairs;
4. `fit_planar_pose` runs RANSAC over them for a `Sim2`.

RANSAC (`align.visual.ransac_sim2`, 2000 iterations here) draws **three
correspondences** at a time, fits the same closed-form Umeyama similarity the
ground truth was fitted with, and counts correspondences within the tolerance.
The best consensus is refitted on its inliers, twice. The model is a proper
rotation by construction, so a mirrored correspondence set — the wrong gravity
sign — never grows a consensus and the fit simply fails. **That is the mechanism
that resolves the up-sign at inference**, where no oracle is available. A
`Sim2` with a non-finite or non-positive scale, or fewer than `min_inliers`
(12) supporters, is rejected outright.

The **inlier tolerance** is the interesting part. It is quoted in metres
(`tolerance_m`, 0.5) and converted with the sheet scale, but on a whole-viewport
crop one patch can be several metres, and no matcher can beat its own
resolution. So the tolerance is

```
tolerance_inches = max( tolerance_m / metres_per_inch,
                        tolerance_patches × patch × inches_per_crop_pixel )
```

with `tolerance_patches = 0.5`. The winner is the pose with the most inliers,
ties broken by inlier fraction. Every candidate is recorded, so a failed
placement can be read afterwards.

### 8.2 The mutual-nearest-neighbour check

A pair is usually kept only if the patch it chose also prefers that query. Done
exactly, with a thousand queries over a few dozen patches, this keeps one query
per patch and threw away all but 6 of 512 matches when measured. So the check is
**spatial**: the patch's favourite query must lie within `mutual_radius_patches`
(1.5) of this query on the density map. And it is applied only on the zoomed
pass, not on the whole-viewport pass where the capture is a few patches wide.

### 8.3 Stage two — refine on the footprint

The winning pose says where the capture lies on the sheet. The sheet is
re-cropped around that footprint with a 1.6× margin
(`transforms.footprint_zoom_crop`), which is the regime the zoom crops trained
for, and the match is re-fitted with the mutual check on.

The refined pose is accepted only if it **agrees** with the coarse one — the
median camera shift is within `refine_agreement_patches` (2) coarse patches —
and is supported by at least `max(min_inliers, 0.7 × coarse inliers)`. Otherwise
the coarse pose stands. Refinement improves precision; it must not be allowed to
invent a different answer.

### 8.4 Pose to `tm`

`utils/geometry.py:pose_transform_matrix` lifts the `Sim2` to a `Sim3` using the
gravity rotation, sets `tz` so the capture's floor lands on paper `z = 0`
(cameras sit `camera_height_above_floor` above it), and composes with the
structure's `design_tm` when it has one. Metrics are horizontal; the height only
keeps a georeferenced prediction from sitting a hundred metres under the
building.

---

## 9. Evaluation

Two different measurements, deliberately kept apart.

**Correspondence quality**, logged every validation pass: `PCK@{1,2,4}` patches,
the median error in patches, and the mean confidence. Cheap, dense, and a poor
proxy for the thing that matters.

**Pose quality**, `lightning/callbacks.py:PoseEvaluation`: run the full
propose-and-RANSAC path on validation captures, place every camera, and measure
in **metres**. `tpr@X` is the fraction of a capture's cameras landing within X
metres; `solved_frac`, `rmse_xy_median`, `viewport_correct_frac` and the median
inlier count come with it. **Checkpoints are selected on `val/tpr@1.0_mean`**,
not on the loss, because the loss can fall while the pose stays wrong.

`evaluate.py` is the offline version and writes through
`align.eval.reality.ResultWriter`, so its output is the same three files the
rest of the repo's methods write.

**The pose in the shape a viewer reads.** `--images` adds one
`<reality>_<capture>_images.json` per reality, flat under `<stem>_images/` so a
folder of them sorts by id: a record per photograph carrying a
position in the target frame and a heading in three.js `Euler` XYZ degrees. That
convention is not guessable and is not re-derived here -- `align.io.floormap` is
production's own converter, pinned against delivered files by
`tests/test_floormap.py`, and the only thing this method contributes is the
transform it is given. `--no-score` turns off every ground-truth read, so the
same command serves a capture that has no truth to be scored against.

### The pictures

Numbers do not show a mirror. `val/matches/<n>` in TensorBoard draws the density
map beside the crop with a line from each query to the patch the matcher chose,
green within 2 patches of the truth and red beyond, with an amber ring on the
truth. `train/pairs/<n>` draws the same panels with the **ground-truth** pairs,
which is what the network is being asked to learn. A mirror, a shift, or a
collapsed match field is obvious in one glance and invisible in a loss curve.

Offline, `evaluate.py --qualitative N` writes two per capture flat under
`<stem>_viz/`, prefixed `<reality>_<capture>_`, drawn by `viz.py` from arrays
the caller already has,
so nothing in the drawing path reads ground truth on its own:

| File | What it shows | What it diagnoses |
|---|---|---|
| `matches.png` | the density map beside the winning sheet crop, with the correspondences the pose was actually fitted from, green where RANSAC kept them and red where it did not, sampled in proportion to the inlier fraction | whether the matcher found the right patches. Every line converging on one spot is a collapsed match field; a wide red spray with a thin green band is a fit surviving on a minority of its matches |
| `placement.png` | the capture's wall-band points and its camera walk, pushed through the predicted pose onto the sheet itself, with the ground-truth walk in amber, cropped to what was drawn | whether the pose put the walk in the right room. The two walks lying on top of each other is a correct placement; two walks in different rooms is not |

The split is deliberate. A capture can match well and be placed wrongly, when
the matches are locally right and globally inconsistent, and it can be placed
correctly on matches that look poor, when a minority of them agree. The first
picture answers the first question and the second answers the second.

---

## 10. What has actually been measured

**Overfit sanity, 2026-09-25** (DINOv2 fallback, 1008 px, batch 2, two captures,
no augmentation, 200 steps in 8 min 16 s on a T4). Validation is the same two
captures; the pose figures come from the full propose-and-RANSAC path:

| step | `val/pck@2p` | median error | `val/tpr@1.0_mean` | `val/rmse_xy_median` | inliers |
|---|---|---|---|---|---|
| 50 | 0.09 | 7.8 patches | 0.00 | 23.6 m | 60 |
| 100 | 0.99 | 0.40 | 1.00 | 0.32 m | 382 |
| 150 | 1.00 | 0.22 | 1.00 | 0.28 m | 494 |
| 200 | 1.00 | 0.15 | 1.00 | 0.10 m | 509 |

This is a wiring test, not accuracy: it shows the affine chain, the curriculum,
the soft-argmax and RANSAC all agreeing on one answer. **No model has been
trained on the full set**, because the paper's DINOv3 weights are gated and the
local card is a 16 GB T4.

**Dataset**, 2026-09-24, over `100_prod_current.csv` + `100_prod_prior.csv`: 412
of 470 rows usable; 36 `bad_fit`, 11 `no_floor`, 10 `gt_outside_sheet`, 1
`thin_band`. Preprocessing costs a median 1.3 s of worker time per capture.

---

## 11. Where the method could go wrong

Ranked by how quietly it would fail.

| Failure | What it looks like | What catches it |
|---|---|---|
| a mirrored chain | trains happily, poses are nonsense | `determinant > 0` assertions; the signed sine in the topology loss; RANSAC's proper rotation |
| the wrong gravity sign | a plausible pose on the wrong side | both hypotheses tried, inlier count decides |
| duplicate InfoNCE targets | loss rises or plateaus near log Q | near-duplicate masking, `loss.near_duplicate_radius` |
| zooming out too far | the capture spans one patch, no signal | `min_footprint_patches` cap |
| tolerance finer than the matcher | RANSAC finds nothing on a big viewport | tolerance floored at half a patch |
| refinement inventing a pose | a confident, different, wrong answer | agreement and inlier-count gates |
| a shrinking denominator | accuracy that improves by dropping captures | `manifest.csv` records every skip and its reason |
