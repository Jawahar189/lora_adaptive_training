# Prior-capture alignment

Place a construction capture onto its building's coordinate system by matching
it against the **previous capture of the same building**, instead of against the
architect's floor plan.

A capture's point cloud arrives in an arbitrary frame with arbitrary scale. The
job is to recover `tm`, the 4x4 that carries it into the building's coordinates.
The prior capture is already registered, so its transform is a known input — and
matching two point clouds from the same camera a median of one day apart is a
far easier problem than matching a point cloud to a drawing.

`docs/METHOD.md` describes the original floor-plan method.
`docs/DEPLOY.md` is the two modules a service needs, and what to expect of them.
`docs/PRIOR_CAPTURE.md` explains this one — what the substitution changes, and
what each stage isolates. **This** file is how to run it: one command under
[The one command](#the-one-command), or the stages individually under
[The seven steps](#the-seven-steps).

The learned floor-plan matcher this branch is named for is a separate program
with its own runbook: `README.md`.

---

## Before anything

No CSV path is hardcoded anywhere in the code. Every stage is told which data to
use, and refuses to run rather than guess. Set it once:

```bash
cd /home/raghava.m/alignment
export ALIGN_CURRENT_CSV=datasets/dataset_all_local_current.csv
export ALIGN_PRIOR_CSV=datasets/dataset_all_local_prior.csv
```

or pass `--current-csv` / `--prior-csv` on any command — flags beat the
environment. Both accept several files, so indexes can be combined without
editing anything:

```bash
--current-csv a_current.csv b_current.csv --prior-csv a_prior.csv b_prior.csv
```

Every sample command below is the one actually used to produce the figures
quoted beside it, updated only where the CLI has changed since. The `timeout`
wrappers are real: several stages run for over an hour, and a wrapper that kills
a wedged run beats discovering it the next morning.

**Shared by every stage below:**

| Argument | What it does |
|---|---|
| `--current-csv CSV...` | the captures being placed. Defaults to `$ALIGN_CURRENT_CSV`; required if that is unset |
| `--prior-csv CSV...` | the already-registered captures to align against. Defaults to `$ALIGN_PRIOR_CSV` |
| `--limit N` | only the first N realities — use it for a smoke test before a long run |
| `--stride N` | keep every Nth point of a cloud. Lower is denser and slower |

---

## The one command

`scripts/run_alignment.py` runs every stage against one reality at a time, in
one process — reference, cloud, search, pose, score, visuals — and writes a
row-major table. It is what to run; the seven stages further down are how to
find out *why* a run looks the way it does.

Nothing shells out. `align/runner.py` holds the stages as functions and the
script is argparse over its `Params`, so the same run is reproducible from
Python:

```python
from align.runner import Params, run
for r in run(Params(current_csv=["cur.csv"], prior_csv=["prior.csv"])):
    print(r.case.reality_id, r.poses["predicted"]["rmse_xy"])
```

```bash
.venv/bin/python scripts/run_alignment.py \
  --current-csv datasets/dataset_all_local_current.csv \
  --prior-csv   datasets/dataset_all_local_prior.csv \
  --visualise --stem out/run/alignment
```

**~40 min** for 131 realities, so usually in the background:

```bash
nohup .venv/bin/python scripts/run_alignment.py --visualise \
  --stem out/run/alignment > out/run.log 2>&1 &
tail -f out/run.log
```

Smoke test first — 4 realities, about a minute:

```bash
.venv/bin/python scripts/run_alignment.py --limit 4 --visualise \
  --stem /tmp/smoke/alignment
```

### What it writes

```
<stem>.csv                  one row per reality — prior side, current side, outcome
<stem>.jsonl                the same plus warnings, drop reasons, per-stage timings
<stem>_transforms.jsonl     {"reality_id", "pose", "tm"} for every pose scored
<stem>_viz/<reality_id>/    R1_reference, R2_placement, R3_alternatives, run.png
<stem>_images/<reality_id>/ images.json and floormap.png, with --images
```

Row-major: one row per reality carrying both sides, so a result reads without
joining anything.

| group | columns |
|---|---|
| identity | `reality_id` `capture_id` `structure` `root` `status` `georeferenced` |
| prior side | `prior_capture_ids` `prior_splits` `prior_dropped` `prior_wall_cells` `prior_same_day` `prior_capture_date` `overlap_5m` |
| current side | `capture_date` `n_images` `rig_height_m` `rig_implausible` `metres_per_inch` `scale_source` `sign_ambiguous` `yaw_fallback` `n_hypotheses` `scale` `yaw_deg` `score` |
| was it right | `top1_correct` `pose_error_m` `yaw_error_deg` `scale_ratio` |
| accuracy | `rmse_xy` `rmse_z` `median_xy` `p90_xy` `max_xy` `tpr@0.1` `tpr@0.5` `tpr@1.0` `tpr@1.5` `tpr@2.0` |
| the ceiling | `oracle_rmse_xy` `oracle_rmse_z` `oracle_tpr@1.0` |
| cost | `seconds` `stages` |

`stages` marks a stage that stopped the run with `!`, so a bad row says where it
gave up: `reference cloud search! `.

### Arguments, by the stage that reads them

Selection:

| Argument | Default | What it does |
|---|---|---|
| `--reality ID...` | all | only these reality ids |
| `--limit N` | all | only the first N realities |

Stage: reference — the prior BEV. Changing any of these except stride and cell
size also needs `--reference-force`, or the cache returns one built with the old
value.

| Argument | Default | What it does |
|---|---|---|
| `--reference-stride` | `6` | keep every Nth point of each prior cloud |
| `--reference-cell-m` | `0.10` | raster cell, metres |
| `--reference-pad-m` | `2.0` | margin around the merged extent |
| `--coverage-dilate-m` | `3.0` | how wide the prior's walk becomes a corridor |
| `--level-gap-m` | `2.0` | floor-height gap that separates two storeys |
| `--consistency-tol-m` | `5.0` | drop a prior split disagreeing with a sibling by more |
| `--max-rel-mad` | `0.5` | reject a floor fit more dispersed than this |
| `--height-band LO HI` | `0.55 1.65` | fraction of the expected rig height still believable |
| `--max-reference-px` | `6000` | cap on the raster's largest dimension |
| `--min-wall-cells` | `2000` | below this the reference is flagged thin |
| `--no-cache` | off | build in memory, cache nothing |
| `--reference-force` | off | rebuild even where a cached reference exists |

Stage: current cloud.

| Argument | Default | What it does |
|---|---|---|
| `--cloud-stride` | `3` | keep every Nth point of the current cloud |
| `--min-source-frames` | `2` | drop points seen by fewer source frames |
| `--single-gravity` | off | carry one up-vector; the default carries both where the floor/ceiling test ties |

Stage: search.

| Argument | Default | What it does |
|---|---|---|
| `--search-cell-m` | `0.20` | correlation cell, metres |
| `--max-search-px` | `4000` | cap on the correlated raster |
| `--yaw-window-deg` | `10.0` | half-width swept around each Manhattan direction |
| `--yaw-step-deg` | `3.0` | spacing of heading candidates |
| `--yaw-min-strength` | `0.08` | below this the heading estimate is unusable and the search sweeps the full circle |
| `--scale-steps` | `9` | scale candidates across the rig-height bracket |
| `--w-wall` | `1.0` | weight of wall agreement; several values sweep |
| `--w-free` | `0.1` | weight of the coverage term; several values sweep |
| `--w-scale` | `0.25` | weight of scale plausibility; several values sweep |
| `--w-aspect` | `0.25` | weight of local oriented-bbox aspect agreement; several values sweep |
| `--top-k` | `5` | peaks kept per scale/heading combination |
| `--backend` | `auto` | `auto` or `numpy` |
| `--rerank` | off | re-sort every proposal by wall-segment inlier count (`align.rerank`); measured 22 vs 20 of 94 correct on `100_prod`, so off by default |

The three weights were re-measured on the `100_prod` prior-capture set over
40 random 50/50 splits, selecting on the train half only: top-1 within 2 m
went 20 → 31 of 94 (test halves 10.6 → 14.9 of 47, train–test gap 0.25).

Visual prior: where the panoramas say the cameras are (`align/visual.py`).
Every prior panorama has a known place on the plan; a current panorama that
looks like one was taken near it. A frozen DINOv2 describes both sides (four
yaw rolls per prior image, since a camera yaw is a column roll of an ERP), each
current camera retrieves its look-alike prior camera, the walk filters
look-alike corridors, RANSAC fits the planar pose those places imply, and the
correlation refines it over a small neighbourhood. Measured on `100_prod`
against the independent run: top-1 correct 31 vs 30 of 94, realities with
`tpr@1.0` > 0.95 **18 vs 10**, mean `tpr@1.0` 0.268 vs 0.231, median `rmse_xy`
9.3 vs 21.8 m; hold-out 10 vs 9 of 30. The gain is precision, from the
"accepted" path (>= half the cameras agree): 15 vs 8 sub-metre realities.
Per-reality cost is unchanged (14.7 s vs 15.4 s) once descriptors are cached;
describing a dataset the first time is ~1 h on a T4.

| Argument | Default | What it does |
|---|---|---|
| `--visual-prior` | off | ask the panoramas first; take their neighbourhood as the search when enough cameras agree, else let their candidates compete |
| `--visual-model` | `facebook/dinov2-base` | Hugging Face model id (`dinov2-small` is ~4x faster to describe) |
| `--visual-cache` | `cache/desc` | per-split descriptor cache |
| `--visual-top-k` | `5` | prior cameras retrieved per current camera |
| `--visual-accept-frac` | `0.5` | fraction of cameras agreeing with the fit above which its neighbourhood replaces the global search; measured right 4 times in 5 |
| `--visual-min-inliers` | `3` | below this the panoramas say nothing and the search runs alone |
| `--visual-radius-m` | `5.0` | walk-consistency tolerance, and the refinement disc |
| `--visual-workers` | `16` | JPEG decode processes |

Stage: scoring. All of it is optional — `--no-score` solves and writes
transforms without reading ground truth at all.

| Argument | Default | What it does |
|---|---|---|
| `--no-score` | off | solve only, never read the truth |
| `--no-oracle` | off | skip the best-planar-pose ceiling |
| `--thresholds` | `0.1 0.5 1.0 1.5 2.0` | metres, for `tpr@X` |
| `--overlap-radius-m` | `5.0` | a camera counts as covered within this of a prior one |
| `--pos-tol-m` | `2.0` | a pose counts as correct within this of truth |
| `--yaw-tol-deg` | `10.0` | ...and within this heading error |
| `--scale-tol` | `1.15` | ...and this scale ratio |
| `--debug-candidates` | off | write `<stem>_gt_probe.csv` (per reality: does the grid's scale bracket / yaw window reach the truth, and what the score function makes of the true pose itself) and `<stem>_grid_hypotheses.csv` (every evaluated candidate with its own distance from truth). Tells an enumeration miss from a ranking miss; large over a full batch, pairs well with `--reality` |

Stage: visualisation.

| Argument | Default | What it does |
|---|---|---|
| `--visualise` | off | render the three panels per reality |
| `--viz-dir` | `<stem>_viz` | where they go |
| `--viz-width` | `1100` | panel width in pixels |
| `--viz-top-k` | `5` | candidates drawn in R3 |

Output:

| Argument | Default | What it does |
|---|---|---|
| `--stem` | `out/run/alignment` | path prefix for every output |
| `--html-report` | off | write `<stem>.html`: a filterable, sortable gallery of every reality's metrics and viz images, all shown by default (needs `--visualise` for the images) |
| `--quiet` | off | suppress the per-reality line |

### The panels

Three per reality, titled so a PNG found on its own still explains itself.

| Panel | Shows |
|---|---|
| `R1_reference` | what this walk is matched against — the merged prior BEV, with the prior's own coverage behind it |
| `R2_placement` | where the walk was put: the capture in red, the truth in blue, with scale, heading and the row's metrics in the caption |
| `R3_alternatives` | the top-N candidates, warm for the chosen one and cooling with rank. A tight cluster means the score decided; a scatter means it did not |

### Sweeping

Passing several score weights runs the pipeline once per combination, each to
its own stem:

```bash
.venv/bin/python scripts/run_alignment.py --w-free 0 0.25 0.5 --limit 20 \
  --stem out/sweep/alignment
# -> out/sweep/alignment_wall1_free0_scale1.csv, ..._free0.25_..., ...
```

---

## The seven steps

The pipeline above in pieces, each holding four unknowns still and varying one,
so a bad number can only mean one thing. Steps **1, 4 and 7** produce results;
2, 3, 5 and 6 are diagnostics — skip them on a re-run, but not on a first one,
and not after the data changes.

### 1. Tests — is the boundary intact?

Checks the frame algebra, and the rule that keeps the current capture's own
answer out of the solver. That rule can no longer be a filename rule: a prior's
transform and the current capture's answer live in files with the same name, so
half the enforcement is a run-time guard and an audit hook that watches real file
opens. If this fails, stop — nothing downstream means anything.

```bash
.venv/bin/python -m pytest tests/ -q
```

Expect `1316 passed, 6 skipped` with the environment set (`130 passed,
18 skipped` without it — the dataset-backed tests skip rather than guess).
**~45 s.**

### 2. `check_shots.py` — is the trajectory a real input?

The solver needs per-image camera poses. These used to be synthesised by
un-applying the ground-truth transform — sanctioned, but still a path from the
answer into the inputs. Production writes the same poses to `shots.geojson`.
This proves the two agree before anything relies on it.

| Argument | Default | What it does |
|---|---|---|
| `--tol` | `1e-06` | metres; a split above this fails the check |
| `--out` | `out/check_shots.jsonl` | per-split record of the comparison |

```bash
.venv/bin/python scripts/check_shots.py
```

Must end `all within 1e-06 m`. Run over both indexes at once — the prior side
matters as much, since its poses are what the reference is built from.
Result: **273 splits, median RMS 3.39e-11 m, max 1.21e-9 m**, zero failures.
**~1 min.**

### 3. `e5_prior_quality.py` — how sharp is the reference?

Splits of one prior capture overlap. Where two contain the same photograph they
should place it identically; the gap between them is the reference's own blur,
and a floor under any accuracy the method can reach. Reads no point clouds.

| Argument | Default | What it does |
|---|---|---|
| `--min-shared` | `3` | images two splits must share to be comparable |
| `--out` | `out/e5_prior_quality.jsonl` | one record per comparable pair |

```bash
timeout 900 .venv/bin/python scripts/e5_prior_quality.py
```

Result: 55 comparable pairs across 37 structures; within one capture, median
**1.41 m**, p75 2.22, p90 3.71, max 10.78. Two pairs above 5 m are the same
outlier `docs/METHOD.md` §11 already records at 10.8 m. **~2 min.**

### 4. `reference_build.py` — build the thing being matched against

Pushes every split of a prior capture through its own transform into paper
inches, cuts the wall band, merges them into one raster, and adds a coverage
mask of where the prior actually walked. Cached per prior set — 131 realities
share 58 references.

| Argument | Default | What it does |
|---|---|---|
| `--cell-m` | `0.10` | raster cell, metres. The prior disagrees with itself by ~1.4 m, so finer buys little |
| `--stride` | `6` | keep every Nth point of each prior cloud |
| `--force` | off | rebuild even where a cached reference exists |
| `--out` | `out/reference_build.jsonl` | what was kept, dropped and corrected per reference |

```bash
timeout 3000 .venv/bin/python scripts/reference_build.py
```

Read the tail, not just the count: splits dropped and why, floor fits overridden
with the known 2.11 m rig height, references spanning more than one storey.
Result: **57 built, 73 shared a cached reference, 0 failed, 1 abstained, in
63 s**; rig height median 2.11 m, raster cells median 0.39M / max 11.0M, 40 floor
fits overridden, 2 references with more than one storey.

After changing anything in `align/reference.py`, clear the cache or the old
rasters are silently reused:

```bash
rm -rf cache/reference && timeout 3000 .venv/bin/python scripts/reference_build.py
```

### 5. `e3_prior_oracle.py` — can the reference be ranked at all?

The gate the whole approach rests on, and it runs no search. Place each capture
at its true scale and yaw and measure where that pose sits on the score surface.
If the answer does not rank highly when handed over for free, no pipeline work
will find it.

| Argument | Default | What it does |
|---|---|---|
| `--w-free` | `0.0 0.25 0.5 1.0` | weights to sweep for the coverage term; 0 disables it |
| `--out` | `out/e3_prior_oracle.jsonl` | per-reality percentiles, ceiling and overlap |

Six realities first, to see the shape before spending twelve minutes:

```bash
timeout 1200 .venv/bin/python scripts/e3_prior_oracle.py --limit 6
```

then the full set:

```bash
timeout 5400 .venv/bin/python scripts/e3_prior_oracle.py > out/e3_prior_oracle.log 2>&1
tail -30 out/e3_prior_oracle.log
```

Threshold, fixed before the first run: median percentile above **99.5**. The
floor-plan baseline for the identical measurement is 94 / 98 / 19.5.

Result: 109 of 131 scored in 697 s. Median percentile **99.962** at
`w_free=0.25` (99.764 with the coverage term off, which is what it is worth),
73/109 above 99.5, **PASS**. Ceiling median **0.71 m**, 69/109 under 1 m. Of the
22 unscored: 11 true pose off the reference, 10 no usable wall band, 1 no prior
cloud. **~12 min.**

### 6. `e6_scale_yaw.py` — does the search find scale and heading?

Step 5 handed those over. This asks whether the search discovers them, and
separates two failures that need different fixes: the truth not being among the
candidates at all (fix the grid) versus being there and out-ranked (fix the
score).

| Argument | Default | What it does |
|---|---|---|
| `--w-free` | `0.25` | weight of the coverage term in the search score |
| `--out` | `out/e6_scale_yaw.jsonl` | grid coverage and rank of the truth, per reality |

Three realities first — this one is slow enough that a typo is expensive:

```bash
timeout 900 .venv/bin/python scripts/e6_scale_yaw.py --limit 3 --out /tmp/e6t.jsonl
```

then the full set, in the background:

```bash
nohup timeout 5400 .venv/bin/python scripts/e6_scale_yaw.py \
  > out/e6_scale_yaw.log 2>&1 &
tail -f out/e6_scale_yaw.log
```

Result: 120 of 131 scored in 4511 s. Truth in the grid **76%** (scale 80%, yaw
92%), top-1 correct **37%**, median rank-when-present **0**. Split by whether
the grid held the truth: both covered, 48% correct and median error 1.44 m;
yaw missed, **0%**; scale missed, **0%**. The 18 realities that fell back to a
full 360 deg yaw sweep scored 0% — a sweep gives coverage and no
discrimination. **~75 min.**

### 7. `e7_reality_metrics.py` — the numbers

Takes the pose the search picks, places every photograph with it, and measures
how far each lands from where it belongs.

| Argument | Default | What it does |
|---|---|---|
| `--w-free` | `0.25` | weight of the coverage term in the search score |
| `--stem` | `out/e7_reality_metrics` | path prefix; three files are written from it |

Four realities first, which is what has been run so far:

```bash
timeout 900 .venv/bin/python scripts/e7_reality_metrics.py --limit 4 --stem /tmp/e7s
```

then the full set:

```bash
nohup timeout 5400 .venv/bin/python scripts/e7_reality_metrics.py \
  > out/e7.log 2>&1 &
tail -f out/e7.log
```

**The full run has not been done yet** — only the four-reality smoke test, which
is why no headline RMSE or TPR figure is quoted anywhere in this file. The
per-stage diagnostics above are complete; this one is not. **~40 min.**

---

## What step 7 writes

```
<stem>.csv                   one row per (reality, pose) — the reporting format
<stem>.jsonl                 the same, plus the 16 floats of every transform
<stem>_predictions.jsonl     {"reality_id", "tm", "confidence"}
```

The transforms are kept deliberately: a new threshold or a new metric then costs
nothing instead of a forty-minute re-solve.

**Columns.** `rmse_xy`, `median_xy`, `p90_xy`, `max_xy`, `tpr@0.1`, `tpr@0.5`,
`tpr@1.0`, `tpr@1.5`, `tpr@2.0`, `rmse_z`, `n_images`, plus `rig_height_m`,
`rig_implausible`, `yaw_fallback`, `overlap_5m`, `scale_source`.

**Error is horizontal.** `rmse_xy` and `tpr@X` measure distance in plan;
`rmse_z` sits beside them and is never folded in. The search solves a planar
similarity and never touches height — z is set afterwards by levelling against
the prior — so a combined figure lets a bad floor fit masquerade as a bad match.
At the oracle pose, where the horizontal fit is exact by construction, a 3-D
number still read as high as 2 m on nothing but the height estimate.

`tpr@X` is the fraction of a reality's photographs landing within X metres.
Five thresholds because one hides the shape: a pose can be metre-accurate on
every image, or right on half and wild on the rest.

**Three poses per reality, and the first is not readable alone:**

| Pose | What it is |
|---|---|
| `predicted` | what the search chose — the honest end-to-end result |
| `oracle` | the best 4-DOF planar fit to the truth — the parameterisation's own limit |
| `ceiling` | the same pose reached through the prior's registration, inheriting its error |

A predicted transform is built on top of the prior's. Reading `predicted`
without `ceiling` beside it credits the previous capture's registration error to
this work.

---

## Smoke test first

Every long stage was developed this way — a handful of realities to a scratch
output, then the full set once the shape looked right:

```bash
.venv/bin/python scripts/run_alignment.py --limit 4 --visualise \
  --stem /tmp/smoke/alignment
column -s, -t /tmp/smoke/alignment.csv | cut -c1-200   # readable in a terminal
```

The individual stages take `--limit` too:

```bash
.venv/bin/python scripts/reference_build.py --limit 10
timeout 900 .venv/bin/python scripts/e7_reality_metrics.py --limit 4 --stem /tmp/e7s
```

## Re-running from scratch

```bash
rm -rf cache/reference
rm -f out/check_shots.jsonl out/e5_prior_quality.jsonl out/reference_build.jsonl \
      out/e3_prior_oracle.* out/e6_scale_yaw.* out/e7_reality_metrics.* out/e*.log
rm -rf out/run
```

Then either the one command, or steps 1-7.

`cache/reference` must go whenever `align/reference.py` changes — it was the
cause of one confusing hour when a new field was added to the cached object and
the loader kept returning the old one.

Then steps 1-7. The reference cache is the only expensive state; everything else
is rewritten each run.

## The floor-plan baseline

The original method, for comparison, writing the same result format:

```bash
.venv/bin/python scripts/solve.py --all --results-stem out/solve_results
```

## Reading the results

Three things to know before the numbers surprise you.

**Roughly a third of realities come out correct.** Step 6 locates the
bottleneck: the scale and yaw grid, not the scoring. Miss either and the result
is unrecoverable — measured at 0% — while hitting both gives a sub-half-metre
candidate on the median reality. The `rig_implausible` and `yaw_fallback`
columns mark most of the failures.

**Abstentions are results, not zeros.** A reality with no usable prior, no wall
band, or a true pose outside the reference still gets a row carrying its reason.
Averaging them in as zeros would shrink the denominator and flatter everything.

**`tpr@1` is bounded by the prior.** The inherited ceiling is a median 0.71 m,
so on roughly a third of realities a flawless matcher still misses a one-metre
threshold. Report it beside its ceiling, or at two metres, or as error against
the ceiling rather than against the answer.

---

## Using a saved transform

`<stem>_transforms.jsonl` holds the answer: one line per (reality, pose), the
`tm` as 16 floats row-major.

```json
{"reality_id": "RLT...", "pose": "predicted", "tm": [16 floats]}
```

**Its frame is not the same for every reality.** `compose_tm` wrote ply → UTM
metres where the structure has a `design_tm`, and ply → paper inches where it
does not. The file records only `reality_id`, `pose` and `tm`, so the frame has
to come from the `georeferenced` column of `<stem>.csv`, joined on
`reality_id`. Inverting a `design_tm` against a `tm` that never had one
composed in produces no error, just a result in UTM-sized units.

### ply → paper inches

```
M = design_tm⁻¹ · tm      georeferenced
M = tm                    not georeferenced

paper_xyz = M · [ply_xyz, 1]
```

That is `decompose_tm`, which takes `design_tm=None` for the second case and so
makes the branch hard to get wrong:

```python
import json, numpy as np
from align.frames import load_design_tm, decompose_tm, apply, SheetGeometry

tms = {r["reality_id"]: np.asarray(r["tm"], float).reshape(4, 4)
       for r in map(json.loads, open("out/deploy/run_alignment_transforms.jsonl"))
       if r["pose"] == "predicted"}

design_tm = load_design_tm(path) if georeferenced else None
M = decompose_tm(tms[reality_id], design_tm)        # ply -> paper inches
paper_xyz = apply(M, ply_xyz)
```

For points already in the target frame, rather than a matrix to compose, use
`target_to_paper(pts, design_tm)`.

### paper inches → floormap.png

Paper has its origin bottom-left with y up; the image has it top-left with y
down. `SheetGeometry` owns that flip, and `check()` fails loudly on an aspect or
ppi mismatch rather than skewing silently:

```python
geom = SheetGeometry(paper_w=paper_w, paper_h=paper_h, img_w=img_w, img_h=img_h)
geom.check()
pix = geom.paper_to_pixel(paper_xyz[:, :2])
```

### Three things not to hardcode

| | measured across the dataset |
|---|---|
| `ppi` | 87.9, 150, 200, 300, 600 — five distinct values over 37 structures |
| `design_scale` | 2.44 .. 78.5 metres per paper inch |
| `design_z_datum` | 8 distinct values over 18 files; it is often not 100 |

And the `offset` field in the dict-shaped `design_tm` files is deliberately not
applied — `design_tm` already maps to absolute UTM, so applying it shifts twice.

A reality that abstained has no line here at all. Its row in the CSV carries the
reason.

### Per-image poses, and the picture on the sheet

`--images` turns the transform into the shape the floormap viewer consumes — one
record per photograph — and draws it on the architectural sheet:

```bash
.venv/bin/python scripts/run_alignment.py --current-csv ... --prior-csv ... \
  --no-score --images --stem out/deploy/run_alignment
# -> out/deploy/run_alignment_images/<reality_id>/images.json
# -> out/deploy/run_alignment_images/<reality_id>/floormap.png
```

```json
{"VID_..._IMG_00001.JPG": {"position": [x, y, z], "rotation": [x, y, z]}}
```

Positions are in the target frame, the same one `tm` maps to. Rotations are
degrees in three.js `Euler` order `XYZ`, which is **production's own
convention** — `align/io/floormap.py` is vendored from the converter that built
the files on disk, and `tests/test_floormap.py` pins it by pushing each split's
shots through its own registration and checking the result equals what was
delivered. It does, to 1e-12 degrees.

| flag | |
|---|---|
| `--images-every-nth 3` | keeps indices 1, 4, 7 … — the subset production ships |
| `--images-z 104.6` | pins every z, as the viewer's own file does |
| `--images-gt` | draws ground truth beneath the prediction, in grey |
| `--no-images-plot` | write the JSON, skip the picture |
| `--images-name`, `--images-dir`, `--images-plot-width` | the obvious |

Defaults write **all** shots with the **solved** z, which is the honest output of
the transform; the first two flags together reproduce production's file shape
exactly.

**`--images-gt` needs scoring.** Ground truth reaches the plot only through the
scorer, which has already read and converted it, so under `--no-score` the flag
is silently a no-op and the stage says so in its detail line. That is the
boundary working, not a limitation: a deployment run cannot draw an answer it
never opened.

One asymmetry worth knowing: nothing under `align/` may contain the string
`images.json` — `tests/test_no_leakage.py` forbids it outright, and does not
distinguish reading production's file from writing our own. So the library
default is `images_predicted.json` and `scripts/run_alignment.py`, which sits
outside the scanned tree, defaults the flag to `images.json`. The CLI writes
what a consumer expects; the guard stays intact.
