# Deploying prior-capture alignment

Two modules. One turns an already-registered capture into a reference; the other
places a new capture against it. They are separate because their lifetimes are:
a reference is built once per prior capture and reused by every capture that
follows it, while the aligner runs once per arrival.

`docs/PRIOR_CAPTURE_RUNBOOK.md` is how to run the research pipeline. This is how to run the two
pieces that matter in production. `docs/PRIOR_CAPTURE.md` explains the method.

**Read [What to expect](#what-to-expect) before wiring this to anything.** It
places roughly a third of captures correctly and currently cannot tell you
which third.

---

## What it needs on disk

Nothing else. In particular, no floor plan image and nothing of the new
capture's own registration.

| | file | used for |
|---|---|---|
| per prior split | `point_cloud.ply` | the reference's walls |
| | `shots.geojson` | its camera track |
| | `tm.json` | its known transform |
| per new capture | `point_cloud.ply` | what is being placed |
| | `shots.geojson` | its camera track |
| per structure | `tm_local_to_global.json` | the georeference, when there is one |
| | `paper_width`, `paper_height` | from the index, for the paper frame |

The new capture's `tm.json`, `images_full.json`, `images.json` and `geo.txt` are
**never opened**. That is enforced rather than asserted: `align/priors.py`
refuses any read outside the declared prior pool, and
`tests/test_solver_reads.py` runs the real pipeline under a `sys.addaudithook`
and fails if a single one of those files is touched.

---

## Module A — the prior reference

`align/priors.py` + `align/reference.py`

Pushes every split of a prior capture through its own transform into a common
frame, cuts the wall band, merges them into one raster, and records where the
prior actually walked. Build once per prior capture, cache, reuse.

```python
from align import reference as refmod
from align.io.cases import PairedIndex
from align.io.pairing import prior_reference_for

index = PairedIndex.load()                      # or PairedIndex.load(spec)
case  = index.current.get("RLT1788173473510")

ref     = prior_reference_for(case, index)      # the guarded pool, or None
ref_map = refmod.build_cached(ref, case.structure, cache_dir="cache/reference")
```

`build_cached` writes `cache/reference/<key>_s<stride>_c<cell>.npz`, where the
key is a hash of the prior split ids — so every capture naming the same prior
shares one build. Observed: 131 captures, 58 builds.

```
key 5575445ae0c6c4ed   cache 5575445ae0c6c4ed_s6_c0.1000.npz
cold  0.1 - 13 s depending on how many splits merge
warm  0.003 s
```

Two returns mean "no reference", and both are ordinary outcomes rather than
errors: `prior_reference_for` gives `None` when no prior split has a cloud, and
`build_cached` gives `None` when none of them survives the floor and
consistency checks.

Worth reading off the result before trusting it:

| field | meaning |
|---|---|
| `prior_reality_ids` | which splits actually made it in |
| `dropped` | which did not, and why |
| `warnings` | floor fits overridden with the known 2.11 m rig height, extra storeys, a thin reference |
| `thin` | fewer than 2000 wall cells — too little to match against |
| `scale_source` | `design_tm` (exact), `prior_camera_height` (estimated), `fallback` (a guess) |

The cache key carries stride and cell size only. **Changing any other build
parameter needs `force=True`**, or a stale raster comes back silently.

Batch it ahead of time:

```bash
.venv/bin/python scripts/reference_build.py \
  --current-csv CUR.csv --prior-csv PRIOR.csv
```

~60 s for 58 references. The tail prints splits dropped, fits overridden and
multi-storey references — the numbers worth watching between data drops.

---

## Module B — align the new capture

`align/runner.py`

Reads the new cloud, fits gravity, cuts its wall band, searches every scale and
heading against the reference, and composes the winner into a transform.

```python
from align.runner import Params, run_reality

p = Params(scoring=False)          # no ground truth, anywhere
r = run_reality(case, index, p)

if r.status == "ok":
    tm = r.poses["predicted"]["tm"]        # 4x4, ply -> target frame
else:
    ...     # r.status says why; r.stages says which stage stopped it
```

`~4 s` on a small reference, `~23 s` on a large one, with the reference cached.

`r.status` is `ok`, or one of `no prior cloud`, `no usable prior split`,
`no point cloud`, `no usable wall band`, `no hypotheses`. Each is an abstain.

To run the stages yourself — the runner is only their sequence:

```python
from align.runner import stage_reference, stage_cloud, stage_search, stage_pose, RealityRun

run = RealityRun(case=case)
stage_reference(run, index, p)     # module A, through the same cache
stage_cloud(run, p)
stage_search(run, p)
stage_pose(run, p)
```

---

## What the transform is

```
tm :  ply  ──▶  target
```

`target` is UTM metres where the structure has a design transform and paper
inches where it does not. **`design_tm` is already inside `tm`** — the same
convention as the stored `tm.json`, so the two are directly comparable.

To put cameras on the floor plan, `design_tm` therefore comes off, not on:

```python
from align.frames import apply as apply4, target_to_paper

pos   = apply4(tm, shots)                                  # ply -> target
paper = target_to_paper(pos, structure.design_tm)          # -> paper inches
px    = structure.geometry().paper_to_pixel(paper[:, :2])  # -> floormap pixels
```

`target_to_paper` inverts `design_tm` when there is one and does nothing when
there is not, so do not branch on `georeferenced` yourself. Applying `design_tm`
a second time puts the cameras thousands of kilometres away and 0% of them on
the sheet — it fails loudly in a plot and silently in a database.

The z column is the weakest part of the transform: the search solves a planar
pose and height is set afterwards by levelling against the prior. Plan views are
unaffected.

---

## Batch

```bash
.venv/bin/python scripts/run_alignment.py \
  --current-csv CUR.csv --prior-csv PRIOR.csv \
  --no-score --stem out/deploy/run
```

`--no-score` is the deployment flag: it skips every ground-truth read and still
writes the transforms.

```
out/deploy/run.csv                  one row per capture, with the reject signals
out/deploy/run_transforms.jsonl     {"reality_id", "pose", "tm"}
out/deploy/run.jsonl                warnings, drop reasons, per-stage timings
```

Add `--visualise` to get three panels per capture. Useful when a result is
disputed, too slow to leave on for every arrival.

---

## What to expect

**About a third of captures are placed correctly.** Top-1 was correct on 37% of
120 realities, within 2 m, 10° and a 1.15× scale ratio. The bottleneck is the
candidate grid, not the scoring: when the grid contained the true scale and
heading, 48% were correct with a median error of 1.44 m; when it missed either,
**0%** were, with median errors of 11 m and 39 m.

**There is no confidence signal yet on this path.** Nothing in
`run_alignment.py`'s output separates a 0.3 m answer from a 40 m one, so a
consumer cannot tell them apart. This is the single biggest gap between this
code and something servable, and it is a known piece of work rather than a
surprise.

The learned path has one. `bevaligner/inference/confidence.py` writes a
`confidence` column on every row and beside every transform in
`<stem>_predictions.jsonl`: the calibrated probability that the reality is
placed within 2 m, from the inlier fraction, the zoomed pass, the margin over
the best rival and the rig height. Measured over 1073 scored realities, rows at
0.85 and above are right 92–94% of the time against a base rate of 0.23, and
about 60% of rows score below 0.1. It does not transfer to this path: the
classical solver's signals are a correlation score and a wall fraction, not an
inlier fraction, so it needs its own fit.

Three reject rules are available today, and together they are all you have:

| signal | what it caught |
|---|---|
| `status != "ok"` | 11 of 131 abstained outright |
| `yaw_fallback` | the heading estimate was too weak, so the search swept the full circle: **0% correct** across all 18 such realities |
| `rig_implausible` | the capture's own floor fit is wrong, which corrupts scale and height together |

Rejecting on those three is worth doing. It will not get you to a trustworthy
service on its own.

**Accuracy is bounded by the prior, not only by the matcher.** The transform is
built on top of the prior's, so the prior's registration error is inherited
whole: median 0.71 m, p90 3.04 m. On roughly a third of captures a *flawless*
match would still miss a one-metre threshold. If a downstream consumer needs
metre accuracy, that bound applies to it too.

**Prior quality is the highest-value thing to improve.** The poorly-placed
captures have the same camera overlap with their prior as the good ones — 1.00
against 1.00 — and differ only in inherited error, 2.80 m against 0.50 m. They
are not failing to match; they are matching correctly against a reference that
is in the wrong place.
