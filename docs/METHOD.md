# Method: point cloud → architectural sheet alignment

Recover the 4×4 transform `tm` that places a construction capture's point cloud
onto its architectural sheet (and, where the sheet is georeferenced, into UTM).

This document is the whole method: the frames, the pipeline, and stage-by-stage
pseudocode with the real constants. Every stage is implemented. What is *not*
yet true is the accuracy — see §11.

---

## 1. The quantity being solved

The search space is **4-DOF**, not 7: a planar similarity `Sim2 = (s, θ, tx, ty)`
applied after a tilt correction that gravity supplies for free.

```
tm = design_tm ∘ (ply → paper)        georeferenced structure  (ply → UTM metres)
tm =             (ply → paper)        otherwise                (ply → paper inches)
```

`align/frames.py:30` (`Sim2`), `align/frames.py:163` (`compose_tm`).
We always solve in **paper inches** and compose to UTM only at the very end —
`design_tm` scales span 2.44 … 78.5 m/inch, so solving in metres would make one
tolerance mean seven different things.

### Frames

| frame   | units | origin | notes |
|---------|-------|--------|-------|
| `ply`   | arbitrary | arbitrary | SfM/panovggt output; GT `tm` scales span 0.168 … 122.3 (2.9 decades) |
| `paper` | inches | bottom-left, y **up** | the sheet; the frame we solve in |
| `pixel` | px | top-left, y **down** | `floormap.png`; ppi ∈ {87.9, 150, 200, 300, 600}, always derived |
| `utm`   | metres | CRS in `geo_first_point.json` | only where `design_tm` exists and is non-identity |

```
px = paper_x / paper_w * img_w
py = (1 - paper_y / paper_h) * img_h          # align/frames.py:118
```

Raster convention, shared by `segments.rasterize`, `sheet.resample_to` and
`search`: cell (0,0) at the bounds origin, **row 0 at minimum paper y**. A cloud
raster and a sheet patch built on the same bounds line up index for index.

---

## 2. Inputs, outputs, and the leakage boundary

**Per split (reality):** `panovggt_point_cloud.ply`, and a camera trajectory in
the `ply` frame. **Per capture:** `geo_first_point.json`, one surveyed point.
**Per structure:** `floormap.png` + `paper_width`/`paper_height`, optional
`design_tm.json`.
**Output:** one JSONL row per split `{"reality_id": ..., "tm": [16 floats]}`.

`align/groundtruth.py` is the **only** module permitted to open `tm.json` or
`images_full.json`, enforced by `tests/test_no_leakage.py` with an AST scan over
every solver module (including a check that `align/io/paths.py` may *name* those
files but never read them). Production's SfM does supply per-image poses in the
`ply` frame, so the solver is entitled to a trajectory; here it is synthesized as
`inv(tm) @ images_full`. **Positions and image indices only** — the euler
convention in `images_full` is undocumented and a wrong guess fails silently, so
gravity, scale, Manhattan and chaining all work from positions alone.

`scripts/solve.py` scores each case against ground truth as it finishes, but
strictly **after** the search, through `align.groundtruth`, feeding nothing back;
`--no-gt` turns it off.

Dataset: **131 splits / 64 captures / 37 structures** (132 before
`RLT1788540996423` was removed); 61 georeferenced, 71 paper-frame. Measurements
quoted below keep the denominator they were taken with.

---

## 3. Pipeline

Two axes, deliberately:

* **stage-major** (`scripts/e1_scale_anchor.py`, `sheets.py`,
  `e2_oracle_inliers.py`, `consistency.py`) — one stage across all splits, for
  measuring a stage;
* **reality-major** (`align/pipeline.py:RealityPipeline`) — every stage for one
  split, each memoized, every intermediate left on the object, so a split can be
  stepped through in a debugger. `scripts/solve.py` drives it;
* **capture-major** (`align/chain.py`, `solve.py --chain`) — wraps the above and
  registers a capture's splits in walk order.

```
                 ┌──────────────────────── per STRUCTURE (cached) ─────────────┐
 floormap.png ──▶│ A. SHEET → WALL MAP                                         │
 paper_w/h       │   A1 binarize   A2 drop glyphs   A3 long runs                │
 metres/inch     │   A4 text+dense masks   A5 wall pairs   A6 border+viewports  │
                 └──────────────┬──────────────────────────────────────────────┘
                                │ wall mask (full res) + ≤5 viewports
                                ▼
  cloud.ply ─────▶┌── B. GRAVITY, FLOOR, SCALE ───────────────────────────────┐
  traj (ply)      │   B1 up vector   B2 local floor   B3 up-sign (may tie)     │
                  │   B4 s = 2.11 m / h, bracket      B5 sheet m/inch          │
                  └──────────────┬────────────────────────────────────────────┘
                                 │ tilt R, h, scale bracket, 1–2 up hypotheses
                                 ▼
                  ┌── C. CLOUD → WALL RASTER ─────────────────────────────────┐
                  │   C1 band vs NEAREST camera   C2 top-down raster           │
                  │   C3 Manhattan direction (4th-harmonic circular mean)      │
                  └──────────────┬────────────────────────────────────────────┘
                                 │ wall points, walk, yaw estimate
                                 ▼
                  ┌── D. COARSE SEARCH (proposes) ────────────────────────────┐
                  │   grid = scales × yaws × viewports × up-hypotheses         │
                  │   per grid point, one FFT pass over ALL translations:      │
                  │     w_wall·maskedNCC − w_free·(on-wall − inside) + w_scale·prior
                  │   optional translation prior disc (chain / geo)            │
                  │   top-5 peaks per map, pooled and sorted                   │
                  └──────────────┬────────────────────────────────────────────┘
                                 │ ≤300 proposals
                                 ▼
                  ┌── E. SEGMENT RERANK (disposes) ───────────────────────────┐
                  │   place cloud, extract segments, count inliers against the │
                  │   sheet by distance AND orientation; re-sort; keep top 20  │
                  └──────────────┬────────────────────────────────────────────┘
                                 │ best pose + gate verdict
                                 ▼
                  ┌── F. CHAIN (capture-major) ───────────────────────────────┐
                  │   order splits by image index, seed at the geo anchor,     │
                  │   walk outward; shared images → exact transfer, else a     │
                  │   prior disc sized by junction kind; re-seed on gate fail  │
                  └──────────────┬────────────────────────────────────────────┘
                                 ▼
                  ┌── G. COMPOSE & EVALUATE ──────────────────────────────────┐
                  │   Sim2 + tilt → 4×4 → compose through design_tm;           │
                  │   pixels always, metres only where georeferenced           │
                  └───────────────────────────────────────────────────────────┘
```

Two feasibility gates were run before any matcher existed — **E1** (does the
metric anchor pin the scale?) and **E2** (is the evidence present at all?). Both
pass; §10.

---

## 4. Stage A — sheet → wall map  (`align/sheet.py`)

A `floormap.png` is a full CAD sheet, not a clean plan: title block, notes,
dimension strings, leader lines, grid bubbles, hatching, and often several plan
viewports of which one matches the capture. Correlating against raw ink finds the
title block every time, so this cleaning is what makes matching possible at all.

**Two threshold families, and mixing them destroys the sheet.** Text is drawn at
a fixed size *on paper* → glyph thresholds in **inches**. Walls are sized in the
*world* → length/thickness in **metres**, converted through the sheet's own scale
(0.5 in is 2.4 m at 1/16″=1′ but 0.6 m at 1/4″=1′). All cleaning happens at
**full resolution** before any downsampling.

```
FUNCTION extract_sheet(png, paper_w, paper_h, metres_per_inch mpi = 3.0 default):

  gray ← load(png) as luminance;  ppi ← img_w / paper_w          # derived, never assumed

  # A1 — ink
  ink ← gray < 160
  IF mean(ink) < 0.005:                        # e.g. STR1782849228036: <1% ink, darkest ~250
      ink ← gray < percentile(gray, 0.5) + 20

  # photo-backed sheets (site plan over aerial imagery) have large smooth
  # mid-grey regions; pure line-work does not
  photo ← fraction of 1/8-scale pixels with 60<g<225 AND local σ(9×9) < 18
  photo_backed ← photo > 0.10

  # A2 — split ink into line-work vs glyph-like, by connected component
  FOR each 8-connected component:
      longest ← max(w,h);  fill ← area/(w·h)
      drop IF longest < 0.22·ppi                    # GLYPH_MAX: text, tick marks
           OR (fill > 0.55 AND longest < 1.00·ppi)  # SYMBOL_MAX: filled blobs
  keep, glyphs ← ink split by that rule

  # A3 — only long strokes can be walls
  run_px ← 0.80 m / mpi · ppi                 # MIN_LINE_RUN_M
  long_ink ← runs_at_least(keep, run_px, horizontal) ∪ (… vertical)

  # A4 — clutter masks. A text PANEL is dense in glyphs AND sparse in long
  # line-work; a tagged floor plan is dense in both, so glyph density alone
  # would delete the building.
  glyph_density ← boxfilter(glyph centroids, 1.0 in window) per square inch
  text ← (glyph_density > 25) AND (long_ink density in same window < 0.04)
  text ← dilate(text, 0.15 in)                # TEXT_HALO: the rules boxing a panel
  dense ← boxfilter(keep, 1.20 in) > 0.35     # schedules, dense hatch
  wall ← long_ink AND NOT text AND NOT dense

  # A5 — walls are drawn as PAIRS of parallel lines; dimension, leader, grid and
  # match lines are single. Strongest single discriminator on a CAD sheet.
  pairs ← per axis, ink runs facing another long run across a white gap in
          [0.05 m, 0.60 m], both runs ≥ 1.00 m
  IF |pairs| > 0.02·|wall|: wall ← pairs

  # A6 — frame and viewports
  border ← innermost rows/cols with ink fraction > 0.45 in the outer quarters
  wall ← wall ∩ border
  viewports ← find_viewports(wall)
  RETURN SheetFeatures(ink, wall, ppi, viewports, photo_backed, ink_fraction, border)
```

```
FUNCTION find_viewports(wall, max_depth=3, min_area=4 in²):
  # recursive X–Y cut on wall mass — the standard document-layout split, and
  # exactly right for sheets laid out as whitespace-separated viewports
  small ← downsample wall to 0.5 in cells, masked to the border box
  RECURSE(box, depth):
      FOR axis in (rows, cols):
          empty runs ≥ 0.8 in that are INTERIOR (not touching an edge)
          IF any: split at the middle of the widest, recurse both halves
  tighten each box to the ink it contains
  discard area < 4 in² or aspect > 8;  keep the 4 heaviest by wall mass
  ALWAYS append the bounding box of all retained wall ink
      # insurance: a plan cut into pieces that each fail the filters would
      # otherwise yield no viewport at all, and the split would be unsolvable
      # for a mechanical reason rather than a data one (STR1782849228036 did this)
```

`resample_to(feats, geom, bounds, res, shape)` puts the sheet's wall map onto a
paper-frame grid: crop by `paper_to_pixel`, `INTER_AREA`, then **threshold > 0** —
pool with *any*, not nearest, so a wall one sheet-pixel wide survives landing in a
cell six pixels across. Extraction is cached per `(structure, mpi)` as a
bit-packed `.npz`: a sheet takes up to 42 s and one sheet serves as many as 19
splits.

---

## 5. Stage B — gravity, floor, scale  (`align/gravity.py`, `align/scales.py`)

The cloud's scale is arbitrary across 2.9 decades, which makes a blind scale
search hopeless. A 360 camera on a monopod sits at a near-constant height above
the floor, so camera-height-above-floor in cloud units divided by that constant
pins the scale to ~±15%. `CAMERA_HEIGHT_M = 2.11`, measured over the
georeferenced sign-decisive subset (n=27: p10 1.88 / median 2.11 / p90 2.66).
Everything here is **scale-free** until a scale is applied: distances are
fractions of the trajectory's own horizontal extent.

```
FUNCTION estimate_gravity(cloud_xyz, traj_xyz):
  SVD(traj − mean);  up ← smallest singular vector       # a walked floor is a plane
  planarity ← σ₃ / √N / trajectory_extent
  IF planarity > 0.02:                                   # ramps, stairs, multi-storey
      up ← gravity_from_normals(cloud)
      # the direction least represented among local surface normals: walls
      # dominate, and every wall normal is perpendicular to up
  up, ambiguous, floor ← resolve_up_sign(cloud, traj, up)
  RETURN GravityFit(up, R = rotation taking up → +z, planarity, sign_ambiguous)
```

```
FUNCTION resolve_up_sign(cloud, traj, up):
  # Mass-below-the-trajectory fails in tall industrial and airport spaces where
  # roof structure outweighs the slab. No distance rule works either: in a hangar
  # the floor is much nearer; in a 3.6 m room a camera at 2.1 m is nearer the
  # ceiling. The reliable fact: NOTHING LIES BENEATH A FLOOR.
  FOR cand in (up, −up): fit ← floor_below_trajectory(cloud·R(cand)ᵀ, traj·R(cand)ᵀ)
  ambiguous ← |fit₀.beyond − fit₁.beyond| < 0.004         # SIGN_TIE_TOL
  RETURN the fit with less mass past its surface, flagged or not
  # 49 of 132 splits tie. GravityFit.alternatives() then yields BOTH up vectors
  # and the search scores both — an inverted up mirrors the top-down projection,
  # and a mirrored building matches no plan at any pose. Sampled 6 ambiguous
  # splits: 2 had the sign the ground truth disagrees with.
```

```
FUNCTION floor_below_trajectory(cloud_up, traj_up, radius_frac = 0.06):
  # The floor must be measured LOCALLY under the walk. A global z-histogram picks
  # up outdoor ground, mezzanines and ceilings and is useless.
  L ← trajectory extent;  radius ← 0.06·L;  tree ← KDTree(cloud XY)
  REPEAT twice:                    # pass 1 finds the scale, pass 2 refines on it
      window ← (0, 0.35·L)  then  (0.45·h₀, 1.7·h₀)
      FOR ~120 evenly spaced cameras c:
          idx  ← points within radius of c in XY     SKIP if < 40
          drop ← c.z − point.z, inside window        SKIP if < 25
          hist ← 60-bin histogram(drop)              SKIP if peak < 10
          heights ← append centre of the tallest mode   # nearest strong mode =
                                                        # the floor under the feet
      h₀ ← median(heights)
  h ← median(heights);  mad ← median|heights − h|
  beyond ← median over cameras of mean(drop > h + 0.08·(p98−p2 of drop))
      # mass PAST the surface, margin scaled by local vertical extent so it means
      # the same in a corridor and in a hangar. True floor ≤0.008, ceilings ≥0.03.
```

```
FUNCTION scale_prior(floor, camera_height = 2.11):
  s ← camera_height / floor.height            # target units per cloud unit
  bracket ← s·[0.85, 1.30] if rel_mad ≤ 0.15 else s·[0.65, 1.60]
  # In the pipeline both are divided by the sheet's metres-per-inch, giving
  # PAPER INCHES per cloud unit — the frame the search works in.
```

**Sheet scale** (`align/scales.py`). Metres per paper inch is a property of the
*sheet*, not of any one walk:

```
FOR each structure:
    design_tm present      → design_scale(design_tm)                  exact
    else pooled E1 splits  → median over sign-decisive splits of
                             2.11 / height_in_inches                  "e1 (n=…)"
    else                   → 3.0                                      "fallback"
# Only sign-decisive splits contribute: where the floor/ceiling test tied, the
# height could be either surface and the implied scale either value.
# `is_trusted` is False for the fallback, and solve.py prints SHEET SCALE GUESSED
# rather than letting every metre-based threshold downstream be silently wrong.
```

A standard-scale ladder (1/4″=1′ … 1:480) is carried as a **validator, not a
constraint** — the georeferenced structures include 3.733, 4.492 and 4.608 m/inch,
which sit on no standard rung. Measured: within 15% of a standard scale on 14 of
15 un-georeferenced structures, and the five 48×36 sheets of one project agree
with each other to about 2%.

---

## 6. Stage C — cloud → wall raster  (`align/cloud.py`)

```
FUNCTION wall_band(points_up, traj_up, floor):
  # Band each point against its NEAREST camera, so a walk that climbs stairs or
  # crosses levels does not smear the band into a slab spanning both.
  rel ← P.z − C[nearest camera in XY].z
  RETURN (rel > −0.85·h) AND (rel < 0.25·h)          # BAND_LO / BAND_HI
  # below the floor is nothing; a quarter-height above the camera is where useful
  # wall still is without reaching ceilings and roof structure
  # ABORT the split if fewer than 200 points survive

FUNCTION raster_bounds(points_xy, res, max_px = 4000):
  coarsen res until the raster fits max_px on a side
  # the site-plan structure sits at 78.5 m per inch, where a tolerance-derived
  # resolution would ask for tens of thousands of pixels

FUNCTION manhattan(mask, percentile = 97):
  # Buildings and sheets are overwhelmingly rectilinear, so the strongest
  # gradient direction modulo 90° pins yaw to four candidates instead of a sweep.
  θ ← gradient directions of the strongest 3% of edges
  RETURN (atan2(Σw·sin4θ, Σw·cos4θ) / 4) mod 90°, strength = |Σ| / Σw
  # fourth-harmonic circular mean: the standard way to average directions with
  # 90-degree symmetry
```

The coarse working cell is `COARSE_CELL_M = 0.20` m. Resolution is not cosmetic:
at 2 px across the tolerance two test splits scored 0 and 29 oracle inliers; at
6 px the same two scored 73 and 111.

---

## 7. Stage D — coarse search: propose  (`align/search.py`)

One FFT pass per (scale, yaw, viewport, up-hypothesis) evaluates **every
translation at once**:

```
score(s, θ, viewport)[t] =
      W_WALL  · maskedNCC(cloud_wall, sheet_wall)[t]         # 1.0
    − W_FREE  · (walk on wall − walk inside)[t]              # 0.5
    + W_SCALE · scale_plausibility(s, prior)                 # 1.0, per hypothesis
```

Weights are **not fitted** — 27 splits is far too few to fit three numbers
without memorising them. They come from measured separations:

* Plain binary correlation puts the peak on the **title block** and lands ~25 in
  from truth on 5 of 6 test cases. Normalising by the local energy under the
  template footprint — masked NCC — drops 4 of those to 1.8–7.8 in. Normalisation
  is not a refinement here, it is the term.
* A blind search over scale and yaw still got only 1 of 3 right; the other two
  locked onto a 90°-rotated, 1.36×-rescaled pose, because filled occupancy
  overlap is nearly invariant to exactly that transformation on a rectangular
  building. **Scoring where the walk lands breaks it**: a person does not walk
  through walls, and the cloud alone does not carry that.
* The walk term needs **both halves**. Penalising ink under the walk alone is
  worse than useless — blank paper carries no ink, so the lowest-penalty
  placement is off the drawing entirely. Measured: the top-ranked pose scored the
  *best* walk penalty of any candidate (0.006) precisely because it sat on empty
  sheet, 114 m from truth. Hence "inside the building AND not through its walls".

```
FUNCTION search(cloud_xy, traj_xy, sheet_map, sheet_origin, res, grid, …):

  # build every template first: the padding and the shared FFT plan depend on
  # the largest one
  FOR s in grid.scales, yaw in grid.yaws:
      template ← binary stamp of rotate(cloud_xy, yaw, s) at `res`
      traj_template ← the walk stamped on the same grid
  pad the sheet map by the largest template on every side
      # without it, only placements wholly inside the viewport are representable,
      # and a capture running past the drawn area — common, since viewports are
      # cut to the ink and walks are not — can have NO expressible true pose
  interior ← interior_mask(sheet_map)
      # close small gaps, flood in from all four corners, keep what the flood
      # could not reach. Blank margin outside the building IS reachable and is
      # therefore not interior — which is the point.
  F_sheet, F_sheet², F_inside ← one forward FFT each, shared by every hypothesis

  FOR each template:
      ncc   ← maskedNCC over all translations
              # footprint = the covered RECTANGLE, not the occupied cells:
              # inside the area the capture covers, "no wall here" is evidence.
              # (Using occupied cells makes the mean-subtracted template
              # identically zero and the whole correlation zero.)
      free  ← (walk·wall)/mass − (walk·interior)/mass
      sp    ← exp(−½·(log(s/prior) / 0.18)²)      # SCALE_SIGMA_LOG, the E1 spread
      total ← W_WALL·ncc − W_FREE·free + W_SCALE·sp

      IF a TranslationPrior is given:              # from the chain or the geo fix
          centres ← prior.centre − rotate(prior.ref, yaw, s)      # one per ref
              # the constraint is on POINTS, not on the translation: what "this
              # camera lands here" permits depends on the scale and yaw tried.
              # `ref` is a WINDOW of cameras (±20 around the named photograph
              # for a geo fix), so the admissible set is the UNION of their
              # discs — one distance transform, not a hypot per camera.
          dist  ← distance to the nearest centre
          total ← total + 0.5·exp(−½·(dist/radius)²), then −∞ outside the disc
          IF the disc holds nothing representable:
              fall back to the unrestricted map and set outside_prior
      keep the top 5 local maxima (NMS radius 3 cells)
  RETURN hypotheses sorted by score
```

`yaw_candidates`: both Manhattan angles are known only modulo 90°, leaving four
rotations; each gets a ±10° window at 3° steps (the estimator was within 5° on
16 of 27 splits and within 10° on 19, and ground truth's own rotation is off by
as much as 14.5°, so a narrower window would chase noise — and its median
rotation error of 1.57° is why the step is 3° and not the 1.5° it started at,
which doubled the grid to resolve the same noise twice). Below a strength
threshold it falls back to a 4° sweep over the full circle.
`scale_candidates`: 9 log-spaced steps across the E1 bracket.
Backends: numpy reference and a torch/GPU path, `test_backends_agree` pins them
together.

`aspect_agreement(cloud_xy, sheet_wall, res)` — width-to-width and height-to-height
as an ordered pair, deliberately not a max/min ratio, which would be invariant
under the very 90° flip it exists to catch — is implemented and tested but **not
currently wired into the score**; the third term in code is the scale prior. The
module docstring still describes the aspect variant.

---

## 8. Stage E — segment rerank: dispose  (`align/rerank.py`)

**The correlation score cannot rank.** Measured with scale and yaw held at ground
truth, the true translation sits at the 94th, 98th and **19.5th** percentile of
the masked-NCC surface on three test splits — on one, the term prefers 80% of all
other placements. Softening the sheet map and changing the footprint were both
tried; neither repaired it. Correlating ink overlap simply does not distinguish a
correct partial match from a coincidental dense one.

Segment matching does — that is exactly what E2 measured before any matcher
existed. So the search **proposes** and this **disposes**:

```
FUNCTION rerank(hypotheses, wall_points_by_up, sheet, geom, mpi):
  # EVERY hypothesis, not a pool. Nothing below depends on a candidate except
  # through the similarity applied to segment endpoints, so it is prepared once.
  res   ← (0.30 m / mpi) / 6                      # TOL_M / TOL_PX
  scorer over the region covering every placement these poses produce:
      dist ← distanceTransform(NOT plan),  ang ← orientation_map(plan)
                                        # properties of the SHEET; pose-free
      segs ← LSD(rasterize(wall_xy), ≥ 8 px)
                                        # ONE detection for the whole split
  FOR each hypothesis:
      h.inliers ← how many segs, carried to this pose, agree in distance AND
                  orientation (the same test match_inliers applies)
  re-sort by (−inliers, −score)     # correlation is the tie-break, not the rank
  keep the top 20
```

There used to be a pool, because reranking cost **344 ms a candidate**: it
rasterized the cloud, ran LSD, resampled the sheet and built a distance
transform for every one. Sweeping the pool: 60 → 16.0 m from truth, 300 →
12.6 m, 1200 → 12.8 m, at 99 s → 173 s → 473 s. That plateau looked like the
ranking converging. It was the pool running out of candidates — on the same
split the pose closest to truth sat at correlation **rank 671 of 12,600**.

Preparing the context removes the reason for a pool. Two facts make it exact
rather than approximate:

  * the sheet's distance and orientation maps do not depend on the pose at all;
  * `count_inliers` rasterizes over bounds derived from the PLACED cloud, so a
    translation shifts the raster origin by exactly as much as the points —
    LSD's input, and therefore its output, depends only on yaw and scale.

So segments can be cached per (yaw, scale): 253 LSD runs instead of 1260, and
the per-candidate cost falls to two array lookups. Measured against
`count_inliers` on the four test splits, the segment sets are **identical** and
inlier counts differ by at most 5 of ~100 (Spearman 0.93–1.00) — the residue is
grid phase, since one shared grid cannot share every pose's origin.

Going further, segments can be detected **once for the whole split** and carried
to each pose. That is no longer exact — LSD on a rotated raster is not a
rotation of LSD on an unrotated one — but it is arguably the better measure: a
wall is a property of the building, not of the guess being tested, and
re-detecting per pose makes the segment count swing (333…362 on one split at a
single scale) purely from raster aliasing, so two candidates are not scored
against the same denominator. Side by side over every candidate of the four test
splits, reranking cost 3.1–55.1 s per-pose against 0.5–1.7 s once-per-split, and
the top-ranked pose landed within 5 m of the same place either way. `SEGMENT_MODE
= "cloud"` records that; `"pose"` remains, pinned to `count_inliers` by test.

Read the tie weakly. **The ranking is poor in both modes**: on three of the four
splits the true pose scores no better than a candidate tens of metres away, so
"no measurable difference" partly means there was little to measure.

```
FUNCTION match_inliers(cloud_segs, plan_mask, tol_px, ang_tol_deg = 10):
  dist ← distanceTransform(NOT plan_mask)      # L2, pixels
  ang  ← orientation_map(plan_mask)            # structure tensor, mod π:
                                               # a wall has direction, not sense
  a segment is an inlier when ≥70% of 12 samples along it lie within tol_px of
  line-work whose local direction agrees to ang_tol_deg
  # matched against a distance transform, NOT against extracted plan segments: a
  # wall drawn as one long line and seen as three occluded fragments should count
  # as three matches, not fail a one-to-one assignment
  # 10° not 5°: at the GT pose the cloud-vs-sheet Manhattan difference has median
  # 1.6° but p90 9.7° and max 14.5°, so a tighter gate rejects correct matches on
  # the strength of ground-truth error alone
```

---

## 9. Stage F — chaining a capture  (`align/chain.py`)

A capture is one continuous walk cut into pieces. Ordered by image index, where
one ends the next begins — so once any piece is placed, its neighbour's start is
known to within a few metres. Measured over the 68 adjacent pairs, end-to-start
distance in paper inches:

| junction | pairs | median | max | search radius used |
|----------|-------|--------|-----|--------------------|
| contiguous (consecutive indices) | 40 | 0.18 | 1.22 | 5 m |
| overlapping / nested | 9 | 0.81 | 8.58 | 2 m (or exact transfer) |
| index gap (recording stopped) | 19 | 1.04 | 9.54 | 15 m |

against the GPS start point's median 8.7 m, p90 35 m, max 157 m. **The GPS fix is
only the bootstrap; the chain is what pins things down.**

```
FUNCTION solve_capture(cases, solve_fn, mpi, trajectories):
  order  ← splits sorted by first image index
  geo    ← the capture's start point, from WHICHEVER member carries it
           # Case.geo is set only where source_reality_id matches the split,
           # so capture-level code has to consult every member
  anchor ← find_anchor(members, trajectories, geo)
           # the fix identifies its moment TWICE — a reality id and a
           # photograph — and they disagree: 21 of 59 splits do not contain the
           # image their own start point names, and 19 have it in a sibling.
           # The photograph decides; the reality id only says which capture.
  IF no anchor: solve every split independently and return
           # including when the image is in NO split (2 captures). The previous
           # code fell back silently to the split's first camera — and inside a
           # 4-inch radius that EXCLUDES the true pose, leaving a third of
           # anchored splits worse off than with no prior at all.

  seed  ← the split HOLDING the photograph
          # not necessarily the earliest, and for 19 of 59 not the one the
          # start point is filed under, so the chain walks BOTH ways
  prior ← TranslationPrior(centre = geo point in paper inches, radius = 4.0 in,
                           ref = the ±20 cameras around that photograph)
  run(seed, prior, "seed (geo anchor, ±20 images)")

  FOR rid, from_id in walk_outward(order, seed):        # forward, then backward
      IF from_id did not pass the gate:
          run(rid, no prior, "independent (neighbour not trusted)")
          # let it re-seed the chain rather than carry a bad pose through
          CONTINUE
      link ← classify_junction(from, rid)
      IF link is SHARED and ≥3 images in common:
          sim ← umeyama_sim2(rid's ply positions → from's solved paper positions)
          accept EXACTLY, no search at all, record the RMS residual
          CONTINUE
      centre ← from's LAST camera, placed        # splits are index-ordered, so
                                                 # A's last camera meets B's first
      run(rid, TranslationPrior(centre, link.radius, ref = rid's first camera),
          "chained")
```

```
FUNCTION gate_result(result, mpi):              # may this pose seed the rest?
  rival ← the best hypothesis more than 5 m away OR more than 10° turned
  margin ← best.score − rival.score
  FAIL IF best.wall < 0.02          OR  margin < 0.01
  # a margin test, not a model: if a materially different pose scores about the
  # same, the score did not actually choose, and propagating from it would carry
  # a coin-flip into every split after it
```

Two results worth keeping, both measured while building this:

* **The prior works, on translation.** Given the true start position, two of
  three test splits went from 64.5 m and 12.1 m of error to 2.1 m and 4.3 m.
  Translation was the dimension the coarse score could not resolve; a disc around
  a known point resolves it.
* **It does not constrain yaw, and heading continuity cannot.** A single-point
  prior is satisfied just as well by a pose rotated about that point — the third
  split stayed wrong and flipped 178°. The obvious fix, requiring walk direction
  to carry across the seam, does not survive the data: the turn at a contiguous
  junction has median **52.8°**, and only 18 of 40 are under 45°. People turn
  where the recording breaks. **Yaw has to come from the wall term.**

The geo fix lives in the structure's target frame — UTM metres where a design
transform exists, **paper inches where it does not**, with `crs` set either way
and misleading on the second group (a paper-frame fix reads x=28.6, y=16.1 on a
42×30 sheet, z=1.7 for every split, which is the rig height, not an elevation).
Lifting the original UTM-only assumption took the anchored splits from 28 to 60.
Radius 4.0 in covers p90 twice over: against the camera each fix names,
georeferenced splits are median 0.63 in / p90 2.03 / max 2.99, paper-frame splits
median 0.59 / p90 1.35 with a single 33.7 outlier.

---

## 10. Stage G — compose and evaluate  (`align/eval/metrics.py`)

```
ply_to_paper ← Sim2(s, θ, tx, ty).to_sim3(tilt_R = R_up, tz = 0)
tm           ← compose_tm(ply_to_paper, design_tm)

FUNCTION evaluate(pred_tm, gt_tm, gt_pos, sfm_pos, design_tm, geom):
  pred_pos ← sfm_pos · pred_tm                     # forward the SIMULATED input
  d_px  ← ‖paper_to_pixel(pred) − paper_to_pixel(gt)‖        # ALWAYS
  d_utm ← ‖pred_pos − gt_pos‖  ONLY IF design_tm is non-identity
          # else the target frame IS the paper: pixels alone, not fake units
  scale_log_err ← |log(scale(P)/scale(G))|;  yaw_err, and yaw_err mod 90°
  tpr@1m ← mean(d_utm ≤ 1.0)                       # 3-D and 2-D variants
  centroid_err ← ‖mean(pred_pos) − mean(gt_pos)‖
      # never ‖t_pred − t_gt‖: the raw translation column is entangled with scale
      # and rotation about a UTM origin 250 km away

FUNCTION rank_of_truth(result, gt, …, tol 15 m / 15°):
  # the number that matters at this stage: a correct pose ranked 10th means the
  # search FOUND it and the score could not tell; no correct pose at all means
  # the ENUMERATION missed it. Those call for different fixes.
  RETURN closest_rank, closest_pos_m, found_rank, n_within
```

Thresholds: px ∈ {5, 10, 25, 50, 100}; metres ∈ {0.25, 0.5, 1, 2, 5}. Splits
absent from `predictions.jsonl` count as **abstained**, not as errors.

---

## 11. Where it stands

**Feasibility (both PASS, run before any matcher existed).**

*E1 — metric anchor.* Falsified if the p10–p90 spread exceeds ±25%, or if more
than ~15 splits yield no floor mode. → n=27 georeferenced sign-decisive: median
**2.11 m**, −11%/+26%. 10 splits with no usable floor mode; **49 sign-ambiguous**,
carried as two hypotheses.

*E2 — is the evidence there?* Place each cloud at its **ground-truth** pose and
count wall segments with a counterpart on the sheet — what a perfect search would
score. Falsified if median < 8 or more than a third of splits fall below 4. →
n=123: median **12**, p90 43, **29% under 4** (permanent abstain bucket).
Position-only: median 22, 18 under 4 → for 18 splits the wall is present but the
orientation disagrees, i.e. ground-truth rotation error, not missing evidence.

*Ground-truth self-consistency.* 10 overlapping split pairs disagree by a median
of **50 px**, worst 331 px (10.8 m). That is a hard floor on any achievable score.
Separately, 6 splits carry `"reg_tool": true` — hand-registered, so they score
tpr=1.0 against their own `tm` tautologically and must stay out of headline
numbers.

**End-to-end accuracy: not there yet, and the smoke runs say so.** On the two
6-split runs currently on disk:

| run | n | median pixel err | closest hypothesis to truth | a pose within 15 m/15° | ranked 1st |
|-----|---|------------------|-----------------------------|------------------------|------------|
| independent (`out/indep_metrics.jsonl`) | 6 | 953 px | 8.2 m | 3/6 | 1 |
| chained (`out/chain_metrics.jsonl`) | 6 | 955 px | 12.6 m | 2/6 | 2 |

Within the chained run: seeds n=3 (closest median 18.3 m, 2 of 3 gates passed),
chained n=2 (8.8 m, both gates passed), independent n=1 (0.1 m, 9 px). tpr@1m is
0.167 on both. A full `--chain --root data2` sweep over 37 captures is the
current run; it writes `out/predictions_data2_chain.jsonl` and
`out/metrics_data2_chain.jsonl` at the end.

Read that table with `rank_of_truth` in mind: *closest hypothesis* is the
enumeration's best achievable, and it is still 8–13 m out, so at present both the
enumeration and the ranking need work — not the ranking alone.

---

## 12. Running it

```bash
python scripts/run_all.py                       # every stage, all splits
python scripts/run_all.py --root data           # data/ only (CSV restricted to match)
python scripts/run_all.py --stages solve evaluate
python scripts/run_all.py --dry-run             # print the commands, run nothing

python scripts/solve.py --reality RLT1788196494610          # one split, verbose
python scripts/solve.py --reality RLT1788195772679 --viz    # render the decision
python scripts/solve.py --chain --root data                 # capture-major
```

| stage | script | what it does |
|-------|--------|--------------|
| `tests` | `pytest tests/` | frame algebra, leakage boundary, search, rerank, chain |
| `e1` | `e1_scale_anchor.py` | gravity, local floor, camera-height scale prior → `out/e1_*.jsonl` |
| `sheets` | `sheets.py` | wall extraction per structure → `cache/sheet/` |
| `e2` | `e2_oracle_inliers.py` | oracle inliers at the GT pose → `out/e2_*.jsonl` |
| `consistency` | `consistency.py` | GT self-disagreement → `out/gt_consistency.jsonl` |
| `solve` | `solve.py` | search + rerank (+ chain) → `predictions_*.jsonl`, `hypotheses_*.jsonl` |
| `evaluate` | `evaluate.py` | image-to-image error of those predictions |
| `viz` | `viz_stages.py` | every stage as a panel, per split |

`align/io/paths.py:case_paths(reality_id)` resolves every file for one split from
either CSV — sheet, cloud, `tm`, `images_full`, the geo fix, `design_tm`, paper
size — and flags the two traps: `shots_path`/`images_prod_path` name files that
are not on disk, and a capture's `geo_first_point.json` anchors **one** split, so
`start_point_is_ours` says whether it is this one's (60 of 132).

---

## 13. Known failure modes

| failure | scale | cause | handling |
|---------|-------|-------|----------|
| no floor mode | 10 splits | multi-level walk, ramps, cloud sparse beneath the cameras | no scale prior → the split yields no hypothesis |
| sign-ambiguous floor | 49 splits | flat floor and flat ceiling both fully captured | both up-vectors searched; the rerank decides |
| under 4 oracle inliers | 36 splits | as-built differs from the sheet, photo-backed site plans, plan not on this sheet | permanent abstain bucket — no geometric method solves these |
| orientation-limited | 18 splits | ground-truth rotation error, not missing evidence | position-only scoring as a diagnostic; never tighten the angle gate below 10° against GT |
| guessed sheet scale | structures with no `design_tm` and no E1 estimate | every metre-based threshold derives from it | `scale_trusted=False`, printed as SHEET SCALE GUESSED |
| gate failure | frequent on seeds | the score did not actually choose between rivals | the chain re-seeds instead of propagating a coin-flip |
| yaw flip at a prior | seen on 1 of 3 test splits | a point prior is invariant to rotation about that point | unfixable from the chain; yaw must come from the wall term |
| GT disagrees with itself | 5 pairs > 50 px | at least one of the two transforms is wrong | a hard floor on any score; report it rather than chase it |
