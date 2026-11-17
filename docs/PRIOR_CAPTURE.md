# Prior-capture alignment — the method, and how it is measured

Place a capture onto its building's coordinate system by matching it against the
**previous capture of the same building**, instead of against the architect's
drawing.

`docs/METHOD.md` describes the original floor-plan method and remains the
reference for everything the two share — frames, gravity, the wall band, the
search. This file covers what the substitution changes, and the seven-stage
diagnostic pipeline built around it. `docs/PRIOR_CAPTURE_RUNBOOK.md` is how to run it.

Live versions of these two halves, with proper diagrams:

* the method — https://claude.ai/artifact/Q6j4oBb99ijD7xK6vDCVQJ
* the stages — https://claude.ai/artifact/E9uuEcG3bXJFxQdFkxG4gn

Every number below is a measurement over 131 current walks, 191 prior walks and
77 prior captures across 37 buildings, except where it says a stage has not been
run.

---

# Part one — the method

## 1. What is being solved

A surveyor walks a site with a 360° camera on a monopod. Photogrammetry turns
that walk into a point cloud and a camera trajectory, but in an *arbitrary*
frame with arbitrary scale. It knows the shape of what was walked, not where on
Earth it sits.

`tm` is the 4×4 that carries that frame onto the building's coordinates. Gravity
is recoverable from the walk itself, so the tilt comes free and what is left is a
**planar similarity** — scale, heading, two translations. Four numbers.

```
   ply  ──── tm ────▶  target  ── design_tm⁻¹ ──▶  paper  ── ppi ──▶  pixel
   │                   │                           │                  │
   arbitrary SfM       UTM metres on 50            inches,            floormap.png
   scale 0.168         structures, paper           y up, origin       y down,
   … 122.3             inches on 59                bottom-left        87.9…600 ppi
                       ▲
                       └─ what we solve for
```

The target frame being metres on some structures and inches on others is the
single most dangerous fact in this document. See §5.

## 2. Why the drawing is a hostile target

A drawing and a point cloud share no sensor and almost no content. The drawing
has door swings, dimension strings, hatching, a title block, furniture and text;
most of its ink is not wall. The cloud has scaffolding, stacked material and
temporary props. The paper scale is often unrecorded. One describes the building
as it will be, the other as it currently is.

Measured, on the same buildings:

| | |
|---|---|
| median `tpr@1m`, 94 splits | **0.000** |
| median `tpr@1m`, easier 27-split set | 0.407 |
| end-to-end median error (`METHOD.md` §11) | 8.2 m |
| true pose's percentile on the score surface | **94th / 98th / 19.5th** |

That last row is the diagnostic one: handed the correct position for free, the
score still could not distinguish it from thousands of wrong ones.

## 3. The substitution

Sites are captured repeatedly, and by the time a new walk arrives the same
building has usually been walked before — and *already placed*. That transform
sits in the production database, so it is an input, not an answer.

```
  BEFORE                                  AFTER
  ┌──────────────┐                        ┌──────────────┐
  │ current BEV  │                        │ current BEV  │
  └──────┬───────┘                        └──────┬───────┘
         │ match                                 │ match
         ▼                                       ▼
  ┌──────────────────────┐              ┌──────────────────────┐
  │ floorplan line-work  │              │ prior capture BEV    │
  │ drawn, not measured  │              │ same sensor          │
  │ text · symbols       │              │ median 1 day earlier │
  │ scale often unknown  │              │ scale already known  │
  └──────────────────────┘              └──────────────────────┘
   true pose at 19.5th–98th pct          true pose at 99.96th pct

  UNCHANGED  gravity · wall band · FFT search · frame algebra · evaluation
  CHANGED    what the cloud is compared against · the free-space term · scale
  REMOVED    sheet cleaning · paper-scale estimation · the surveyed-point seed
```

Both sides are now the same kind of object: two point clouds of the same rooms,
from the same camera, a median of **one day** apart (max 14). The same
scaffolding is in both.

## 4. Building the reference

A capture is a walk cut into *splits*, each reconstructed separately, so a prior
capture arrives as 1–14 clouds each with its own known transform.

```
 split A ──tm_A──┐
 split B ──tm_B──┼──▶ paper inches ──▶ band cut ──▶ merge ──▶ wall raster
 split C ──tm_C──┘      one common        per          │        + coverage mask
                        frame            camera        │
                                                       ▼
                                          shaped exactly like a cleaned
                                          sheet, so every downstream
                                          stage is unchanged
```

Cached per prior set: 131 realities share **58** references, ~1 s each.

### Only a horizontal slice is useful

A raw top-down projection is mostly floor and ceiling, which are the same
everywhere. What locates a capture is its walls, so the cloud is cut to a band
around the walker's height — measured against the **nearest camera**, not a
global height, so a ramp doesn't smear it into a wedge.

```
        ·  ·   ·   ·  ·   ·   ·  ·   ·   ·  ·        +0.25 h
     ░░░░░░░░░░░░░░░░ wall band — the only slice kept ░░░░░░
        ●  camera                                            −0.85 h
        │
        │  h = 2.11 m
   ─────┴──────────────────────  floor  ──────╲________
                                                ramp — the band follows it
```

### The rig height is known; the fit often isn't

The floor is found by histogramming the drop from each camera. On a site that
surface is frequently stacked plasterboard or a scaffold deck instead.

| | |
|---|---|
| prior splits with a known scale | 98 |
| implied rig height outside 1.2–3.5 m | **23**, *every one too low* |
| effect of substituting the known 2.11 m | one split went 0.47% → **3.44%** usable wall |

Dropping those would cost a quarter of the reference, so the known height is
substituted and the override is recorded.

**The same failure exists on the current side and is not yet corrected.** Four of
14 sampled realities imply a 4.7–9.0 m rig. That corrupts the wall band *and* the
scale prior — `s = h_prior / h_cloud`, so a doubled `h_cloud` halves the scale,
which is why the measured scale p10 is 0.448 ≈ 1/2.2. Flagged per row as
`rig_implausible`; fixing it is the highest-value open item.

### The prior disagrees with itself

Splits of one capture overlap; where two contain the same photograph they should
place it identically.

| | |
|---|---|
| median disagreement, 43 pairs | **1.41 m** |
| p75 / p90 | 2.22 / 3.71 m |
| worst | 10.78 m |

This is the reference's intrinsic blur, and it showed that a one-metre
consistency filter was discarding ordinary data — 13 of 22 splits on one capture.
The filter now catches only gross mis-registration, at 5 m.

## 5. Two consequences

### The scale constant cancels

Traditionally: divide the monopod's 2.11 m by the camera height in cloud units,
then convert through the paper scale — which over half the buildings never
recorded. Against a prior capture the same monopod appears on both sides:

```
    s = (H / h_cloud) / (H / h_prior) = h_prior / h_cloud
```

One rig measured in two frames. No constant, no paper-scale estimate.
Measured against truth: median **0.993**, p10 0.696, p90 1.267 — and the tail is
the buildings where the prior's own rig height is itself estimated.

### The units are a trap

The prior's transform lands in the target frame, which is metres for some
buildings and inches for others. A reference built "in the target frame" makes
`COARSE_CELL_M = 0.20` mean two different physical distances across one dataset,
**silently** — no exception, just a plausible answer in the wrong place on half
the data. So everything is carried into paper inches inside `align/priors.py`,
every existing constant keeps its meaning, and the result composes back out
unchanged.

## 6. Propose, then dispose

For each candidate scale and heading, stamp the current wall band into a raster
and slide it across the reference at every position at once by FFT. Score has
three terms; only the middle one needed rebuilding.

```
   wall agreement   normalised against local density
 − walk term        the walk must lie where walking is possible  ← rebuilt
 + scale term       distance from the rig-height ratio
   ───────────────────────────────────────────────
   = a score surface, one value per position, per scale, per heading
```

"Where walking is possible" was a flood-fill inward from the drawing's edge —
valid because a floor plan is a **closed outline**. A point cloud is not: the
flood leaks through and marks almost nothing, so the term stops rewarding "inside
the building" and becomes a pure penalty for crossing a wall. The lowest-penalty
position is then blank space, and the search is pushed away from the answer.

The replacement: the prior already tells you where walking is possible, because
someone walked there. Dilate its path into a corridor.

| | |
|---|---|
| flood-fill coverage of a prior reference | **0.08 – 13.3 %** |
| true pose distance from top-1, old term at 0.5 | 16 – 121 m |
| …with the term off | 0.16, 1.90 m |
| median percentile, term off → coverage term | 99.764 → **99.962** |

## 7. Using the answer without using the answer

A prior's transform and the current capture's answer live in files with the same
name. The only thing separating them is which capture the path points at — a
run-time value, invisible to source inspection. So the boundary is two halves:

| module | role | may open |
|---|---|---|
| `align/groundtruth.py` | the **current** capture's answer | scoring only, after the search |
| `align/priors.py` | a **prior**'s transform | a solver input, guarded |
| everything else | the solver | nothing ground-truth-named |

`align/priors.py` imports neither the evaluator nor the case index, so a current
`Case` is unreachable from inside it; `align/io/pairing.py` is the only bridge.
The guard refuses the current split, any sibling of its capture, anything outside
the declared pool, any path resolving to the current directory, and any capture
dated later.

Statically: exactly two modules may name the transform file, asserted as an
**equality** so a third fails loudly. At run time: the guard is exercised against
every reality, and a `sys.addaudithook` watches real file opens during a real
run — the only check that sees a read inside a compiled library.

A side effect worth noting. Tightening this exposed that the solver's trajectory
was synthesised by *un-applying* the answer. Production writes the same poses to
`shots.geojson`; they agree to **1.2 × 10⁻⁹ m** across all 273 walks, so the
synthesis was deleted and the solver now reads no part of the current answer.

## 8. The ceiling

A predicted transform is built on top of the prior's, so the prior's error is
inherited whole. Place the current capture where its own answer says, the prior
where its answer says, and measure the gap: that is what a *perfect* matcher
would still show.

| | |
|---|---|
| median | **0.71 m** |
| p75 / p90 | 1.46 / 3.04 m |
| a perfect match lands under 1 m | 69 / 109 |
| …under 2 m | 87 / 109 |

On 40 of 109 realities a flawless matcher misses a one-metre threshold. Report
`tpr@1m` beside its ceiling, or at two metres, or as error against the ceiling —
otherwise the figure is the previous capture's registration quality wearing this
work's name.

---

# Part two — the seven stages

An end-to-end accuracy figure cannot say what to fix. Nine metres of error could
be a blurry reference, wrong scale, wrong heading, weak position scoring, or no
overlap — each with a different remedy. So each stage **holds four unknowns still
and varies one**.

```
                        reference   scale   heading  position  overlap
  3  e5_prior_quality    TESTED       –        –         –        –
  5  e3_prior_oracle     measured   GIVEN    GIVEN    TESTED   measured
  6  e6_scale_yaw         fixed     TESTED   TESTED   from e3     –
  7  e7_reality_metrics  ─────────── ALL OF IT AT ONCE ───────────────

  1  pytest              is the boundary intact?
  2  check_shots         is the trajectory a real input?
  4  reference_build     construct the target (not a test)
```

## Stage 3 — `e5_prior_quality.py`

**Asks:** before matching anything, does the reference agree with itself?
**Reads no point clouds. ~2 min.**

Result: median **1.41 m**, p90 3.71, max 10.78 over 43 pairs.

*What it means:* a floor, not an error. Nothing downstream can beat it, so it is
the first thing to fix if the result must be sharper than a metre.

## Stage 5 — `e3_prior_oracle.py`

**Asks:** given the right scale and heading for free, does the correct position
score highest? **Runs no search. ~12 min.**

```
  against the floorplan     ├──────────────┬────────────────────────┬──┤
                            19.5th                               98th
  against the prior         ├───────────────────────────────────────┬─┤
                                                               99.96th
  worst ◀─────────────── every candidate position ───────────────▶ best
```

Result: 109 of 131 scored, median percentile **99.962** at `w_free=0.25`,
73/109 above 99.5. Threshold of 99.5 was fixed before the first run. **PASS.**

*What it means:* the scoring function is not the problem. Whatever fails
end-to-end, it is not that the score cannot recognise a correct position.

## Stage 6 — `e6_scale_yaw.py`

**Asks:** take scale and heading back — does the search discover them? **~75 min.**

```
  truth INSIDE the grid          truth OUTSIDE the grid
  ┌───┬───┬───┬───┐              ┌───┬───┬───┬───┐
  │   │   │   │   │              │   │   │   │   │
  ├───┼───┼───┼───┤              ├───┼───┼───┼───┤        ● truth
  │   │ ● │   │   │  truth       │   │   │   │   │   ⋯⋯▶
  ├───┼───┼───┼───┤              ├───┼───┼───┼───┤
  │   │   │   │   │              │   │   │   │   │
  └───┴───┴───┴───┘              └───┴───┴───┴───┘
  76% of realities               24% of realities
  48% ranked first                0% ranked first
  median error 1.44 m             not 5%, not 1% — zero
```

| outcome | realities | top-1 correct | median error |
|---|---:|---:|---:|
| grid held the truth | 91 | 48% | 1.44 m |
| heading missed | 10 | **0%** | 11.35 m |
| scale missed | 24 | **0%** | 39.13 m |

Overall: truth in the grid **76%**, top-1 correct **37%**, median
rank-when-present **0**. Heading is implicated in 70% of failures, scale in 54%.

A related finding: 18 realities had wall directions too weak to propose a
heading, so the search swept all 360°. Perfect coverage, **0% correct** — ninety
near-identical candidates drown the right one. Abstaining would beat sweeping.

*What it means:* **this is the bottleneck.** Enumeration failure and ranking
failure look identical in an accuracy number and have nothing in common as
problems.

## Stage 7 — `e7_reality_metrics.py`

**Asks:** everything varying at once — how far from truth does each photograph
land? **~40 min. Not yet run in full** — only a four-reality smoke test, which is
why no headline accuracy figure appears in this document.

```
  every photo ──▶ horizontal distance ──▶ rmse_xy + tpr@{0.1,0.5,1,1.5,2}
  predicted vs     height kept separate    five thresholds, because one
  true position                            hides the shape
```

Three poses scored per reality:

| pose | what it is |
|---|---|
| `predicted` | what the search chose — the honest end-to-end result |
| `oracle` | the best planar pose — the parameterisation's own limit |
| `ceiling` | the same pose through the prior's registration, inheriting its error |

**Error is horizontal.** `rmse_xy` and every `tpr@X` measure distance in plan;
`rmse_z` sits beside them, never folded in. The search solves a flat four-number
pose and never touches height — z is set afterwards by levelling against the
prior — so a combined figure lets a bad floor estimate masquerade as a bad match.
It did: at the oracle pose, where the horizontal fit is exact by construction, a
3-D number still read as high as **2 m** on nothing but the height estimate.

Levelling by matching camera heights beats landing floor on floor, measured at
the oracle pose over 14 realities: median `rmse_z` **0.077 m** against 0.150 m.

## Stages 1, 2, 4 — the ones that are not measurements

**1 · `pytest`** — the boundary of §7 above, enforced. `1316 passed, 6 skipped`
with a dataset in the environment; `130 passed, 18 skipped` without one, because
the dataset-backed tests skip rather than guess.

**2 · `check_shots.py`** — the trajectory check of §7 above. Must end
`all within 1e-06 m`.

**4 · `reference_build.py`** — constructs the target. Read the tail, not the
count: splits dropped and why, floor fits overridden, references spanning more
than one storey.

---

## Reading the three verdicts together

```
  is the target sharp?     ──▶  1.41 m blur     a floor under everything
  can the score recognise? ──▶  top 0.04%       this part works
  can the search find?     ──▶  76% of the time the bottleneck
```

Same end-to-end symptom, three different causes — and only this decomposition
says which one to spend the next month on.

**The conclusion is the shape, not any single figure.** Effort belongs on the
candidate grid and on prior quality, not on the scoring function. Concretely, in
order:

1. **Fix the current-side floor fit.** It corrupts the wall band, the scale prior
   and the height at once, and is the common root of the scale tail and the worst
   `rmse_z` rows.
2. **Abstain instead of sweeping** when the Manhattan estimate is too weak to
   propose a heading — 18 realities, currently 0%.
3. **Resolve the 90°/180° heading ambiguity**, which the wall term alone cannot.
4. **Filter bad priors.** The 31 poorly-ranked realities have *identical* camera
   overlap to the good ones (1.00 vs 1.00) and differ only in inherited error
   (2.80 m vs 0.50 m).
