"""Panels for one reality's run: the reference, the placement, the alternatives.

Rendered from the intermediates a `RealityRun` already carries, so nothing is
recomputed to draw it. Three panels, answering the three questions worth asking
of a result that looks wrong:

  R1  what was it matched against?     the merged prior BEV and where the prior walked
  R2  where did it end up?             the placed wall band, over the reference
  R3  what else was considered?        the runners-up, warm for rank 0, cooling down

Every title names the reality and says in plain words what the panel shows, so
a PNG found on its own a week later still explains itself.

Ground truth is drawn as a separate layer when it is available and simply
omitted when it is not, so the same panels serve a scored run and an unscored
one.

A fourth, debug-only panel:

  R0  does truth even land somewhere sane?   ground truth's own image poses,
                                              alone, over the full reference

Deliberately excludes the chosen pose -- R2 already overlays chosen-vs-truth,
and a bad reference (e.g. two physically disjoint prior areas merged onto one
raster) is easiest to see with nothing else drawn on top of it. Needs ground
truth, so -- like R2/R3's blue layer -- it exists only on a scored run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from align import viz as V

MAX_PX = 1400            # a reference can be 11M cells; nobody needs that on screen
GT = V.PATH              # blue, the ground-truth walk
PRED = V.RED             # the placed cloud
PRIOR = (120, 132, 146)  # the reference, deliberately quiet behind everything
COVER = (214, 226, 236)


def _fit(mask: np.ndarray, width: int = MAX_PX):
    """Shrink a reference raster for display, keeping thin walls alive."""
    if mask.shape[1] <= width:
        return mask, 1.0
    small = V.downsample_mask(mask, width)
    return small, small.shape[1] / mask.shape[1]


def _who(run) -> str:
    """Which reality this is, for a panel that may be read on its own."""
    c = run.case
    return f"{c.reality_id} · {c.structure.structure_id}"


def _to_px(xy, res, scale, height_px):
    """Local solve-frame inches -> display pixels, y flipped for the screen."""
    q = np.asarray(xy, dtype=np.float64)[:, :2] / res * scale
    return np.stack([q[:, 0], height_px - 1 - q[:, 1]], axis=1)


def ground_truth_panel(run, width: int = MAX_PX):
    """R0 -- truth's own image poses, alone, over the full reference.

    No chosen pose drawn: the question this answers is whether the reference
    itself makes sense for this capture, not how the search did. A walk that
    lands nowhere near any of the reference's own wall mass -- or in a patch
    disjoint from where most of it sits -- points at the reference, not the
    scorer.
    """
    ref = run.reference
    wall = np.flipud(ref.feats.wall)
    small, scale = _fit(wall, width)
    h = small.shape[0]
    img = V.layers(small.shape, (small, PRIOR))
    gt_local = run.geometry["gt_local"]
    img = V.draw_path(img, _to_px(gt_local, ref.res, scale, h), color=GT,
                      width=2, dot=4)
    return V.panel(
        img,
        f"R0 · WHERE TRUTH SAYS THIS CAPTURE BELONGS — debug, no placement drawn  ·  {_who(run)}",
        "blue = ground truth's own per-image poses, over the full merged reference  ·  "
        "a walk sitting apart from the reference's own wall mass means the "
        "reference is the problem, not the ranking",
        width=width)


def reference_panel(run, width: int = MAX_PX):
    """R1 -- the thing being matched against."""
    ref = run.reference
    wall = np.flipud(ref.feats.wall)          # back to y-up for display
    cover = np.flipud(ref.free.wall)
    small, _ = _fit(wall, width)
    cover_s, _ = _fit(cover, width)
    img = V.layers(small.shape, (cover_s, COVER), (small, PRIOR))
    mpi = ref.metres_per_inch
    span = np.array(wall.shape[::-1]) * ref.res * mpi
    return V.panel(
        img,
        f"R1 · WHAT IT IS MATCHED AGAINST — prior capture, bird's-eye  ·  {_who(run)}",
        f"{len(ref.prior_reality_ids)} prior split(s) merged  ·  "
        f"{ref.wall_cells:,} wall cells  ·  {ref.res * mpi:.2f} m/cell  ·  "
        f"{span[0]:.0f} x {span[1]:.0f} m  ·  scale from {ref.scale_source}"
        + ("  ·  THIN" if ref.thin else ""),
        width=width)


def placement_panel(run, width: int = MAX_PX):
    """R2 -- where the capture was put, with the truth beside it when known."""
    ref = run.reference
    wall = np.flipud(ref.feats.wall)
    small, scale = _fit(wall, width)
    h = small.shape[0]
    img = V.layers(small.shape, (small, PRIOR))

    best = run.hypotheses[0]
    v = run.geometry["predicted_view"]
    placed = best.sim2.apply(v["wall_xy"])
    step = max(len(placed) // 30000, 1)
    pts = _to_px(placed[::step], ref.res, scale, h)
    px = pts.astype(int)
    ok = ((px[:, 0] >= 0) & (px[:, 0] < small.shape[1])
          & (px[:, 1] >= 0) & (px[:, 1] < h))
    arr = np.asarray(img).copy()
    arr[px[ok, 1], px[ok, 0]] = PRED
    img = V.Image.fromarray(arr)

    img = V.draw_path(img, _to_px(best.sim2.apply(v["traj_xy"]), ref.res, scale, h),
                      color=PRED, width=2, dot=4)
    gt_local = run.geometry.get("gt_local")
    if gt_local is not None:
        img = V.draw_path(img, _to_px(gt_local, ref.res, scale, h),
                          color=GT, width=2, dot=4)

    m = run.poses.get("predicted", {})
    sub = (f"red = this capture placed by the chosen pose"
           + (", blue = where it truly belongs" if gt_local is not None else "")
           + f"  ·  scale {best.sim2.s:.4f}  ·  heading {best.yaw_deg:.1f} deg  ·  "
           f"score {best.score:.3f}")
    if "rmse_xy" in m:
        sub += (f"  ·  rmse_xy {m['rmse_xy']:.2f} m  ·  rmse_z {m['rmse_z']:.2f} m"
                f"  ·  tpr@1 {m['tpr@1.0']:.2f}  ·  tpr@2 {m['tpr@2.0']:.2f}")
    return V.panel(
        img,
        f"R2 · WHERE THIS CAPTURE WAS PLACED — chosen pose vs truth  ·  {_who(run)}",
        sub, width=width)


def alternatives_panel(run, top_k: int = 5, width: int = MAX_PX):
    """R3 -- the runners-up. A tight cluster means the score chose; a scatter means it did not."""
    ref = run.reference
    wall = np.flipud(ref.feats.wall)
    small, scale = _fit(wall, width)
    h = small.shape[0]
    img = V.layers(small.shape, (small, PRIOR))
    keep = run.hypotheses[:top_k]
    for i, hyp in enumerate(reversed(keep)):          # rank 0 drawn last, on top
        rank = len(keep) - 1 - i
        v = run.geometry["views"].get(hyp.up_index)
        if v is None:
            continue
        img = V.draw_path(img, _to_px(hyp.sim2.apply(v["traj_xy"]), ref.res, scale, h),
                          color=V.rank_color(rank, max(len(keep), 2)),
                          width=3 if rank == 0 else 2, dot=4 if rank == 0 else 0)
    gt_local = run.geometry.get("gt_local")
    if gt_local is not None:
        img = V.draw_path(img, _to_px(gt_local, ref.res, scale, h), color=GT,
                          width=2, dot=4)
    spread = ""
    if len(keep) > 1:
        c = np.array([[hyp.sim2.tx, hyp.sim2.ty] for hyp in keep])
        spread = (f"  ·  top-{len(keep)} spread "
                  f"{np.linalg.norm(c - c[0], axis=1).max() * ref.metres_per_inch:.1f} m")
    return V.panel(
        img,
        f"R3 · WHAT ELSE THE SEARCH CONSIDERED — top {len(keep)} candidates  ·  {_who(run)}",
        f"one walk per candidate, warm = the chosen one, cooling with rank"
        + (", blue = truth" if gt_local is not None else "")
        + spread + "  ·  a tight cluster means the score decided; a scatter means it did not",
        width=width)


def render(run, out_dir, top_k: int = 5, width: int = MAX_PX) -> Path:
    """All panels plus a caption, written under <out_dir>/<reality_id>/.

    R0 only exists on a scored run (needs ground truth) and is written
    separately from the R1-R3 grid: it is a debug aid, not part of the
    normal read-this-in-order sequence `run.png` walks through.
    """
    dest = Path(out_dir) / run.case.reality_id
    has_gt = run.geometry.get("gt_local") is not None
    if has_gt:
        V.save(ground_truth_panel(run, width), dest / "R0_ground_truth.png")

    panels = [reference_panel(run, width), placement_panel(run, width),
              alternatives_panel(run, top_k, width)]
    for name, img in zip(("R1_reference", "R2_placement", "R3_alternatives"),
                         panels):
        V.save(img, dest / f"{name}.png")
    entries = [("prior reference", PRIOR), ("prior coverage", COVER),
               ("this capture, placed", PRED)]
    if has_gt:
        entries.append(("ground truth", GT))
    panels.append(V.legend(entries, width=width))
    V.save(V.grid(panels, cols=1), dest / "run.png")
    return dest
