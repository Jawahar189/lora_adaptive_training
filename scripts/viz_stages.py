"""Render every pipeline stage for one split, as individual panels and a sheet.

  python scripts/viz_stages.py --reality RLT1788196494610     # one split
  python scripts/viz_stages.py --root data                    # every split under data/
  python scripts/viz_stages.py --all                          # all 132

Stages
  S1-S6  sheet: raw -> ink -> line-work kept -> clutter masks -> wall pairs -> viewports
  C1-C4  cloud: raw top-down -> gravity + floor -> wall band -> wall raster
  M1-M2  match at the ground-truth pose: overlay, then segment inliers
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import sheet as S
from align import viz as V
from align.frames import decompose_tm, design_scale, target_to_paper
from align.gravity import (CAMERA_HEIGHT_M, estimate_gravity,
                           floor_below_trajectory, trajectory_extent)
from align.groundtruth import gt_trajectory, load_tm
from align.io.shots import load_shots
from align.io.cases import CaseIndex
from align.io.ply import read_ply
from align.segments import detect_segments, match_inliers, orientation_map, rasterize
from e2_oracle_inliers import TOL_M, TOL_PX, sheet_patch, structure_scales

W = 900


def sheet_stages(st, mpi, out, traj_paper):
    """The sheet cleaning pipeline, stage by stage, at full resolution."""
    with Image.open(st.sheet_path) as im:
        gray = np.asarray(im.convert("L"))
    ppi = st.geometry().ppi
    panels = []

    raw = Image.fromarray(gray)
    panels.append(V.panel(raw, "S1 · Raw sheet",
                          f"{gray.shape[1]}x{gray.shape[0]} px · {ppi:.0f} ppi · "
                          f"{st.paper_w:g}x{st.paper_h:g} in", W))

    ink = S._binarize(gray)
    panels.append(V.panel(V.mask_rgb(V.downsample_mask(ink, W)), "S2 · Ink",
                          f"{ink.mean():.2%} of sheet", W))

    keep, glyphs = S._drop_small_components(ink, ppi)
    panels.append(V.panel(
        V.layers(V.downsample_mask(ink, W).shape,
                 (V.downsample_mask(glyphs, W), (214, 219, 225)),
                 (V.downsample_mask(keep, W), V.INK)),
        "S3 · Glyphs removed",
        f"text/ticks under {S.GLYPH_MAX}\" dropped (grey)", W))

    run_px = max(int(S.MIN_LINE_RUN_M / mpi * ppi), 2)
    long_ink = (S._runs_at_least(keep, run_px, axis=1)
                | S._runs_at_least(keep, run_px, axis=0))
    text = S._text_blocks(glyphs, ppi, long_ink)
    halo = max(int(S.TEXT_HALO * ppi), 3)
    text = cv2.dilate(text.astype(np.uint8), np.ones((halo, halo), np.uint8)).astype(bool)
    dwin = max(int(S.DENSITY_WIN * ppi), 8)
    dense = cv2.boxFilter(keep.astype(np.float32), -1, (dwin, dwin)) > S.DENSITY_MAX
    panels.append(V.panel(
        V.layers(V.downsample_mask(long_ink, W).shape,
                 (V.downsample_mask(long_ink, W), V.INK),
                 (V.downsample_mask(text, W), (240, 190, 185)),
                 (V.downsample_mask(dense, W), (250, 226, 170))),
        "S4 · Long runs, clutter flagged",
        f"runs >= {S.MIN_LINE_RUN_M}m = {S.MIN_LINE_RUN_M/mpi:.3f}\" · "
        f"pink = text panel, amber = dense", W))

    wall = long_ink & ~text & ~dense
    pairs = S._double_line_walls(wall, ppi, mpi)
    panels.append(V.panel(
        V.layers(V.downsample_mask(wall, W).shape,
                 (V.downsample_mask(wall, W), (205, 211, 218)),
                 (V.downsample_mask(pairs, W), V.INK)),
        "S5 · Double-line wall pairs",
        f"gap {S.WALL_GAP_MIN_M}-{S.WALL_GAP_MAX_M}m · grey = single lines rejected", W))

    feats = S.extract_cached(st.sheet_path, st.paper_w, st.paper_h, st.key, mpi)
    base = V.mask_rgb(V.downsample_mask(feats.wall, W)).convert("RGB")
    scale = W / feats.wall.shape[1]
    d = ImageDraw.Draw(base)
    for i, v in enumerate(feats.viewports):
        d.rectangle([v.x0 * scale, v.y0 * scale, v.x1 * scale, v.y1 * scale],
                    outline=V.BLUE, width=2)
    g = st.geometry()
    px = g.paper_to_pixel(traj_paper[:, :2]) * scale
    for x, y in px:
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=V.RED)
    panels.append(V.panel(base, "S6 · Wall map + viewports",
                          f"{feats.wall.mean():.3%} retained · blue = viewports · "
                          f"red = this split's walk", W))
    return panels, feats


def cloud_stages(case, st, mpi, out):
    cloud = read_ply(case.cloud_path, stride=2, min_source_frames=2)
    traj_ply = load_shots(case.rlt_dir)
    panels = []

    # C1 -- raw cloud, arbitrary frame, no gravity correction
    xy = cloud.xyz[:, :2]
    r0 = rasterize(xy, res=(xy.max(0) - xy.min(0)).max() / 700, min_count=1)
    c1 = V.mask_rgb(r0.mask[::-1])
    c1 = V.draw_path(c1, V.raster_to_display(traj_ply.positions[:, :2], r0.origin,
                                             r0.res, r0.mask.shape[0], flipped=False))
    panels.append(V.panel(c1, "C1 · Raw cloud, top-down",
                          f"{len(cloud):,} pts · arbitrary frame, no gravity · "
                          f"blue = ground-truth walk", W))

    # C2 -- gravity applied, coloured by height, trajectory drawn
    g = estimate_gravity(cloud.xyz, traj_ply.positions)
    P = cloud.xyz @ g.R.T
    C = traj_ply.positions @ g.R.T
    f = floor_below_trajectory(P, C)
    res = (P[:, :2].max(0) - P[:, :2].min(0)).max() / 700
    lo = P[:, :2].min(0) - res
    hi = P[:, :2].max(0) + res
    shape = (int((hi[1] - lo[1]) / res) + 1, int((hi[0] - lo[0]) / res) + 1)
    ix = ((P[:, 0] - lo[0]) / res).astype(int).clip(0, shape[1] - 1)
    iy = ((P[:, 1] - lo[1]) / res).astype(int).clip(0, shape[0] - 1)
    zmax = np.zeros(shape, np.float32) - 1e9
    np.maximum.at(zmax, (iy, ix), P[:, 2])
    hit = zmax > -1e8
    zz = np.zeros(shape, np.float32)
    if hit.any():
        v = zmax[hit]
        lo_z, hi_z = np.percentile(v, [2, 98])
        zz[hit] = np.clip((zmax[hit] - lo_z) / max(hi_z - lo_z, 1e-9), 0, 1)
    ramp = (np.stack([1 - zz * 0.85, 1 - zz * 0.55, 1 - zz * 0.15], -1) * 255).astype(np.uint8)
    ramp[~hit] = 255
    imgc = Image.fromarray(ramp[::-1])
    dc = ImageDraw.Draw(imgc)
    tx = (C[:, 0] - lo[0]) / res
    ty = shape[0] - (C[:, 1] - lo[1]) / res
    dc.line([tuple(p) for p in np.stack([tx, ty], 1)], fill=V.RED, width=2)
    panels.append(V.panel(imgc, "C2 · Gravity corrected",
                          f"up via {g.method} · planarity {g.planarity:.4f}"
                          f"{' · SIGN AMBIGUOUS' if g.sign_ambiguous else ''} · "
                          f"red = walk", W))

    # C3 -- height histogram under the trajectory, with the floor marked
    drop = []
    tree = cKDTree(P[:, :2])
    L = trajectory_extent(C)
    for i in range(0, len(C), max(1, len(C) // 120)):
        idx = tree.query_ball_point(C[i, :2], 0.06 * L)
        if len(idx) >= 40:
            drop.append(C[i, 2] - P[idx, 2])
    dd = np.concatenate(drop) if drop else np.array([0.0])
    span = (-0.6 * f.height, 2.0 * f.height)
    hb, edges = np.histogram(dd, bins=140, range=span)
    hi_img = Image.new("RGB", (W, 250), (255, 255, 255))
    dh = ImageDraw.Draw(hi_img)
    x0, x1 = 40, W - 30
    if hb.max() > 0:
        for i, v in enumerate(hb):
            x = x0 + i * (x1 - x0) / len(hb)
            bh = v / hb.max() * 170
            dh.rectangle([x, 205 - bh, x + (x1 - x0) / len(hb) - 1, 205],
                         fill=(150, 160, 172))
    fx = x0 + (f.height - span[0]) / (span[1] - span[0]) * (x1 - x0)
    dh.line([(fx, 16), (fx, 209)], fill=V.RED, width=2)
    # The camera height is the ONE metric constant available, so the arrow runs
    # from it to the scale, not the other way round.
    lines = [f"floor at h = {f.height:.4f} cloud units",
             f"assume {CAMERA_HEIGHT_M} m rig height",
             f"→ scale = {CAMERA_HEIGHT_M / f.height:.3f} m per cloud unit"]
    tw = max(dh.textlength(t, font=V._font(13)) for t in lines)
    lx = fx + 8 if fx + tw + 16 < W else fx - tw - 10
    for i, t in enumerate(lines):
        dh.text((lx, 18 + i * 19), t, font=V._font(13, i == 0), fill=V.RED)
    dh.line([(x0, 205), (x1, 205)], fill=V.MUTED, width=1)
    dh.text((x0, 214), "drop below camera, cloud units  →", font=V._font(12),
            fill=(110, 118, 129))
    panels.append(V.panel(hi_img, "C3 · Local floor under the walk",
                          f"beyond={f.beyond:.4f} (a floor has ~0 mass past it) · "
                          f"rel MAD {f.rel_mad:.3f}", W))
    return panels, g, f, cloud, traj_ply


def match_stages(case, st, mpi, feats, out):
    """Everything from here uses the GROUND-TRUTH pose -- this is the oracle view."""
    M = decompose_tm(load_tm(case.rlt_dir), st.design_tm)
    cloud = read_ply(case.cloud_path, stride=2, min_source_frames=2)
    P = cloud.xyz @ M[:3, :3].T + M[:3, 3]
    C = target_to_paper(gt_trajectory(case.rlt_dir).positions, st.design_tm)
    f = floor_below_trajectory(P, C)
    h = f.height
    nn = cKDTree(C[:, :2]).query(P[:, :2])[1]
    rel = P[:, 2] - C[nn, 2]
    band = (rel > -0.85 * h) & (rel < 0.25 * h)
    panels = []

    tol_in = TOL_M / mpi
    res = tol_in / TOL_PX
    lo = P[band][:, :2].min(0) - 0.5
    hi = P[band][:, :2].max(0) + 0.5
    if max((hi - lo) / res) > 4000:
        res = float(max(hi - lo) / 4000)
    ras = rasterize(P[band][:, :2], res=res, bounds=(lo[0], lo[1], hi[0], hi[1]),
                    min_count=1)
    allr = rasterize(P[:, :2], res=res, bounds=(lo[0], lo[1], hi[0], hi[1]), min_count=1)
    band_img = V.layers(ras.mask.shape,
                        (allr.mask[::-1], (216, 221, 227)),
                        (ras.mask[::-1], V.INK))
    band_img = V.draw_path(band_img, V.raster_to_display(
        C[:, :2], (lo[0], lo[1]), res, ras.mask.shape[0], flipped=False))
    panels.append(V.panel(band_img, "C4 · Wall band selected",
                          "band -0.85h..+0.25h above local floor · grey = all points · "
                          "blue = ground-truth walk", W))

    segs = detect_segments(ras.mask, min_len_px=8.0)
    plan = sheet_patch(feats, st.geometry(), (lo[0], lo[1], hi[0], hi[1]), res,
                       ras.mask.shape)
    inl, frac = match_inliers(segs, plan, tol_px=max(tol_in / res, 1.5), ang_tol_deg=10.0)
    pos, _ = match_inliers(segs, plan, tol_px=max(tol_in / res, 1.5), ang_tol_deg=90.0)

    ov = V.layers(ras.mask.shape,
                  (plan, (150, 158, 168)),
                  (ras.mask, (200, 57, 43)),
                  (plan & ras.mask, (47, 122, 90)))
    ov = V.draw_path(ov, V.raster_to_display(C[:, :2], (lo[0], lo[1]), res,
                                             ras.mask.shape[0], flipped=True))
    panels.append(V.panel(ov.transpose(Image.FLIP_TOP_BOTTOM),
                          "M1 · Cloud over plan, at ground-truth pose",
                          "grey = sheet wall · red = cloud · green = agreeing · "
                          "blue = ground-truth walk", W))

    sc = W / max(ras.mask.shape[1], 1)
    base = V.layers(ras.mask.shape, (plan, (218, 223, 229)))
    base = base.resize((W, max(1, round(ras.mask.shape[0] * sc))), Image.NEAREST)
    colors = [V.GREEN if inl[i] else (V.AMBER if pos[i] else V.RED)
              for i in range(len(segs))]
    base = V.draw_path(base, V.raster_to_display(
        C[:, :2], (lo[0], lo[1]), res, ras.mask.shape[0], flipped=True, scale=sc),
        width=max(int(sc * 1.5), 2))
    seg_img = V.draw_segments(base, segs, sc, colors, width=2)
    panels.append(V.panel(seg_img.transpose(Image.FLIP_TOP_BOTTOM),
                          "M2 · Segment verdict",
                          f"{len(segs)} segments · {int(inl.sum())} inliers · "
                          f"{int(pos.sum())} position-only", W))
    panels.append(V.panel(V.legend([("inlier (position + angle)", V.GREEN),
                                    ("position only — rotation error", V.AMBER),
                                    ("no match", V.RED),
                                    ("ground-truth walk", V.PATH)], W, 34),
                          "", "", W, bar=0))
    return panels, int(len(segs)), int(inl.sum()), int(pos.sum())


PANEL_NAMES = ["S1_raw", "S2_ink", "S3_glyphs", "S4_runs", "S5_pairs", "S6_wall",
               "C1_raw_cloud", "C2_gravity", "C3_floor", "C4_band", "M1_overlay",
               "M2_segments", "M3_legend"]


def render(case, mpi, out_root, contact=True):
    st = case.structure
    out = Path(out_root) / case.reality_id
    out.mkdir(parents=True, exist_ok=True)
    traj_paper = target_to_paper(gt_trajectory(case.rlt_dir).positions, st.design_tm)

    sp, feats = sheet_stages(st, mpi, out, traj_paper)
    cp, *_ = cloud_stages(case, st, mpi, out)
    mp, nseg, ninl, npos = match_stages(case, st, mpi, feats, out)

    for name, p in zip(PANEL_NAMES, sp + cp + mp):
        V.save(p, out / f"{name}.png")
    if contact:
        V.save(V.grid(sp + cp + mp[:3], cols=2), out / "contact_sheet.png")
    return nseg, ninl, npos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reality", nargs="+", help="one or more reality ids")
    g.add_argument("--root",
                   help="every split under this data folder")
    g.add_argument("--all", action="store_true", help="all 132 splits")
    ap.add_argument("--out", default="out/stages")
    ap.add_argument("--no-contact-sheet", "--no-contact", dest="no_contact_sheet",
                    action="store_true",
                    help="skip contact_sheet.png, the single image tiling all 13 "
                         "panels together (that file is most of the output size)")
    args = ap.parse_args()

    index = CaseIndex.load_all()
    if args.reality:
        wanted = set(args.reality)
        cases = [c for c in index if c.reality_id in wanted]
        missing = wanted - {c.reality_id for c in cases}
        if missing:
            raise SystemExit(f"unknown reality ids: {sorted(missing)}")
    elif args.root:
        cases = [c for c in index if c.structure.root == args.root]
        if not cases:
            raise SystemExit(
                f"no cases under root {args.root!r}; this index has "
                f"{index.roots()}")
    else:
        cases = list(index)
    # Group by structure so each sheet is extracted once and stays cached.
    cases.sort(key=lambda c: (c.structure.key, c.reality_id))

    scales = structure_scales(index)
    print(f"rendering {len(cases)} splits across "
          f"{len({c.structure.key for c in cases})} structures -> {args.out}\n")
    ok = failed = 0
    for i, case in enumerate(cases, 1):
        st = case.structure
        mpi = scales.get(st.key) or (design_scale(st.design_tm)
                                     if st.design_tm is not None else 3.0)
        try:
            nseg, ninl, npos = render(case, mpi, args.out, not args.no_contact_sheet)
            ok += 1
            print(f"[{i:3d}/{len(cases)}] {case.reality_id} {st.key:26s} "
                  f"mpi={mpi:5.2f} segments={nseg:4d} inliers={ninl:4d} "
                  f"pos-only={npos:4d}", flush=True)
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"[{i:3d}/{len(cases)}] {case.reality_id} {st.key:26s} "
                  f"FAILED {exc!r}"[:140], flush=True)
    print(f"\ndone: {ok} rendered, {failed} failed -> {args.out}")


if __name__ == "__main__":
    main()
