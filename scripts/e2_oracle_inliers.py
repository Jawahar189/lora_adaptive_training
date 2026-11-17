"""E2 -- is the evidence even present?

Places each cloud at its GROUND-TRUTH pose and counts how many of its wall
segments have a counterpart on the sheet. This is what a perfect search would
score, so it separates "our algorithm is bad" from "the information is not
there" -- a distinction you cannot afford to confuse on a 132-example dataset.

Falsified if the median oracle inlier count is under ~8, or if more than a
third of splits fall below 4.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import sheet as sheetmod
from align.frames import decompose_tm, design_scale, target_to_paper
from align.gravity import CAMERA_HEIGHT_M, floor_below_trajectory
from align.groundtruth import gt_trajectory, load_tm
from align.io.cases import CaseIndex
from align.io.ply import read_ply
from align.segments import detect_segments, match_inliers, rasterize

TOL_M = 0.30          # how close a cloud wall must sit to plan line-work
# The ground truth's own rotation is imprecise: comparing the cloud's Manhattan
# direction to the sheet's AT THE GT POSE gives median 1.6 deg but p90 9.7 and
# max 14.5 over dataset_28. A 5 deg gate would therefore reject correct matches
# on the strength of ground-truth error alone.
ANG_TOL_DEG = 10.0
TOL_PX = 6.0          # pixels per tolerance radius; the raster resolution
                      # follows from this. At 2 px neither the distance nor the
                      # orientation of a wall can be measured.
MAX_RASTER_PX = 4000
E1_PATH = Path(__file__).resolve().parents[1] / "out" / "e1_all132.jsonl"


def structure_scales(index) -> dict:
    """Metres per paper inch for every structure.

    Exact from design_tm where the structure is georeferenced; otherwise the
    camera-height anchor inverted (E1), pooled across all splits of the
    structure since the drawing scale is a property of the sheet, not the walk.
    """
    out, pooled = {}, {}
    if E1_PATH.exists():
        for line in E1_PATH.read_text().splitlines():
            r = json.loads(line)
            if r.get("height_target") and not r.get("georeferenced") \
                    and not r.get("sign_ambiguous"):
                pooled.setdefault(r["structure"], []).append(
                    CAMERA_HEIGHT_M / r["height_target"])
    for case in index:
        st = case.structure
        if st.key in out:
            continue
        if st.design_tm is not None:
            out[st.key] = design_scale(st.design_tm)
        else:
            vals = pooled.get(st.key)
            out[st.key] = float(np.median(vals)) if vals else None
    return out


def sheet_patch(feats, geom, bounds, res, shape):
    """The sheet's wall mask resampled onto the cloud raster's grid.

    Downsampling thin line-work, so pool with `any` rather than nearest: a
    one-pixel wall must survive.
    """
    x0, y0, x1, y1 = bounds
    p0 = geom.paper_to_pixel(np.array([[x0, y1]]))[0]      # top-left in pixels
    p1 = geom.paper_to_pixel(np.array([[x1, y0]]))[0]      # bottom-right
    c0 = np.clip(np.floor(p0).astype(int), 0, None)
    c1 = np.array([min(int(np.ceil(p1[0])), feats.wall.shape[1]),
                   min(int(np.ceil(p1[1])), feats.wall.shape[0])])
    if c1[0] <= c0[0] or c1[1] <= c0[1]:
        return np.zeros(shape, bool)
    crop = feats.wall[c0[1]:c1[1], c0[0]:c1[0]].astype(np.float32)
    h, w = shape
    import cv2
    pooled = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    # INTER_AREA averages; anything non-zero means the cell contained wall ink.
    return np.flipud(pooled > 0)                            # paper y is up


def run_case(case, mpi_sheet, stride=3):
    st = case.structure
    gt = load_tm(case.rlt_dir)
    M = decompose_tm(gt, st.design_tm)                      # ply -> paper inches

    cloud = read_ply(case.cloud_path, stride=stride, min_source_frames=2)
    P = cloud.xyz @ M[:3, :3].T + M[:3, 3]
    C = target_to_paper(gt_trajectory(case.rlt_dir).positions, st.design_tm)

    floor = floor_below_trajectory(P, C)
    if not np.isfinite(floor.height) or floor.height <= 0:
        return dict(error="no floor mode")
    h = floor.height                                        # inches

    # Band each point relative to its NEAREST camera, so a walk that changes
    # level does not smear the band.
    cam = cKDTree(C[:, :2])
    _, nn = cam.query(P[:, :2])
    rel = P[:, 2] - C[nn, 2]
    band = (rel > -0.85 * h) & (rel < 0.25 * h)
    W = P[band][:, :2]
    if len(W) < 200:
        return dict(error="empty wall band")

    # Sheet scale: exact where georeferenced, else the structure-level
    # consensus from E1, falling back to this split's own camera height.
    mpi = mpi_sheet or (design_scale(st.design_tm) if st.design_tm is not None
                        else CAMERA_HEIGHT_M / h)
    tol_in = TOL_M / mpi
    res = tol_in / TOL_PX

    lo = W.min(0) - 0.5
    hi = W.max(0) + 0.5
    if max((hi - lo) / res) > MAX_RASTER_PX:
        res = float(max((hi - lo)) / MAX_RASTER_PX)
    ras = rasterize(W, res=res, bounds=(lo[0], lo[1], hi[0], hi[1]), min_count=1)

    segs = detect_segments(ras.mask, min_len_px=8.0)
    plan = sheet_patch(sheetmod.extract_cached(st.sheet_path, st.paper_w,
                                               st.paper_h, st.key, mpi),
                       st.geometry(), (lo[0], lo[1], hi[0], hi[1]), res,
                       ras.mask.shape)
    inl, frac = match_inliers(segs, plan, tol_px=max(tol_in / res, 1.5),
                              ang_tol_deg=ANG_TOL_DEG)
    # Position-only variant, to separate "the wall is not there" from
    # "the ground truth's rotation is off".
    inl_pos, _ = match_inliers(segs, plan, tol_px=max(tol_in / res, 1.5),
                               ang_tol_deg=90.0)
    return dict(
        oracle_inliers_pos_only=int(inl_pos.sum()),
        n_cloud_segments=int(len(segs)),
        oracle_inliers=int(inl.sum()),
        inlier_fraction=float(inl.mean()) if len(segs) else 0.0,
        plan_ink_fraction=float(plan.mean()),
        cloud_ink_fraction=float(ras.mask.mean()),
        metres_per_inch=float(mpi), tol_inches=float(tol_in), res_inches=res,
        raster_px=[int(ras.mask.shape[1]), int(ras.mask.shape[0])],
        camera_height_in=float(h),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--root", default=None,
                    help="restrict to one data root, e.g. data or data_v")
    ap.add_argument("--out", default="out/e2_oracle.jsonl")
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    index = CaseIndex.load(args.csv) if args.csv else CaseIndex.load_all()
    if args.root:
        index = CaseIndex([c for c in index if c.structure.root == args.root])
        if not len(index):
            raise SystemExit(f"no cases under root {args.root!r}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scales = structure_scales(index)
    unknown = [k for k, v in scales.items() if v is None]
    if unknown:
        print(f"no sheet scale for {len(unknown)} structures, "
              f"falling back per split: {unknown}")
    rows = []
    with open(args.out, "w") as fh:
        for i, case in enumerate(index, 1):
            base = dict(reality_id=case.reality_id, structure=case.structure.key,
                        status=case.structure.status,
                        georeferenced=case.structure.georeferenced)
            try:
                base.update(run_case(case, scales.get(case.structure.key),
                                     args.stride))
            except Exception as exc:                          # noqa: BLE001
                base["error"] = repr(exc)
            rows.append(base)
            fh.write(json.dumps(base) + "\n")
            fh.flush()
            print(f"[{i:3d}/{len(index)}] {case.reality_id} "
                  f"segs={base.get('n_cloud_segments','-')} "
                  f"inliers={base.get('oracle_inliers','-')} "
                  f"frac={base.get('inlier_fraction',0):.2f} "
                  f"{base.get('error','')}", flush=True)

    summarize(rows)


def summarize(rows):
    ok = [r for r in rows if "oracle_inliers" in r]
    inl = np.array([r["oracle_inliers"] for r in ok])
    pos = np.array([r["oracle_inliers_pos_only"] for r in ok])
    print("\n" + "=" * 78)
    print("E2 RESULT -- oracle inlier count at the ground-truth pose")
    if len(inl):
        print(f"  n={len(inl)}  p10 {np.percentile(inl,10):.0f}  "
              f"median {np.median(inl):.0f}  p90 {np.percentile(inl,90):.0f}  "
              f"max {inl.max()}")
        print(f"  under 4 inliers (unsolvable by any geometric method): "
              f"{(inl < 4).sum()}/{len(inl)} "
              f"({(inl < 4).mean():.0%}; falsified above 33%)")
        print(f"  median >= 8: {'PASS' if np.median(inl) >= 8 else 'FAIL'}")
        print(f"  position-only (angle gate off): median {np.median(pos):.0f}, "
              f"under 4: {(pos < 4).sum()}/{len(pos)}")
        gap = (pos >= 4) & (inl < 4)
        print(f"  wall present but orientation disagrees: {gap.sum()} splits "
              f"-- ground-truth rotation error, not missing evidence")
    errs = [r for r in rows if "error" in r]
    print(f"  errored: {len(errs)}/{len(rows)}")

    print("\n  per structure (median inliers, n splits):")
    bys = {}
    for r in ok:
        bys.setdefault(r["structure"], []).append(r["oracle_inliers"])
    byp = {}
    for r in ok:
        byp.setdefault(r["structure"], []).append(r["oracle_inliers_pos_only"])
    for k, v in sorted(bys.items(), key=lambda kv: np.median(kv[1])):
        mp = np.median(byp[k])
        flag = "  <-- NO EVIDENCE" if mp < 4 else ("  (rotation)" if np.median(v) < 4 else "")
        print(f"    {k:30s} angle {np.median(v):6.1f}   pos-only {mp:6.1f}  "
              f"n={len(v):3d}{flag}")
    print("=" * 78)


if __name__ == "__main__":
    main()
