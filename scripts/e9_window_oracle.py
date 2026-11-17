"""Was the right window ever OFFERED, and if so did the score pick it?

The learned matcher places a capture to 0.76 m median when it lands on the right
part of the prior, and lands there about one time in six. That is a search
failure, not a precision failure, and it splits in two:

  * the window holding the true pose was never among the candidates -- the
    tiling is too coarse, or too few windows survive `max_windows`;
  * it was a candidate and something else outranked it -- the inlier count
    cannot tell right from wrong, and the fix is reranking.

This decides which, and it reads the answer AFTER the fact, exactly as
`scripts/e3_prior_oracle.py` does for the classical solver: the model has
already run, its candidates and its choice are in the eval jsonl, and nothing
here feeds back into a pose.

No model and no point cloud are loaded. `inference.candidate_windows` is
deterministic in (pool, side), the pool comes from the same store the eval used,
and `side` is rebuilt from the cached band and the pool's rig height -- the same
scale cancellation `propose_pose` uses, so the windows come out in the same
order and `window_index` from the jsonl indexes straight into them.

    python scripts/e9_window_oracle.py --eval out/bevaligner/eval/prior_bev_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from align.io import datasets as datasets_mod                    # noqa: E402
from align.io.cases import PairedIndex                           # noqa: E402
from bevaligner.config import REPO_ROOT, load_config           # noqa: E402
from bevaligner.data.density import square_extent              # noqa: E402
from bevaligner.data.datamodule import build_capture_store     # noqa: E402
from bevaligner.inference.propose import (PoseProposalOptions,  # noqa: E402
                                            candidate_windows)
from bevaligner.utils import affine                            # noqa: E402
from bevaligner.utils.gravity import BAND_HI, BAND_LO          # noqa: E402

# A window could have produced the right answer if the walk is substantially
# inside it. Not "contains every point": the training windows never did either,
# and `sample_ground_truth_pairs` drops the queries that fall outside.
INSIDE = 0.5


def truth_and_span(store, reality_id, pool):
    """(the walk's true band in paper inches, its size in paper inches).

    The span is the estimate `propose_pose` makes WITHOUT the answer -- the
    walk's own extent times the prior's rig height over this walk's camera
    height -- so the windows reproduce. The band position is the answer, used
    only to ask afterwards which window held it.
    """
    stored = np.load(store.paths.capture_npz(reality_id), allow_pickle=False)
    heights = stored["band_relative_height"]
    band = stored["band_xy"][(heights > BAND_LO) & (heights < BAND_HI)]
    if len(band) < 2:
        return None, None
    camera_height = float(stored["camera_height_above_floor"])
    low, high = np.percentile(band, 2.5, axis=0), np.percentile(band, 97.5, axis=0)
    span = (float((high - low).max()) * pool.rig_height_in / camera_height
            if camera_height > 0 else float(pool.span))
    scale, theta, tx, ty = (float(v) for v in stored["gt_sim2_params"])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    gt = np.array([[scale * cos_t, -scale * sin_t, tx],
                   [scale * sin_t, scale * cos_t, ty]])
    return affine.apply(gt, band), span


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="an evaluate.py stem")
    ap.add_argument("--config", default="bevaligner/configs/prior_bev_l4.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    config = load_config(REPO_ROOT / args.config)
    index = PairedIndex.load(datasets_mod.resolve(
        ["/mnt/align-data/datasets/manifest_all_19_aug_23_sep_current_with_prior.csv"],
        [str(REPO_ROOT / p) for p in config.data.prior_csv_files], name="oracle"))
    store = build_capture_store(config)
    store.index = index
    by_id = {c.reality_id: c for c in index.current}
    options = PoseProposalOptions(resolution=config.data.eval_resolution)

    records = [json.loads(line) for line in open(f"{args.eval}.jsonl")]
    scored = pd.read_csv(f"{args.eval}.csv").set_index("reality_id")
    rows = []
    for record in records:
        if record.get("status") != "ok":
            continue
        rid = record["reality_id"]
        case = by_id.get(rid)
        pool = store.pool_for_case(case) if case is not None else None
        if pool is None:
            continue
        truth, span = truth_and_span(store, rid, pool)
        if truth is None:
            continue
        windows = candidate_windows(pool, options.search_zoom_out * span, options)
        holds = np.array([
            float(((truth[:, 0] >= w[0]) & (truth[:, 0] < w[2])
                   & (truth[:, 1] >= w[1]) & (truth[:, 1] < w[3])).mean()) >= INSIDE
            for w in windows])
        # the per-window inlier counts the model produced, in window order
        inliers = np.zeros(len(windows))
        for cand in record.get("candidates", []):
            i = cand.get("window_index")
            if i is not None and i < len(windows):
                inliers[i] = max(inliers[i], cand.get("num_inliers") or 0)
        chosen = record.get("window")
        order = np.argsort(-inliers)                       # how the score ranked them
        truth_rank = (int(np.where(holds[order])[0][0]) + 1) if holds.any() else None
        rows.append(dict(
            reality_id=rid, n_windows=len(windows), offered=bool(holds.any()),
            chose_truth=bool(chosen is not None and chosen < len(holds) and holds[chosen]),
            truth_rank=truth_rank,
            rmse_xy=float(scored.rmse_xy.get(rid, np.nan)),
            good=bool(scored["tpr@1.0"].get(rid, 0) > 0.5)))
        if args.limit and len(rows) >= args.limit:
            break

    d = pd.DataFrame(rows)
    if args.out:
        d.to_csv(args.out, index=False)
    n = len(d)
    print(f"\n{n} placed captures examined\n")
    print(f"  windows offered per capture: median {d.n_windows.median():.0f}  "
          f"max {d.n_windows.max()}")
    print(f"  a window HOLDING the true pose was offered : "
          f"{d.offered.sum():>4} / {n}  ({d.offered.mean():.1%})")
    off = d[d.offered]
    print(f"  ... and the score chose it                 : "
          f"{d.chose_truth.sum():>4} / {len(off)}  "
          f"({d.chose_truth.sum()/max(len(off),1):.1%} of those offered)")
    print(f"  ... and it ranked 1st by inliers           : "
          f"{(off.truth_rank==1).sum():>4} / {len(off)}  "
          f"({(off.truth_rank==1).mean():.1%})")
    if len(off):
        print(f"\n  where the true window ranked, when offered: "
              f"median {off.truth_rank.median():.0f}  "
              f"p90 {off.truth_rank.quantile(.9):.0f}")
    print(f"\n  of the {d.good.sum()} captures placed within 1 m, "
          f"{d[d.good].offered.mean():.0%} had the true window offered")
    print(f"  of the {(~d.good).sum()} that were not, "
          f"{d[~d.good].offered.mean():.0%} did\n")
    print("VERDICT: " + (
        "the search rarely offers the right window -- widen the tiling"
        if d.offered.mean() < 0.6 else
        "the right window is usually offered but outranked -- rerank"
        if d.chose_truth.sum() / max(len(off), 1) < 0.6 else
        "the search is fine; the error is in the fit"))


if __name__ == "__main__":
    main()
