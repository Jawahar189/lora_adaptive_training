"""Place recognition between the current capture's panoramas and the prior's.

The prior capture is registered, so every prior panorama has a known place on
the plan. A current panorama that looks like one of them was taken near it --
and that places the current camera to within a few metres from the picture
alone, before any point cloud is correlated. It is a source of information
the search does not have, aimed at what the search gets wrong: which of four
Manhattan headings, and which scale.

Mechanics. A frozen DINOv2 (from the Hugging Face hub; the torch-hub weights
host is unreachable here) turns a panorama, downsampled to 518x266, into one
L2-normalised vector: its CLS token joined with a GeM pool of the patch
tokens. Two captures of the same corridor differ in camera yaw, and for an
equirectangular image a yaw rotation is exactly a circular roll of the
columns -- so a prior image is described at ROLLS evenly spaced rolls and a
match takes the best roll. Repeated structure is disambiguated by the walk:
consecutive current cameras must retrieve consecutive-ish places.

Nothing here names or reads a ground-truth file. Image lists come from
`shots.geojson` (`Trajectory.names`), never from listing `images_processed/`,
which also holds `geo.txt`. Descriptors are cached per split under
`cache_dir`, keyed by model and roll count.
"""
from __future__ import annotations

import multiprocessing as mp
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from align.chain import _umeyama_sim2

IMG_DIR = "images_processed"
DEFAULT_MODEL = "facebook/dinov2-base"
INPUT_HW = (266, 518)          # 19 x 37 patches of 14 px; keeps the 2:1 aspect
ROLLS = 4                      # yaw steps a prior image is described at
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def image_path(rlt_dir, name: str) -> Path:
    return Path(rlt_dir) / IMG_DIR / name


def load_erp(path, hw=INPUT_HW) -> np.ndarray:
    """A panorama as (H, W, 3) uint8 at the network's input size.

    `draft` lets libjpeg decode at a reduced DCT scale, which is most of the
    saving: a full 4096x2048 decode is ~84 ms, this ~50."""
    im = Image.open(path)
    im.draft("RGB", (hw[1] * 2, hw[0] * 2))
    im = im.convert("RGB").resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def roll_yaw(img: np.ndarray, k: int, rolls: int = ROLLS) -> np.ndarray:
    """The same panorama seen after a camera yaw of k/rolls of a turn.
    Works on one image (H, W, 3) or a batch (N, H, W, 3): width is axis -2."""
    return np.roll(img, shift=(img.shape[-2] * k) // rolls, axis=-2)


def gem_pool(desc: np.ndarray, patches: np.ndarray, p: float = 3.0) -> np.ndarray:
    """CLS joined with a GeM pool over the patch tokens, unit length. Pure
    numpy so the pooling is testable without a network."""
    pooled = (np.clip(patches, 1e-6, None) ** p).mean(axis=1) ** (1.0 / p)
    out = np.concatenate([desc, pooled], axis=1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class Backbone:
    """DINOv2 through `transformers`, fp16 on the GPU when there is one.

    `pixel_values` are built by hand: the library's image processor pulls in
    torchvision, which this environment does not have.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, device=None, batch: int = 64):
        import torch
        from transformers import Dinov2Model
        self.torch, self.model_id, self.batch = torch, model_id, batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Dinov2Model.from_pretrained(model_id).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()

    def embed(self, imgs: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8 -> (N, D) float32, unit length."""
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(imgs), self.batch):
                x = (imgs[i:i + self.batch].astype(np.float32) / 255.0 - MEAN) / STD
                x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
                x = x.to(self.device, dtype=torch.half if self.device == "cuda" else torch.float32)
                h = self.model(pixel_values=x).last_hidden_state.float().cpu().numpy()
                out.append(gem_pool(h[:, 0], h[:, 1:]))
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def _cache_path(cache_dir, model_id: str, reality_id: str, rolls: int) -> Path:
    tag = model_id.replace("/", "__")
    return Path(cache_dir) / tag / f"{reality_id}_r{rolls}.npz"


def describe(rlt_dir, names, backbone: Backbone, cache_dir, reality_id: str,
             rolls: int = 1, workers: int = 16) -> np.ndarray:
    """(N, rolls, D) descriptors for `names` in `rlt_dir`, from the cache when
    every name is already there, else computed and cached."""
    path = _cache_path(cache_dir, backbone.model_id, reality_id, rolls)
    names = list(names)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        have = {n: i for i, n in enumerate(z["names"].tolist())}
        if all(n in have for n in names):
            return z["desc"][[have[n] for n in names]].astype(np.float32)
    paths = [str(image_path(rlt_dir, n)) for n in names]
    if workers > 1 and len(paths) > 8:
        with mp.Pool(workers) as pool:
            imgs = np.stack(pool.map(load_erp, paths, chunksize=4))
    else:
        imgs = np.stack([load_erp(p) for p in paths])
    desc = np.stack([backbone.embed(roll_yaw(imgs, k, rolls) if k else imgs)
                     for k in range(rolls)], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, names=np.array(names), desc=desc.astype(np.float16),
             model_id=backbone.model_id, rolls=rolls)
    return desc


def retrieve(query: np.ndarray, pool: np.ndarray, top_k: int = 5):
    """Best `top_k` pool entries per query, over the pool's rolls.

    `query` (Nq, D) unrolled; `pool` (Np, R, D). Returns (indices (Nq, k),
    similarities (Nq, k)), best first."""
    sims = np.einsum("qd,prd->qpr", query, pool).max(axis=2)
    k = min(top_k, sims.shape[1])
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-s, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(s, order, axis=1)


def consistent_anchors(src_xy: np.ndarray, cand_xy: np.ndarray, scale: float,
                       radius: float, window: int = 3, min_support: int = 2):
    """Which retrievals the walk itself corroborates.

    A camera's candidate place is believed when its neighbours along the walk
    (within `window` cameras) have a candidate that sits where the walk says
    it should: about `scale` times the cloud-frame step away, within `radius`.
    Returns (camera index, chosen candidate index) pairs -- the anchors.
    """
    n, k, _ = cand_xy.shape
    anchors = []
    for i in range(n):
        best = None
        for j in range(k):
            support = 0
            for d in range(-window, window + 1):
                m = i + d
                if d == 0 or m < 0 or m >= n:
                    continue
                expected = scale * np.linalg.norm(src_xy[m] - src_xy[i])
                dist = np.linalg.norm(cand_xy[m] - cand_xy[i, j], axis=1)
                if np.any(np.abs(dist - expected) <= radius):
                    support += 1
            if best is None or support > best[0]:
                best = (support, j)
        if best[0] >= min_support:
            anchors.append((i, best[1]))
    return anchors


@lru_cache(maxsize=512)
def _triplet_draws(n: int, iters: int, seed: int) -> np.ndarray:
    """The (iters, 3) triples `ransac_sim2` samples, for a match set of `n`.

    Drawn one at a time because `choice(..., replace=False)` has no batched
    spelling that produces the same sequence, and the sequence is the thing
    that has to be preserved. But it depends on nothing except `(n, iters,
    seed)` -- and inference fits thousands of windows at a fixed seed and
    iteration count, with the match count repeating constantly -- so the draw
    is done once per distinct match count and shared. Read-only: callers index
    with it and must not write through it.
    """
    rng = np.random.default_rng(seed)
    picks = np.array([rng.choice(n, 3, replace=False) for _ in range(iters)])
    picks.setflags(write=False)
    return picks


def _rank_ok(s0: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Which centred triples span the plane, without a bad one taking the batch.

    `matrix_rank` is an SVD, and numpy raises for the WHOLE batch when any one
    matrix fails to converge, so the batch is tried first and only retried one
    at a time if it refuses.
    """
    out = np.zeros(len(s0), dtype=bool)
    index = np.flatnonzero(usable)
    if not index.size:
        return out
    try:
        out[index] = np.linalg.matrix_rank(s0[index]) >= 2
        return out
    except np.linalg.LinAlgError:
        pass
    for i in index:
        try:
            out[i] = bool(np.linalg.matrix_rank(s0[i]) >= 2)
        except np.linalg.LinAlgError:
            out[i] = False
    return out


def _svd_or_skip(cov: np.ndarray, usable: np.ndarray):
    """Batched SVD of 2x2 covariances, degrading a bad matrix to a skipped
    hypothesis rather than an exception.

    One triple whose covariance LAPACK cannot decompose used to raise
    `LinAlgError` out of the batch, out of `ransac_sim2`, out of the matcher and
    out of the capture: `bevaligner.evaluate`'s chain catches at the CAPTURE,
    so all 22 splits of one capture were written `error:` because of a single
    three-point sample (measured on SNPUp7WaDxhcfarLZAG,
    `out/bevaligner/eval/mp_geo_chain.log`). A hypothesis that cannot be
    fitted is exactly what `-1` already means here.

    Skipped entries come back as the identity and zero singular values so the
    arithmetic downstream stays finite; `ok` is what decides their count.
    """
    u = np.tile(np.eye(2), (len(cov), 1, 1))
    d = np.zeros((len(cov), 2))
    vt = np.tile(np.eye(2), (len(cov), 1, 1))
    ok = usable.copy()
    index = np.flatnonzero(usable)
    if not index.size:
        return u, d, vt, ok
    try:
        u[index], d[index], vt[index] = np.linalg.svd(cov[index])
        return u, d, vt, ok
    except np.linalg.LinAlgError:
        pass
    for i in index:
        try:
            u[i], d[i], vt[i] = np.linalg.svd(cov[i])
        except np.linalg.LinAlgError:
            ok[i] = False
    return u, d, vt, ok


def _sim2_hypothesis_counts(src: np.ndarray, dst: np.ndarray, picks: np.ndarray,
                            tol: float, budget: int = 4_000_000):
    """Inliers each three-point hypothesis in `picks` wins, as the loop counted them.

    The vectorised form of the body of `ransac_sim2`'s search: `_umeyama_sim2`
    on `src[pick], dst[pick]`, then `Sim2.apply` over every point. It mirrors
    that pair operation for operation -- the rotation is rebuilt from
    `arctan2` exactly as `Sim2.R` rebuilds it, while the translation keeps the
    SVD's own R, because that is what `_umeyama_sim2` stores -- so the counts
    are bit-identical to the ones the loop produced. A degenerate triple counts
    -1, which is the vectorised spelling of the loop's `continue`.

    Hypotheses are processed in chunks so a large match set cannot turn the
    (iters, n, 2) prediction array into hundreds of megabytes.
    """
    chunk = max(1, budget // max(2 * len(src), 1))
    counts = np.empty(len(picks), dtype=np.int64)
    for start in range(0, len(picks), chunk):
        index = picks[start:start + chunk]
        a, b = src[index], dst[index]                           # (c,3,2)
        with np.errstate(invalid="ignore"):
            mu_s, mu_d = a.mean(1), b.mean(1)
            s0, d0 = a - mu_s[:, None], b - mu_d[:, None]
        # A non-finite coordinate anywhere in a triple makes a covariance no
        # SVD can decompose. It is ONE unusable hypothesis, not a broken run.
        finite = np.isfinite(a).all(axis=(1, 2)) & np.isfinite(b).all(axis=(1, 2))
        rank_ok = _rank_ok(s0, finite)
        cov = np.einsum('kip,kiq->kpq', d0, s0) / 3.0
        cov = np.where(np.isfinite(cov), cov, 0.0)
        u, d, vt, rank_ok = _svd_or_skip(cov, rank_ok)
        sign = np.ones((len(index), 2))
        sign[np.linalg.det(u) * np.linalg.det(vt) < 0, 1] = -1.0
        rotation = (u * sign[:, None, :]) @ vt
        var = (s0 ** 2).sum((1, 2)) / 3.0
        scale = np.where(var > 0, (d * sign).sum(1) / np.where(var > 0, var, 1.0), 1.0)
        translation = mu_d - scale[:, None] * np.einsum('kpq,kq->kp', rotation, mu_s)
        # Sim2 stores a yaw, not a matrix, so `apply` turns it back into one;
        # the round trip is part of the arithmetic the loop did.
        theta = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
        cos, sin = np.cos(theta), np.sin(theta)
        applied = np.empty((len(index), 2, 2))
        applied[:, 0, 0], applied[:, 0, 1] = cos, sin          # already R.T
        applied[:, 1, 0], applied[:, 1, 1] = -sin, cos
        predicted = scale[:, None, None] * (src @ applied) + translation[:, None, :]
        with np.errstate(invalid="ignore"):
            # a skipped hypothesis predicts NaN; NaN <= tol is False, which is
            # the answer wanted, and the warning is noise in a production log
            inside = np.linalg.norm(predicted - dst, axis=-1) <= tol
        counts[start:start + chunk] = np.where(rank_ok, inside.sum(1), -1)
    return counts


def ransac_sim2(src: np.ndarray, dst: np.ndarray, tol: float, bracket=None,
                iters: int = 500, seed: int = 0):
    """A similarity taking `src` onto `dst` that most points agree with.

    Sample three, fit (`align.chain._umeyama_sim2`), count points landing
    within `tol`, refit on the largest agreeing set. `bracket` (lo, hi)
    rejects a fitted scale the rig could not have produced. Returns
    (Sim2, inlier mask) or None.

    The hypotheses are drawn one at a time, so the sequence is the one this
    function has always drawn, but they are SCORED all at once
    (`_sim2_hypothesis_counts`) rather than in a Python loop. The winner is the
    first with the most inliers -- `argmax` takes the first maximum, which is
    what the loop's strict `>` did -- and its mask is then recomputed by the
    scalar path below, so everything after the search is untouched. The loop's
    `best_n == n` early exit needs no equivalent: a hypothesis every point
    agrees with is already maximal, and the first such one still wins.
    """
    src = np.asarray(src, dtype=np.float64)[:, :2]
    dst = np.asarray(dst, dtype=np.float64)[:, :2]
    n = len(src)
    if n < 3 or np.linalg.matrix_rank(src - src.mean(0)) < 2:
        return None
    picks = _triplet_draws(n, iters, seed)
    counts = _sim2_hypothesis_counts(src, dst, picks, tol)
    best = int(counts.argmax())
    if counts[best] < 3:
        return None
    sim = _umeyama_sim2(src[picks[best]], dst[picks[best]])
    best_mask = np.linalg.norm(sim.apply(src) - dst, axis=1) <= tol
    sim = _umeyama_sim2(src[best_mask], dst[best_mask])
    mask = np.linalg.norm(sim.apply(src) - dst, axis=1) <= tol
    if mask.sum() < 3:
        return None
    sim = _umeyama_sim2(src[mask], dst[mask])
    if bracket is not None and not (bracket[0] <= sim.s <= bracket[1]):
        return None
    return sim, mask


def pose_from_retrieval(views, positions, q_desc, pool_desc, pool_xy, brackets,
                        top_k: int, radius: float, tol: float):
    """The planar pose a split's retrieved places imply, over its gravity views.

    `positions` are the split's cameras in the ply frame; each view rotates
    them with its own `R`, so the fit is tried per view and the one most
    cameras agree with wins. Two fits per view: on every camera's top-1 place,
    and on the walk-consistent anchors only. Returns a dict (up_index, sim2,
    n_inliers, inlier_frac, how, n_anchors, top1_sim) or None.
    """
    idx, sims = retrieve(q_desc, pool_desc, top_k)
    cand = pool_xy[idx]
    best = None
    for ui, v in views.items():
        s0, bracket = brackets.get(ui, (float("nan"), None))
        src = (np.asarray(positions) @ v["R"].T)[:, :2]
        fits = []
        f = ransac_sim2(src, cand[:, 0], tol, bracket)
        if f is not None:
            fits.append((f[0], f[1], "raw"))
        anchors = consistent_anchors(src, cand, s0 if np.isfinite(s0) else 1.0, radius)
        if len(anchors) >= 3:
            ai = np.array([a for a, _ in anchors]); aj = np.array([j for _, j in anchors])
            f = ransac_sim2(src[ai], cand[ai, aj], tol, bracket)
            if f is not None:
                mask = np.zeros(len(src), bool); mask[ai[f[1]]] = True
                fits.append((f[0], mask, "anchored"))
        for sim, mask, how in fits:
            n = int(mask.sum())
            if best is None or n > best["n_inliers"]:
                best = dict(up_index=ui, sim2=sim, n_inliers=n,
                            inlier_frac=n / len(src), how=how,
                            n_anchors=len(anchors),
                            top1_sim=float(np.median(sims[:, 0])))
    return best
