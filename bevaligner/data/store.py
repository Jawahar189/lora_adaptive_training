"""The capture store: CSV rows in, training records out, with a cache in between.

The dataset is driven by a pair of CSVs -- current rows and prior rows, joined
on `prior_capture_id -> capture_id` by `align.io.cases.PairedIndex`.
`CaptureStore.ensure_prepared(cases, index)` makes sure every row's slow work
exists on disk, on BOTH sides (a prior split needs the same .npz a current row
does), corrects the prior rig heights the floor fit got wrong, gates each
current row on how much of its walk the prior actually covered, and then
`capture_records(cases)` hands back the rows that survived, in CSV order.

Nothing here is keyed on a dataset name; the same cache serves any CSV pair
whose rows point at the same captures.
"""
from __future__ import annotations

import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from align.frames import design_scale

from . import prior_bev
from .preprocess import CachePaths, CaptureResult, prepare_capture, write_manifest

# `align/reference.py`'s monopod, the length every prior threshold is measured in.
CAMERA_HEIGHT_M = 2.11


class CaptureStore:
    def __init__(self, cache_dir: str | Path, stride: int = 1, min_source_frames: int = 2,
                 max_residual_m: float = 0.5, data_config=None):
        from ..config import DataConfig
        self.paths = CachePaths(cache_dir)
        self.stride, self.min_source_frames = stride, min_source_frames
        self.max_residual_m = max_residual_m
        self.data = data_config if data_config is not None else DataConfig()
        self.pools = prior_bev.PriorPoolStore(
            self.paths, cache_size=self.data.prior_pool_cache,
            mosaic=self.data.prior_mosaic, mosaic_cell_m=self.data.prior_mosaic_cell_m)
        self.index = None
        self._pool_members: dict[tuple, tuple] = {}
        self._overlap_masks: dict[str, np.ndarray] = {}
        self._bboxes: dict[str, object] = {}
        self._bands: dict[str, object] = {}
        self._occupancy: dict[str, float] = {}

    # ------------------------------------------------------------------ cache
    def capture_result(self, reality_id: str) -> CaptureResult | None:
        path = self.paths.capture_json(reality_id)
        return CaptureResult(**json.loads(path.read_text())) if path.exists() else None

    def _save_result(self, result: CaptureResult) -> None:
        self.paths.capture_json(result.reality_id).write_text(
            json.dumps(dataclasses.asdict(result)))

    # ------------------------------------------------------------------ pairing
    def pool_members(self, case) -> tuple:
        """(reality_id, capture_id, capture_date) for every prior split this
        case may see: cloud-bearing, and never a sibling of its own capture.

        The sibling filter is the one that matters. 1,457 capture ids appear on
        both sides of these manifests, and a sibling split's registration is as
        much the answer as the case's own -- `align/io/cases.py:337` explains
        why this cannot be answered by "it came out of the prior index".

        Dates are NOT filtered. The two sources in the manifests -- the current
        row's `prior_capture_date` and the prior row's own `capture_date` --
        disagree on a third of the edges, and 98 prior captures carry more than
        one `capture_date`, so the metadata cannot carry a rule. It is recorded
        on the result instead and the geometry gate does the work. This is a
        deliberate divergence from `align/priors.py:_check`, which refuses a
        later-dated prior outright.
        """
        # Keyed by the priors it names as well as the id: a reality can be a
        # current row (with priors) and a prior row (with none) in the same index,
        # and preparing the prior side first must not leave the current side an
        # empty pool.
        key = (case.reality_id, tuple(case.prior_capture_ids))
        if key in self._pool_members:
            return self._pool_members[key]
        if self.index is None:
            return ()
        forbidden = set(self.index.siblings_of(case))
        members = tuple(
            (split.reality_id, split.capture_id, split.capture_date or "")
            for split in self.index.priors_for(case)
            if split.reality_id not in forbidden)
        self._pool_members[key] = members
        return members

    def current_bbox(self, reality_id: str):
        """This walk's own camera box in paper inches, memoised."""
        if reality_id not in self._bboxes:
            npz = self.paths.capture_npz(reality_id)
            box = None
            if npz.exists():
                cameras = np.load(npz, allow_pickle=False)["camera_paper_xy"]
                if len(cameras) >= 3:
                    box = prior_bev.trimmed_bbox(cameras)
            self._bboxes[reality_id] = box
        return self._bboxes[reality_id]

    def prepared_pool_members(self, case) -> tuple:
        """Every named prior split that has a wall band on disk.

        `prepared`, never `ok`. A split is part of a target because it has a
        band, not because it would make a good training row -- and most prior
        splits are also current rows, so the two verdicts meet on the same
        reality constantly. Testing `ok` here makes a row's target depend on
        that row's own fitness as a sample, which is the cycle
        `align/io/cases.py:287` says is not a well-formed question, and it
        cascades: a gated row empties the next row's pool, gating that row too.
        Measured, it took a full run from ~9,000 usable rows to 30.

        This is also the pool PRODUCTION has. It is everything known about the
        previous visit, with no reference to where the new walk turned out to
        be.
        """
        out = []
        for reality_id, capture_id, capture_date in self.pool_members(case):
            result = self.capture_result(reality_id)
            if result is not None and result.prepared:
                out.append((reality_id, capture_id, capture_date))
        return tuple(out)

    def usable_pool_members(self, case) -> tuple:
        """The prepared splits that also stand on ground this walk covered.

        Narrower than `prepared_pool_members`, and narrower using the ANSWER:
        the filter compares each split against `current_bbox`, which comes from
        `camera_paper_xy`, which is the walk's ground-truth placement. That is
        right for building training pairs -- the point of a training pair is to
        show the model the piece it should match -- and wrong for measuring,
        because production does not know where the walk is. Pass
        `whole_capture=True` to `pool_for_case` for the deployable pool.

        A pool is a median of 6 splits of the named captures and up to 70, of
        which only 17% meet the walk; the rest are other parts of the site.
        """
        prepared = self.prepared_pool_members(case)
        if not prepared:
            return ()
        overlapping = set(self.pools.overlapping(
            [m[0] for m in prepared], self.current_bbox(case.reality_id)))
        return tuple(m for m in prepared if m[0] in overlapping)

    def lost_pool_members(self, case) -> tuple:
        """(reality_id, why) for each prior split PREPARATION could not use."""
        out = []
        for reality_id, _capture_id, _date in self.pool_members(case):
            result = self.capture_result(reality_id)
            if result is None:
                out.append((reality_id, "not in the capture cache"))
            elif not result.prepared:
                out.append((reality_id, result.status))
        return tuple(out)

    def pool_for_case(self, case, whole_capture: bool = False
                      ) -> prior_bev.PriorPool | None:
        """The target for one row.

        `whole_capture=False` is the training pool: the splits that overlap the
        walk, chosen with the walk's true position. `whole_capture=True` is what
        production holds -- every prepared split of the named prior captures,
        chosen with nothing the answer knows.
        """
        members = (self.prepared_pool_members(case) if whole_capture
                   else self.usable_pool_members(case))
        return self.pools.pool(members) if members else None

    def current_band_paper(self, reality_id: str):
        """This walk's wall band in paper inches. Small LRU -- a band is ~0.5 MB."""
        if reality_id in self._bands:
            return self._bands[reality_id]
        npz = self.paths.capture_npz(reality_id)
        band = None
        if npz.exists():
            stored = np.load(npz, allow_pickle=False)
            scale, theta, tx, ty = (float(v) for v in stored["gt_sim2_params"])
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            gt = np.array([[scale * cos_t, -scale * sin_t, tx],
                           [scale * sin_t, scale * cos_t, ty]])
            band = stored["band_xy"] @ gt[:, :2].T + gt[:, 2]
        if len(self._bands) > 8:
            self._bands.clear()
        self._bands[reality_id] = band
        return band

    def occupancy(self, case) -> float:
        """The share of this walk's wall the prior also has. The criterion.

        Not the bounding box and not camera proximity: a box around a diagonal
        walk is mostly empty, so a walk sitting in that emptiness scores 1.00 on
        it and shares no wall at all; and a prior that walked the corridors
        while the current walked inside a room puts their cameras far apart and
        their walls exactly together. `prior_bev.band_occupancy` carries the
        six measured rows that show both failures.
        """
        if case.reality_id in self._occupancy:
            return self._occupancy[case.reality_id]
        pool = self.pool_for_case(case)
        band = self.current_band_paper(case.reality_id)
        share = 0.0
        if pool is not None and band is not None:
            _mask, share = prior_bev.band_occupancy(band, pool.band_paper_xy,
                                                    pool.rig_height_in)
        self._occupancy[case.reality_id] = share
        return share

    def overlap_mask(self, case, result: CaptureResult) -> np.ndarray:
        """Which of this walk's cameras stand on ground the pool covers.

        Nothing gates on this any more -- membership is the gate, and a row with
        a pool at all has one by construction. It is kept because it is what
        `CaptureRecord.anchor_paper_xy` puts the target window on and what
        `data.camera_window_overlap_bias` steers the query's camera window
        towards, which together are what make a walk that leaves its prior's
        area part-way through trainable on the part that did not.

        A camera counts when the prior has wall in its cell, which is the same
        test `occupancy` applies to the band -- one notion of overlap, asked of
        cameras instead of points.
        """
        if case.reality_id in self._overlap_masks:
            return self._overlap_masks[case.reality_id]
        pool = self.pool_for_case(case)
        if pool is None:
            mask = np.zeros(0, dtype=bool)
        else:
            cameras = np.load(self.paths.capture_npz(case.reality_id),
                              allow_pickle=False)["camera_paper_xy"]
            mask, _ = prior_bev.band_occupancy(cameras, pool.band_paper_xy,
                                               pool.rig_height_in)
        self._overlap_masks[case.reality_id] = mask
        return mask

    # ------------------------------------------------------------------ build
    def ensure_prepared(self, cases, index, workers: int = 8, force: bool = False,
                        verbose: bool = True) -> dict[str, CaptureResult]:
        """Compute what the cache lacks for these rows and their priors."""
        self.index = index
        self.paths.captures.mkdir(parents=True, exist_ok=True)
        self._pool_members.clear()
        self._overlap_masks.clear()
        self._bboxes.clear()
        self._bands.clear()
        self._occupancy.clear()

        prior_cases = self._prior_cases(cases, index)
        results = self._prepare_all(list(cases) + prior_cases, workers, force, verbose)
        self._correct_prior_heights(cases, prior_cases, results, workers, verbose)
        self._pair(cases, results, verbose)
        write_manifest(self.paths.manifest, [results[c.reality_id] for c in cases])
        return results

    def _prior_cases(self, cases, index) -> list:
        """Every prior-side row any of these cases needs, once."""
        wanted, out = set(), []
        by_reality = {c.reality_id: c for c in index.prior}
        for case in cases:
            for reality_id, _capture, _date in self.pool_members(case):
                if reality_id in wanted or reality_id not in by_reality:
                    continue
                wanted.add(reality_id)
                out.append(by_reality[reality_id])
        return out

    def _prepare_all(self, cases, workers: int, force: bool, verbose: bool,
                     overrides: dict[str, float] | None = None,
                     label: str = "captures") -> dict[str, CaptureResult]:
        results: dict[str, CaptureResult] = {}
        to_build = []
        for case in cases:
            cached = None if force else self.capture_result(case.reality_id)
            if cached is None or (overrides and case.reality_id in overrides):
                to_build.append(case)
            else:
                results[case.reality_id] = cached
        if verbose:
            print(f"store: {len(cases)} rows, {len(to_build)} {label} to build")
        if not to_build:
            return results
        started = time.time()
        with ProcessPoolExecutor(max(1, min(workers, len(to_build)))) as pool:
            futures = {
                pool.submit(prepare_capture, case, self.paths, self.stride,
                            self.min_source_frames, self.max_residual_m,
                            (overrides or {}).get(case.reality_id)): case
                for case in to_build}
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results[result.reality_id] = result
                self._save_result(result)
                if verbose and (done % 25 == 0 or done == len(futures)):
                    print(f"  {label} [{done}/{len(futures)}] {result.reality_id} "
                          f"{result.status} ({time.time() - started:.0f}s)")
        return results

    def _correct_prior_heights(self, cases, prior_cases, results, workers: int,
                               verbose: bool) -> None:
        """Re-cut the wall band of prior splits whose floor fit is implausible.

        The rig height is the one thing about a walk known in advance. Measured
        over 98 georeferenced prior splits, the implied height has median 1.94 m
        but p10 0.58, and 23 land outside 1.2-3.5 m -- every one too LOW, which
        cuts the band to a slab. Dropping them would cost a quarter of the
        reference, so the known height is substituted and recorded.

        The expectation is per PRIOR CAPTURE: one session, one rig. Where the
        sheet scale is known the expected height is exact; where it is not, the
        capture's own median is the only reference there is, which makes this a
        consistency test between splits rather than against 2.11 m.
        """
        by_capture: dict[str, list] = {}
        for case in prior_cases:
            result = results.get(case.reality_id)
            if result is not None and not result.height_corrected:
                by_capture.setdefault(case.capture_id, []).append((case, result))
        overrides: dict[str, float] = {}
        for members in by_capture.values():
            heights = np.array([r.rig_height_in for _c, r in members if r.prepared])
            good = heights[np.isfinite(heights) & (heights > 0)]
            structure = members[0][0].structure
            if structure.georeferenced:
                expected = CAMERA_HEIGHT_M / design_scale(structure.design_tm)
            elif len(good):
                expected = float(np.median(good))
            else:
                continue
            low, high = prior_bev.HEIGHT_BAND
            for case, result in members:
                # A split that failed the floor fit outright is the one this
                # helps most: `no_floor` means the histogram found nothing, and
                # nothing is not a height the band can be cut at.
                rescuable = result.status in ("no_floor", "thin_band")
                height = result.rig_height_in
                plausible = (result.prepared and np.isfinite(height) and height > 0
                             and low * expected <= height <= high * expected)
                if rescuable or (result.prepared and not plausible):
                    overrides[case.reality_id] = float(expected)
        if not overrides:
            return
        if verbose:
            print(f"store: {len(overrides)} prior splits re-banded at the known rig height")
        corrected = self._prepare_all(
            [c for c in prior_cases if c.reality_id in overrides], workers, force=False,
            verbose=verbose, overrides=overrides, label="height corrections")
        results.update(corrected)

    def _pair(self, cases, results, verbose: bool) -> None:
        """Attach each current row's pool and overlap, and gate on it."""
        started = time.time()
        for done, case in enumerate(cases, 1):
            result = results.get(case.reality_id)
            if result is None or not result.prepared:
                continue
            # recomputed every run, never inherited: the pool this row can see
            # grows as the prior side gets built
            result.pair_status = "ok"
            result.prior_splits = result.prior_dropped = result.prior_levels = 0
            named = self.pool_members(case)
            lost = self.lost_pool_members(case)
            result.same_day = any(d and d == case.capture_date for _r, _c, d in named)
            result.later_dated = any(d and case.capture_date and d > case.capture_date
                                     for _r, _c, d in named)
            if not named:
                result.pair_status = "no_prior"
                self._save_result(result)
                continue
            pool = self.pool_for_case(case)
            if pool is None:
                # two different failures, kept apart because they say different
                # things: nothing the named captures offered was PREPARED, or
                # what was prepared stands somewhere this walk never went
                prepared_any = any(
                    (r := self.capture_result(m[0])) is not None and r.prepared
                    for m in named)
                result.pair_status = ("no_overlap" if prepared_any
                                      else "no_usable_prior_split")
                result.prior_dropped = len(lost)
                self._save_result(result)
                continue
            result.prior_splits = len(pool.splits)
            result.prior_dropped = len(lost) + len(pool.dropped)
            result.prior_levels = pool.num_levels
            result.overlap = self.occupancy(case)
            if result.overlap < self.data.min_occupancy:
                result.pair_status = "low_overlap"
            self._save_result(result)
            if verbose and done % 500 == 0:
                print(f"  pairing [{done}/{len(cases)}] ({time.time() - started:.0f}s)")

    # ---------------------------------------------------------------- records
    def capture_records(self, cases, results: dict[str, CaptureResult] | None = None):
        """Usable `CaptureRecord`s for these rows, in CSV order.

        Gated on `prepared` plus a pool rebuilt HERE, never on the cached
        `pair_status`. Recomputing it is not redundant -- the cached verdict and
        the live pool disagree whenever the cache was written by a different
        run. A reality prepared as somebody's PRIOR carries no pairing verdict
        at all (1,457 capture ids appear on both sides of these manifests), a
        pool shrinks when one of its splits fails, and it grows as the prior side
        gets built, so before `_pair` has ever run every row would look unusable.
        That last one is not hypothetical: it is what makes this work mid-build,
        which is when the diagnostics want it.

        The dataset must not contain a row it cannot build a sample from -- the
        `__getitem__` hop is a belt for the occasional bad draw, not a filter.
        """
        from .dataset import CaptureRecord
        records = []
        for case in cases:
            result = (results.get(case.reality_id) if results
                      else self.capture_result(case.reality_id))
            if result is None or not result.prepared:
                continue
            members = self.usable_pool_members(case)
            if not members or self.pool_for_case(case) is None:
                continue
            if self.occupancy(case) < self.data.min_occupancy:
                continue
            record = CaptureRecord.load(self.paths.capture_npz(case.reality_id))
            record.capture_id = case.capture_id
            record.pool_members = members
            record.overlap_mask = self.overlap_mask(case, result)
            record.metres_per_inch_source = result.metres_per_inch_source
            record.same_day = result.same_day
            record.later_dated = result.later_dated
            records.append(record)
        return records

    def pool_for(self, record) -> prior_bev.PriorPool | None:
        return self.pools.pool(record.pool_members) if record.pool_members else None


def summarise_capture_results(results: dict[str, CaptureResult], reality_ids=None,
                              label: str = "current") -> str:
    """The denominator, for one side of the dataset at a time.

    `reality_ids` is which rows to count -- the caller's own list, because the
    cached `role` field cannot answer it. `role` records which SIDE first
    prepared a reality, written once and read back from the cache forever, and
    on this export every prior split is also a current row (the union of the two
    manifests is exactly the current row count). So the label says which index
    reached a reality first, not which index it belongs to, and splitting a
    summary on it puts `no_prior` -- a verdict only a current row can receive --
    under the prior heading.

    Two tables, because there are two verdicts. PREPARATION is whether the slow
    work produced a wall band; PAIRING is whether a current row found a target.
    A prior split failing preparation is not a training row lost, it is a piece
    of somebody's target lost, and it shows up on the current side as
    `no_usable_prior_split` or as a lower overlap. Counting the two together
    makes both numbers unreadable -- and conflating them in the CACHE once took
    a full run from ~9,000 usable rows to 30.
    """
    if reality_ids is None:
        all_results = list(results.values())
    else:
        seen, all_results = set(), []
        for reality_id in reality_ids:
            if reality_id in seen or reality_id not in results:
                continue
            seen.add(reality_id)
            all_results.append(results[reality_id])
    if not all_results:
        return f"no {label} rows"
    prepared = [r for r in all_results if r.prepared]
    lines = [f"{len(prepared)}/{len(all_results)} {label} captures prepared"]
    reasons: dict[str, int] = {}
    for result in all_results:
        if not result.prepared:
            reasons[result.status] = reasons.get(result.status, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        lines.append(f"  not prepared {count:5d}  {reason}")
    if prepared:
        residuals_m = np.array([r.residual_m for r in prepared])
        lines.append(f"  GT fit residual (m): median {np.median(residuals_m):.3f}  "
                     f"p90 {np.percentile(residuals_m, 90):.3f}")
        lines.append(f"  sign ambiguous: {sum(r.sign_ambiguous for r in prepared)}   "
                     f"rig implausible: {sum(r.rig_height_implausible for r in prepared)}   "
                     f"height corrected: {sum(r.height_corrected for r in prepared)}")

    paired = [r for r in prepared if r.pair_status]
    if not paired:
        return "\n".join(lines)
    usable = [r for r in paired if r.ok]
    lines.append(f"{len(usable)}/{len(paired)} prepared rows found a prior target")
    pair_reasons: dict[str, int] = {}
    for result in paired:
        if not result.ok:
            pair_reasons[result.pair_status] = pair_reasons.get(result.pair_status, 0) + 1
    for reason, count in sorted(pair_reasons.items(), key=lambda item: -item[1]):
        lines.append(f"  no target   {count:5d}  {reason}")
    if usable:
        overlaps = np.array([r.overlap for r in usable if np.isfinite(r.overlap)])
        if len(overlaps):
            lines.append(f"  prior overlap: median {np.median(overlaps):.2f}  "
                         f"p10 {np.percentile(overlaps, 10):.2f}  "
                         f"p90 {np.percentile(overlaps, 90):.2f}")
        splits = np.array([r.prior_splits for r in usable])
        lines.append(f"  prior splits merged: median {np.median(splits):.0f}  "
                     f"max {splits.max()}   splits lost "
                     f"{sum(r.prior_dropped for r in usable)}   "
                     f"multi-level {sum(r.prior_levels > 1 for r in usable)}")
        lines.append(f"  same-day prior: {sum(r.same_day for r in usable)}   "
                     f"later-dated prior in pool: "
                     f"{sum(r.later_dated for r in usable)}  (recorded, not filtered)")
    return "\n".join(lines)
