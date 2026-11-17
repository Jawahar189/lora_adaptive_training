"""Place every capture in a CSV with a trained matcher; score it like the solver.

Nothing here reads the training cache for the CURRENT capture: cloud, cameras,
gravity and both up hypotheses are computed live, so the script works on
captures the cache has never seen. The prior pool does come from the cache --
it is the target, and building it means every prior split of the capture.

By default the target is the prior splits that OVERLAP the walk, which is the
pool training used -- and choosing it needs the walk's true position. That makes
the default an upper bound rather than a forecast. `--whole-prior` scores against
every prepared split of the named prior captures instead, which is what
production holds, and the gap between the two is how much of the problem is
search rather than matching.

Two error numbers are reported. The pose is solved in the prior's frame; the
transform matrix composes out through `design_tm` (`align/frames.py:compose_tm`,
via `pose_transform_matrix_for`) so `reality_metrics` scores it against the
building's own coordinates exactly as the floor-plan runs did. Where a structure
carries no design transform the two frames coincide and the metre scale rests on
a rig-height estimate -- `scale_source` is written on every row for that reason.

    python -m bevaligner.evaluate --ckpt out/bevaligner/gpu24/checkpoints/last.ckpt \
        --csv datasets/manifest_..._current.csv \
        --prior-csv datasets/manifest_..._prior.csv --stem out/bevaligner/eval/gpu24
    python -m bevaligner.evaluate --ckpt ... --split val

Output: <stem>.jsonl / .csv / _predictions.jsonl via `SheetResultWriter`,
the same shape `scripts/run_alignment.py` writes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import REPO_ROOT
from .data import prior_bev
from .data.datamodule import build_capture_store, load_paired_index
from .data.splits import load_splits
from .inference import (MATCHERS, MODES, PoseProposal, PoseProposalOptions,
                        pose_transform_matrix_for, propose_pose)
from .inference.chain import (add_chain_arguments, add_oracle_seed_argument,
                              chain_policy, place_capture_chain)
from .inference.confidence import confidence_for
from .infer import (capture_groups, chain_callbacks, prepare_capture,
                    solve_prepared)
from .lightning.module import BEVAlignerModule
from .utils import affine, geometry, gravity, groundtruth, io
from .utils.metrics import (CHAIN_COLUMNS, RIG_PLAUSIBLE_M, SheetResultWriter,
                            metre_aliases, pixel_metrics, reality_metrics,
                            rig_height_m, summarise, summarise_chain,
                            summarise_confidence, summarise_pixels)
from .viz import render_inference_matches, render_placement_image


def write_qualitative(out_dir: Path, case, proposal, view, pool, data_config,
                      gt_paper_xy=None, metrics: dict | None = None) -> list[Path]:
    """Two pictures for one placed capture: the matches it was fitted from, and
    the capture drawn on the PRIOR it was placed against. Both land flat in
    `out_dir`, named `<reality>_<capture>_{matches,placement}.png`, so a folder
    of them can be flicked through."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{case.reality_id}_{case.capture_id}_"
    written = []
    scored = ("" if metrics is None
              else f"  rmse_xy {metrics['rmse_xy']:.2f} m  tpr@1.0 {metrics['tpr@1.0']:.2f}")
    # "12/340 inliers" when nothing was verified, "12/180/340" when something
    # was: inliers of verified of proposed, so the two filters are separable.
    support = (f"{proposal.num_inliers}/{proposal.num_matches}" if proposal.num_verified is None
               else f"{proposal.num_inliers}/{proposal.num_verified}/{proposal.num_matches}")
    caption = (f"{case.reality_id}  up {proposal.up_index}  window "
               f"{proposal.window_index}  {support} "
               f"inliers{'  refined' if proposal.refined else ''}{scored}")

    if proposal.matches is not None and proposal.density_map is not None \
            and proposal.crop is not None:
        image = render_inference_matches(
            proposal.density_map.image, proposal.crop.image, proposal.matches.query_px,
            proposal.matches.target_crop_px, inlier_mask=proposal.inlier_mask,
            confidence=proposal.matches.confidence,
            verified_mask=proposal.verified_mask, caption=caption)
        path = out_dir / f"{prefix}matches.png"
        image.save(path)
        written.append(path)

    # the placement, drawn on the whole prior rather than on a drawing
    backdrop = prior_bev.render_target(pool, pool.extent, 1280, None, data_config)
    if backdrop is not None:
        to_crop = backdrop.paper_to_crop
        grey = (np.clip(backdrop.image, 0, 1) * 255).astype(np.uint8)
        image = render_placement_image(
            Image.fromarray(np.stack([grey] * 3, axis=-1)),
            affine.apply(to_crop, proposal.sim2.apply(view.select_band_points())),
            affine.apply(to_crop, proposal.sim2.apply(view.camera_xy)),
            None if gt_paper_xy is None else affine.apply(to_crop, gt_paper_xy),
            caption=caption)
        path = out_dir / f"{prefix}placement.png"
        image.save(path)
        written.append(path)
    return written


def write_images(case, transform_matrix, out_dir: Path, *, name: str,
                 every_nth: int = 1, override_z: float | None = None,
                 plot: bool = False, plot_name: str = "floormap.png",
                 plot_width: int = 1800, gt_target=None) -> tuple[Path, int]:
    """Every photograph's predicted pose, in the shape the floormap viewer reads.

    One record per image: a position in the target frame and a heading in
    three.js `Euler` XYZ degrees. The conversion is production's own, borrowed
    through `utils/io.py`, so a file written here is directly comparable with a
    delivered one. This is a pure function of the pose and the shots -- it reads
    nothing the solver has not already read.
    """
    poses = io.shot_poses(case.rlt_dir)
    images = io.poses_to_images(poses, transform_matrix, override_z=override_z)
    if every_nth > 1:
        images = io.every_nth_image(images, every_nth)
    if not images:
        raise ValueError("no shots carried a pose")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(images, indent=2))
    if plot:
        io.render_floormap(case, images, out_dir / plot_name, gt_target_xyz=gt_target,
                           width=plot_width)
    return path, len(images)


def record_capture(writer, case, proposal, trajectory, metres_per_inch,
                   context: dict, *, score: bool = True, images: dict | None = None):
    """Score one placed split, write its row, and hand back what a drawing needs.

    Shared by the split-major loop and the capture-major one so the two cannot
    drift: a chained run and an independent one are scored by the same code, or
    the comparison between them means nothing.
    """
    transform_matrix = pose_transform_matrix_for(proposal, case.structure.design_tm)
    metrics: dict = {}
    gt_target = None
    if score:
        sfm_positions, gt_positions = groundtruth.matched_camera_positions(
            case, trajectory)
        gt_target = gt_positions
        metrics = reality_metrics(transform_matrix, sfm_positions, gt_positions,
                                  metres_per_inch, case.structure.georeferenced)
        # The same per-image error measured on the drawing instead of in
        # metres: both sides carried to floormap pixels first. Empty when
        # the sheet cannot be measured -- never a zero.
        metrics.update(pixel_metrics(transform_matrix, sfm_positions,
                                     gt_positions, case.structure))
        # The metres again under a name that says so, beside the pixels.
        metrics.update(metre_aliases(metrics))
    camera_rig_height_m = rig_height_m(proposal.camera_height_above_floor,
                                       proposal.sim2.s, metres_per_inch)
    metrics.update(rig_height_m=camera_rig_height_m,
                   rig_implausible=not (RIG_PLAUSIBLE_M[0] <= camera_rig_height_m
                                        <= RIG_PLAUSIBLE_M[1]),
                   confidence=confidence_for(proposal, metres_per_inch))
    context.update(up_index=proposal.up_index, window=proposal.window_index,
                   n_inliers=proposal.num_inliers,
                   inlier_frac=round(proposal.inlier_fraction, 3),
                   n_matches=proposal.num_matches,
                   n_verified=proposal.num_verified, refined=proposal.refined,
                   n_crops=len(proposal.candidates),
                   candidates=proposal.candidates)
    if images:
        try:
            prefix = f"{case.reality_id}_{case.capture_id}_"
            images_path, num_images = write_images(
                case, transform_matrix,
                Path(images["dir"]), name=prefix + images["name"],
                every_nth=images["every_nth"], override_z=images["z"],
                plot=images["plot"], plot_name=f"{prefix}floormap.png",
                plot_width=images["plot_width"],
                gt_target=gt_target if images["plot_gt"] else None)
            context.update(images_path=str(images_path), images_n=num_images)
        except Exception as error:                                # noqa: BLE001
            print(f"    images skipped: {type(error).__name__}: {error}")
    writer.write(case, {"predicted": dict(metrics, tm=transform_matrix)},
                 status="ok", **context)
    return metrics, gt_target


def scored_line(metrics: dict, score: bool) -> str:
    """The `rmse / tpr` fragment of a progress line, empty when unscored."""
    if not score:
        return ""
    in_pixels = ("" if metrics.get("rmse_px") is None
                 else f" / {metrics['rmse_px']:.0f} px")
    return (f"rmse_xy {metrics['rmse_xy']:.2f} m{in_pixels}  "
            f"tpr@1.0 {metrics['tpr@1.0']:.2f}  ")


def evaluate(module: BEVAlignerModule, cases, stem: Path, store, index,
             options: PoseProposalOptions, device, stride: int = 1,
             min_source_frames: int = 2, qualitative: int = 0,
             score: bool = True, images: dict | None = None,
             whole_prior: bool = False):
    writer = SheetResultWriter(stem)
    model = module.model.eval().to(device)
    drawn = 0
    for position, case in enumerate(cases, 1):
        started = time.time()
        context: dict = {}
        try:
            trajectory = io.load_camera_trajectory(case)
            cloud_xyz = io.load_point_cloud_xyz(case, stride, min_source_frames)
            gravity_fit, views = gravity.gravity_views(cloud_xyz, trajectory.positions)
            usable_views = {up_index: view for up_index, view in views.items()
                            if not isinstance(view, str)}
            context.update(sign_ambiguous=bool(gravity_fit.sign_ambiguous),
                           num_views=len(usable_views))
            if not usable_views:
                writer.write(case, {}, status="no_floor", **context)
                continue
            if case.structure.georeferenced:
                metres_per_inch = geometry.design_scale(case.structure.design_tm)
                scale_source = "design_tm"
            else:
                metres_per_inch, scale_source = io.metres_per_inch_for(case.structure, np.nan)
            context.update(metres_per_inch=metres_per_inch, scale_source=scale_source)
            # the FILTERED pool -- prepared, and overlapping this walk. Built
            # from `pool_members` this matched against every named split, so
            # evaluation saw the whole site while training saw the part the walk
            # covered: a different target, a median of 6 splits rather than 2,
            # and most of why placing one capture took 25 s instead of 4.
            pool = store.pool_for_case(case, whole_capture=whole_prior)
            if pool is None:
                writer.write(case, {}, status="no_prior",
                             seconds=round(time.time() - started, 1), **context)
                print(f"[{position}/{len(cases)}] {case.reality_id}: no usable prior")
                continue
            context.update(prior_splits=len(pool.splits), prior_key=pool.key,
                           whole_prior=whole_prior)
            proposal: PoseProposal | None = propose_pose(
                model, usable_views, pool, module.config.data, metres_per_inch,
                options, device=device, patch_size=module.patch_size)
            if proposal is None:
                writer.write(case, {}, status="no_pose",
                             seconds=round(time.time() - started, 1), **context)
                print(f"[{position}/{len(cases)}] {case.reality_id}: no pose")
                continue
            context["seconds"] = round(time.time() - started, 1)
            metrics, _gt_target = record_capture(
                writer, case, proposal, trajectory, metres_per_inch, context,
                score=score, images=images)
            if qualitative and drawn < qualitative:
                try:
                    _, gt_paper_xy = groundtruth.camera_positions_in_paper(case, trajectory)
                    write_qualitative(Path(f"{stem}_viz"), case, proposal,
                                      usable_views[proposal.up_index], pool,
                                      module.config.data, gt_paper_xy, metrics)
                    drawn += 1
                except Exception as error:                            # noqa: BLE001
                    print(f"    qualitative skipped: {type(error).__name__}: {error}")
            print(f"[{position}/{len(cases)}] {case.reality_id}: "
                  f"{scored_line(metrics, score)}"
                  f"inliers {proposal.num_inliers}/{proposal.num_matches}  "
                  f"up {proposal.up_index} window {proposal.window_index}  "
                  f"({time.time() - started:.0f}s)")
        except Exception as error:                                    # noqa: BLE001
            import traceback
            writer.write(case, {},
                         status=f"error: {type(error).__name__}: {error}"[:200], **context)
            print(f"[{position}/{len(cases)}] {case.reality_id}: "
                  f"ERROR {type(error).__name__}: {error}")
            traceback.print_exc()
    writer.close()
    if score:
        print(summarise(writer.rows))
        print(summarise_pixels(writer.rows))
    else:
        # ResultWriter.rows holds one row per placed capture: it appends only
        # the pose named "predicted", and only when there is one.
        print(f"{len(writer.rows)}/{len(cases)} captures placed; "
              f"no scoring was asked for")
    # Printed whether or not the run was scored: on an unscored one it is the
    # only statement about quality there is.
    print(summarise_confidence(writer.rows))
    return writer.rows


class OracleTrack:
    """Ground-truth camera positions, wearing the seed interface `geo.txt` uses.

    Deliberately the same shape as `utils.io.GeoTrack`, so the oracle run
    exercises the SAME seeding path as the real one and differs only in what the
    fix says. A separate oracle code path would be testing itself.
    """

    def __init__(self, names, paper_xy):
        self.names, self.paper_xy = list(names), np.asarray(paper_xy)

    def first_of(self, trajectory):
        index = {name: row for row, name in enumerate(self.names)}
        for position in np.argsort(trajectory.idx):
            row = index.get(trajectory.names[int(position)])
            if row is not None:
                return self.paper_xy[row]
        return None


def oracle_track_for(case):
    """The true paper position of every photograph of `case`, or None."""
    try:
        trajectory = io.load_camera_trajectory(case)
        _, paper_xy = groundtruth.camera_positions_in_paper(case, trajectory)
    except Exception:                                             # noqa: BLE001
        return None
    if len(paper_xy) == 0:
        return None
    # `matched_camera_positions` keeps only the images ground truth and the
    # trajectory agree on, in trajectory order, so the names line up with it.
    names = [n for n in trajectory.names][:len(paper_xy)]
    return OracleTrack(names, paper_xy)


def evaluate_chained(module: BEVAlignerModule, cases, stem: Path, store,
                     options: PoseProposalOptions, device, policy,
                     stride: int = 1, qualitative: int = 0, score: bool = True,
                     images: dict | None = None, whole_prior: bool = False):
    """`evaluate`, capture-major, each split seeded from the one before it.

    The scoring and the row are `record_capture`'s, the same ones the
    split-major loop uses, so the only difference between a chained run and an
    independent one is how the pose was arrived at. Rows are written as each
    CAPTURE finishes rather than each split, and in capture order rather than
    CSV order.
    """
    writer = SheetResultWriter(stem, extra_columns=CHAIN_COLUMNS)
    model = module.model.eval().to(device)
    policy.keep_prepared = qualitative > 0
    prepare, solve = chain_callbacks(model, module.config, store, device, stride,
                                     whole_prior=whole_prior)
    seed_track = (oracle_track_for if policy.seed_source == "oracle"
                  else io.load_geo_track)
    groups = capture_groups(cases)
    drawn = 0
    for position, (capture_id, members) in enumerate(groups, 1):
        started = time.time()
        print(f"[{position}/{len(groups)}] {capture_id} -- {len(members)} split(s)",
              flush=True)
        try:
            results = place_capture_chain(members, prepare, solve, options, policy,
                                          geo_track_for=seed_track)
        except Exception as error:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            for case in members:
                writer.write(case, {},
                             status=f"error: {type(error).__name__}: {error}"[:200],
                             chain_mode="chain", chain_how="capture failed")
            continue
        seconds = round((time.time() - started) / max(len(members), 1), 1)
        for case in members:
            result = results.get(case.reality_id)
            if result is None:
                writer.write(case, {}, status="no_pose", chain_mode="chain",
                             chain_how="not reached")
                continue
            context = dict(result.context, seconds=seconds)
            if result.proposal is None:
                writer.write(case, {}, status=result.status, **context)
                print(f"      {case.reality_id}: {context.get('chain_how', '?')}")
                continue
            try:
                metrics, gt_target = record_capture(
                    writer, case, result.proposal, result.trajectory,
                    result.metres_per_inch, context, score=score, images=images)
            except Exception as error:                            # noqa: BLE001
                writer.write(case, {},
                             status=f"error: {type(error).__name__}: {error}"[:200],
                             **context)
                print(f"      {case.reality_id}: scoring failed: {error}")
                continue
            if qualitative and drawn >= qualitative:
                # The quota is full, so stop paying to hold wall bands for
                # captures that will never be drawn.
                policy.keep_prepared = False
            if qualitative and drawn < qualitative and result.prepared is not None:
                try:
                    _, gt_paper_xy = groundtruth.camera_positions_in_paper(
                        case, result.trajectory)
                    write_qualitative(Path(f"{stem}_viz"), case, result.proposal,
                                      result.prepared.views[result.proposal.up_index],
                                      result.prepared.pool, module.config.data,
                                      gt_paper_xy, metrics)
                    drawn += 1
                except Exception as error:                        # noqa: BLE001
                    print(f"    qualitative skipped: {type(error).__name__}: {error}")
            gate = "gate ok" if context.get("gate_passed") else "GATE FAILED"
            print(f"      {case.reality_id}: {scored_line(metrics, score)}"
                  f"{context.get('chain_how', '?'):<30} {gate}", flush=True)
    writer.close()
    if score:
        print(summarise(writer.rows))
        print(summarise_pixels(writer.rows))
    # Printed whether or not the run was scored: on an unscored one it is the
    # only statement about quality there is.
    print(summarise_confidence(writer.rows))
    print(summarise_chain(writer.rows))
    return writer.rows


def _mtime(path) -> str:
    """When the splits file was last written -- an eval reads whatever the most
    recent training run left there, so the date is the thing that says whether
    it is the split this checkpoint was trained against."""
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--csv", nargs="+", default=None,
                        help="current-side CSV(s) to place; default: the training config's")
    parser.add_argument("--prior-csv", nargs="+", default=None,
                        help="prior-side CSV(s) to align against; default: the "
                             "training config's")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all",
                        help="restrict to the structures of a split from the cache")
    parser.add_argument("--realities", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--tolerance-m", type=float, default=0.5)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--matcher", choices=MATCHERS, default="dense",
                        help="dense: each query's soft-argmax over the prior; "
                             "keypoints: points drawn on both maps, matched one to one")
    parser.add_argument("--geometric-verification", choices=MODES, default="none",
                        help="COLMAP LO-RANSAC over the matches before the Sim2 "
                             "fit, dropping the ones no planar motion explains. "
                             "Off by default: it changes results, and "
                             "scripts/e10_colmap_verification.py is what decides "
                             "whether it should not be")
    parser.add_argument("--no-refine", action="store_true",
                        help="skip the zoomed second pass")
    windows = parser.add_argument_group("stage-one windows")
    windows.add_argument("--search-zoom-out", type=float,
                         default=PoseProposalOptions.search_zoom_out,
                         help="window side as a multiple of the walk's estimated "
                              "paper size")
    windows.add_argument("--max-windows", type=int,
                         default=PoseProposalOptions.max_windows,
                         help="windows kept, richest in prior wall first")
    windows.add_argument("--window-stride", type=float,
                         default=PoseProposalOptions.window_stride_fraction,
                         help="step between windows, as a fraction of their side")
    parser.add_argument("--whole-prior", action="store_true",
                        help="score the way production must: the target is every "
                             "prepared split of the named prior captures, not the "
                             "ones that overlap the walk. Selecting the overlapping "
                             "splits needs the walk's true position, which is the "
                             "thing being solved for, so the default is an upper "
                             "bound and this is the deployable number")
    parser.add_argument("--prior-mosaic", action="store_true",
                        help="a prior drawn from several dates: each 5 m cell keeps "
                             "the walls of the newest date only, instead of every "
                             "split stacked")
    parser.add_argument("--qualitative", type=int, default=0, metavar="N",
                        help="write <reality>_<capture>_{matches,placement}.png under "
                             "<stem>_viz/ for the first N placed captures (0 = off)")
    parser.add_argument("--no-score", action="store_true",
                        help="solve and write transforms without reading ground truth")
    # The production filename lives here rather than in the package, the way
    # scripts/run_alignment.py keeps it out of align/.
    images = parser.add_argument_group("per-image poses")
    images.add_argument("--images", action="store_true",
                        help="write each photograph's predicted pose per reality")
    images.add_argument("--images-dir", default=None,
                        help="default: <stem>_images/")
    images.add_argument("--images-name", default="images.json",
                        help="filename suffix: each reality writes "
                             "<reality>_<capture>_<this>")
    images.add_argument("--images-every-nth", type=int, default=1,
                        help="3 keeps indices 1, 4, 7 ... the subset production ships")
    images.add_argument("--images-z", type=float, default=None,
                        help="pin every z to this, as the viewer's own file does; "
                             "default is the solved z")
    images.add_argument("--images-plot", action="store_true",
                        help="also draw <reality>_<capture>_floormap.png beside the JSON")
    images.add_argument("--images-gt", action="store_true",
                        help="draw ground truth beneath; needs scoring, so it is "
                             "silently absent under --no-score")
    images.add_argument("--images-plot-width", type=int, default=1800)
    add_chain_arguments(parser)
    add_oracle_seed_argument(parser)
    args = parser.parse_args(argv)
    if args.chain_oracle_seed:
        if args.no_score:
            parser.error("--chain-oracle-seed reads ground truth, which "
                         "--no-score exists to avoid")
        args.chain_seed = "oracle"

    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "cache" / "hf"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = BEVAlignerModule.from_checkpoint(args.ckpt)
    config = module.config
    if args.csv:
        config.data.csv_files = list(args.csv)
    if args.prior_csv:
        config.data.prior_csv_files = list(args.prior_csv)
    config.data.prior_mosaic = args.prior_mosaic
    index = load_paired_index(config)
    cases = list(index.current)
    store = build_capture_store(config)
    store.index = index
    if args.realities:
        wanted = set(args.realities)
        cases = [c for c in cases if c.reality_id in wanted]
    split_source = "--split all: every row of the CSV, split ignored"
    if args.split != "all":
        splits = load_splits(store.paths.splits)
        split_structures = set(splits[args.split])
        cases = [c for c in cases if c.structure.key in split_structures]
        # Worth naming: this is whatever the LAST training run against this
        # cache wrote, not the CSV the checkpoint was trained from. A run that
        # re-cut its split by structure has moved it since.
        split_source = (f"{args.split} ({len(split_structures)} structures) from "
                        f"{store.paths.splits}, written "
                        f"{_mtime(store.paths.splits)}")
    if args.limit:
        cases = cases[:args.limit]
    stem = (Path(args.stem) if args.stem
            else config.run_dir / "eval" / f"{Path(args.ckpt).stem}_{args.split}")
    options = PoseProposalOptions(
        resolution=args.resolution or config.data.eval_resolution,
        num_queries=args.queries, tolerance_m=args.tolerance_m,
        min_inliers=args.min_inliers, refine_on_footprint=not args.no_refine,
        geometric_verification=args.geometric_verification, matcher=args.matcher,
        refine_footprint_margin=config.data.refine_footprint_margin,
        search_zoom_out=args.search_zoom_out, max_windows=args.max_windows,
        window_stride_fraction=args.window_stride,
        keep_match_artifacts=args.qualitative > 0)
    images = None
    if args.images:
        images = dict(dir=args.images_dir or f"{stem}_images", name=args.images_name,
                      every_nth=args.images_every_nth, z=args.images_z,
                      plot=args.images_plot, plot_gt=args.images_gt and not args.no_score,
                      plot_width=args.images_plot_width)
    print(f"{len(cases)} captures -> {stem}   ({index.coverage()})")
    print(f"  checkpoint: {args.ckpt}  (run {config.run_name})")
    own = "the checkpoint's own"
    print(f"  current csv: {', '.join(config.data.csv_files)}"
          f"  [{'--csv' if args.csv else own}]")
    print(f"  prior csv  : {', '.join(config.data.prior_csv_files)}"
          f"  [{'--prior-csv' if args.prior_csv else own}]")
    print(f"  split      : {split_source}")
    print(f"  cache      : {config.cache_dir}")
    if args.chain:
        policy = chain_policy(args)
        print(f"  chaining   : seed {policy.seed_source}, radii "
              f"x{policy.radius_scale}, {policy.max_windows_with_prior}"
              f"+{policy.insurance_windows} windows, {policy.retries} retries")
        evaluate_chained(module, cases, stem, store, options, device, policy,
                         stride=args.stride, qualitative=args.qualitative,
                         score=not args.no_score, images=images,
                         whole_prior=args.whole_prior)
    else:
        evaluate(module, cases, stem, store, index, options, device,
                 stride=args.stride, qualitative=args.qualitative,
                 score=not args.no_score, images=images,
                 whole_prior=args.whole_prior)


if __name__ == "__main__":
    main()
