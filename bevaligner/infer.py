"""Production inference: place every capture of a CSV against its prior, one 4x4 each.

    python -m bevaligner.infer --ckpt best.ckpt \\
        --csv datasets/current.csv --prior-csv datasets/prior.csv \\
        --stem out/bevaligner/infer/run

This is the placement `bevaligner.evaluate --whole-prior --no-score` makes,
fit to deploy:

- ground truth is never read, so it runs on captures that have none;
- the CURRENT capture's cloud, cameras, gravity and both up hypotheses are
  computed live; the target is every prepared split of its named prior
  captures, read from the cache `bevaligner.data.preprocess` fills;
- Lightning is not needed. A checkpoint is the run's config plus the model's
  tensors, so `load_matcher` builds the matcher from the one and loads the other.

Output, via `SheetResultWriter` (the shape `bevaligner.evaluate` writes):
<stem>_predictions.jsonl holds one {"reality_id", "tm", "confidence"} per placed
capture -- the transform taking its point cloud into the structure's frame, and
how likely that placement is to be within 2 m (`inference/confidence.py`; absent
on a pose no search scored). <stem>.jsonl and <stem>.csv hold every capture,
with the reason for any that could not be placed. The confidence is the only
quality signal an unscored run has, so it is also the last line it prints.
To score afterwards, where ground truth exists:

    python scripts/evaluate.py --pred <stem>_predictions.jsonl --csv datasets/current.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from align.io import datasets as datasets_mod
from align.io.cases import PairedIndex

from .config import REPO_ROOT, config_from_dict
from .data.store import CaptureStore
from .inference import (MATCHERS, MODES, PoseProposal, PoseProposalOptions,
                        pose_transform_matrix_for, propose_pose)
from .inference.chain import (ChainPolicy, add_chain_arguments,
                              chain_policy, place_capture_chain)
from .inference.confidence import confidence_for
from .inference.gate import GateThresholds
from .models import DensityPriorMatcher, DinoPatchEncoder
from .models.backbone import LoraSettings
from .utils import geometry, gravity, io
from .utils.metrics import (CHAIN_COLUMNS, RIG_PLAUSIBLE_M, SheetResultWriter,
                            rig_height_m, summarise_chain,
                            summarise_confidence)


@dataclass
class Placement:
    """What placing one capture came to: a status, and the pose when it is "ok"."""
    status: str
    pose: dict | None = None
    # The proposal behind `pose`. Production writes the 4x4 and throws the rest
    # away; chaining needs the Sim2 and the gravity rotation to seed the next
    # split, and the candidate list to decide whether it should.
    proposal: PoseProposal | None = None


# ----------------------------------------------------------------------------
# the model

def _checkpoint_config(blob: dict):
    """The run's config, under whichever key the checkpoint's era wrote it."""
    stored = (blob.get("bevaligner_config") or blob.get("bevaligner_cfg")
              or blob["hyper_parameters"]["config"])
    return config_from_dict(stored)


def load_matcher(checkpoint, device="cpu", backbone: str | None = None):
    """(model, config) from a training checkpoint, without Lightning.

    `backbone` overrides where the DINO weights are read from; by default it is
    the path the checkpoint was trained with, as `bevaligner.evaluate` reads
    it -- a relative one is relative to where the command runs. The frozen backbone comes from
    those weights, so a backbone key absent from the checkpoint is expected; a
    missing LoRA adapter, or a key the model does not have, means the
    checkpoint is not this model.
    """
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _checkpoint_config(blob)
    settings = config.model
    encoder = DinoPatchEncoder(
        family=settings.backbone_family,
        pretrained_id=backbone or settings.pretrained_id,
        lora=LoraSettings(settings.lora_rank, settings.lora_alpha, settings.lora_dropout))
    model = DensityPriorMatcher(encoder, temperature=settings.temperature)
    state = {key[len("model."):]: value for key, value in blob["state_dict"].items()
             if key.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_lora = [name for name in missing if ".lora_" in name]
    if missing_lora or unexpected:
        raise KeyError(f"checkpoint mismatch: {missing_lora[:3]} / {list(unexpected)[:3]}")
    return model.eval().to(device), config


# ----------------------------------------------------------------------------
# the captures

def load_index_and_store(config, csv_files, prior_csv_files) -> tuple[PairedIndex,
                                                                      CaptureStore]:
    """The current/prior pairing of the CSVs, and the cache that holds the priors.

    Built here rather than through `data.datamodule`, which imports Lightning.
    """
    index = PairedIndex.load(datasets_mod.resolve(
        [str(REPO_ROOT / path) for path in csv_files],
        [str(REPO_ROOT / path) for path in prior_csv_files], name="infer"))
    data = config.data
    store = CaptureStore(config.cache_dir, stride=data.cloud_stride,
                         min_source_frames=data.min_source_frames,
                         max_residual_m=data.max_gt_fit_residual_m, data_config=data)
    store.index = index
    return index, store


def select_cases(index: PairedIndex, realities=None, limit: int | None = None) -> list:
    """The current captures to place: all of them, or the named ones, or the first N."""
    cases = list(index.current)
    if realities:
        wanted = set(realities)
        cases = [case for case in cases if case.reality_id in wanted]
    return cases[:limit] if limit else cases


# ----------------------------------------------------------------------------
# placing one capture

def _sheet_scale(structure, pool) -> tuple[float, str]:
    """Metres per paper inch, and where that number came from: the georeference, else the prior's
    rig height on this sheet (the one physical length a sheet without metres has)."""
    if structure.georeferenced:
        return geometry.design_scale(structure.design_tm), "design_tm"
    return io.metres_per_inch_for(structure, pool.rig_height_in if pool is not None else np.nan)


def _pose_record(proposal: PoseProposal, case, metres_per_inch: float) -> dict:
    """The transform the service ships, the rig height that sanity-checks it, and
    how likely the placement is to be within 2 m.

    `confidence` is the only quality signal an unscored production run has -- no
    ground truth means no rmse -- and it is computed here rather than by the
    caller so this and `evaluate.record_capture` cannot drift apart. The two
    have to agree: a split placed by `infer` and the same split placed by
    `evaluate` are the same pose.
    """
    camera_rig_height_m = rig_height_m(proposal.camera_height_above_floor,
                                       proposal.sim2.s, metres_per_inch)
    low, high = RIG_PLAUSIBLE_M
    return dict(tm=pose_transform_matrix_for(proposal, case.structure.design_tm),
                rig_height_m=camera_rig_height_m,
                rig_implausible=not low <= camera_rig_height_m <= high,
                confidence=confidence_for(proposal, metres_per_inch))


def _proposal_diagnostics(proposal: PoseProposal) -> dict:
    """What the search decided, for the row: which hypothesis, which window, how firmly."""
    return dict(up_index=proposal.up_index, window=proposal.window_index,
                n_inliers=proposal.num_inliers,
                inlier_frac=round(proposal.inlier_fraction, 3),
                n_matches=proposal.num_matches, n_verified=proposal.num_verified,
                refined=proposal.refined, n_crops=len(proposal.candidates),
                candidates=proposal.candidates)


@dataclass
class Prepared:
    """One split reduced to everything a search needs, before any search runs.

    Split out of `place_capture` so a capture can be searched more than once --
    the chain's recovery ladder retries with a wider prior, and reloading the
    cloud and refitting gravity between attempts would cost more than the
    retries themselves.
    """
    case: object
    trajectory: object
    views: dict                       # up_index -> GravityView, usable only
    metres_per_inch: float
    scale_source: str
    pool: object


def prepare_capture(case, store: CaptureStore, context: dict, *, stride: int = 1,
                    min_source_frames: int = 2,
                    whole_prior: bool = True) -> Prepared | Placement:
    """`Prepared` for a split that can be searched, or the `Placement` saying why not.

    `context` is filled as it goes, so a split that stops early still reports
    how far it got. A split with an IMU scale (`case.imu_scale`, the current CSV's
    `imu_scale`) is banded at the camera height it implies, in place of the floor
    fit's (`gravity.metric_scale_views`).
    """
    trajectory = io.load_camera_trajectory(case)
    cloud_xyz = io.load_point_cloud_xyz(case, stride, min_source_frames)
    gravity_fit, views = gravity.gravity_views(cloud_xyz, trajectory.positions)
    if case.imu_scale is not None:
        pool = store.pool_for_case(case, whole_capture=whole_prior)
        if pool is not None:
            rig_height_m = pool.rig_height_in * _sheet_scale(case.structure, pool)[0]
            views = gravity.metric_scale_views(cloud_xyz, trajectory.positions, gravity_fit, views,
                                               rig_height_m, case.imu_scale)
        context.update(imu_scale=case.imu_scale)
    usable_views = {up_index: view for up_index, view in views.items()
                    if not isinstance(view, str)}
    context.update(sign_ambiguous=bool(gravity_fit.sign_ambiguous),
                   num_views=len(usable_views))
    if not usable_views:
        return Placement("no_floor")

    pool = store.pool_for_case(case, whole_capture=whole_prior)
    metres_per_inch, scale_source = _sheet_scale(case.structure, pool)
    context.update(metres_per_inch=metres_per_inch, scale_source=scale_source)
    if pool is None:
        return Placement("no_prior")
    context.update(prior_splits=len(pool.splits), prior_key=pool.key,
                   whole_prior=whole_prior)
    return Prepared(case=case, trajectory=trajectory, views=usable_views,
                    metres_per_inch=metres_per_inch, scale_source=scale_source,
                    pool=pool)


def solve_prepared(model, data_config, prepared: Prepared,
                   options: PoseProposalOptions, device) -> PoseProposal | None:
    """The search itself, over an already-prepared split."""
    return propose_pose(model, prepared.views, prepared.pool, data_config,
                        prepared.metres_per_inch, options, device=device,
                        patch_size=model.patch_size)


def place_capture(model, config, case, store: CaptureStore, options: PoseProposalOptions,
                  device, context: dict, *, stride: int = 1,
                  min_source_frames: int = 2) -> Placement:
    """Solve one capture against every prepared split of its prior.

    `context` is filled as the solve goes, so a capture that stops early -- or
    raises -- still reports how far it got.
    """
    prepared = prepare_capture(case, store, context, stride=stride,
                               min_source_frames=min_source_frames)
    if isinstance(prepared, Placement):
        return prepared
    proposal = solve_prepared(model, config.data, prepared, options, device)
    if proposal is None:
        return Placement("no_pose")
    context.update(_proposal_diagnostics(proposal))
    return Placement("ok", pose=_pose_record(proposal, case, prepared.metres_per_inch),
                     proposal=proposal)


# ----------------------------------------------------------------------------
# per-image poses

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


def record_images(case, transform_matrix, images: dict | None, context: dict) -> None:
    """Write this split's images.json when asked; note where on its row.

    A failure here never costs the pose: the transform is already decided, and
    the row says the images were skipped rather than the run stopping.
    """
    if not images:
        return
    prefix = f"{case.reality_id}_{case.capture_id}_"
    try:
        path, count = write_images(
            case, transform_matrix, Path(images["dir"]), name=prefix + images["name"],
            every_nth=images["every_nth"], override_z=images["z"],
            plot=images["plot"], plot_name=f"{prefix}floormap.png",
            plot_width=images["plot_width"])
        context.update(images_path=str(path), images_n=count)
    except Exception as error:                                        # noqa: BLE001
        print(f"    images skipped: {type(error).__name__}: {error}")


# ----------------------------------------------------------------------------
# placing them all

def _describe(placement: Placement, context: dict) -> str:
    """One progress line's worth of a placement."""
    if placement.status == "ok":
        return (f"inliers {context['n_inliers']}/{context['n_matches']}  "
                f"up {context['up_index']} window {context['window']}  "
                f"({context['seconds']:.0f}s)")
    return {"no_floor": "no floor", "no_prior": "no usable prior",
            "no_pose": "no pose"}.get(placement.status, placement.status)


def place_captures(model, config, cases, store: CaptureStore, options: PoseProposalOptions,
                   stem, device, *, stride: int = 1,
                   images: dict | None = None) -> list[dict]:
    """Place every case; write `<stem>.jsonl`, `.csv` and `_predictions.jsonl`.

    A capture that cannot be placed still gets a row carrying its reason
    (`no_floor`, `no_prior`, `no_pose`, `error: ...`), so the denominator stays
    visible, and one that raises does not end the run. Returns the writer's
    rows: one per placed capture.
    """
    writer = SheetResultWriter(Path(stem))
    for position, case in enumerate(cases, 1):
        started = time.time()
        context: dict = {"matcher": options.matcher}
        failure = None
        try:
            placement = place_capture(model, config, case, store, options, device,
                                      context, stride=stride)
        except Exception as error:                                    # noqa: BLE001
            placement = Placement(f"error: {type(error).__name__}: {error}"[:200])
            failure = traceback.format_exc()
        context["seconds"] = round(time.time() - started, 1)
        if placement.pose:
            record_images(case, placement.pose["tm"], images, context)
        poses = {"predicted": placement.pose} if placement.pose else {}
        writer.write(case, poses, status=placement.status, **context)
        print(f"[{position}/{len(cases)}] {case.reality_id}: {_describe(placement, context)}",
              flush=True)
        if failure:
            print(failure)
    writer.close()
    return writer.rows


# ----------------------------------------------------------------------------
# placing them all, capture by capture

def capture_groups(cases) -> list[tuple[str, list]]:
    """The selected cases grouped by capture, captures in first-seen order.

    Order is kept rather than sorted so a `--limit` or `--realities` run still
    walks the CSV in the order it was written; what changes under `--chain` is
    that a capture's splits arrive together, which is the point.
    """
    groups: dict[str, list] = {}
    for case in cases:
        groups.setdefault(case.capture_id or case.reality_id, []).append(case)
    return list(groups.items())


def chain_callbacks(model, config, store, device, stride, *, whole_prior=True):
    """`(prepare, solve)` for `place_capture_chain`, over the real pipeline.

    Both swallow their exceptions. One split whose cloud is unreadable, or whose
    matcher dies on a degenerate crop, must not take the rest of its capture
    with it -- the chain simply treats it as a split that could not be placed
    and goes on leaning on the last one that could.
    """
    def prepare(case, context):
        try:
            prepared = prepare_capture(case, store, context, stride=stride,
                                       whole_prior=whole_prior)
        except Exception as error:                                # noqa: BLE001
            traceback.print_exc()
            return None, f"error: {type(error).__name__}: {error}"[:200]
        if isinstance(prepared, Placement):
            return None, prepared.status
        return prepared, None

    def solve(prepared, options):
        try:
            return solve_prepared(model, config.data, prepared, options, device)
        except Exception:                                         # noqa: BLE001
            traceback.print_exc()
            return None

    return prepare, solve


def place_captures_chained(model, config, cases, store: CaptureStore,
                           options: PoseProposalOptions, stem, device,
                           policy: ChainPolicy, *, stride: int = 1,
                           whole_prior: bool = True,
                           images: dict | None = None) -> list[dict]:
    """Place every case capture-major, each split leaning on the one before.

    Rows are still one per reality and are still written as each capture
    finishes, so `_predictions.jsonl` and everything downstream of it are
    unchanged in shape. What changes is their ORDER -- capture-major rather than
    CSV order -- and the chain columns.
    """
    writer = SheetResultWriter(Path(stem), extra_columns=CHAIN_COLUMNS)
    prepare, solve = chain_callbacks(model, config, store, device, stride,
                                     whole_prior=whole_prior)
    groups = capture_groups(cases)
    placed = 0
    for position, (capture_id, members) in enumerate(groups, 1):
        started = time.time()
        print(f"[{position}/{len(groups)}] {capture_id} -- {len(members)} split(s)",
              flush=True)
        try:
            results = place_capture_chain(members, prepare, solve, options, policy,
                                          geo_track_for=io.load_geo_track)
        except Exception as error:                                # noqa: BLE001
            traceback.print_exc()
            results = {}
            failure = f"error: {type(error).__name__}: {error}"[:200]
            for case in members:
                writer.write(case, {}, status=failure, matcher=options.matcher,
                             chain_mode="chain", chain_how="capture failed")
            continue
        seconds = round((time.time() - started) / max(len(members), 1), 1)
        for case in members:
            result = results.get(case.reality_id)
            if result is None:
                writer.write(case, {}, status="no_pose", matcher=options.matcher,
                             chain_mode="chain", chain_how="not reached")
                continue
            poses = {}
            if result.proposal is not None:
                poses = {"predicted": _pose_record(result.proposal, case,
                                                   result.metres_per_inch)}
                placed += 1
            context = dict(result.context, matcher=options.matcher, seconds=seconds)
            if result.proposal is not None:
                context.update(_proposal_diagnostics(result.proposal))
                record_images(case, poses["predicted"]["tm"], images, context)
            writer.write(case, poses, status=result.status, **context)
            print(f"      {case.reality_id}  {context.get('chain_how', '?'):<34} "
                  f"{'gate ok' if context.get('gate_passed') else 'GATE FAILED'}",
                  flush=True)
    writer.close()
    print(summarise_chain(writer.rows))
    return writer.rows


# ----------------------------------------------------------------------------
# command line

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="the training checkpoint")
    parser.add_argument("--csv", nargs="+", required=True, help="the captures to place")
    parser.add_argument("--prior-csv", nargs="+", required=True,
                        help="the prior captures they are placed against")
    parser.add_argument("--stem", required=True, help="output path prefix")
    parser.add_argument("--matcher", choices=MATCHERS, default="keypoints",
                        help="keypoints: points drawn on both maps, matched one to one; "
                             "dense: each query's soft-argmax over the prior")
    parser.add_argument("--geometric-verification", choices=MODES, default="none",
                        help="COLMAP LO-RANSAC over the matches before the Sim2 fit")
    parser.add_argument("--cache-dir", default=None,
                        help="the preprocessed prior captures; default: the checkpoint "
                             "config's data.cache_dir")
    parser.add_argument("--prior-mosaic", action=argparse.BooleanOptionalAction, default=True,
                        help="a prior drawn from several dates: each 5 m cell keeps "
                             "the walls of the newest date only (the default); "
                             "--no-prior-mosaic stacks every split. A single-date "
                             "prior is the same either way")
    parser.add_argument("--backbone", default=None,
                        help="DINO weights to load; default: the checkpoint's own path")
    parser.add_argument("--realities", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None,
                        help="default: the checkpoint config's eval_resolution")
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--tolerance-m", type=float, default=0.5)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--no-refine", action="store_true", help="skip the zoomed second pass")
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
    parser.add_argument("--stride", type=int, default=1)
    images = parser.add_argument_group("per-image poses")
    images.add_argument("--images", action="store_true",
                        help="write each photograph's predicted pose per reality")
    images.add_argument("--images-dir", default=None, help="default: <stem>_images/")
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
    images.add_argument("--images-plot-width", type=int, default=1800)
    add_chain_arguments(parser)
    return parser.parse_args(argv)


def images_options(args: argparse.Namespace, stem: Path) -> dict | None:
    """What `record_images` needs, or None when --images is off."""
    if not args.images:
        return None
    return dict(dir=args.images_dir or f"{stem}_images", name=args.images_name,
                every_nth=args.images_every_nth, z=args.images_z,
                plot=args.images_plot, plot_width=args.images_plot_width)


def proposal_options(args: argparse.Namespace, data_config) -> PoseProposalOptions:
    return PoseProposalOptions(
        resolution=args.resolution or data_config.eval_resolution,
        num_queries=args.queries, tolerance_m=args.tolerance_m,
        min_inliers=args.min_inliers, refine_on_footprint=not args.no_refine,
        refine_footprint_margin=data_config.refine_footprint_margin,
        geometric_verification=args.geometric_verification, matcher=args.matcher,
        search_zoom_out=args.search_zoom_out, max_windows=args.max_windows,
        window_stride_fraction=args.window_stride)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "cache" / "hf"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_matcher(args.ckpt, device, backbone=args.backbone)
    if args.cache_dir:
        config.data.cache_dir = args.cache_dir
    config.data.prior_mosaic = args.prior_mosaic
    index, store = load_index_and_store(config, args.csv, args.prior_csv)
    cases = select_cases(index, args.realities, args.limit)
    options = proposal_options(args, config.data)
    stem = Path(args.stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    images = images_options(args, stem)

    print(f"{len(cases)} captures -> {stem}   ({index.coverage()})")
    print(f"  checkpoint: {args.ckpt}  (run {config.run_name})")
    print(f"  matcher {options.matcher}, verification {options.geometric_verification}, "
          f"resolution {options.resolution}, cache {config.cache_dir}")
    if args.chain:
        policy = chain_policy(args)
        print(f"  chaining: seed {policy.seed_source}, radii x{policy.radius_scale}, "
              f"{policy.max_windows_with_prior}+{policy.insurance_windows} windows, "
              f"{policy.retries} retries")
        rows = place_captures_chained(model, config, cases, store, options, stem,
                                      device, policy, stride=args.stride, images=images)
    else:
        rows = place_captures(model, config, cases, store, options, stem, device,
                              stride=args.stride, images=images)
    # Said for both branches, and said LAST: nothing here reads ground truth, so
    # the confidence is the only thing this entry point can say about whether the
    # captures it just placed were placed well.
    print(summarise_confidence(rows))
    print(f"{len(rows)}/{len(cases)} captures placed -> {stem}_predictions.jsonl")


if __name__ == "__main__":
    main()
