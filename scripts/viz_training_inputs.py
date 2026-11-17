#!/usr/bin/env python3
"""Draw what the training loader actually hands the network, from the cache.

Builds the train dataset exactly as `BEVAlignerDataModule` does -- same
config, same splits, same `structure_balance` draw, same augmentation -- and
renders each sample as `current density map | prior BEV window` with the
ground-truth correspondences across the seam. No model is loaded and nothing
touches the GPU, so this is safe to run beside a training job.

    .venv-a100/bin/python scripts/viz_training_inputs.py \
        --config bevaligner/configs/prior_bev.yaml --num 24

Each sample also gets the numbers that decide whether it CAN be learned:

  occupancy      fraction of density pixels with any point in them. A nearly
                 empty map is a capture rendered into too large an extent.
  footprint      how many patches across the capture spans. `min_footprint_
                 patches` (4) is the floor below which there is no signal.
  valid          target patches the prior has something in -- wall, or ground
                 it walked. The softmax is masked to these, so a low number is
                 a sample whose truth may have nowhere legal to land.
  overlap        the share of this walk's cameras the prior went near. Below
                 `data.min_overlap` the row is not in the dataset at all; just
                 above it, the window is placed on the covered part and most of
                 the density map has no counterpart.
  duplication    the mean over queries of log(queries sharing its target
                 patch). This is the InfoNCE FLOOR: pairs that land on one
                 patch are indistinguishable, so with
                 `loss.near_duplicate_radius: 0` no amount of training pushes
                 the feature loss below it. Compare it to the feature loss the
                 run is actually sitting at.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bevaligner.config import load_config  # noqa: E402
from bevaligner.data.datamodule import BEVAlignerDataModule  # noqa: E402
from bevaligner.utils import image as image_utils  # noqa: E402
from bevaligner.viz import render_match_image  # noqa: E402


def panel(tensor) -> np.ndarray:
    return image_utils.tensor_to_rgb_uint8(image_utils.imagenet_denormalize(tensor))


def duplication_nats(target_px: np.ndarray, patch_size: int) -> tuple[float, float]:
    """Mean log(queries sharing a target patch), and the worst query's count.

    A query whose target patch holds `d` queries cannot be told from the other
    `d - 1` by any feature, so the best achievable InfoNCE for it is log(d).
    """
    cells = [tuple(v) for v in np.floor(target_px / patch_size).astype(int)]
    counts = Counter(cells)
    per_query = np.array([counts[c] for c in cells], dtype=np.float64)
    return float(np.log(per_query).mean()), float(per_query.max())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="bevaligner/configs/prior_bev.yaml")
    parser.add_argument("--out-dir", default="out/training_inputs")
    parser.add_argument("--num", type=int, default=24)
    parser.add_argument("--start", type=int, default=0, help="first loader index to draw")
    parser.add_argument("--resolution", type=int, default=None,
                        help="fix the resolution; default cycles the config's")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-lines", type=int, default=40)
    parser.add_argument("--no-augment", action="store_true",
                        help="render the evaluation view of the same records")
    parser.add_argument("--stats-only", action="store_true",
                        help="skip the drawing, which is most of the time, and "
                             "report the numbers over many more samples")
    parser.add_argument("--csv", default=None, help="write the per-sample table here")
    parser.add_argument("overrides", nargs="*",
                        help="key=value YAML overrides, e.g. data.cache_dir=cache/smoke")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    if args.no_augment:
        config.data.augment = False
    datamodule = BEVAlignerDataModule(config, patch_size=args.patch_size)
    datamodule.setup()
    dataset = datamodule.train_dataset
    resolutions = ([args.resolution] if args.resolution else list(config.data.resolutions))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'idx':>6} {'reality':>20} {'res':>5} {'occupancy':>10} {'footprint':>10} "
          f"{'valid':>7} {'overlap':>8} {'dup nats':>9} {'worst':>6}")
    rows = []
    for offset in range(args.num):
        index = args.start + offset
        resolution = resolutions[offset % len(resolutions)]
        sample = dataset[(index, resolution)]
        meta = sample["meta"]
        density, target = panel(sample["density"]), panel(sample["target"])
        query_px = sample["query_px"].numpy()
        target_px = sample["target_px"].numpy()

        occupancy = float((density.max(axis=2) > 0).mean())
        span = target_px.max(axis=0) - target_px.min(axis=0)
        footprint = float(np.hypot(*span) / args.patch_size)
        valid = int(sample["target_patch_valid"].sum())
        dup, worst = duplication_nats(target_px, args.patch_size)

        if not args.stats_only:
            caption = (f"{meta['reality_id']}  R{meta['resolution']}  "
                       f"occ {occupancy:.2f}  footprint {footprint:.0f}p  "
                       f"overlap {meta['overlap']:.2f}  dup {dup:.2f} nats")
            image = render_match_image(density, target, query_px, target_px,
                                       num_lines=args.num_lines, caption=caption)
            image.save(out_dir / f"{index:05d}_{meta['reality_id']}_R{meta['resolution']}.png")
        print(f"{index:>6} {meta['reality_id']:>20} {meta['resolution']:>5} "
              f"{occupancy:>10.3f} {footprint:>10.1f} {valid:>7} "
              f"{meta['overlap']:>8.2f} {dup:>9.2f} {worst:>6.0f}")
        rows.append((index, meta["reality_id"], meta["resolution"], occupancy,
                     footprint, valid, meta["overlap"], dup, worst))

    columns = ["index", "reality_id", "resolution", "occupancy", "footprint_patches",
               "valid_patches", "overlap", "duplication_nats", "worst_patch_queries"]
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    occupancy, footprint, valid, dup, worst = (
        np.array([row[i] for row in rows], dtype=float) for i in (3, 4, 5, 6, 7))
    queries = query_px.shape[0]
    chance = float(np.log(queries))
    print(f"\nover {len(rows)} samples")
    for name, values in (("occupancy", occupancy), ("footprint patches", footprint),
                         ("valid patches", valid), ("duplication nats", dup)):
        q = np.percentile(values, [5, 25, 50, 75, 95])
        print(f"  {name:<18} min {values.min():7.2f}  p5 {q[0]:7.2f}  p25 {q[1]:7.2f}  "
              f"med {q[2]:7.2f}  p75 {q[3]:7.2f}  p95 {q[4]:7.2f}  max {values.max():7.2f}")

    floor = float(dup.mean())
    print(f"\nInfoNCE floor from duplication alone: {floor:.2f} nats  "
          f"(chance is ln({queries}) = {chance:.2f})")
    print(f"  so the most a perfect matcher could win here is {chance - floor:.2f} nats.")
    print(f"\n  samples below min_footprint_patches "
          f"({config.data.min_footprint_patches:g}): "
          f"{(footprint < config.data.min_footprint_patches).sum():>4} "
          f"({(footprint < config.data.min_footprint_patches).mean():.1%})")
    for limit in (3.0, 5.0, 6.0):
        n = (dup > limit).sum()
        print(f"  samples whose floor exceeds {limit:.0f} nats:            {n:>4} "
              f"({n / len(dup):.1%})")
    degenerate = (worst >= queries * 0.9).sum()
    print(f"  samples with ~every query on ONE patch:     {degenerate:>4} "
          f"({degenerate / len(dup):.1%})  <- no learnable signal at all")
    if not args.stats_only:
        print(f"\nimages in {out_dir}")


if __name__ == "__main__":
    main()
