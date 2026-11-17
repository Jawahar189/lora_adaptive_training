"""Per-photograph poses from a set of predicted transforms, one JSON per reality.

    python scripts/predicted_images.py \
        --pred out/bevaligner/infer/full_predictions.jsonl \
        --csv datasets/splits/prior_bev_test.csv \
        --out out/bevaligner/infer/images

`bevaligner.infer` writes transforms, not per-image poses -- only
`bevaligner.evaluate --images` does, which is the scored path. This closes
that gap: it takes any `{reality_id, tm}` JSONL and writes the file the floormap
viewer reads, `{filename: {"position": [x,y,z], "rotation": [x,y,z]}}` with the
rotation in three.js Euler XYZ degrees.

It is a PURE FUNCTION of the transform and the capture's own shots -- the same
`shot_poses` + `poses_to_images` pair `evaluate.write_images` calls, and the same
pair `scripts/fetch_dataset.py` used to write the delivered `images.json`. So
re-solving to obtain these files buys nothing: the transform already determines
them, and a file written here is directly comparable with a delivered one.

`--score` compares against that delivered `images.json`, which is the same
function applied to the registered transform. Position error is therefore the
only thing that differs, and it is reported per reality and pooled. Note the
unit: `images.json` lives in the structure's TARGET frame, which is UTM metres
where a design transform exists and paper inches where it does not -- so the two
groups are reported apart and never averaged together.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.io.cases import CaseIndex                      # noqa: E402
from align.io.floormap import every_nth as every_nth_image  # noqa: E402
from align.io.floormap import poses_to_images, shot_poses   # noqa: E402


def load_predictions(path: Path) -> dict:
    """`{reality_id: 4x4}` from a predictions JSONL, skipping abstentions."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tm") is None:
            continue
        out[row["reality_id"]] = np.asarray(row["tm"], dtype=np.float64).reshape(4, 4)
    return out


def predicted_images(case, tm: np.ndarray, *, every_nth: int = 1,
                     override_z: float | None = None) -> dict:
    """The viewer's file for one reality, under the predicted transform."""
    images = poses_to_images(shot_poses(case.rlt_dir), tm, override_z=override_z)
    return every_nth_image(images, every_nth) if every_nth > 1 else images


def position_error(predicted: dict, delivered: dict) -> np.ndarray | None:
    """Per-photograph distance between two of these files, over shared names.

    Keyed on the filename both carry, so a prediction written at `--every-nth`
    still scores against the images it actually holds rather than counting the
    ones it skipped as misses.
    """
    shared = sorted(set(predicted) & set(delivered))
    if not shared:
        return None
    a = np.array([predicted[n]["position"][:2] for n in shared], dtype=np.float64)
    b = np.array([delivered[n]["position"][:2] for n in shared], dtype=np.float64)
    return np.linalg.norm(a - b, axis=1)


def summarise(rows: list[dict]) -> str:
    """Pooled error, kept apart by frame -- metres and paper inches do not mix."""
    lines = []
    for geo, unit in ((True, "UTM metres"), (False, "paper inches")):
        group = [r for r in rows if r["georeferenced"] is geo and r["n"] > 0]
        if not group:
            continue
        med = np.array([r["median"] for r in group])
        rmse = np.array([r["rmse"] for r in group])
        images = sum(r["n"] for r in group)
        lines.append(f"  {unit}  ({len(group)} realities, {images:,} photographs)")
        lines.append(f"    median-of-medians {np.median(med):8.2f}   p90 {np.percentile(med, 90):8.2f}")
        lines.append(f"    median rmse       {np.median(rmse):8.2f}   p90 {np.percentile(rmse, 90):8.2f}")
        if geo:
            for t in (1.0, 2.0, 5.0):
                share = float(np.mean([r["within"][t] for r in group]))
                lines.append(f"    images within {t:>4.1f} m : {share:.3f}")
    return "\n".join(lines) or "  nothing scored"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", required=True, type=Path,
                        help="JSONL of predicted transforms: {reality_id, tm}")
    parser.add_argument("--csv", required=True, type=Path,
                        help="the dataset CSV naming the realities")
    parser.add_argument("--out", required=True, type=Path,
                        help="directory for the per-reality JSON files")
    parser.add_argument("--name", default="{reality_id}_{capture_id}_images.json",
                        help="filename template; {reality_id} and {capture_id} expand")
    parser.add_argument("--every-nth", type=int, default=1,
                        help="3 keeps indices 1, 4, 7 ... the subset production ships")
    parser.add_argument("--images-z", type=float, default=None,
                        help="pin every z, as the viewer's own file does")
    parser.add_argument("--score", action="store_true",
                        help="also compare against the delivered images.json beside "
                             "each capture, and report position error")
    args = parser.parse_args(argv)

    preds = load_predictions(args.pred)
    cases = {c.reality_id: c for c in CaseIndex.load(args.csv, repo_root=Path.cwd())}
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(preds)} predicted transforms, {len(cases)} realities in the CSV")

    written = skipped = 0
    rows: list[dict] = []
    for reality_id, tm in sorted(preds.items()):
        case = cases.get(reality_id)
        if case is None:
            skipped += 1
            continue
        try:
            images = predicted_images(case, tm, every_nth=args.every_nth,
                                      override_z=args.images_z)
        except Exception as error:                                # noqa: BLE001
            print(f"  {reality_id}: {type(error).__name__}: {error}")
            skipped += 1
            continue
        if not images:
            skipped += 1
            continue
        path = args.out / args.name.format(reality_id=reality_id,
                                           capture_id=case.capture_id)
        path.write_text(json.dumps(images, indent=2))
        written += 1

        if args.score:
            delivered_path = Path(case.rlt_dir) / "images.json"
            if not delivered_path.exists():
                continue
            try:
                delivered = json.loads(delivered_path.read_text())
            except json.JSONDecodeError:
                continue
            err = position_error(images, delivered)
            if err is None:
                continue
            rows.append(dict(reality_id=reality_id, n=len(err),
                             georeferenced=bool(case.structure.georeferenced),
                             median=float(np.median(err)),
                             rmse=float(np.sqrt(np.mean(err ** 2))),
                             p90=float(np.percentile(err, 90)),
                             within={t: float(np.mean(err <= t)) for t in (1.0, 2.0, 5.0)}))

    print(f"wrote {written} files -> {args.out}" + (f"   ({skipped} skipped)" if skipped else ""))
    if args.score:
        print(f"\nposition error against the delivered images.json "
              f"({len(rows)} realities compared)")
        print(summarise(rows))
        out = args.out.parent / (args.out.name + "_error.jsonl")
        with open(out, "w") as fh:
            for r in rows:
                fh.write(json.dumps({k: v for k, v in r.items() if k != "within"}) + "\n")
        print(f"\nper-reality error -> {out}")


if __name__ == "__main__":
    main()
