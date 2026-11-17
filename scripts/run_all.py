"""Run the alignment pipeline end to end, stage by stage.

  python scripts/run_all.py                 # every stage, all 132 splits
  python scripts/run_all.py --root data     # every stage, data/ only
  python scripts/run_all.py --stages e1 e2  # just those
  python scripts/run_all.py --dry-run       # print the commands, run nothing

Stages run in order because each consumes the previous one's output: the sheet
scale E1 derives is what E2 uses to convert its metre-based thresholds, and the
visualiser reads both.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable

STAGES = {
    "tests": dict(
        desc="frame algebra, leakage boundary and identity checks",
        cmd=lambda a: [PY, "-m", "pytest", "tests/", "-q"]),
    "e1": dict(
        desc="metric anchor: gravity, local floor, camera-height scale prior",
        cmd=lambda a: [PY, "scripts/e1_scale_anchor.py",
                       "--out", f"out/e1_{a.tag}.jsonl"] + a.csv_args),
    "sheets": dict(
        desc="wall extraction over every sheet, cached per structure",
        cmd=lambda a: [PY, "scripts/sheets.py"]
        + (["--root", a.root] if a.root else ["--all"])),
    "e2": dict(
        desc="oracle inlier count at the ground-truth pose",
        cmd=lambda a: [PY, "scripts/e2_oracle_inliers.py",
                       "--out", f"out/e2_{a.tag}.jsonl"] + a.csv_args),
    "consistency": dict(
        desc="ground-truth self-consistency in pixels and metres",
        cmd=lambda a: [PY, "scripts/consistency.py"]),
    "solve": dict(
        desc="coarse pose search through the reality-major pipeline",
        cmd=lambda a: [PY, "scripts/solve.py"]
        + (["--root", a.root] if a.root else ["--all"])
        + ["--out", f"out/predictions_{a.tag}.jsonl",
           "--hypotheses-out", f"out/hypotheses_{a.tag}.jsonl"]),
    "evaluate": dict(
        desc="image-to-image error of those predictions, in px and UTM",
        cmd=lambda a: [PY, "scripts/evaluate.py",
                       "--pred", f"out/predictions_{a.tag}.jsonl",
                       "--out", f"out/metrics_{a.tag}.jsonl"] + a.csv_args),
    "viz": dict(
        desc="render all pipeline stages as panels",
        cmd=lambda a: [PY, "scripts/viz_stages.py"]
        + (["--root", a.root] if a.root else ["--all"])
        + ["--out", "out/stages"]
        + (["--no-contact-sheet"] if a.no_contact_sheet else [])),
}
ORDER = ["tests", "e1", "sheets", "e2", "consistency", "solve", "evaluate",
         "viz"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=ORDER, default=ORDER)
    ap.add_argument("--root", default=None,
                    help="restrict to one data folder (viz and the CSV-driven stages)")
    ap.add_argument("--no-contact-sheet", "--no-contact", dest="no_contact_sheet",
                    action="store_true",
                    help="skip contact_sheet.png in the viz stage; the individual "
                         "panels are still written")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a stage fails")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each stage's command without running it")
    args = ap.parse_args()

    args.tag = args.root or "all"
    # In the older generation each CSV was exactly one data folder, so
    # restricting by root meant swapping the CSV. That correspondence is gone --
    # the current index holds both roots in one file -- so the CSV-driven stages
    # take the root itself and filter on it, the same way the others already do.
    args.csv_args = ["--root", args.root] if args.root else []

    print(f"pipeline: {' -> '.join(args.stages)}   (root={args.root or 'all'})\n")
    failed = []
    for name in [s for s in ORDER if s in args.stages]:
        stage = STAGES[name]
        cmd = stage["cmd"](args)
        print("=" * 78)
        print(f"STAGE {name}  —  {stage['desc']}")
        print(f"$ {' '.join(cmd)}")
        print("=" * 78, flush=True)
        if args.dry_run:
            print()
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT)
        dt = time.time() - t0
        if rc != 0:
            failed.append(name)
            print(f"\n!! stage {name} exited {rc} after {dt:.0f}s", flush=True)
            if not args.keep_going:
                break
        else:
            print(f"\n-- stage {name} ok in {dt:.0f}s\n", flush=True)

    print("=" * 78)
    print(f"done. {len(args.stages) - len(failed)}/{len(args.stages)} stages ok"
          + (f", failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
