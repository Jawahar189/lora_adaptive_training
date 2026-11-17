"""Compare two chained evaluations, or a chained one against an unchained baseline.

    python scripts/e12_chain_compare.py --before out/bevaligner/chain/a_base \
        --after out/bevaligner/chain/e_full

Reads the `<stem>.jsonl` both runs wrote -- the CSV drops `candidates`, and the
metrics live under `poses.predicted` rather than at the top level, so the JSONL
is the only complete record. `--merge` first concatenates a sharded run, since a
capture-major run parallelises by splitting captures across processes and the
pieces have to be put back together before anything is pooled.

Every comparison here is PAIRED on `reality_id`. A headline count can hide equal
numbers of gains and regressions, and on a change that re-seeds whole captures
that is not a hypothetical: the populations either side are the same realities,
so they are the only honest control for each other.

Cost is reported in full-search equivalents rather than seconds, because seconds
move with GPU contention and with how many runs share the box. One unconstrained
search is 1.0; a search with a prior renders `max_windows_with_prior +
insurance_windows` of the 32 windows the unconstrained one does, so it is about
0.22, and the widened rung about 0.37. Those three numbers are the whole model,
and when the wall clock disagrees with it the difference is everything the model
leaves out -- `prepare`'s cloud load and gravity fit above all.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

# Rung cost in full-search equivalents; see the module docstring.
FSE = {1: 0.22, 2: 0.37, 3: 1.00}
GOOD_T1 = 0.5          # a reality is placed when more than half its images land <1 m
GOOD_T2 = 0.9


def merge(stems: list[Path], out: Path) -> Path:
    """Concatenate a sharded run into one stem. Returns the merged stem."""
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    with open(out.with_suffix(".jsonl"), "w") as fh:
        for stem in stems:
            for line in Path(f"{stem}.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["reality_id"] in seen:
                    continue
                seen.add(row["reality_id"])
                fh.write(line + "\n")
    with open(out.parent / (out.name + "_predictions.jsonl"), "w") as fh:
        for stem in stems:
            path = Path(f"{stem}_predictions.jsonl")
            if path.exists():
                fh.write(path.read_text())
    print(f"merged {len(stems)} shards -> {out}.jsonl  ({len(seen)} realities)")
    return out


def load(stem: Path) -> dict:
    """Rows keyed by reality id, with the metrics lifted out of `poses.predicted`."""
    path = stem if stem.suffix == ".jsonl" else Path(f"{stem}.jsonl")
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        predicted = (row.get("poses") or {}).get("predicted")
        metrics = {k: v for k, v in (predicted or {}).items() if k != "tm"}
        out[row["reality_id"]] = dict({k: v for k, v in row.items() if k != "poses"},
                                      **metrics, _placed=bool(predicted))
    return out


def bucket(row) -> str:
    """`chain_how` reduced to a bucket, with reach-backs kept separate.

    A split leaning on its neighbour and one reaching back past a failure have
    different expected error; fusing them makes the run unreadable.
    """
    how = (row.get("chain_how") or "unchained")
    name = (how.split(")")[0] + ")" if how.startswith(("seed", "independent"))
            else how.split(" (")[0])
    try:
        if int(row.get("chain_reach") or 0) > 1:
            name += " (reached)"
    except (TypeError, ValueError):
        pass
    return name


def cost_fse(rows: dict) -> float:
    total = 0.0
    for row in rows.values():
        attempts = max(int(row.get("chain_attempt") or 0), 1)
        if not row.get("chain_prior_radius_in"):
            total += 1.00 * attempts           # unconstrained rungs
        else:
            total += sum(FSE.get(i, 1.0) for i in range(1, attempts + 1))
    return total


def headline(before: dict, after: dict, keys: list[str]) -> None:
    g1 = lambda r: (r.get("tpr@1.0") or 0) > GOOD_T1          # noqa: E731
    g2 = lambda r: (r.get("tpr@2.0") or 0) > GOOD_T2          # noqa: E731
    print(f"\n{len(keys)} realities placed in both runs\n")
    print(f'{"":24} {"before":>10} {"after":>10} {"delta":>8}')
    for label, test in (("tpr@1.0 > 0.5", g1), ("tpr@2.0 > 0.9", g2)):
        a = sum(test(before[k]) for k in keys)
        b = sum(test(after[k]) for k in keys)
        print(f"{label:24} {a:>10} {b:>10} {b - a:>+8}")
        gained = sum(1 for k in keys if test(after[k]) and not test(before[k]))
        lost = sum(1 for k in keys if test(before[k]) and not test(after[k]))
        print(f'{"  gained / lost":24} {"":>10} {"":>10} {gained:>5} /{lost:>2}')
    ra = np.array([before[k]["rmse_xy"] for k in keys], dtype=float)
    rb = np.array([after[k]["rmse_xy"] for k in keys], dtype=float)
    for label, fn in (("rmse_xy median", np.median),
                      ("rmse_xy p90", lambda x: np.percentile(x, 90))):
        print(f"{label:24} {fn(ra):>9.1f}m {fn(rb):>9.1f}m {fn(rb) - fn(ra):>+7.1f}m")
    print(f'{"  better / worse":24} {"":>10} {"":>10} '
          f"{(rb < ra - 0.01).sum():>5} /{(rb > ra + 0.01).sum():>2}")


def buckets(rows: dict) -> None:
    groups = collections.defaultdict(list)
    for row in rows.values():
        if row.get("tpr@1.0") is not None:
            groups[bucket(row)].append(row)
    stats = []
    for name, group in groups.items():
        n = len(group)
        stats.append((np.median([r["rmse_xy"] for r in group]), name, n,
                      sum(1 for r in group if r["tpr@1.0"] > GOOD_T1) / n,
                      sum(1 for r in group if (r.get("tpr@2.0") or 0) > GOOD_T2) / n,
                      sum(1 for r in group if r.get("gate_passed")) / n))
    stats.sort(reverse=True)
    print(f'\n{"bucket":<38}{"n":>5}{"rmse med":>10}{"1.0>0.5":>9}'
          f'{"2.0>0.9":>9}{"gated":>7}')
    print("-" * 78)
    for med, name, n, f05, t09, gated in stats:
        print(f"{name:<38}{n:>5}{med:>9.2f}m{f05:>9.2f}{t09:>9.2f}{gated:>7.2f}")


def transitions(before: dict, after: dict, keys: list[str]) -> None:
    """Where each bucket's realities ended up, and what it cost them."""
    moves = collections.defaultdict(list)
    for k in keys:
        moves[(bucket(before[k]), bucket(after[k]))].append(after[k])
    print(f'\n{"from":<38}{"to":<30}{"n":>5}{"rmse med":>10}')
    print("-" * 83)
    for (src, dst), group in sorted(moves.items(), key=lambda kv: -len(kv[1]))[:14]:
        if src == dst:
            continue
        print(f"{src:<38}{dst:<30}{len(group):>5}"
              f'{np.median([r["rmse_xy"] for r in group]):>9.1f}m')


def propagation(after: dict) -> None:
    """Whether a reach-back pose that gated then served as a bad anchor."""
    reach = {k for k, r in after.items() if _reach(r) > 1 and r.get("gate_passed")}
    children = collections.defaultdict(list)
    for row in after.values():
        parent = row.get("chain_seeded_from")
        if parent and row.get("tpr@1.0") is not None:
            children["reached" if parent in reach else "adjacent"].append(row)
    print("\nchildren of each kind of anchor:")
    for kind, group in sorted(children.items()):
        if not group:
            continue
        print(f'  anchored on a {kind:<9} anchor: n={len(group):<4} '
              f'rmse med {np.median([r["rmse_xy"] for r in group]):>7.1f}m  '
              f'correct {sum(1 for r in group if r["tpr@1.0"] > GOOD_T1) / len(group):.2f}')
    bad = [r for r in after.values()
           if r.get("gate_passed") and (r.get("rmse_xy") or 0) > 20]
    print(f"  gated but >20 m out: {len(bad)}"
          f'  (of which reach-backs: {sum(1 for r in bad if _reach(r) > 1)})')


def _reach(row) -> int:
    try:
        return int(row.get("chain_reach") or 0)
    except (TypeError, ValueError):
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--merge", nargs="*", type=Path, default=None,
                        help="shard stems to concatenate into --after first")
    args = parser.parse_args(argv)

    if args.merge:
        merge(list(args.merge), args.after)
    before, after = load(args.before), load(args.after)
    keys = [k for k in after if k in before
            and before[k].get("rmse_xy") is not None
            and after[k].get("rmse_xy") is not None]
    print(f"before {args.before}: {len(before)} rows, {sum(r['_placed'] for r in before.values())} placed")
    print(f"after  {args.after}: {len(after)} rows, {sum(r['_placed'] for r in after.values())} placed")
    headline(before, after, keys)
    buckets(after)
    transitions(before, after, keys)
    propagation(after)
    ca, cb = cost_fse(before), cost_fse(after)
    print(f"\ncost: {ca:.0f} -> {cb:.0f} FSE  ({cb / max(ca, 1):.2f}x)")


if __name__ == "__main__":
    main()
