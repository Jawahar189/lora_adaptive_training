"""Ground-truth self-consistency, in pixels and metres.

Where two splits of one capture share image names, both transforms claim a
position for the same photograph. The disagreement is an image-to-image error
that needs no prediction and no matcher -- and it is a hard floor on what any
method can score, because at least one of the two transforms is wrong.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align.frames import is_identity, target_to_paper
from align.groundtruth import gt_trajectory
from align.io.cases import CaseIndex


def main():
    index = CaseIndex.load_all()
    print(f"{'capture':38s} {'shared':>7s} {'px med':>9s} {'px max':>9s} "
          f"{'m med':>8s} {'m max':>8s}")
    rows = []
    for cap, cases in sorted(index.by_capture().items()):
        if len(cases) < 2:
            continue
        traj = {c.reality_id: gt_trajectory(c.rlt_dir) for c in cases}
        for i in range(len(cases)):
            for j in range(i + 1, len(cases)):
                a, b = cases[i], cases[j]
                ta, tb = traj[a.reality_id], traj[b.reality_id]
                shared = sorted(set(ta.names) & set(tb.names))
                if not shared:
                    continue
                ia = {n: k for k, n in enumerate(ta.names)}
                ib = {n: k for k, n in enumerate(tb.names)}
                pa = ta.positions[[ia[n] for n in shared]]
                pb = tb.positions[[ib[n] for n in shared]]
                st = a.structure
                g = st.geometry()
                qa = g.paper_to_pixel(target_to_paper(pa, st.design_tm)[:, :2])
                qb = g.paper_to_pixel(target_to_paper(pb, st.design_tm)[:, :2])
                d_px = np.linalg.norm(qa - qb, axis=1)
                d_m = (np.linalg.norm(pa - pb, axis=1)
                       if not is_identity(st.design_tm) else None)
                rows.append(dict(capture=cap, structure=st.key, n=len(shared),
                                 px_med=float(np.median(d_px)),
                                 px_max=float(d_px.max()),
                                 m_med=float(np.median(d_m)) if d_m is not None else None,
                                 m_max=float(d_m.max()) if d_m is not None else None))
                r = rows[-1]
                print(f"{Path(cap).name[:38]:38s} {r['n']:7d} "
                      f"{r['px_med']:9.1f} {r['px_max']:9.1f} "
                      + (f"{r['m_med']:8.2f} {r['m_max']:8.2f}"
                         if r['m_med'] is not None else f"{'—':>8s} {'—':>8s}"))
    Path("out").mkdir(exist_ok=True)
    with open("out/gt_consistency.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    px = np.array([r["px_med"] for r in rows])
    print(f"\n{len(rows)} overlapping split pairs | median disagreement "
          f"{np.median(px):.1f} px | worst {px.max():.0f} px")
    bad = [r for r in rows if r["px_med"] > 50]
    print(f"pairs disagreeing by more than 50 px: {len(bad)}")
    for r in sorted(bad, key=lambda r: -r["px_med"]):
        m = f"{r['m_med']:.1f} m" if r["m_med"] is not None else "no georef"
        print(f"   {r['structure']:26s} {r['px_med']:7.0f} px = {m}  ({r['n']} images)")
    print("-> out/gt_consistency.jsonl")


if __name__ == "__main__":
    main()
