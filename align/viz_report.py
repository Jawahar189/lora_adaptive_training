"""Turn a run's detail JSONL into a self-contained HTML viewer.

Renders `viz_prior.render`'s per-reality PNGs (ground truth, reference,
placement, alternatives, run summary) as a filterable, sortable gallery, one
card per reality, all shown by default. Nothing is embedded -- the page links
to the PNGs by a path relative to itself, so it stays cheap to generate and
diff even over a thousand-reality run.
"""
import json
import os
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "viz_report_template.html"

FIELDS = [
    "reality_id", "capture_id", "structure", "status", "georeferenced",
    "capture_date", "prior_capture_date", "prior_same_day", "overlap_5m",
    "n_images", "top1_correct", "pose_error_m", "yaw_error_deg",
    "scale_ratio", "score", "rmse_xy", "median_xy", "p90_xy", "max_xy",
    "tpr@1.0", "seconds", "n_hypotheses", "sign_ambiguous", "sign_uncertain",
    "yaw_fallback", "rig_implausible", "floor_fit_corrected", "scale_source",
]


def _rows(detail_jsonl: Path, viz_dir: Path) -> list[dict]:
    rows = []
    with open(detail_jsonl) as f:
        for line in f:
            d = json.loads(line)
            row = {k: d.get(k) for k in FIELDS}
            row["warnings"] = d.get("warnings") or []
            row["has_viz"] = (viz_dir / d["reality_id"]).is_dir()
            rows.append(row)
    return rows


def write(detail_jsonl, viz_dir, out_path, title=None, subtitle=None) -> Path:
    """Render the gallery for one run's detail JSONL to `out_path`.

    `viz_dir` is the directory `--visualise` wrote its per-reality PNGs into
    (`viz_prior.render`'s output); a reality missing from it renders with a
    "no visualization available" placeholder instead of broken images.
    """
    detail_jsonl = Path(detail_jsonl)
    viz_dir = Path(viz_dir)
    out_path = Path(out_path)
    rows = _rows(detail_jsonl, viz_dir)
    viz_base = (os.path.relpath(viz_dir, start=out_path.parent)
                if viz_dir.exists() else viz_dir.name)

    html = TEMPLATE_PATH.read_text()
    html = html.replace("__TITLE__", title or f"Alignment results — {out_path.stem}")
    html = html.replace("__SUBTITLE__", subtitle or
                         f"{detail_jsonl.name} · all results shown by default, "
                         "filter with the dropdowns below")
    html = html.replace("__RESULTS_JSON__", json.dumps(rows))
    html = html.replace("__VIZ_BASE_JSON__", json.dumps(viz_base))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
