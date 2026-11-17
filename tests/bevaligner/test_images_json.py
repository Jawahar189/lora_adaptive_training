"""The per-photograph pose file: layout, contents, and the knobs that shape it.

`tests/test_floormap.py` pins the conversion itself against production's own
files, and skips without mounted data. This covers what the evaluator adds on
top -- where the file lands, which images survive, and what the pipeline asks
for -- with a synthetic shots dict, so it needs no data and no GPU.
"""
import json
import math

import numpy as np
import pytest

from bevaligner import evaluate
from bevaligner.utils import io

REALITY = "RLT0000000000001"


class _Structure:
    key = "data_local/STR1"
    sheet_path = "/nonexistent/sheet.png"


class _Case:
    reality_id = REALITY
    capture_id = "SNP1"
    rlt_dir = "/nonexistent/rlt"
    structure = _Structure()


def _poses(count=10):
    """`shot_poses`-shaped: a straight walk, each camera looking the same way."""
    return {f"VID_x_IMG_{index:05d}.JPG":
            {"translation": [float(index), 0.5 * index, 1.4],
             "rotation": [0.0, 0.0, 0.1 * index]}
            for index in range(1, count + 1)}


@pytest.fixture
def shots(monkeypatch):
    monkeypatch.setattr(io, "shot_poses", lambda rlt_dir: _poses())


def _write(tmp_path, **kwargs):
    kwargs.setdefault("name", "images.json")
    path, count = evaluate.write_images(_Case(), np.eye(4), tmp_path / REALITY, **kwargs)
    return path, count, json.loads(path.read_text())


def test_it_lands_under_the_reality_and_parses(shots, tmp_path):
    path, count, images = _write(tmp_path)
    assert path == tmp_path / REALITY / "images.json"
    assert count == len(images) == 10


def test_every_record_is_a_position_and_a_heading(shots, tmp_path):
    _, _, images = _write(tmp_path)
    for name, record in images.items():
        assert name.endswith(".JPG") and "/" not in name
        assert len(record["position"]) == 3 and len(record["rotation"]) == 3
        assert all(math.isfinite(value) for value in record["position"])
        assert all(math.isfinite(value) for value in record["rotation"])


def test_every_nth_keeps_the_indices_production_keeps(shots, tmp_path):
    _, count, images = _write(tmp_path, every_nth=3)
    kept = sorted(int(name.split("_")[-1].split(".")[0]) for name in images)
    assert kept == [1, 4, 7, 10] and count == 4


def test_z_can_be_pinned_to_the_viewers_convention(shots, tmp_path):
    _, _, solved = _write(tmp_path)
    _, _, pinned = _write(tmp_path, override_z=104.6)
    assert {record["position"][2] for record in pinned.values()} == {104.6}
    assert {record["position"][2] for record in solved.values()} != {104.6}


def test_the_transform_is_applied_to_the_positions(shots, tmp_path):
    """The pose is what makes this a prediction rather than a re-export."""
    moved_by = np.eye(4)
    moved_by[:3, 3] = [100.0, 200.0, 300.0]
    path, _ = evaluate.write_images(_Case(), moved_by, tmp_path / "moved" / REALITY,
                                    name="images.json")
    moved = json.loads(path.read_text())
    _, _, plain = _write(tmp_path / "plain")
    first = "VID_x_IMG_00001.JPG"
    for axis, offset in enumerate((100.0, 200.0, 300.0)):
        assert moved[first]["position"][axis] == pytest.approx(
            plain[first]["position"][axis] + offset)


def test_no_picture_unless_one_is_asked_for(shots, tmp_path, monkeypatch):
    drawn = []
    monkeypatch.setattr(io, "render_floormap",
                        lambda *args, **kwargs: drawn.append(args[2]))
    _write(tmp_path)
    assert not drawn and not list((tmp_path / REALITY).glob("*.png"))
    _write(tmp_path, plot=True)
    assert drawn == [tmp_path / REALITY / "floormap.png"]


def test_a_reality_with_no_posed_shots_is_an_error_the_caller_can_catch(tmp_path,
                                                                        monkeypatch):
    monkeypatch.setattr(io, "shot_poses", lambda rlt_dir: {})
    with pytest.raises(ValueError):
        _write(tmp_path)


def test_the_pipeline_asks_for_them(tmp_path):
    from bevaligner.pipeline import Pipeline, PipelineConfig
    pipeline = Pipeline(PipelineConfig())
    argv = pipeline.eval_argv(must_exist=False)
    assert "--images" in argv and "--images-plot" not in argv
    pipeline.config.eval.images = False
    assert "--images" not in pipeline.eval_argv(must_exist=False)


def test_the_pipeline_can_ask_for_the_picture_too():
    from bevaligner.pipeline import Pipeline, PipelineConfig
    pipeline = Pipeline(PipelineConfig())
    pipeline.config.eval.images_plot = True
    assert "--images-plot" in pipeline.eval_argv(must_exist=False)
    # the picture is drawn beside the JSON, so it needs the JSON asked for
    pipeline.config.eval.images = False
    assert "--images-plot" not in pipeline.eval_argv(must_exist=False)
