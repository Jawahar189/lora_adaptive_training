"""The only module in this package that reads ground truth.

Wraps `align.groundtruth` (the answer for the current split) and the oracle
fit in `align.eval.oracle`. `data/preprocess.py`, the evaluation script and
the validation callback import it; `models`, `losses` and `inference` must
not, so that they remain usable inside the solver.
"""
from __future__ import annotations

import numpy as np

from align.eval.oracle import best_planar_fit
from align.groundtruth import gt_trajectory, is_manually_registered  # noqa: F401
from align.frames import Sim2, target_to_paper


def matched_camera_positions(case, trajectory) -> tuple[np.ndarray, np.ndarray]:
    """SfM (point-cloud frame) and ground-truth (target frame) camera positions
    for the same images in the same order. Mirrors `scripts/e7_reality_metrics.py`."""
    ground_truth = gt_trajectory(case.rlt_dir)
    index_of_image = {name: i for i, name in enumerate(trajectory.names)}
    shared_names = [n for n in ground_truth.names if n in index_of_image]
    sfm_positions = np.asarray(trajectory.positions)[[index_of_image[n] for n in shared_names]]
    gt_positions = np.asarray(ground_truth.positions)[
        [i for i, n in enumerate(ground_truth.names) if n in index_of_image]]
    return sfm_positions, gt_positions


def camera_positions_in_paper(case, trajectory) -> tuple[np.ndarray, np.ndarray]:
    """(sfm_xyz (N,3), ground-truth paper xy (N,2)) for the shared images."""
    sfm_positions, gt_positions = matched_camera_positions(case, trajectory)
    return sfm_positions, target_to_paper(gt_positions, case.structure.design_tm)[:, :2]


def camera_paper_z_median(case, trajectory) -> float:
    """The cameras' height in the paper frame, as one number.

    `gt_sim2` is planar, so nothing else in the cache carries a paper-frame z --
    and without one, splits of a prior capture cannot be clustered into storeys.
    A median is all that clustering needs, so a median is all that is stored.
    """
    _, gt_positions = matched_camera_positions(case, trajectory)
    if len(gt_positions) == 0:
        return float("nan")
    paper = target_to_paper(gt_positions, case.structure.design_tm)
    return float(np.median(paper[:, 2]))


def ground_truth_planar_fit(case, trajectory, rotations_per_hypothesis: dict[int, np.ndarray]):
    """The ground-truth planar pose:
    `(residual_inches, up_index, rotation_to_gravity_up, Sim2)` or None.

    `Sim2` takes gravity-up point-cloud xy (under that rotation) to paper
    inches, fitted by Umeyama on the cameras exactly as
    `align.eval.oracle.best_planar_fit` does, and the up hypothesis with the
    smaller residual wins.
    """
    sfm_positions, gt_paper_xy = camera_positions_in_paper(case, trajectory)
    if len(sfm_positions) < 3:
        return None
    return best_planar_fit(gt_paper_xy, rotations_per_hypothesis, sfm_positions)


def sim2_to_params(sim2: Sim2) -> np.ndarray:
    """`Sim2` -> the [scale, theta, tx, ty] array the cache stores."""
    return np.array([sim2.s, sim2.theta, sim2.tx, sim2.ty], dtype=np.float64)


def sim2_from_params(params) -> Sim2:
    values = np.asarray(params, dtype=np.float64)
    return Sim2(s=float(values[0]), theta=float(values[1]),
                tx=float(values[2]), ty=float(values[3]))
