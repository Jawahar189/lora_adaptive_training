"""BEVAligner (arXiv 2605.22581) on this repo's captures.

A gravity-aligned point cloud becomes a top-down density map; the density map
and the floor-plan sheet go through one frozen DINOv3 encoder with LoRA; dense
correspondences fall out of a soft-argmax over floor-plan patches; RANSAC turns
them into the planar similarity the rest of `align` already knows how to score.

Layout (a conventional deep-learning package):

    data/        preprocessing cache, density rendering, sheet cropping,
                 augmentation, correspondence sampling, dataset + sampler
    models/      LoRA adapters, the DINO patch encoder, the matcher
    losses/      the four paper losses and the curriculum that gates them
    optim/       optimiser + learning-rate schedule
    lightning/   LightningModule and callbacks
    inference/   matching, RANSAC, pose proposal
    utils/       every borrow from `align`, wrapped once
    train.py / evaluate.py   entry points

Ground truth is read in exactly one place, `utils/groundtruth.py`; `models`,
`losses` and `inference` never import it, so they can be used by the solver.
"""
