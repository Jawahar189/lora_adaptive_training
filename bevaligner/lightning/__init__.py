"""Training loop glue: the LightningModule and its callbacks."""
from .callbacks import (LogPredictedMatchImages, PoseEvaluation,  # noqa: F401
                        SaveTrainingPairImages, render_match_image)
from .module import BEVAlignerModule  # noqa: F401
