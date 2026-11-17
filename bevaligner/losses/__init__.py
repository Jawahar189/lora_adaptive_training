"""The four losses of the paper and the schedule that switches them on."""
from .curriculum import (DEFAULT_LOSS_GATES, DEFAULT_LOSS_WEIGHTS,  # noqa: F401
                         LossSettings, BEVAlignerLoss, active_loss_weights)
from .feature import feature_infonce_loss  # noqa: F401
from .geometry import scale_consistency_loss  # noqa: F401
from .regression import coordinate_regression_loss  # noqa: F401
from .topology import angle_topology_loss  # noqa: F401
