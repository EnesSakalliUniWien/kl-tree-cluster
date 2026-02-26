"""KL divergence-based statistical tests for hierarchical clustering.

This subpackage provides statistical tests based on KL divergence for
evaluating significance of splits and divergences in hierarchical trees.
"""

from .chi_square_test import (
    kl_divergence_chi_square_test,
    kl_divergence_chi_square_test_batch,
)
from .edge_calibration import (
    EdgeCalibrationModel,
    calibrate_edges_from_sibling_neighborhood,
    predict_edge_inflation_factor,
)
from .edge_significance import annotate_child_parent_divergence
from .utils import get_local_kl_series


__all__ = [
    "kl_divergence_chi_square_test",
    "kl_divergence_chi_square_test_batch",
    "annotate_child_parent_divergence",
    "EdgeCalibrationModel",
    "calibrate_edges_from_sibling_neighborhood",
    "predict_edge_inflation_factor",
    "get_local_kl_series",
]
