"""KL divergence-based statistical tests for hierarchical clustering.

This subpackage provides statistical tests based on KL divergence for
evaluating significance of splits and divergences in hierarchical trees.
"""

from .chi_square_test import kl_divergence_chi_square_test
from .edge_significance import annotate_child_parent_divergence
from .root_significance import annotate_root_node_significance
from .utils import get_local_kl_series


__all__ = [
    "kl_divergence_chi_square_test",
    "annotate_child_parent_divergence",
    "annotate_root_node_significance",
    "get_local_kl_series",
]
