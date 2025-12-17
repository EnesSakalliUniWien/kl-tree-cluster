"""
Hierarchy analysis with statistical testing and visualization.

This package provides functions for hierarchical clustering analysis including:
- Distribution calculations across tree structures
- KL divergence metrics for significance testing
- Decomposition-based clustering via conditional independence
- Visualization and correlation analysis helpers
"""

from kl_clustering_analysis.information_metrics import (
    calculate_kl_divergence_vector,
    compute_node_divergences,
)
from .tree_decomposer import ClusterDecomposer
from .statistics import (
    annotate_child_parent_divergence,
    annotate_sibling_independence_cmi,
    benjamini_hochberg_correction,
    kl_divergence_chi_square_test,
)

__all__ = [
    "calculate_kl_divergence_vector",
    "compute_node_divergences",
    "ClusterDecomposer",
    "benjamini_hochberg_correction",
    "kl_divergence_chi_square_test",
    "annotate_child_parent_divergence",
    "annotate_sibling_independence_cmi",
]
