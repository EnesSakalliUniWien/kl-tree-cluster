"""
Hierarchy analysis with statistical testing and visualization.

This package provides functions for hierarchical clustering analysis including:
- Distribution calculations across tree structures
- KL divergence metrics for significance testing
- Decomposition-based clustering via conditional independence
- Visualization and correlation analysis helpers
"""

from .divergence_metrics import (
    calculate_kl_divergence_vector,
    calculate_hierarchy_kl_divergence,
)
from .cluster_decomposition import ClusterDecomposer
from .kl_correlation_analysis import (
    calculate_kl_divergence_mutual_information_matrix,
    get_highly_similar_nodes,
    get_node_mutual_information_summary,
)
from .mutual_info_utils import (
    _binary_pattern,
    _binary_entropy,
    _mutual_info_binary_normalized,
    estimate_global_mi_threshold,
)
from .statistics import (
    annotate_child_parent_divergence,
    annotate_nodes_with_statistical_significance_tests,
    annotate_sibling_independence_cmi,
    apply_benjamini_hochberg_correction,
    binary_threshold,
    calculate_chi_square_test,
    kl_divergence_deviation_from_zero_test,
    test_feature_independence_conservative,
    test_feature_independence_liberal,
)

__all__ = [
    "calculate_kl_divergence_vector",
    "calculate_hierarchy_kl_divergence",
    "ClusterDecomposer",
    "calculate_kl_divergence_mutual_information_matrix",
    "get_highly_similar_nodes",
    "get_node_mutual_information_summary",
    "_binary_pattern",
    "_binary_entropy",
    "_mutual_info_binary_normalized",
    "estimate_global_mi_threshold",
    "apply_benjamini_hochberg_correction",
    "binary_threshold",
    "calculate_chi_square_test",
    "test_feature_independence_conservative",
    "test_feature_independence_liberal",
    "kl_divergence_deviation_from_zero_test",
    "annotate_nodes_with_statistical_significance_tests",
    "annotate_child_parent_divergence",
    "annotate_sibling_independence_cmi",
]
