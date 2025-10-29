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
from .similarity_analysis import (
    analyze_hierarchy_similarity_patterns,
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

__all__ = [
    "calculate_kl_divergence_vector",
    "calculate_hierarchy_kl_divergence",
    "analyze_hierarchy_similarity_patterns",
    "ClusterDecomposer",
    "calculate_kl_divergence_mutual_information_matrix",
    "get_highly_similar_nodes",
    "get_node_mutual_information_summary",
    "_binary_pattern",
    "_binary_entropy",
    "_mutual_info_binary_normalized",
    "estimate_global_mi_threshold",
]
