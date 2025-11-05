"""
Compatibility layer exporting statistical testing helpers.

The implementations now live under hierarchy_analysis.statistics.* modules.
"""

from __future__ import annotations

from .statistics.shared_utils import (
    apply_benjamini_hochberg_correction,
    binary_threshold,
    calculate_chi_square_test,
)
from .statistics.root_significance import (
    annotate_nodes_with_statistical_significance_tests,
    kl_divergence_deviation_from_zero_test,
    test_feature_independence_conservative,
    test_feature_independence_liberal,
)
from .statistics.edge_significance import annotate_child_parent_divergence
from .statistics.sibling_independence import annotate_sibling_independence_cmi

# Backwards-compatible aliases
_calculate_chi_square_test = calculate_chi_square_test
_binary_threshold = binary_threshold
annotate_local_child_parent_significance = annotate_child_parent_divergence

__all__ = [
    "apply_benjamini_hochberg_correction",
    "binary_threshold",
    "calculate_chi_square_test",
    "test_feature_independence_conservative",
    "test_feature_independence_liberal",
    "kl_divergence_deviation_from_zero_test",
    "annotate_nodes_with_statistical_significance_tests",
    "annotate_child_parent_divergence",
    "annotate_local_child_parent_significance",
    "annotate_sibling_independence_cmi",
    "_calculate_chi_square_test",
    "_binary_threshold",
]
