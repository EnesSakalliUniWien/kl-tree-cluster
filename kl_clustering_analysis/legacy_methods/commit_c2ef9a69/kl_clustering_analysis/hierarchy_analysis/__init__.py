"""
Hierarchy analysis with statistical testing and visualization.

This package provides functions for hierarchical clustering analysis including:
- Distribution calculations across tree structures
- KL divergence metrics for significance testing
- Decomposition-based clustering via sibling divergence tests
- Visualization and correlation analysis helpers
"""

from .statistics import (
    annotate_child_parent_divergence,
    annotate_sibling_divergence,
    benjamini_hochberg_correction,
)
from .tree_decomposition import TreeDecomposition

__all__ = [
    "TreeDecomposition",
    "benjamini_hochberg_correction",
    "annotate_child_parent_divergence",
    "annotate_sibling_divergence",
]
