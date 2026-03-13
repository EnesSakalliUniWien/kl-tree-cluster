"""KL divergence-based statistical tests for hierarchical clustering."""

from .edge_significance import annotate_child_parent_divergence


__all__ = [
    "annotate_child_parent_divergence",
]
