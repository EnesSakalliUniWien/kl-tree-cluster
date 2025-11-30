"""KL Divergence calculations for hierarchical clustering.

This subpackage provides KL divergence metrics for analyzing
cluster hierarchies and tree structures.
"""

from .divergence_metrics import (
    calculate_kl_divergence_vector,
    compute_node_divergences,
)

__all__ = [
    "calculate_kl_divergence_vector",
    "compute_node_divergences",
]
