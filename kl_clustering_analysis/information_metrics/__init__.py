"""Information-theoretic metrics and utilities.

This package provides KL divergence metrics for distributions.
"""

from .kl_divergence import (
    calculate_kl_divergence_vector,
)

__all__ = [
    "calculate_kl_divergence_vector",
]
