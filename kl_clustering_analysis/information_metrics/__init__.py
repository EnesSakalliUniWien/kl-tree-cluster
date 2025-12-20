"""Information-theoretic metrics and utilities.

This package provides various information-theoretic calculations including:
- Mutual Information (MI) for binary/discrete data
- KL divergence metrics for distributions
"""

from .mutual_information import (
    _mi_binary_vec,
    _mi_binary_vec_accel,
    _mi_binary_vec_numpy,
    _mi_binary_vec_numba,
)
from .kl_divergence import (
    calculate_kl_divergence_vector,
    compute_node_divergences,
)

__all__ = [
    # MI functions
    "_mi_binary_vec",
    "_mi_binary_vec_accel",
    "_mi_binary_vec_numpy",
    "_mi_binary_vec_numba",
    # KL divergence functions
    "calculate_kl_divergence_vector",
    "compute_node_divergences",
]
