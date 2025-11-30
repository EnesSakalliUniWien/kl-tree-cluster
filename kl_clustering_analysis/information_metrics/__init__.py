"""Information-theoretic metrics and utilities.

This package provides various information-theoretic calculations including:
- Mutual Information (MI) for binary/discrete data
- Conditional Mutual Information (CMI) for binary data
- KL divergence metrics for distributions
- Permutation tests for independence testing
"""

from .mutual_information import (
    _mi_binary_vec,
    _mi_binary_vec_accel,
    _mi_binary_vec_numpy,
    _mi_binary_vec_numba,
    _cmi_binary_vec,
    _perm_cmi_binary_batch,
    _perm_test_cmi_binary,
    _cmi_perm_from_args,
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
    # CMI functions
    "_cmi_binary_vec",
    # Permutation test functions
    "_perm_cmi_binary_batch",
    "_perm_test_cmi_binary",
    "_cmi_perm_from_args",
    # KL divergence functions
    "calculate_kl_divergence_vector",
    "compute_node_divergences",
]
