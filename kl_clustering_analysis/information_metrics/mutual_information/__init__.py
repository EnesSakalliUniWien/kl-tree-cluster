"""Mutual Information and Conditional Mutual Information calculations.

This subpackage provides:
- Mutual Information (MI) for binary/discrete data
- Conditional Mutual Information (CMI)
- Permutation tests for conditional independence
"""

from .mi import (
    _mi_binary_vec,
    _mi_binary_vec_accel,
    _mi_binary_vec_numpy,
    _mi_binary_vec_numba,
)
from .cmi import _cmi_binary_vec
from .permutation import (
    _perm_cmi_binary_batch,
    _perm_test_cmi_binary,
    _cmi_perm_from_args,
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
]
