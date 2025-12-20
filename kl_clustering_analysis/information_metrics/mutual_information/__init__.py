"""Mutual Information calculations.

This subpackage provides:
- Mutual Information (MI) for binary/discrete data
"""

from .mi import (
    _mi_binary_vec,
    _mi_binary_vec_accel,
    _mi_binary_vec_numpy,
    _mi_binary_vec_numba,
)

__all__ = [
    # MI functions
    "_mi_binary_vec",
    "_mi_binary_vec_accel",
    "_mi_binary_vec_numpy",
    "_mi_binary_vec_numba",
]
