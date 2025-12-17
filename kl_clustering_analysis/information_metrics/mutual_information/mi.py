"""Mutual Information (MI) calculations for binary data.

Provides efficient implementations of mutual information for binary/discrete
distributions, with optional Numba acceleration for large-scale computations.
"""

from __future__ import annotations

import numpy as np

from .mi_numba import _NUMBA_AVAILABLE, _mi_binary_vec_numba
from .mi_numpy import _mi_binary_vec_numpy
from .mi_sklearn import _mi_binary_vec_sklearn


def _mi_binary_vec_accel(x_vector: np.ndarray, y_vectors: np.ndarray) -> np.ndarray:
    """
    Dispatcher: use Numba if available, else NumPy.

    Parameters
    ----------
    x_vector : np.ndarray
        Shape (n_samples,), binary values.
    y_vectors : np.ndarray
        Shape (n_vectors, n_samples), binary rows.

    Returns
    -------
    np.ndarray
        Shape (n_vectors,), MI values.
    """
    # Ensure contiguous arrays for Numba/C-level efficiency
    x = np.ascontiguousarray(x_vector, dtype=np.uint8)
    Y = np.ascontiguousarray(y_vectors, dtype=np.uint8)

    if _NUMBA_AVAILABLE:
        return _mi_binary_vec_numba(x, Y)
    return _mi_binary_vec_numpy(x, Y)


def _mi_binary_vec(x_vector: np.ndarray, y_vectors: np.ndarray) -> np.ndarray:
    """
    Primary Mutual Information dispatcher exposed to callers.

    Calculates MI between a discrete vector X and a set of discrete vectors Y.
    Defaults to using scikit-learn's implementation which supports arbitrary
    discrete (categorical) data, not just binary.

    Parameters
    ----------
    x_vector : np.ndarray
        Shape (n_samples,), discrete values (int).
    y_vectors : np.ndarray
        Shape (n_vectors, n_samples), discrete rows (int).

    Returns
    -------
    np.ndarray
        Shape (n_vectors,), MI values in nats.
    """
    return _mi_binary_vec_sklearn(x_vector, y_vectors)


__all__ = [
    "_mi_binary_vec_numpy",
    "_mi_binary_vec_numba",
    "_mi_binary_vec_sklearn",
    "_mi_binary_vec_accel",
    "_mi_binary_vec",
]
