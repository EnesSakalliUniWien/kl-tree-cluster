"""Symmetric eigensolver helpers for correlation representations."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def decompose_symmetric_matrix(
    matrix: np.ndarray,
    *,
    compute_eigenvectors: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run a symmetric eigendecomposition and return descending nonnegative values."""
    if compute_eigenvectors:
        eigenvalues, eigenvectors = linalg.eigh(matrix, check_finite=False)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
    else:
        eigenvalues = linalg.eigvalsh(matrix, check_finite=False)[::-1]
        eigenvectors = None

    return np.maximum(eigenvalues, 0.0), eigenvectors


__all__ = ["decompose_symmetric_matrix"]
