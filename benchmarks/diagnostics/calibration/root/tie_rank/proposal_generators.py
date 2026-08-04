"""Shared invariants for generated binary root tie-rank proposal matrices."""

from __future__ import annotations

import numpy as np


def ensure_nonempty_binary_feature_columns(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ensure every binary feature has at least one observed positive value."""
    fixed = np.asarray(matrix, dtype=int).copy()
    n_samples = fixed.shape[0]
    if n_samples <= 0:
        raise ValueError("Binary proposal matrices require at least one sample.")
    for feature_index in np.where(fixed.sum(axis=0) == 0)[0]:
        fixed[int(rng.integers(0, n_samples)), int(feature_index)] = 1
    return fixed


__all__ = ["ensure_nonempty_binary_feature_columns"]
