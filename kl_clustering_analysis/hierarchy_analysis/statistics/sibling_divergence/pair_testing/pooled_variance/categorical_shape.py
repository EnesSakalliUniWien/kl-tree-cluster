"""Shape helpers for binary vs categorical pooled-variance calculations."""

from __future__ import annotations

import numpy as np


def _is_categorical(arr: np.ndarray) -> bool:
    """Check if array represents categorical distributions (2D)."""
    return arr.ndim == 2 and arr.shape[1] > 1


def _flatten_categorical(arr: np.ndarray) -> np.ndarray:
    """Flatten categorical distribution to 1D for Wald testing."""
    if _is_categorical(arr):
        return arr.ravel()
    return arr


__all__ = ["_flatten_categorical", "_is_categorical"]
