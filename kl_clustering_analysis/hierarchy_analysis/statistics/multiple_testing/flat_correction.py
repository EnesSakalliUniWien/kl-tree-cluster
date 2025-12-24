"""Flat (standard) BH correction across all p-values.

This module provides a simple wrapper that applies BH correction
uniformly across all tests without considering any structure.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import benjamini_hochberg_correction


def flat_bh_correction(
    p_values: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply standard flat BH correction across all p-values.

    This is the simplest multiple testing correction - it treats all
    tests equally regardless of any hierarchical structure.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values for each test
    alpha : float
        Significance level for BH correction

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (reject_null, adjusted_p_values) arrays aligned to input

    Examples
    --------
    >>> import numpy as np
    >>> p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.1])
    >>> rejected, adjusted = flat_bh_correction(p_values, alpha=0.05)
    """
    n = len(p_values)
    if n == 0:
        return np.zeros(0, dtype=bool), np.ones(0, dtype=float)

    reject_null, adjusted_p, _ = benjamini_hochberg_correction(p_values, alpha=alpha)
    return reject_null, adjusted_p


__all__ = ["flat_bh_correction"]
