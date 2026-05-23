"""Core Benjamini-Hochberg FDR correction.

This module provides the fundamental BH correction that other correction
methods build upon.

References
----------
Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery
rate: a practical and powerful approach to multiple testing. Journal of
the Royal Statistical Society Series B, 57, 289-300.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from statsmodels.stats.multitest import multipletests


def benjamini_hochberg_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply Benjamini-Hochberg FDR correction to p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values to correct
    alpha : float, default=0.05
        Significance level for FDR control

    Returns
    -------
    rejected_hypotheses : np.ndarray (bool)
        Boolean array indicating which null hypotheses are rejected
    adjusted_p_values : np.ndarray (float)
        FDR-adjusted p-values
    alpha_threshold : float
        The significance level used

    Notes
    -----
    Returns empty arrays if input is empty (guards against edge cases).
    Uses statsmodels implementation of Benjamini-Hochberg procedure.

    Examples
    --------
    >>> import numpy as np
    >>> p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.1])
    >>> rejected, adjusted, alpha = benjamini_hochberg_correction(p_values)
    >>> rejected
    array([ True,  True,  True, False, False])
    """
    p_values_array = np.asarray(p_values, dtype=float)
    alpha_threshold = float(alpha)
    if not np.isfinite(alpha_threshold) or not (0.0 < alpha_threshold <= 1.0):
        raise ValueError(f"BH alpha must be finite and in (0, 1]; got {alpha!r}.")
    if p_values_array.ndim != 1:
        raise ValueError(f"BH p-values must be a 1-D array; got shape {p_values_array.shape}.")
    if np.any(~np.isfinite(p_values_array)):
        raise ValueError("BH p-values must be finite before correction.")
    if np.any((p_values_array < 0.0) | (p_values_array > 1.0)):
        raise ValueError("BH p-values must lie in [0, 1].")

    if p_values_array.size == 0:
        empty_rejected = np.array([], dtype=bool)
        empty_adjusted = np.array([], dtype=float)
        return empty_rejected, empty_adjusted, alpha_threshold

    rejected, adjusted, _, _ = multipletests(
        p_values_array,
        alpha=alpha_threshold,
        method="fdr_bh",
        is_sorted=False,
        returnsorted=False,
    )

    rejected_hypotheses = rejected.astype(bool)
    adjusted_p_values = adjusted.astype(float)

    return rejected_hypotheses, adjusted_p_values, alpha_threshold


__all__ = ["benjamini_hochberg_correction"]
