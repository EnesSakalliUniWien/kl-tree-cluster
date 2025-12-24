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
    # Convert to float array for numerical stability
    p_values_array = np.asarray(p_values, dtype=float)

    # Handle empty input edge case
    if p_values_array.size == 0:
        empty_rejected = np.array([], dtype=bool)
        empty_adjusted = np.array([], dtype=float)
        alpha_threshold = float(alpha)
        return empty_rejected, empty_adjusted, alpha_threshold

    # Apply Benjamini-Hochberg FDR correction using statsmodels
    rejected, adjusted, _, _ = multipletests(
        p_values_array,
        alpha=alpha,
        method="fdr_bh",
        is_sorted=False,
        returnsorted=False,
    )

    # Convert to appropriate output types
    rejected_hypotheses = rejected.astype(bool)
    adjusted_p_values = adjusted.astype(float)
    alpha_threshold = float(alpha)

    return rejected_hypotheses, adjusted_p_values, alpha_threshold


__all__ = ["benjamini_hochberg_correction"]
