"""Eigenvalue-whitened chi-square p-value for projected test statistics.

Provides :func:`compute_projected_pvalue`, used by both the edge test (Gate 2)
and the sibling test (Gate 3) to avoid code duplication.

Whitened mode: ``T = Σ (vᵢᵀz)²/λᵢ ~ χ²(k)`` — exact under H₀.

When eigenvalues cover fewer components than the projected vector, a
"split" strategy is used: PCA components get eigenvalue whitening,
while remaining (random-padding) components are treated as plain ``χ²(1)`` each.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import chi2


def compute_projected_pvalue(
    projected_vector: np.ndarray,
    degrees_of_freedom: int,
    eigenvalues: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    """Compute test statistic and p-value from a projected z-score vector.

    Parameters
    ----------
    projected_vector
        The projected vector ``R @ z`` of length ``k``.
    degrees_of_freedom
        Nominal degrees of freedom (projection dimension ``k``).
    eigenvalues
        PCA eigenvalues for the first ``k_pca ≤ k`` components.  When
        provided, those components are whitened (divided by λᵢ).  Any
        remaining components (random-padding) are treated as plain
        ``χ²(1)`` each.  Pass ``None`` for a plain ``χ²(k)`` test.

    Returns
    -------
    Tuple[float, float, float]
        ``(test_statistic, effective_df, p_value)``
    """
    if eigenvalues is not None and len(eigenvalues) > 0:
        n_pca_components = len(eigenvalues)
        # Whitened: T = Σ wᵢ²/λᵢ ~ χ²(k)
        whitened_statistic = float(
            np.sum(projected_vector[:n_pca_components] ** 2 / eigenvalues)
        )
        random_padding_statistic = (
            float(np.sum(projected_vector[n_pca_components:] ** 2))
            if n_pca_components < len(projected_vector)
            else 0.0
        )
        test_statistic = whitened_statistic + random_padding_statistic
        p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))
        return test_statistic, float(degrees_of_freedom), p_value
    else:
        test_statistic = float(np.sum(projected_vector**2))
        p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))
        return test_statistic, float(degrees_of_freedom), p_value


__all__ = ["compute_projected_pvalue"]
