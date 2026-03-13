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
    projected: np.ndarray,
    degrees_of_freedom: int,
    eigenvalues: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    """Compute test statistic and p-value from a projected z-score vector.

    Parameters
    ----------
    projected
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
        k_pca = len(eigenvalues)
        # Whitened: T = Σ wᵢ²/λᵢ ~ χ²(k)
        stat_pca = float(np.sum(projected[:k_pca] ** 2 / eigenvalues))
        stat_rand = float(np.sum(projected[k_pca:] ** 2)) if k_pca < len(projected) else 0.0
        stat = stat_pca + stat_rand
        return stat, float(degrees_of_freedom), float(chi2.sf(stat, df=degrees_of_freedom))
    else:
        stat = float(np.sum(projected**2))
        return stat, float(degrees_of_freedom), float(chi2.sf(stat, df=degrees_of_freedom))


__all__ = ["compute_projected_pvalue"]
