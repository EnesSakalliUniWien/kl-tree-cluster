"""Shared Satterthwaite chi-square approximation for projected test statistics.

Provides :func:`compute_projected_pvalue`, which handles both eigenvalue-whitened
and Satterthwaite-corrected modes.  Used by both the edge test (Gate 2) and the
sibling test (Gate 3) to avoid code duplication.

The two modes are controlled by ``config.EIGENVALUE_WHITENING``:

* **Whitened** (``True``): ``T = Σ (vᵢᵀz)²/λᵢ ~ χ²(k)``.
  Exact under H₀ but divides signal by large eigenvalues → lower power.
* **Satterthwaite** (``False``): ``T = Σ (vᵢᵀz)²``, approximated as
  ``c·χ²(ν)`` where ``c = Σλ²/Σλ``, ``ν = (Σλ)²/Σλ²``.
  Preserves power because signal in high-eigenvalue directions is not dampened.

When eigenvalues cover fewer components than the projected vector, a
"split" strategy is used: PCA components get the eigenvalue-aware treatment,
while remaining (random-padding) components are treated as plain ``χ²(1)`` each.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import chi2

from kl_clustering_analysis import config


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
        provided, those components are treated via the eigenvalue-aware
        path (whitened or Satterthwaite).  Any remaining components
        (random-padding) are treated as plain ``χ²(1)`` each.
        Pass ``None`` for a plain ``χ²(k)`` test.

    Returns
    -------
    Tuple[float, float, float]
        ``(test_statistic, effective_df, p_value)``
    """
    if eigenvalues is not None and len(eigenvalues) > 0:
        k_pca = len(eigenvalues)

        if config.EIGENVALUE_WHITENING:
            # --- Whitened mode: T = Σ wᵢ²/λᵢ ~ χ²(k) ---
            stat_pca = float(np.sum(projected[:k_pca] ** 2 / eigenvalues))
            stat_rand = float(np.sum(projected[k_pca:] ** 2)) if k_pca < len(projected) else 0.0
            stat = stat_pca + stat_rand
            return stat, float(degrees_of_freedom), float(chi2.sf(stat, df=degrees_of_freedom))
        else:
            # --- Satterthwaite mode: T_pca = Σ wᵢ² ~ Σ λᵢ·χ²(1) ---
            stat_pca = float(np.sum(projected[:k_pca] ** 2))
            k_rand = len(projected) - k_pca
            stat_rand = float(np.sum(projected[k_pca:] ** 2)) if k_rand > 0 else 0.0
            stat = stat_pca + stat_rand

            # Satterthwaite: approximate Σ λᵢ·χ²(1) as c·χ²(ν)
            # c = Σλ²/Σλ,  ν = (Σλ)²/Σλ²
            eigs = np.asarray(eigenvalues, dtype=np.float64)
            sum_eig = float(np.sum(eigs))
            sum_eig2 = float(np.sum(eigs**2))

            if sum_eig2 > 0 and sum_eig > 0:
                c = sum_eig2 / sum_eig
                nu_pca = sum_eig**2 / sum_eig2
                # Combined df: Satterthwaite ν for PCA + k_rand for random padding
                nu_total = nu_pca + k_rand
                # Scale only PCA component; random part is already χ²
                stat_scaled = stat_pca / c + stat_rand
                return stat, float(nu_total), float(chi2.sf(stat_scaled, df=nu_total))
            else:
                # Degenerate eigenvalues — fall back to plain χ²(k)
                return stat, float(degrees_of_freedom), float(
                    chi2.sf(stat, df=degrees_of_freedom)
                )
    else:
        stat = float(np.sum(projected**2))
        return stat, float(degrees_of_freedom), float(chi2.sf(stat, df=degrees_of_freedom))


__all__ = ["compute_projected_pvalue"]
