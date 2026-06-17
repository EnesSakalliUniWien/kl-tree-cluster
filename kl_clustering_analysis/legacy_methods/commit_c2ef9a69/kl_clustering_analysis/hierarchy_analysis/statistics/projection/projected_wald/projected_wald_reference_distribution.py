"""Satterthwaite-calibrated chi-square reference for projected test statistics.

Provides :func:`compute_projected_pvalue`, used by both the edge test (Gate 2)
and the sibling test (Gate 3) to avoid code duplication.

When eigenvalues are available, the projected quadratic form is kept unwhitened:

``T = Σ (vᵢᵀz)²``

and referenced against a moment-matched ``c × χ²(ν)`` where
``c = Σλ²/Σλ`` and ``ν = (Σλ)²/Σλ²``.

When eigenvalues cover fewer components than the projected vector, a split
strategy is used: PCA components receive Satterthwaite calibration, while
remaining random-padding components are treated as plain ``χ²(1)`` each.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import chi2


def compute_projected_pvalue(
    projected_vector: np.ndarray,
    degrees_of_freedom: int | float,
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
        provided, those components are Satterthwaite-calibrated. Any
        remaining components (random-padding) are treated as plain
        ``χ²(1)`` each.  Pass ``None`` for a plain ``χ²(k)`` test.

    Returns
    -------
    Tuple[float, float, float]
        ``(test_statistic, effective_df, p_value)``
    """
    if eigenvalues is not None and len(eigenvalues) > 0:
        n_pca = len(eigenvalues)
        pca_part = projected_vector[:n_pca]
        padding_part = projected_vector[n_pca:]
        padding_statistic = float(np.sum(padding_part**2)) if len(padding_part) > 0 else 0.0
        n_padding = len(padding_part) if len(padding_part) > 0 else 0

        return _compute_satterthwaite_pvalue(
            pca_part,
            eigenvalues,
            padding_statistic,
            n_padding,
        )
    else:
        test_statistic = float(np.sum(projected_vector**2))
        p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))
        return test_statistic, float(degrees_of_freedom), p_value


def _compute_satterthwaite_pvalue(
    projected_pca_components: np.ndarray,
    eigenvalues: np.ndarray,
    padding_statistic: float,
    n_padding: int,
) -> Tuple[float, float, float]:
    """Moment-matched chi-squared for unwhitened PCA projections.

    The PCA part ``T_pca = Σ (vᵢᵀz)²`` is a weighted sum of χ²(1)
    with weights λᵢ.  Satterthwaite: ``T_pca ≈ c_pca × χ²(ν_pca)``.

    The random-padding part is plain χ²(n_padding), which is added
    as ``c=1, ν=n_padding`` to the combined Satterthwaite moments.
    """
    component_eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    component_eigenvalues = np.maximum(component_eigenvalues, 1e-12)

    # PCA Satterthwaite moments
    sum_eigenvalues = float(np.sum(component_eigenvalues))
    sum_squared_eigenvalues = float(np.sum(component_eigenvalues**2))

    # Combined moments (PCA + padding):
    # Padding is Σ rᵢᵀz² ~ χ²(n_padding) ⟹ each weight = 1
    combined_statistic_mean = sum_eigenvalues + n_padding  # E[T]
    combined_statistic_variance = 2.0 * sum_squared_eigenvalues + 2.0 * n_padding  # Var[T]

    if combined_statistic_mean <= 0 or combined_statistic_variance <= 0:
        return 0.0, 1.0, 1.0

    satterthwaite_scale = combined_statistic_variance / (2.0 * combined_statistic_mean)
    satterthwaite_degrees_of_freedom = (
        2.0 * combined_statistic_mean**2 / combined_statistic_variance
    )

    # Unwhitened test statistic
    pca_statistic = float(np.sum(projected_pca_components**2))
    test_statistic = pca_statistic + padding_statistic

    # p-value: P(c × χ²(ν) > T) = P(χ²(ν) > T/c)
    p_value = float(
        chi2.sf(test_statistic / satterthwaite_scale, df=satterthwaite_degrees_of_freedom)
    )

    return test_statistic, float(satterthwaite_degrees_of_freedom), p_value


__all__ = ["compute_projected_pvalue"]
