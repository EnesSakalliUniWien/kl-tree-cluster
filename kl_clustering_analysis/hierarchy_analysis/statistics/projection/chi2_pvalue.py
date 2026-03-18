"""Eigenvalue-aware chi-square p-value for projected test statistics.

Provides :func:`compute_projected_pvalue`, used by both the edge test (Gate 2)
and the sibling test (Gate 3) to avoid code duplication.

Two eigenvalue modes are supported:

**Whitened mode** (default, ``whitening="per_component"``):
    ``T = Σ (vᵢᵀz)²/λᵢ ~ χ²(k)`` — exact under H₀.
    Best when eigenvalues are NOT contaminated by the signal being tested
    (e.g., Gate 2 child-vs-parent test, where z captures only a fraction
    of the between-group variance in the eigenvalues).

**Satterthwaite mode** (``whitening="satterthwaite"``):
    ``T = Σ (vᵢᵀz)²`` (unwhitened), referenced against a moment-matched
    ``c × χ²(ν)`` where ``c = Σλ²/Σλ``, ``ν = (Σλ)²/Σλ²``.
    Use when eigenvalues are contaminated by the signal (e.g., Gate 3
    sibling test with parent PCA, where the parent eigenvalues include
    the between-group variance that the test tries to detect).

When eigenvalues cover fewer components than the projected vector, a
"split" strategy is used: PCA components get eigenvalue treatment,
while remaining (random-padding) components are treated as plain ``χ²(1)`` each.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from scipy.stats import chi2

WhiteningMode = Literal["per_component", "satterthwaite"]


def compute_projected_pvalue(
    projected_vector: np.ndarray,
    degrees_of_freedom: int,
    eigenvalues: np.ndarray | None = None,
    *,
    whitening: WhiteningMode = "per_component",
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
        provided, those components are either whitened (divided by λᵢ)
        or Satterthwaite-calibrated, depending on *whitening*.  Any
        remaining components (random-padding) are treated as plain
        ``χ²(1)`` each.  Pass ``None`` for a plain ``χ²(k)`` test.
    whitening
        ``"per_component"`` (default): divide each squared projection by
        its eigenvalue → exact ``χ²(k)`` under H₀.
        ``"satterthwaite"``: keep unwhitened squared projections and use
        moment-matched ``c × χ²(ν)`` reference distribution.

    Returns
    -------
    Tuple[float, float, float]
        ``(test_statistic, effective_df, p_value)``
    """
    if eigenvalues is not None and len(eigenvalues) > 0:
        n_pca = len(eigenvalues)
        pca_part = projected_vector[:n_pca]
        padding_part = projected_vector[n_pca:]
        padding_statistic = float(np.sum(padding_part ** 2)) if len(padding_part) > 0 else 0.0
        n_padding = len(padding_part) if len(padding_part) > 0 else 0

        if whitening == "satterthwaite":
            return _satterthwaite_pvalue(pca_part, eigenvalues, padding_statistic, n_padding)

        # Default: per-component whitening  T = Σ wᵢ²/λᵢ ~ χ²(k)
        whitened_statistic = float(np.sum(pca_part ** 2 / eigenvalues))
        test_statistic = whitened_statistic + padding_statistic
        p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))
        return test_statistic, float(degrees_of_freedom), p_value
    else:
        test_statistic = float(np.sum(projected_vector**2))
        p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))
        return test_statistic, float(degrees_of_freedom), p_value


def _satterthwaite_pvalue(
    pca_projections: np.ndarray,
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
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    eigs = np.maximum(eigs, 1e-12)

    # PCA Satterthwaite moments
    sum_lam = float(np.sum(eigs))
    sum_lam2 = float(np.sum(eigs ** 2))

    # Combined moments (PCA + padding):
    # Padding is Σ rᵢᵀz² ~ χ²(n_padding) ⟹ each weight = 1
    total_mean = sum_lam + n_padding         # E[T]
    total_var = 2.0 * sum_lam2 + 2.0 * n_padding  # Var[T]

    if total_mean <= 0 or total_var <= 0:
        return 0.0, 1.0, 1.0

    c = total_var / (2.0 * total_mean)
    nu = 2.0 * total_mean ** 2 / total_var

    # Unwhitened test statistic
    pca_statistic = float(np.sum(pca_projections ** 2))
    test_statistic = pca_statistic + padding_statistic

    # p-value: P(c × χ²(ν) > T) = P(χ²(ν) > T/c)
    p_value = float(chi2.sf(test_statistic / c, df=nu))

    return test_statistic, float(nu), p_value


__all__ = ["compute_projected_pvalue"]
