"""Satterthwaite-calibrated chi-square reference for projected test statistics.

Provides :func:`compute_projected_pvalue`, used by both the edge test (Gate 2)
and the sibling test (Gate 3) to avoid code duplication.

When eigenvalues are available, the projected quadratic form is kept unwhitened:

``T = Σ (vᵢᵀz)²``

and referenced against a moment-matched ``c × χ²(ν)`` where
``c = Σλ²/Σλ`` and ``ν = (Σλ)²/Σλ²``.

Eigenvalues must cover the projected vector components; the statistic is
calibrated by Satterthwaite moment matching under that PCA basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class ProjectedQuadraticReference:
    """Reference law for one projected quadratic statistic."""

    statistic: float
    reference_scale: float
    degrees_of_freedom: float
    p_value: float


def compute_projected_pvalue(
    projected_vector: np.ndarray,
    eigenvalues: np.ndarray | None = None,
) -> ProjectedQuadraticReference:
    """Compute test statistic and p-value from a projected z-score vector.

    Parameters
    ----------
    projected_vector
        The projected vector ``R @ z`` of length ``k``.
    eigenvalues
        PCA eigenvalues for the projected components.

    Returns
    -------
    Tuple[float, float, float]
        Reference law ``T ~ reference_scale * chi2(degrees_of_freedom)``.
    """
    projected_components = np.asarray(projected_vector, dtype=np.float64)
    if projected_components.ndim != 1:
        raise ValueError(
            "Projected Wald vector must be one-dimensional; "
            f"got shape {projected_components.shape}."
        )
    if not np.isfinite(projected_components).all():
        raise ValueError("Projected Wald vector components must be finite.")

    if eigenvalues is None:
        raise ValueError("Projected Wald p-value requires PCA eigenvalues.")

    component_eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if component_eigenvalues.shape != projected_components.shape:
        raise ValueError(
            "Projected vector and PCA eigenvalues must have the same length: "
            f"{projected_components.shape[0]} component(s) versus "
            f"{component_eigenvalues.shape[0]} eigenvalue(s)."
        )

    return _compute_satterthwaite_pvalue(projected_components, component_eigenvalues)


def _compute_satterthwaite_pvalue(
    projected_pca_components: np.ndarray,
    eigenvalues: np.ndarray,
) -> ProjectedQuadraticReference:
    """Moment-matched chi-squared for unwhitened PCA projections.

    The PCA part ``T_pca = Σ (vᵢᵀz)²`` is a weighted sum of χ²(1)
    with weights λᵢ.  Satterthwaite: ``T_pca ≈ c_pca × χ²(ν_pca)``.

    """
    component_eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if not np.isfinite(component_eigenvalues).all() or np.any(component_eigenvalues <= 0):
        raise ValueError("Projected Wald PCA eigenvalues must be finite and positive.")

    sum_eigenvalues = float(np.sum(component_eigenvalues))
    sum_squared_eigenvalues = float(np.sum(component_eigenvalues**2))

    combined_statistic_mean = sum_eigenvalues
    combined_statistic_variance = 2.0 * sum_squared_eigenvalues

    if combined_statistic_mean <= 0 or combined_statistic_variance <= 0:
        raise ValueError("Projected Wald Satterthwaite moments must be positive.")

    satterthwaite_scale = combined_statistic_variance / (2.0 * combined_statistic_mean)
    satterthwaite_degrees_of_freedom = (
        2.0 * combined_statistic_mean**2 / combined_statistic_variance
    )

    # Unwhitened test statistic
    test_statistic = float(np.sum(projected_pca_components**2))

    # p-value: P(c × χ²(ν) > T) = P(χ²(ν) > T/c)
    p_value = float(
        chi2.sf(test_statistic / satterthwaite_scale, df=satterthwaite_degrees_of_freedom)
    )

    return ProjectedQuadraticReference(
        statistic=test_statistic,
        reference_scale=float(satterthwaite_scale),
        degrees_of_freedom=float(satterthwaite_degrees_of_freedom),
        p_value=p_value,
    )


__all__ = ["ProjectedQuadraticReference", "compute_projected_pvalue"]
