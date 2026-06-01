"""Chi-square reference for orthonormal projected test statistics.

Provides :func:`compute_projected_pvalue`, used by both edge-divergence and
sibling-divergence tests to avoid code duplication.

The projected quadratic form is

``T = Σ (vᵢᵀz)²``

where the rows ``vᵢ`` are orthonormal PCA directions. Under an isotropic
standardized null, PCA eigenvalues choose the subspace but do not weight the
quadratic reference law, so ``T ~ χ²(k)``.

Eigenvalues must cover the projected vector components to prove the PCA context
is complete, but they are not used as null weights.
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
        Reference law ``T ~ chi2(degrees_of_freedom)`` with reference_scale 1.
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

    if not np.isfinite(component_eigenvalues).all() or np.any(component_eigenvalues <= 0):
        raise ValueError("Projected Wald PCA eigenvalues must be finite and positive.")

    return _compute_orthonormal_projection_pvalue(projected_components)


def _compute_orthonormal_projection_pvalue(
    projected_pca_components: np.ndarray,
) -> ProjectedQuadraticReference:
    """Chi-squared reference for an orthonormal PCA subspace."""
    test_statistic = float(np.sum(projected_pca_components**2))
    degrees_of_freedom = float(projected_pca_components.shape[0])
    p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))

    return ProjectedQuadraticReference(
        statistic=test_statistic,
        reference_scale=1.0,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
    )


__all__ = ["ProjectedQuadraticReference", "compute_projected_pvalue"]
