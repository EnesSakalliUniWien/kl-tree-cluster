"""Shared projected Wald test kernel used by edge and sibling tests."""

from __future__ import annotations

import numpy as np

from .chi2_pvalue import WhiteningMode
from .chi2_pvalue import compute_projected_pvalue as _compute_projected_pvalue
from .projection_basis import build_projection_basis_with_padding


def compute_projected_pvalue(
    projected_vector: np.ndarray,
    degrees_of_freedom: int,
    *,
    eigenvalues: np.ndarray | None = None,
    whitening: WhiteningMode = "per_component",
) -> tuple[float, float, float]:
    """Compute projected test statistic and p-value."""
    return _compute_projected_pvalue(
        np.asarray(projected_vector, dtype=np.float64),
        int(degrees_of_freedom),
        eigenvalues=eigenvalues,
        whitening=whitening,
    )


def run_projected_wald_kernel(
    z: np.ndarray,
    *,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    child_pca_projections: list[np.ndarray] | None = None,
    whitening: WhiteningMode = "per_component",
) -> tuple[float, int, float, float]:
    """Project a standardized vector and compute Wald statistic/p-value.

    When ``spectral_k`` is ``None`` or ≤ 0, no valid projection dimension
    exists — returns ``(nan, 0, nan, nan)`` to signal a skip/merge.

    Returns
    -------
    tuple[float, int, float, float]
        ``(statistic, nominal_k, effective_df, p_value)``
    """
    if spectral_k is None or spectral_k <= 0:
        return (float("nan"), 0, float("nan"), float("nan"))

    # PCA basis is required when spectral_k > 0.  When the upstream pipeline
    # could not produce a PCA decomposition for this node (e.g. too few
    # descendant samples), treat the test as invalid — skip/merge.
    if pca_projection is None:
        return (float("nan"), 0, float("nan"), float("nan"))

    standardized_diff = np.asarray(z, dtype=np.float64)
    n_features = int(standardized_diff.shape[0])
    projection_dim = min(int(spectral_k), n_features)

    projection_matrix, whitening_eigenvalues = build_projection_basis_with_padding(
        n_features=n_features,
        k=projection_dim,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
        child_pca_projections=child_pca_projections,
    )

    projected_diff = projection_matrix @ standardized_diff

    test_statistic, effective_df, p_value = compute_projected_pvalue(
        projected_diff,
        projection_matrix.shape[0],
        eigenvalues=whitening_eigenvalues,
        whitening=whitening,
    )
    return float(test_statistic), int(projection_matrix.shape[0]), float(effective_df), float(p_value)


__all__ = [
    "run_projected_wald_kernel",
    "compute_projected_pvalue",
]
