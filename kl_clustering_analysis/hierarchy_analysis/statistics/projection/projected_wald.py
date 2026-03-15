"""Shared projected Wald test kernel used by edge and sibling tests."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .chi2_pvalue import (
    compute_projected_pvalue as _compute_projected_pvalue,
)
from .projection_basis import build_projection_basis_with_padding


def compute_projected_pvalue(
    projected_vector: np.ndarray,
    degrees_of_freedom: int,
    *,
    eigenvalues: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute projected test statistic and p-value."""
    return _compute_projected_pvalue(
        np.asarray(projected_vector, dtype=np.float64),
        int(degrees_of_freedom),
        eigenvalues=eigenvalues,
    )


def run_projected_wald_kernel(
    z: np.ndarray,
    *,
    seed: int,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    k_fallback: Callable[[int], int],
) -> tuple[float, int, float, float]:
    """Project a standardized vector and compute Wald statistic/p-value.

    Returns
    -------
    tuple[float, int, float, float]
        ``(statistic, nominal_k, effective_df, p_value)``
    """
    standardized_diff = np.asarray(z, dtype=np.float64)
    n_features = int(standardized_diff.shape[0])

    if spectral_k is not None and spectral_k > 0:
        projection_dim = min(int(spectral_k), n_features)
    else:
        projection_dim = min(int(k_fallback(n_features)), n_features)

    projection_matrix, whitening_eigenvalues = build_projection_basis_with_padding(
        n_features=n_features,
        k=projection_dim,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
        random_state=seed,
    )

    projected_diff = projection_matrix @ standardized_diff

    test_statistic, effective_df, p_value = compute_projected_pvalue(
        projected_diff,
        projection_dim,
        eigenvalues=whitening_eigenvalues,
    )
    return float(test_statistic), int(projection_dim), float(effective_df), float(p_value)


__all__ = [
    "run_projected_wald_kernel",
    "compute_projected_pvalue",
]
