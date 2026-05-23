"""Shared projected Wald test kernel used by edge and sibling tests."""

from __future__ import annotations

import numpy as np

from .projected_wald_projection_basis import build_pca_projection_basis
from .projected_wald_reference_distribution import compute_projected_pvalue


def run_projected_wald_kernel(
    z: np.ndarray,
    *,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
) -> tuple[float, int, float, float]:
    """Project a standardized vector and compute Wald statistic/p-value.

    Returns
    -------
    tuple[float, int, float, float]
        ``(statistic, nominal_k, effective_df, p_value)``
    """
    if spectral_k is None or spectral_k < 0:
        raise ValueError("Projected Wald kernel requires a non-negative spectral_k.")

    standardized_diff = np.asarray(z, dtype=np.float64)
    if standardized_diff.ndim != 1:
        raise ValueError(
            "Projected Wald z-score vector must be one-dimensional; "
            f"got shape {standardized_diff.shape}."
        )
    if not np.isfinite(standardized_diff).all():
        raise ValueError("Projected Wald z-score vector must be finite.")
    n_features = int(standardized_diff.shape[0])
    projection_dim = int(spectral_k)
    if projection_dim == 0:
        if not np.allclose(standardized_diff, 0.0, atol=1e-12, rtol=0.0):
            raise ValueError(
                "Zero-dimensional spectral context can only be used with a zero "
                "projected-Wald contrast."
            )
        return 0.0, 0, 0.0, 1.0
    if projection_dim > n_features:
        raise ValueError(
            f"Projected Wald spectral_k={projection_dim} exceeds feature count {n_features}."
        )

    projection_matrix, whitening_eigenvalues = build_pca_projection_basis(
        k=projection_dim,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
    )

    projected_diff = projection_matrix @ standardized_diff

    test_statistic, effective_df, p_value = compute_projected_pvalue(
        projected_diff,
        eigenvalues=whitening_eigenvalues,
    )
    return (
        float(test_statistic),
        int(projection_matrix.shape[0]),
        float(effective_df),
        float(p_value),
    )


__all__ = [
    "run_projected_wald_kernel",
]
