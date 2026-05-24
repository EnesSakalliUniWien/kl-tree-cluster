"""Shared projected Wald test kernel used by edge and sibling tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .projected_wald_projection_basis import build_pca_projection_basis
from .projected_wald_reference_distribution import compute_projected_pvalue


@dataclass(frozen=True)
class ProjectedWaldResult:
    """Projected-Wald statistic and its complete reference law."""

    statistic: float
    projection_dimension: int
    reference_scale: float
    degrees_of_freedom: float
    p_value: float


def run_projected_wald_kernel(
    z: np.ndarray,
    *,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
) -> ProjectedWaldResult:
    """Project a standardized vector and compute Wald statistic/p-value.

    Returns
    -------
    ProjectedWaldResult
        Statistic and reference law. For the current orthonormal projection basis,
        ``reference_scale`` is 1 and ``degrees_of_freedom`` is the projection
        dimension.
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
        return ProjectedWaldResult(
            statistic=0.0,
            projection_dimension=0,
            reference_scale=1.0,
            degrees_of_freedom=0.0,
            p_value=1.0,
        )
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

    reference = compute_projected_pvalue(
        projected_diff,
        eigenvalues=whitening_eigenvalues,
    )
    return ProjectedWaldResult(
        statistic=float(reference.statistic),
        projection_dimension=int(projection_matrix.shape[0]),
        reference_scale=float(reference.reference_scale),
        degrees_of_freedom=float(reference.degrees_of_freedom),
        p_value=float(reference.p_value),
    )


__all__ = [
    "ProjectedWaldResult",
    "run_projected_wald_kernel",
]
