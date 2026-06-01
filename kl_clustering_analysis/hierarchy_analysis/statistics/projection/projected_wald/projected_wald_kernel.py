"""Shared projected Wald test kernel used by edge and sibling tests."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from time import perf_counter

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


def _add_elapsed(
    stage_timings: MutableMapping[str, float] | None,
    key: str,
    start_sec: float,
) -> None:
    if stage_timings is None:
        return
    stage_timings[key] = float(stage_timings.get(key, 0.0)) + float(
        perf_counter() - start_sec
    )


def run_projected_wald_kernel(
    z: np.ndarray,
    *,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    stage_timings: MutableMapping[str, float] | None = None,
    timing_prefix: str | None = None,
) -> ProjectedWaldResult:
    """Project a standardized vector and compute Wald statistic/p-value.

    Returns
    -------
    ProjectedWaldResult
        Statistic and reference law. Conditional on the supplied orthonormal
        projection basis as fixed, ``reference_scale`` is 1 and
        ``degrees_of_freedom`` is the projection dimension.
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

    projection_start_sec = perf_counter()
    projection_matrix, whitening_eigenvalues = build_pca_projection_basis(
        k=projection_dim,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
    )
    if projection_matrix.shape[1] != n_features:
        raise ValueError(
            "PCA projection width must match the projected-Wald z-score dimension. "
            f"Got projection width {projection_matrix.shape[1]} for z dimension {n_features}."
        )

    projected_diff = projection_matrix @ standardized_diff
    if timing_prefix is not None:
        _add_elapsed(
            stage_timings,
            f"{timing_prefix}_projection_sec",
            projection_start_sec,
        )

    statistic_start_sec = perf_counter()
    reference = compute_projected_pvalue(
        projected_diff,
        eigenvalues=whitening_eigenvalues,
    )
    if timing_prefix is not None:
        _add_elapsed(
            stage_timings,
            f"{timing_prefix}_wald_statistic_sec",
            statistic_start_sec,
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
