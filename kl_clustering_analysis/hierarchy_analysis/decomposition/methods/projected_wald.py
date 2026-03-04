"""Shared projected Wald test kernel used by edge and sibling tests."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.contracts import ProjectedTestResult
from .projection_basis import build_random_orthonormal_basis
from ...statistics.projection.satterthwaite import compute_projected_pvalue as _compute_projected_pvalue


def compute_projected_pvalue(
    projected: np.ndarray,
    df: int,
    *,
    eigenvalues: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
    """Compute projected test statistic and p-value."""
    return _compute_projected_pvalue(
        np.asarray(projected, dtype=np.float64),
        int(df),
        eigenvalues=eigenvalues,
    )


def run_projected_wald_kernel(
    z: np.ndarray,
    *,
    seed: int,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    k_fallback: Callable[[int], int] | None = None,
) -> tuple[float, int, float, float]:
    """Project a standardized vector and compute Wald statistic/p-value.

    Returns
    -------
    tuple[float, int, float, float]
        ``(statistic, nominal_k, effective_df, p_value)``
    """
    z_vec = np.asarray(z, dtype=np.float64)
    d = int(z_vec.shape[0])

    if spectral_k is not None and spectral_k > 0:
        k = min(int(spectral_k), d)
    else:
        if k_fallback is None:
            raise ValueError("k_fallback must be provided when spectral_k is None.")
        k = min(int(k_fallback(d)), d)

    eig_for_whitening: np.ndarray | None = None

    if pca_projection is not None:
        pca_projection = np.asarray(pca_projection, dtype=np.float64)
        k_pca = int(pca_projection.shape[0])
        if k_pca >= k:
            R = pca_projection[:k]
            eig_for_whitening = (
                np.asarray(pca_eigenvalues[:k], dtype=np.float64)
                if pca_eigenvalues is not None
                else None
            )
        else:
            R_pad = build_random_orthonormal_basis(
                n_features=d,
                k=k - k_pca,
                random_state=seed,
                use_cache=False,
            )
            R = np.vstack([pca_projection, R_pad])
            eig_for_whitening = (
                np.asarray(pca_eigenvalues, dtype=np.float64)
                if pca_eigenvalues is not None
                else None
            )
    else:
        R = build_random_orthonormal_basis(
            n_features=d,
            k=k,
            random_state=seed,
            use_cache=False,
        )

    try:
        if hasattr(R, "dot"):
            projected = R.dot(z_vec)
        else:
            projected = R @ z_vec
    except Exception as exc:  # pragma: no cover - preserves context for callers
        raise RuntimeError(
            f"Projection failed: z.shape={z_vec.shape}, R.shape={R.shape}, "
            f"z_stats={np.min(z_vec)}/{np.max(z_vec)}"
        ) from exc

    stat, effective_df, pval = compute_projected_pvalue(
        projected,
        k,
        eigenvalues=eig_for_whitening,
    )
    return float(stat), int(k), float(effective_df), float(pval)


def run_projected_wald_test(
    z: np.ndarray,
    *,
    k: int,
    seed: int | None = None,
    random_state: int | None = None,
    projection: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
) -> ProjectedTestResult:
    """Run projected Wald test from a pre-standardized z vector.

    Compatibility helper for callers that already preselect ``k``.
    """
    z_vec = np.asarray(z, dtype=np.float64)
    if not np.isfinite(z_vec).all():
        return ProjectedTestResult(np.nan, np.nan, np.nan, invalid=True)

    if projection is None:
        seed_value = seed if seed is not None else random_state
        R = build_random_orthonormal_basis(
            n_features=int(z_vec.shape[0]),
            k=int(k),
            random_state=seed_value,
            use_cache=False,
        )
    else:
        R = np.asarray(projection, dtype=np.float64)

    projected = R @ z_vec
    stat, effective_df, pval = compute_projected_pvalue(
        projected,
        int(k),
        eigenvalues=eigenvalues,
    )
    return ProjectedTestResult(
        statistic=float(stat),
        degrees_of_freedom=float(effective_df),
        p_value=float(pval),
        invalid=False,
    )


__all__ = [
    "run_projected_wald_test",
    "run_projected_wald_kernel",
    "compute_projected_pvalue",
]
