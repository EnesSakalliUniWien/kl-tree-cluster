"""Shared projected Wald test kernel used by edge and sibling tests."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .projection_basis import build_projection_basis_with_padding
from ...statistics.projection.satterthwaite import compute_projected_pvalue as _compute_projected_pvalue


def compute_projected_pvalue(
    projected: np.ndarray,
    degrees_of_freedom: int,
    *,
    eigenvalues: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute projected test statistic and p-value."""
    return _compute_projected_pvalue(
        np.asarray(projected, dtype=np.float64),
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

    # Projection basis policy is centralized in projection_basis.py.
    R, eig_for_whitening = build_projection_basis_with_padding(
        n_features=d,
        k=k,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
        random_state=seed,
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


__all__ = [
    "run_projected_wald_kernel",
    "compute_projected_pvalue",
]
