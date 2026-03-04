"""Projection-dimension estimators exposed via the new decomposition API."""

from __future__ import annotations

import numpy as np

from ...statistics.projection.estimators import (
    count_active_features,
    effective_rank,
    marchenko_pastur_signal_count,
)


def estimate_k_effective_rank(
    eigenvalues: np.ndarray,
    *,
    min_k: int = 1,
    d_active: int | None = None,
) -> int:
    """Estimate k from effective rank with optional floor and cap."""
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    k = int(np.round(effective_rank(eigs)))
    k = max(k, int(min_k))
    if d_active is not None:
        k = min(k, int(d_active))
    return k


def estimate_k_marchenko_pastur(
    eigenvalues: np.ndarray,
    *,
    n_desc: int,
    d_active: int,
    min_k: int = 1,
) -> int:
    """Estimate k via Marchenko-Pastur signal-count thresholding."""
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    k = int(marchenko_pastur_signal_count(eigs, n_desc=n_desc, d_active=d_active))
    k = max(k, int(min_k))
    k = min(k, int(d_active))
    return k


def estimate_k_active_features(
    data_sub: np.ndarray,
    *,
    min_k: int = 1,
) -> int:
    """Estimate k as active-feature count with a hard floor."""
    data = np.asarray(data_sub, dtype=np.float64)
    k = int(count_active_features(data))
    return max(k, int(min_k))


__all__ = [
    "estimate_k_effective_rank",
    "estimate_k_marchenko_pastur",
    "estimate_k_active_features",
]
