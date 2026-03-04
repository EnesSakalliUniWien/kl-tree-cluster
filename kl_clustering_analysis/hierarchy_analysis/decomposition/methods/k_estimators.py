"""Canonical projection-dimension estimators for decomposition methods."""

from __future__ import annotations

import numpy as np


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Continuous effective rank via Shannon entropy of eigenvalue spectrum."""
    eigs = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(np.sum(eigs))
    if total <= 0:
        return 1.0

    p = eigs / total
    p = p[p > 0]
    if p.size == 0:
        return 1.0
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def marchenko_pastur_signal_count(
    eigenvalues: np.ndarray,
    n_desc: int,
    d_active: int,
) -> int:
    """Count eigenvalues above the Marchenko-Pastur upper bound."""
    if n_desc <= 0 or d_active <= 0:
        return 1

    eigs = np.asarray(eigenvalues, dtype=np.float64)
    sigma2 = float(np.median(eigs[eigs > 0])) if np.any(eigs > 0) else 0.0
    if sigma2 <= 0:
        return 1

    q = float(d_active) / float(n_desc)
    mp_upper = sigma2 * (1.0 + np.sqrt(q)) ** 2
    k = int(np.sum(eigs > mp_upper))
    return max(k, 1)


def count_active_features(data_sub: np.ndarray) -> int:
    """Count non-constant feature columns."""
    data = np.asarray(data_sub, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] == 0:
        return 1
    col_var = np.var(data, axis=0)
    return max(int(np.sum(col_var > 0)), 1)


def estimate_k_effective_rank(
    eigenvalues: np.ndarray,
    *,
    minimum_projection_dimension: int = 1,
    d_active: int | None = None,
) -> int:
    """Estimate k from effective rank with optional floor and cap."""
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    k = int(np.round(effective_rank(eigs)))
    k = max(k, int(minimum_projection_dimension))
    if d_active is not None:
        k = min(k, int(d_active))
    return k


def estimate_k_marchenko_pastur(
    eigenvalues: np.ndarray,
    *,
    n_desc: int,
    d_active: int,
    minimum_projection_dimension: int = 1,
) -> int:
    """Estimate k via Marchenko-Pastur signal-count thresholding."""
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    k = int(marchenko_pastur_signal_count(eigs, n_desc=n_desc, d_active=d_active))
    k = max(k, int(minimum_projection_dimension))
    k = min(k, int(d_active))
    return k


def estimate_k_active_features(
    data_sub: np.ndarray,
    *,
    minimum_projection_dimension: int = 1,
) -> int:
    """Estimate k as active-feature count with a hard floor."""
    data = np.asarray(data_sub, dtype=np.float64)
    k = int(count_active_features(data))
    return max(k, int(minimum_projection_dimension))


__all__ = [
    "effective_rank",
    "marchenko_pastur_signal_count",
    "count_active_features",
    "estimate_k_effective_rank",
    "estimate_k_marchenko_pastur",
    "estimate_k_active_features",
]
