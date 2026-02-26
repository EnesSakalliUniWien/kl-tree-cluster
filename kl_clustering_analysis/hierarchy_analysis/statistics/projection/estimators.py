"""Pure eigenvalue-based dimension estimators.

These functions operate on eigenvalue arrays and data matrices only —
they have no dependency on tree structures or NetworkX.

Functions
---------
effective_rank
    Continuous effective rank via Shannon entropy (Roy & Vetterli, 2007).
marchenko_pastur_signal_count
    Count eigenvalues above the Marchenko-Pastur upper bound.
count_active_features
    Count features with non-zero variance.
"""

from __future__ import annotations

import numpy as np


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Continuous effective rank via Shannon entropy of the eigenvalue spectrum.

    Parameters
    ----------
    eigenvalues
        Non-negative eigenvalues, any order.

    Returns
    -------
    float
        exp(−Σ pᵢ log pᵢ). Returns 1.0 when the spectrum is degenerate.
    """
    eigs = np.maximum(eigenvalues, 0.0)
    total = eigs.sum()
    if total <= 0:
        return 1.0
    p = eigs / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def marchenko_pastur_signal_count(
    eigenvalues: np.ndarray,
    n: int,
    d: int,
) -> int:
    """Count eigenvalues above the Marchenko-Pastur upper bound.

    Parameters
    ----------
    eigenvalues
        Sample covariance eigenvalues (descending order preferred but not required).
    n
        Number of observations (descendant leaves).
    d
        Number of features.

    Returns
    -------
    int
        Number of signal eigenvalues exceeding σ² (1 + √(d/n))².
    """
    if n < 2 or d < 1:
        return 0
    gamma = d / n
    sigma2 = float(np.median(eigenvalues[eigenvalues > 0])) if np.any(eigenvalues > 0) else 0.0
    if sigma2 <= 0:
        return 0
    upper = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
    return int(np.sum(eigenvalues > upper))


def count_active_features(data_sub: np.ndarray) -> int:
    """Count features with non-zero variance (not all-0 or all-1).

    Parameters
    ----------
    data_sub
        Binary data matrix (n × d) for a subtree's descendants.

    Returns
    -------
    int
        Number of columns with var > 0.
    """
    if data_sub.shape[0] <= 1:
        return 0
    return int(np.sum(np.var(data_sub, axis=0) > 0))


__all__ = [
    "effective_rank",
    "marchenko_pastur_signal_count",
    "count_active_features",
]
