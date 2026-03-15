"""Canonical projection-dimension estimators for decomposition methods."""

from __future__ import annotations

import numpy as np


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Continuous effective rank via Shannon entropy of eigenvalue spectrum."""
    nonneg_eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    eigenvalue_sum = float(np.sum(nonneg_eigenvalues))
    if eigenvalue_sum <= 0:
        return 1.0

    normalized_spectrum = nonneg_eigenvalues / eigenvalue_sum
    normalized_spectrum = normalized_spectrum[normalized_spectrum > 0]
    if normalized_spectrum.size == 0:
        return 1.0
    shannon_entropy = -float(np.sum(normalized_spectrum * np.log(normalized_spectrum)))
    return float(np.exp(shannon_entropy))


def marchenko_pastur_signal_count(
    eigenvalues: np.ndarray,
    n_samples: int,
    n_features: int,
) -> int:
    """Count eigenvalues above the Marchenko-Pastur upper bound."""
    if n_samples <= 0 or n_features <= 0:
        return 1

    sorted_eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    positive_eigenvalues = sorted_eigenvalues[sorted_eigenvalues > 0]
    bulk_noise_variance = (
        float(np.median(positive_eigenvalues)) if positive_eigenvalues.size > 0 else 0.0
    )
    if bulk_noise_variance <= 0:
        return 1

    dimension_to_sample_ratio = float(n_features) / float(n_samples)
    mp_upper_bound = bulk_noise_variance * (1.0 + np.sqrt(dimension_to_sample_ratio)) ** 2
    n_signal_components = int(np.sum(sorted_eigenvalues > mp_upper_bound))
    return max(n_signal_components, 1)


def estimate_k_marchenko_pastur(
    eigenvalues: np.ndarray,
    *,
    n_samples: int,
    n_features: int,
    minimum_projection_dimension: int = 1,
) -> int:
    """Estimate projection dimension via Marchenko-Pastur signal-count thresholding."""
    sorted_eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    projection_dimension = int(
        marchenko_pastur_signal_count(
            sorted_eigenvalues, n_samples=n_samples, n_features=n_features
        )
    )
    projection_dimension = max(projection_dimension, int(minimum_projection_dimension))
    projection_dimension = min(projection_dimension, int(n_features))
    return projection_dimension


__all__ = [
    "effective_rank",
    "marchenko_pastur_signal_count",
    "estimate_k_marchenko_pastur",
]
