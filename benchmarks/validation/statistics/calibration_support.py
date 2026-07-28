"""Shared numerical primitives for calibration validation programs."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.stats import kstest

from benchmarks.validation.contracts.report_contract import wilson_interval

ContinuousCovarianceProfile = Literal["identity", "ar1_0.6", "ill_conditioned"]


def continuous_covariance(
    *,
    dimension: int,
    profile: ContinuousCovarianceProfile,
) -> np.ndarray:
    """Construct one supported synthetic continuous covariance profile."""

    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if profile == "identity":
        return np.eye(dimension, dtype=np.float64)
    if profile == "ar1_0.6":
        indices = np.arange(dimension)
        return 0.6 ** np.abs(indices[:, None] - indices[None, :])
    if profile == "ill_conditioned":
        eigenvalues = np.geomspace(1.0, 1e-3, num=dimension)
        return np.diag(eigenvalues)
    raise ValueError(f"Unsupported covariance profile: {profile!r}.")


def summarize_p_value_calibration(
    *,
    p_values: np.ndarray,
    n_replicates: int,
    alpha: float,
) -> dict[str, Any]:
    """Return the shared rejection-rate and uniformity report fields."""

    if p_values.shape != (n_replicates,):
        raise ValueError(f"p_values has shape {p_values.shape}; expected {(n_replicates,)}.")
    if not np.isfinite(p_values).all() or np.any(p_values < 0.0) or np.any(p_values > 1.0):
        raise ValueError("Simulated p-values must be finite values in [0, 1].")

    rejection_count = int(np.sum(p_values < alpha))
    rejection_rate = float(rejection_count / n_replicates)
    ci_low, ci_high = wilson_interval(rejection_count, n_replicates)
    ks_result = kstest(p_values, "uniform")
    return {
        "n_replicates": n_replicates,
        "alpha": alpha,
        "rejection_count": rejection_count,
        "rejection_rate": rejection_rate,
        "confidence_interval": {
            "method": "wilson_95",
            "low": ci_low,
            "high": ci_high,
        },
        "effect_estimate": {
            "name": "rejection_rate_minus_alpha",
            "value": float(rejection_rate - alpha),
        },
        "p_value_uniformity_summary": {
            "ks_statistic": float(ks_result.statistic),
            "ks_p_value": float(ks_result.pvalue),
            "mean": float(np.mean(p_values)),
            "median": float(np.median(p_values)),
            "q05": float(np.quantile(p_values, 0.05)),
            "q95": float(np.quantile(p_values, 0.95)),
        },
    }


__all__ = [
    "ContinuousCovarianceProfile",
    "continuous_covariance",
    "summarize_p_value_calibration",
]
