"""Post-selection inflation calibration for the edge (child-parent) test.

Under the null, the projected Wald χ²(k) edge test is systematically
inflated because the tree topology was chosen *from the same data*.
This module estimates the inflation factor ĉ from **null-like edges** —
those whose parent node has no signal eigenvalues above the
Marchenko-Pastur threshold (spectral_k == SPECTRAL_MINIMUM_DIMENSION).

Architecture mirrors ``sibling_divergence/calibration.py``:
  1. Identify null-like edges via spectral oracle
  2. Fit log-linear regression: log(T_i/k_i) = β₀ + β₁·log(n_parent_i)
  3. Deflate ALL edges: T_adj = T / ĉ, p_adj = χ².sf(T_adj, k)
  4. Then BH proceeds on calibrated p-values

Key difference from sibling calibration: we deflate ALL edges (not just
focal), because under pure null every edge is inflated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.stats import chi2

logger = logging.getLogger(__name__)


# =========================================================================
# Data structures
# =========================================================================


@dataclass
class EdgeCalibrationModel:
    """Fitted post-selection inflation model for edge tests."""

    n_calibration: int
    global_inflation_factor: float  # median(T_i / k_i) over null-like edges
    max_observed_ratio: float = 1.0
    beta: Optional[np.ndarray] = None  # [β₀, β₁] regression on (intercept, log(n_parent))
    diagnostics: Dict = field(default_factory=dict)


# =========================================================================
# Fitting
# =========================================================================


def _identify_null_like_edges(
    parent_ids: list[str],
    spectral_dims: dict[str, int] | None,
    invalid_mask: np.ndarray,
) -> np.ndarray:
    """Boolean mask: True for edges whose parent has no signal eigenvalues."""
    n = len(parent_ids)
    if spectral_dims is None:
        return np.zeros(n, dtype=bool)

    min_dim = getattr(config, "SPECTRAL_MINIMUM_DIMENSION", 2)
    mask = np.zeros(n, dtype=bool)
    for i, pid in enumerate(parent_ids):
        if invalid_mask[i]:
            continue
        k = spectral_dims.get(pid)
        if k is not None and k <= min_dim:
            mask[i] = True
    return mask


def fit_edge_inflation_model(
    test_statistics: np.ndarray,
    degrees_of_freedom: np.ndarray,
    parent_leaf_counts: np.ndarray,
    null_like_mask: np.ndarray,
) -> EdgeCalibrationModel:
    """Estimate edge post-selection inflation from null-like edges.

    Uses a simple log-linear regression:
        log(T_i / k_i) = β₀ + β₁ · log(n_parent_i)
    on null-like edges only.
    """
    valid_edge_mask = (
        null_like_mask
        & np.isfinite(test_statistics)
        & (degrees_of_freedom > 0)
        & (test_statistics > 0)
        & (parent_leaf_counts > 0)
    )

    null_test_statistics = test_statistics[valid_edge_mask]
    null_degrees_of_freedom = degrees_of_freedom[valid_edge_mask]
    null_parent_leaf_counts = parent_leaf_counts[valid_edge_mask]
    n_calibration_edges = int(valid_edge_mask.sum())

    diagnostics: dict = {"n_calibration": n_calibration_edges}

    if n_calibration_edges == 0:
        logger.info("Edge calibration: 0 null-like edges — using ĉ = 1.0 (no deflation).")
        diagnostics["fit_status"] = "neutral_no_data"
        return EdgeCalibrationModel(
            n_calibration=0,
            global_inflation_factor=1.0,
            max_observed_ratio=1.0,
            beta=np.zeros(2, dtype=np.float64),
            diagnostics=diagnostics,
        )

    observed_ratios = null_test_statistics / null_degrees_of_freedom
    median_observed_ratio = float(np.median(observed_ratios))
    max_observed_ratio = float(np.max(observed_ratios))

    diagnostics["median_ratio"] = median_observed_ratio
    diagnostics["max_observed_ratio"] = max_observed_ratio

    # Fit log-linear: log(r) = β₀ + β₁·log(n)
    log_ratios = np.log(observed_ratios)
    design_matrix = np.column_stack(
        [
            np.ones(n_calibration_edges),
            np.log(null_parent_leaf_counts.astype(float)),
        ]
    )

    try:
        beta, _res, _rank, _sv = np.linalg.lstsq(design_matrix, log_ratios, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("Edge calibration: regression failed — using median ratio.")
        diagnostics["fit_status"] = "neutral_fit_failure"
        return EdgeCalibrationModel(
            n_calibration=n_calibration_edges,
            global_inflation_factor=median_observed_ratio,
            max_observed_ratio=max_observed_ratio,
            beta=np.zeros(2, dtype=np.float64),
            diagnostics=diagnostics,
        )

    fitted_log_ratios = design_matrix @ beta
    residual_sum_of_squares = float(np.sum((log_ratios - fitted_log_ratios) ** 2))
    total_sum_of_squares = float(np.sum((log_ratios - np.mean(log_ratios)) ** 2))
    r_squared = (
        1.0 - residual_sum_of_squares / total_sum_of_squares if total_sum_of_squares > 0 else 0.0
    )

    diagnostics["fit_status"] = "regression"
    diagnostics["r_squared"] = r_squared
    diagnostics["beta"] = beta.tolist()

    logger.info(
        "Edge calibration: fitted on %d null-like edges. "
        "β = [%.3f, %.3f], R² = %.3f, median T/k = %.3f.",
        n_calibration_edges,
        beta[0],
        beta[1],
        r_squared,
        median_observed_ratio,
    )

    return EdgeCalibrationModel(
        n_calibration=n_calibration_edges,
        global_inflation_factor=median_observed_ratio,
        max_observed_ratio=max_observed_ratio,
        beta=np.asarray(beta),
        diagnostics=diagnostics,
    )


def predict_edge_inflation(
    model: EdgeCalibrationModel,
    n_parent: float,
) -> float:
    """Predict ĉ for one edge given parent sample size.

    Clamped to [1.0, max_observed_ratio].
    """
    if model.beta is None:
        return 1.0

    log_c = float(model.beta[0])
    if n_parent > 0:
        log_c = model.beta[0] + model.beta[1] * np.log(float(n_parent))
    c = float(np.exp(log_c))
    c = min(c, model.max_observed_ratio)
    return max(c, 1.0)


# =========================================================================
# Deflation
# =========================================================================


def deflate_edge_tests(
    test_statistics: np.ndarray,
    degrees_of_freedom: np.ndarray,
    p_values: np.ndarray,
    parent_leaf_counts: np.ndarray,
    model: EdgeCalibrationModel,
    invalid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deflate ALL edges and return calibrated (T_adj, p_adj) arrays.

    Unlike sibling calibration which only deflates focal pairs,
    edge calibration deflates every valid edge. Under pure null
    every edge is inflated by post-selection; under signal, true
    positive edges have T >> ĉ·k so deflation preserves them.
    """
    T_adj = test_statistics.copy()
    p_adj = p_values.copy()

    for i in range(len(test_statistics)):
        if invalid_mask[i] or not np.isfinite(test_statistics[i]) or degrees_of_freedom[i] <= 0:
            continue
        c_hat = predict_edge_inflation(model, float(parent_leaf_counts[i]))
        T_adj[i] = test_statistics[i] / c_hat
        p_adj[i] = float(chi2.sf(T_adj[i], df=degrees_of_freedom[i]))

    return T_adj, p_adj


__all__ = [
    "EdgeCalibrationModel",
    "deflate_edge_tests",
    "fit_edge_inflation_model",
    "predict_edge_inflation",
]
