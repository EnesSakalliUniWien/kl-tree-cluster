"""Edge (Gate 2) calibration via **descendant-balance** weighted Gamma GLM.

For each edge P→C at parent P with children L, R, the null-likelihood
weight is the **split balance**:

    w_i = min(n_L, n_R) / n_parent

This is a purely structural signal from the tree topology:

- Balanced split (w → 0.5): both children have many descendants.
  Under the null (noise tree), balanced splits are common and carry
  post-selection inflation → good calibration references.
- Imbalanced split (w → 0): one tiny child peels off. More likely
  a real cluster boundary → less useful for calibration.

Using descendant counts avoids circular dependency on any test output
(no sibling p-values, no edge p-values — purely topological).

Intercept-only Gamma GLM
-------------------------
Under H₀, ``T ~ c · χ²(k)`` so ``r = T/k`` has ``E[r] = c`` and
``Var(r) = 2c²/k``. The Gamma family with ``V(μ) = μ²`` matches this.
An intercept-only model with log link and frequency weights ``w_i``
yields ``ĉ_edge = exp(β₀)``.

Pipeline ordering
-----------------

    Step 1: Gate 2 RAW (no calibration) — ``annotate_child_parent_divergence()``
    Step 2: Gate 3 (sibling divergence)
    Step 3: Edge calibration post-hoc (descendant-balance weights) — this module
    Step 4: Re-BH correct edges

References
----------
- Efron (2004, 2007): Empirical null estimation for large-scale testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

try:
    import statsmodels.api as sm

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

logger = logging.getLogger(__name__)

# Minimum edges for calibration tiers
_MIN_REGRESSION = 5
_MIN_MEDIAN = 3


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class EdgeCalibrationModel:
    """Result of fitting the edge inflation model.

    Public API — used by annotation and potentially post-hoc merge.
    """

    method: str  # "gamma_glm", "weighted_mean", "none"
    n_calibration: int  # total edges used (with weight > 0)
    global_c_hat: float  # weighted mean of T/k ratios
    max_observed_ratio: float = 1.0  # upper clamp for ĉ
    beta: Optional[np.ndarray] = None  # GLM coefficients
    diagnostics: Dict = field(default_factory=dict)


@dataclass
class _EdgeRecord:
    """Per-edge record with descendant-balance weight."""

    child_id: str
    parent_id: str
    stat: float  # raw Wald T
    df: float  # projection dimension k
    pval: float  # raw Wald p
    weight: float  # min(n_L, n_R) / n_parent, purely structural [0, 0.5]
    is_null_like: bool  # balanced split (weight > 0.3) used for max_c


# =============================================================================
# Inflation estimation
# =============================================================================


def _fit_edge_calibration_model(
    records: List[_EdgeRecord],
) -> EdgeCalibrationModel:
    """Estimate edge post-selection inflation via weighted Gamma GLM.

    Uses ``w_i = p_sibling_raw(parent_i)`` as frequency weights so that
    null-like edges (where siblings look the same) dominate the estimate.

    Architecture mirrors ``cousin_weighted_wald._fit_weighted_inflation_model``.

    ``max_observed_ratio`` is computed from null-like edges only (those
    with ``is_null_like=True``) to prevent signal edges from inflating
    the extrapolation clamp.
    """
    valid = [r for r in records if np.isfinite(r.stat) and r.df > 0 and r.stat > 0 and r.weight > 0]

    if not valid:
        logger.warning("Edge calibration: 0 valid edges — no calibration.")
        return EdgeCalibrationModel(
            method="none",
            n_calibration=0,
            global_c_hat=1.0,
            max_observed_ratio=1.0,
        )

    ratios = np.array([r.stat / r.df for r in valid])
    weights = np.array([r.weight for r in valid])

    # max_observed_ratio from null-like edges only
    null_like_ratios = np.array([r.stat / r.df for r in valid if r.is_null_like])
    max_c = float(np.max(null_like_ratios)) if len(null_like_ratios) > 0 else float(np.max(ratios))

    n_cal = len(ratios)
    global_c = float(np.average(ratios, weights=weights))

    if n_cal < _MIN_MEDIAN:
        logger.warning(
            "Edge calibration: only %d valid edges (need ≥%d) — raw stats used.",
            n_cal,
            _MIN_MEDIAN,
        )
        return EdgeCalibrationModel(
            method="none",
            n_calibration=n_cal,
            global_c_hat=global_c,
            max_observed_ratio=max_c,
        )

    if n_cal < _MIN_REGRESSION:
        logger.info(
            "Edge calibration: %d valid edges (need ≥%d for GLM) — "
            "using weighted mean ĉ = %.3f.",
            n_cal,
            _MIN_REGRESSION,
            global_c,
        )
        return EdgeCalibrationModel(
            method="weighted_mean",
            n_calibration=n_cal,
            global_c_hat=global_c,
            max_observed_ratio=max_c,
        )

    # --- Intercept-only Gamma GLM ---
    X = np.ones((n_cal, 1))
    beta = None
    method = "weighted_mean"  # fallback default
    glm_diagnostics: Dict = {}

    if _HAS_STATSMODELS:
        try:
            glm = sm.GLM(
                ratios,
                X,
                family=sm.families.Gamma(link=sm.families.links.Log()),
                freq_weights=weights,
            )
            result = glm.fit()
            beta = result.params

            glm_diagnostics = {
                "deviance": float(result.deviance),
                "null_deviance": float(result.null_deviance),
                "aic": float(result.aic),
                "scale": float(result.scale),
                "converged": bool(result.converged),
            }
            method = "gamma_glm"

            logger.info(
                "Edge calibration: fitted Gamma GLM (intercept-only) on %d edges "
                "(eff. n=%.1f). β₀ = %.3f → ĉ = %.3f.",
                n_cal,
                float(np.sum(weights) ** 2 / np.sum(weights**2)),
                beta[0],
                float(np.exp(beta[0])),
            )
        except Exception as exc:
            logger.warning(
                "Edge calibration: Gamma GLM failed (%s) — falling back to WLS.",
                exc,
            )
            beta = None

    # --- Fallback: Weighted log-linear regression (WLS) ---
    if beta is None:
        log_r = np.log(ratios)
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, np.newaxis]
        yw = log_r * sqrt_w

        try:
            beta, _residuals, _rank, _sv = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            logger.warning(
                "Edge calibration: WLS also failed — " "using weighted mean ĉ = %.3f.",
                global_c,
            )
            return EdgeCalibrationModel(
                method="weighted_mean",
                n_calibration=n_cal,
                global_c_hat=global_c,
                max_observed_ratio=max_c,
            )

        method = "weighted_regression"
        logger.info(
            "Edge calibration: WLS fallback on %d edges. " "β₀ = %.3f → ĉ = %.3f.",
            n_cal,
            beta[0],
            float(np.exp(beta[0])),
        )

    diagnostics = {
        "n_calibration": n_cal,
        "global_c_hat": float(global_c),
        "max_observed_ratio": float(max_c),
        "total_weight": float(np.sum(weights)),
        "effective_n": float(np.sum(weights) ** 2 / np.sum(weights**2)),
        "n_null_like": int(len(null_like_ratios)),
        **glm_diagnostics,
    }

    return EdgeCalibrationModel(
        method=method,
        n_calibration=n_cal,
        global_c_hat=global_c,
        max_observed_ratio=max_c,
        beta=np.asarray(beta) if beta is not None else None,
        diagnostics=diagnostics,
    )


def predict_edge_inflation_factor(
    model: EdgeCalibrationModel,
) -> float:
    """Predict inflation factor ĉ_edge.

    With intercept-only model, ĉ = exp(β₀) — a single global constant
    (the weighted mean of T/k).

    Clamped to [1.0, max_observed_ratio].
    """
    if model.method == "none":
        return 1.0

    if model.method == "weighted_mean":
        return max(model.global_c_hat, 1.0)

    # Intercept-only GLM or WLS: ĉ = exp(β₀)
    if model.beta is None:
        return max(model.global_c_hat, 1.0)

    c_hat = float(np.exp(model.beta[0]))
    c_hat = min(c_hat, model.max_observed_ratio)
    return max(c_hat, 1.0)


# =============================================================================
# Public API
# =============================================================================


def calibrate_edges_from_sibling_neighborhood(
    tree: nx.DiGraph,
    results_df: pd.DataFrame,
    alpha: float = 0.05,
    fdr_method: str = "tree_bh",
) -> pd.DataFrame:
    """Calibrate edge statistics using descendant-balance weights.

    This is a post-hoc calibration step that runs AFTER Gate 2 (raw).
    It uses purely structural information (descendant leaf counts) to
    identify null-like edges for calibration.

    For each edge P→C at parent P with children L, R:

    1. Retrieves the raw test statistic T and projection dimension k.
    2. Computes weight ``w = min(n_L, n_R) / n_parent``.
    3. Fits an intercept-only Gamma GLM to estimate ĉ_edge.
    4. Deflates: ``T_adj = T / ĉ``, recomputes p-values.
    5. Re-applies BH correction and updates the DataFrame.

    Parameters
    ----------
    tree
        Directed hierarchy.
    results_df
        DataFrame indexed by node id, with Gate 2 columns populated.
    alpha
        Significance level for re-BH correction.
    fdr_method
        FDR correction method: ``"tree_bh"``, ``"flat"``, or ``"level_wise"``.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with calibrated edge statistics.
    """
    from collections import defaultdict

    from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths

    from ..multiple_testing import apply_multiple_testing_correction

    df = results_df

    # --- Retrieve raw edge data stashed by annotate_child_parent_divergence ---
    raw_data = df.attrs.get("_edge_raw_test_data")
    if raw_data is None:
        logger.warning(
            "Edge calibration: no raw test data in attrs — "
            "annotate_child_parent_divergence must run first."
        )
        return df

    child_ids: List[str] = raw_data["child_ids"]
    parent_ids: List[str] = raw_data["parent_ids"]
    test_stats: np.ndarray = raw_data["test_stats"]
    degrees_of_freedom: np.ndarray = raw_data["degrees_of_freedom"]
    p_values_raw: np.ndarray = raw_data["p_values"]
    child_leaf_counts: np.ndarray = raw_data["child_leaf_counts"]
    parent_leaf_counts: np.ndarray = raw_data["parent_leaf_counts"]

    n_edges = len(child_ids)
    if n_edges == 0:
        return df

    # --- Build sibling leaf-count map per parent ---
    # For each parent, collect the leaf counts of its children so we can
    # compute min(n_L, n_R) / n_parent.
    parent_children_leaf_counts: dict[str, list[int]] = defaultdict(list)
    for i in range(n_edges):
        parent_children_leaf_counts[parent_ids[i]].append(int(child_leaf_counts[i]))

    # --- Build per-edge records with descendant-balance weights ---
    records: List[_EdgeRecord] = []

    for i in range(n_edges):
        if not np.isfinite(test_stats[i]) or degrees_of_freedom[i] <= 0:
            continue

        parent_id = parent_ids[i]
        n_parent = int(parent_leaf_counts[i])

        # Weight: min(n_L, n_R) / n_parent
        # Balanced splits (w → 0.5) are more null-like under noise.
        # Imbalanced splits (w → 0) are more likely real boundaries.
        sibling_counts = parent_children_leaf_counts.get(parent_id, [])
        if len(sibling_counts) >= 2 and n_parent > 0:
            weight = float(min(sibling_counts)) / float(n_parent)
        else:
            weight = 0.0  # non-binary or degenerate → skip

        # Null-like flag: balanced split (weight > 0.3 ≈ 60/40 split)
        is_null_like = weight > 0.3

        records.append(
            _EdgeRecord(
                child_id=child_ids[i],
                parent_id=parent_id,
                stat=float(test_stats[i]),
                df=float(degrees_of_freedom[i]),
                pval=float(p_values_raw[i]),
                weight=weight,
                is_null_like=is_null_like,
            )
        )

    if not records:
        logger.warning("Edge calibration: no valid edges to calibrate.")
        return df

    # --- Fit inflation model ---
    model = _fit_edge_calibration_model(records)

    if model.method == "none":
        logger.info("Edge calibration: model is 'none' — no deflation applied.")
        df.attrs["edge_calibration_model"] = model
        df.attrs["edge_calibration_audit"] = model.diagnostics
        return df

    # --- Deflate and re-test ---
    c_hat = predict_edge_inflation_factor(model)
    logger.info("Edge calibration: ĉ = %.3f (method=%s).", c_hat, model.method)

    deflated_stats = test_stats.copy()
    deflated_pvals = p_values_raw.copy()

    for i in range(n_edges):
        if not np.isfinite(test_stats[i]) or degrees_of_freedom[i] <= 0:
            continue
        if c_hat > 1.0:
            t_adj = test_stats[i] / c_hat
            deflated_stats[i] = t_adj
            deflated_pvals[i] = float(chi2.sf(t_adj, df=degrees_of_freedom[i]))

    # --- Re-apply BH correction ---
    p_values_for_correction = np.where(np.isfinite(deflated_pvals), deflated_pvals, 1.0)
    nonfinite_p_mask = ~np.isfinite(deflated_pvals)

    node_depths = compute_node_depths(tree)
    child_depths = np.array([node_depths.get(cid, 0) for cid in child_ids])

    reject_null, p_values_corrected = apply_multiple_testing_correction(
        p_values=p_values_for_correction,
        child_ids=child_ids,
        child_depths=child_depths,
        alpha=alpha,
        method=fdr_method,
        tree=tree,
    )
    reject_null = np.where(nonfinite_p_mask, False, reject_null)

    # --- Update DataFrame columns ---
    df.loc[child_ids, "Child_Parent_Divergence_P_Value"] = deflated_pvals
    df.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = p_values_corrected
    df.loc[child_ids, "Child_Parent_Divergence_Significant"] = reject_null

    # --- Audit ---
    df.attrs["edge_calibration_model"] = model
    df.attrs["edge_calibration_audit"] = {
        "c_hat": c_hat,
        **model.diagnostics,
    }

    logger.info(
        "Edge calibration applied: method=%s, ĉ=%.3f, %d edges, " "%d null-like.",
        model.method,
        c_hat,
        model.n_calibration,
        model.diagnostics.get("n_null_like", 0),
    )

    return df
