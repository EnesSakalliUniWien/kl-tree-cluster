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
    degrees_of_freedom: float  # projection dimension k
    pval: float  # raw Wald p
    weight: float  # min(n_L, n_R) / n_parent, purely structural [0, 0.5]
    is_null_like: bool  # balanced split (weight > 0.3) used for max_c
    parent_sibling_different: bool = False  # parent sibling test rejects similarity


@dataclass(frozen=True)
class _SiblingFilterOutcome:
    """Result of optional sibling-aware filtering in low-information regimes."""

    filtered_records: list[_EdgeRecord]
    sibling_filter_applied: bool
    excluded_sibling_different_count: int


@dataclass(frozen=True)
class _EdgeCalibrationInputs:
    """Prepared numeric vectors and summary values for edge calibration."""

    ratio_values: np.ndarray
    weight_values: np.ndarray
    null_like_ratio_values: np.ndarray
    calibration_edge_count: int
    global_weighted_ratio: float
    max_observed_ratio: float
    total_weight: float
    effective_sample_size: float
    effective_sample_size_pre_filter: float
    sibling_filter_threshold: float
    sibling_filter_applied: bool
    excluded_sibling_different_count: int


@dataclass(frozen=True)
class _RawEdgeCalibrationData:
    """Raw Gate-2 edge arrays required for post-hoc calibration."""

    child_ids: list[str]
    parent_ids: list[str]
    test_statistics: np.ndarray
    degrees_of_freedom: np.ndarray
    raw_p_values: np.ndarray
    child_leaf_counts: np.ndarray
    parent_leaf_counts: np.ndarray

    @property
    def edge_count(self) -> int:
        return len(self.child_ids)


# =============================================================================
# Inflation estimation
# =============================================================================


def _compute_effective_sample_size(weight_values: np.ndarray) -> float:
    """Compute weighted effective sample size: (sum(w)^2 / sum(w^2))."""
    if len(weight_values) == 0:
        return 0.0
    total_weight = float(np.sum(weight_values))
    squared_weight_sum = float(np.sum(weight_values**2))
    if squared_weight_sum <= 0.0:
        return 0.0
    return float((total_weight**2) / squared_weight_sum)


def _filter_valid_edge_records(records: List[_EdgeRecord]) -> List[_EdgeRecord]:
    """Keep records valid for calibration fitting."""
    return [
        record
        for record in records
        if (
            np.isfinite(record.stat)
            and record.degrees_of_freedom > 0
            and record.stat > 0
            and record.weight > 0
        )
    ]


def _apply_low_information_sibling_filter(
    valid_records: List[_EdgeRecord],
    effective_sample_size_pre_filter: float,
    sibling_filter_threshold: float,
) -> _SiblingFilterOutcome:
    """Optionally exclude sibling-different edges when information is low."""
    sibling_filter_applied = False
    excluded_sibling_different_count = 0
    filtered_records = valid_records

    if sibling_filter_threshold > 0.0 and effective_sample_size_pre_filter < sibling_filter_threshold:
        candidate_filtered_records = [
            record for record in valid_records if not record.parent_sibling_different
        ]
        excluded_sibling_different_count = len(valid_records) - len(candidate_filtered_records)

        if len(candidate_filtered_records) >= _MIN_MEDIAN:
            filtered_records = candidate_filtered_records
            sibling_filter_applied = excluded_sibling_different_count > 0
            logger.info(
                "Edge calibration: low effective_n=%.2f (< %.2f), "
                "excluded %d sibling-different edges from fit.",
                effective_sample_size_pre_filter,
                sibling_filter_threshold,
                excluded_sibling_different_count,
            )
        elif excluded_sibling_different_count > 0:
            logger.info(
                "Edge calibration: low effective_n=%.2f but sibling-based exclusion "
                "would leave too few records (%d); using unfiltered fit.",
                effective_sample_size_pre_filter,
                len(candidate_filtered_records),
            )

    return _SiblingFilterOutcome(
        filtered_records=filtered_records,
        sibling_filter_applied=sibling_filter_applied,
        excluded_sibling_different_count=excluded_sibling_different_count,
    )


def _build_edge_calibration_inputs(
    valid_records: List[_EdgeRecord],
    effective_sample_size_pre_filter: float,
    sibling_filter_threshold: float,
    sibling_filter_applied: bool,
    excluded_sibling_different_count: int,
) -> _EdgeCalibrationInputs:
    """Build numeric inputs and summaries used by calibration model fitting."""
    ratio_values = np.array(
        [record.stat / record.degrees_of_freedom for record in valid_records],
        dtype=float,
    )
    weight_values = np.array([record.weight for record in valid_records], dtype=float)
    null_like_ratio_values = np.array(
        [record.stat / record.degrees_of_freedom for record in valid_records if record.is_null_like]
    )

    if len(null_like_ratio_values) > 0:
        max_observed_ratio = float(np.max(null_like_ratio_values))
    else:
        max_observed_ratio = float(np.max(ratio_values))

    calibration_edge_count = len(ratio_values)
    global_weighted_ratio = float(np.average(ratio_values, weights=weight_values))
    total_weight = float(np.sum(weight_values))
    effective_sample_size = _compute_effective_sample_size(weight_values)

    return _EdgeCalibrationInputs(
        ratio_values=ratio_values,
        weight_values=weight_values,
        null_like_ratio_values=null_like_ratio_values,
        calibration_edge_count=calibration_edge_count,
        global_weighted_ratio=global_weighted_ratio,
        max_observed_ratio=max_observed_ratio,
        total_weight=total_weight,
        effective_sample_size=effective_sample_size,
        effective_sample_size_pre_filter=effective_sample_size_pre_filter,
        sibling_filter_threshold=sibling_filter_threshold,
        sibling_filter_applied=sibling_filter_applied,
        excluded_sibling_different_count=excluded_sibling_different_count,
    )


def _fit_edge_gamma_glm(
    ratio_values: np.ndarray,
    weight_values: np.ndarray,
    calibration_edge_count: int,
    effective_sample_size: float,
) -> tuple[np.ndarray | None, Dict]:
    """Fit intercept-only Gamma GLM for edge inflation."""
    design_matrix = np.ones((calibration_edge_count, 1))
    try:
        gamma_glm = sm.GLM(
            ratio_values,
            design_matrix,
            family=sm.families.Gamma(link=sm.families.links.Log()),
            freq_weights=weight_values,
        )
        gamma_glm_result = gamma_glm.fit()
        coefficient_vector = np.asarray(gamma_glm_result.params)

        glm_diagnostics = {
            "deviance": float(gamma_glm_result.deviance),
            "null_deviance": float(gamma_glm_result.null_deviance),
            "aic": float(gamma_glm_result.aic),
            "scale": float(gamma_glm_result.scale),
            "converged": bool(gamma_glm_result.converged),
        }

        logger.info(
            "Edge calibration: fitted Gamma GLM (intercept-only) on %d edges "
            "(eff. n=%.1f). β₀ = %.3f → ĉ = %.3f.",
            calibration_edge_count,
            effective_sample_size,
            coefficient_vector[0],
            float(np.exp(coefficient_vector[0])),
        )
        return coefficient_vector, glm_diagnostics
    except Exception as exception:
        logger.warning(
            "Edge calibration: Gamma GLM failed (%s) — falling back to WLS.",
            exception,
        )
        return None, {}


def _fit_edge_weighted_log_regression(
    ratio_values: np.ndarray,
    weight_values: np.ndarray,
    calibration_edge_count: int,
    global_weighted_ratio: float,
) -> np.ndarray | None:
    """Fit weighted log-linear fallback model for edge inflation."""
    log_ratio_values = np.log(ratio_values)
    square_root_weights = np.sqrt(weight_values)
    design_matrix = np.ones((calibration_edge_count, 1))
    weighted_design_matrix = design_matrix * square_root_weights[:, np.newaxis]
    weighted_targets = log_ratio_values * square_root_weights

    try:
        coefficient_vector, _residuals, _rank, _singular_values = np.linalg.lstsq(
            weighted_design_matrix, weighted_targets, rcond=None
        )
    except np.linalg.LinAlgError:
        logger.warning(
            "Edge calibration: WLS also failed — using weighted mean ĉ = %.3f.",
            global_weighted_ratio,
        )
        return None

    logger.info(
        "Edge calibration: WLS fallback on %d edges. β₀ = %.3f → ĉ = %.3f.",
        calibration_edge_count,
        coefficient_vector[0],
        float(np.exp(coefficient_vector[0])),
    )
    return np.asarray(coefficient_vector)


def _fit_edge_calibration_model(
    records: List[_EdgeRecord],
) -> EdgeCalibrationModel:
    """Estimate edge post-selection inflation via weighted Gamma GLM.

    Uses descendant-balance weights ``w_i = min(n_L, n_R) / n_parent``.
    In low-information regimes (small weighted effective sample size), an
    adaptive sibling-aware filter excludes edges whose parent is already
    sibling-different from the fit to reduce signal contamination.

    ``max_observed_ratio`` is computed from null-like edges only (those
    with ``is_null_like=True``) to prevent signal edges from inflating
    the extrapolation clamp.
    """
    from kl_clustering_analysis import config as runtime_config

    valid_records = _filter_valid_edge_records(records)

    if not valid_records:
        logger.warning("Edge calibration: 0 valid edges — no calibration.")
        return EdgeCalibrationModel(
            method="none",
            n_calibration=0,
            global_c_hat=1.0,
            max_observed_ratio=1.0,
        )

    pre_filter_weights = np.array([record.weight for record in valid_records], dtype=float)
    effective_sample_size_pre_filter = _compute_effective_sample_size(pre_filter_weights)

    sibling_filter_threshold = float(
        getattr(runtime_config, "EDGE_CAL_MIN_EFFECTIVE_N_FOR_SIB_FILTER", 0.0)
    )
    sibling_filter_outcome = _apply_low_information_sibling_filter(
        valid_records,
        effective_sample_size_pre_filter,
        sibling_filter_threshold,
    )
    valid_records = sibling_filter_outcome.filtered_records

    calibration_inputs = _build_edge_calibration_inputs(
        valid_records,
        effective_sample_size_pre_filter,
        sibling_filter_threshold,
        sibling_filter_outcome.sibling_filter_applied,
        sibling_filter_outcome.excluded_sibling_different_count,
    )

    common_diagnostics = {
        "n_calibration": calibration_inputs.calibration_edge_count,
        "global_c_hat": float(calibration_inputs.global_weighted_ratio),
        "max_observed_ratio": float(calibration_inputs.max_observed_ratio),
        "total_weight": calibration_inputs.total_weight,
        "effective_n": calibration_inputs.effective_sample_size,
        "effective_n_pre_filter": calibration_inputs.effective_sample_size_pre_filter,
        "sib_filter_threshold": calibration_inputs.sibling_filter_threshold,
        "sibling_filter_applied": calibration_inputs.sibling_filter_applied,
        "n_excluded_sibling_different": int(calibration_inputs.excluded_sibling_different_count),
        "n_null_like": int(len(calibration_inputs.null_like_ratio_values)),
    }

    if calibration_inputs.calibration_edge_count < _MIN_MEDIAN:
        logger.warning(
            "Edge calibration: only %d valid edges (need ≥%d) — raw stats used.",
            calibration_inputs.calibration_edge_count,
            _MIN_MEDIAN,
        )
        return EdgeCalibrationModel(
            method="none",
            n_calibration=calibration_inputs.calibration_edge_count,
            global_c_hat=calibration_inputs.global_weighted_ratio,
            max_observed_ratio=calibration_inputs.max_observed_ratio,
            diagnostics=common_diagnostics,
        )

    if calibration_inputs.calibration_edge_count < _MIN_REGRESSION:
        logger.info(
            "Edge calibration: %d valid edges (need ≥%d for GLM) — "
            "using weighted mean ĉ = %.3f.",
            calibration_inputs.calibration_edge_count,
            _MIN_REGRESSION,
            calibration_inputs.global_weighted_ratio,
        )
        return EdgeCalibrationModel(
            method="weighted_mean",
            n_calibration=calibration_inputs.calibration_edge_count,
            global_c_hat=calibration_inputs.global_weighted_ratio,
            max_observed_ratio=calibration_inputs.max_observed_ratio,
            diagnostics=common_diagnostics,
        )

    coefficient_vector = None
    calibration_method = "weighted_mean"
    glm_diagnostics: Dict = {}

    if _HAS_STATSMODELS:
        coefficient_vector, glm_diagnostics = _fit_edge_gamma_glm(
            calibration_inputs.ratio_values,
            calibration_inputs.weight_values,
            calibration_inputs.calibration_edge_count,
            calibration_inputs.effective_sample_size,
        )
        if coefficient_vector is not None:
            calibration_method = "gamma_glm"

    if coefficient_vector is None:
        coefficient_vector = _fit_edge_weighted_log_regression(
            calibration_inputs.ratio_values,
            calibration_inputs.weight_values,
            calibration_inputs.calibration_edge_count,
            calibration_inputs.global_weighted_ratio,
        )
        if coefficient_vector is None:
            return EdgeCalibrationModel(
                method="weighted_mean",
                n_calibration=calibration_inputs.calibration_edge_count,
                global_c_hat=calibration_inputs.global_weighted_ratio,
                max_observed_ratio=calibration_inputs.max_observed_ratio,
                diagnostics=common_diagnostics,
            )
        calibration_method = "weighted_regression"

    diagnostics = {
        **common_diagnostics,
        **glm_diagnostics,
    }

    return EdgeCalibrationModel(
        method=calibration_method,
        n_calibration=calibration_inputs.calibration_edge_count,
        global_c_hat=calibration_inputs.global_weighted_ratio,
        max_observed_ratio=calibration_inputs.max_observed_ratio,
        beta=np.asarray(coefficient_vector) if coefficient_vector is not None else None,
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


def _extract_raw_edge_calibration_data(
    annotations_df: pd.DataFrame,
) -> _RawEdgeCalibrationData | None:
    """Load raw edge test arrays stored by Gate-2 annotation."""
    raw_data = annotations_df.attrs.get("_edge_raw_test_data")
    if raw_data is None:
        logger.warning(
            "Edge calibration: no raw test data in attrs — "
            "annotate_child_parent_divergence must run first."
        )
        return None

    return _RawEdgeCalibrationData(
        child_ids=raw_data["child_ids"],
        parent_ids=raw_data["parent_ids"],
        test_statistics=raw_data["test_stats"],
        degrees_of_freedom=raw_data["degrees_of_freedom"],
        raw_p_values=raw_data["p_values"],
        child_leaf_counts=raw_data["child_leaf_counts"],
        parent_leaf_counts=raw_data["parent_leaf_counts"],
    )


def _build_parent_children_leaf_count_map(
    raw_edge_data: _RawEdgeCalibrationData,
) -> dict[str, list[int]]:
    """Map each parent to its children's descendant leaf counts."""
    from collections import defaultdict

    parent_children_leaf_counts: dict[str, list[int]] = defaultdict(list)
    for edge_index in range(raw_edge_data.edge_count):
        parent_identifier = raw_edge_data.parent_ids[edge_index]
        child_leaf_count = int(raw_edge_data.child_leaf_counts[edge_index])
        parent_children_leaf_counts[parent_identifier].append(child_leaf_count)
    return parent_children_leaf_counts


def _extract_parent_sibling_difference_map(annotations_df: pd.DataFrame) -> dict[str, bool]:
    """Build per-parent sibling-difference flags from Gate-3 output, if available."""
    from kl_clustering_analysis.core_utils.data_utils import extract_bool_column_dict

    if "Sibling_BH_Different" not in annotations_df.columns:
        return {}
    return extract_bool_column_dict(
        annotations_df,
        "Sibling_BH_Different",
        null_policy="false",
    )


def _build_edge_records_for_calibration(
    raw_edge_data: _RawEdgeCalibrationData,
    parent_children_leaf_counts: dict[str, list[int]],
    parent_sibling_difference_map: dict[str, bool],
) -> List[_EdgeRecord]:
    """Construct calibration records with descendant-balance weights."""
    edge_records: List[_EdgeRecord] = []

    for edge_index in range(raw_edge_data.edge_count):
        raw_test_statistic = raw_edge_data.test_statistics[edge_index]
        raw_degrees_of_freedom = raw_edge_data.degrees_of_freedom[edge_index]
        if not np.isfinite(raw_test_statistic) or raw_degrees_of_freedom <= 0:
            continue

        parent_identifier = raw_edge_data.parent_ids[edge_index]
        parent_leaf_count = int(raw_edge_data.parent_leaf_counts[edge_index])
        sibling_leaf_counts = parent_children_leaf_counts.get(parent_identifier, [])

        if len(sibling_leaf_counts) >= 2 and parent_leaf_count > 0:
            split_balance_weight = float(min(sibling_leaf_counts)) / float(parent_leaf_count)
        else:
            split_balance_weight = 0.0

        is_null_like_split = split_balance_weight > 0.3
        edge_records.append(
            _EdgeRecord(
                child_id=raw_edge_data.child_ids[edge_index],
                parent_id=parent_identifier,
                stat=float(raw_test_statistic),
                degrees_of_freedom=float(raw_degrees_of_freedom),
                pval=float(raw_edge_data.raw_p_values[edge_index]),
                weight=split_balance_weight,
                is_null_like=is_null_like_split,
                parent_sibling_different=bool(
                    parent_sibling_difference_map.get(parent_identifier, False)
                ),
            )
        )

    return edge_records


def _deflate_edge_p_values(
    raw_edge_data: _RawEdgeCalibrationData,
    calibration_factor: float,
) -> np.ndarray:
    """Deflate edge statistics and recompute p-values."""
    deflated_p_values = raw_edge_data.raw_p_values.copy()

    for edge_index in range(raw_edge_data.edge_count):
        raw_test_statistic = raw_edge_data.test_statistics[edge_index]
        raw_degrees_of_freedom = raw_edge_data.degrees_of_freedom[edge_index]
        if not np.isfinite(raw_test_statistic) or raw_degrees_of_freedom <= 0:
            continue
        if calibration_factor <= 1.0:
            continue

        adjusted_test_statistic = raw_test_statistic / calibration_factor
        deflated_p_values[edge_index] = float(chi2.sf(adjusted_test_statistic, df=raw_degrees_of_freedom))

    return deflated_p_values


def _apply_edge_multiple_testing_correction(
    tree: nx.DiGraph,
    raw_edge_data: _RawEdgeCalibrationData,
    deflated_p_values: np.ndarray,
    alpha: float,
    fdr_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-run multiple testing correction on calibrated edge p-values."""
    from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths

    from ..multiple_testing import apply_multiple_testing_correction

    p_values_for_correction = np.where(np.isfinite(deflated_p_values), deflated_p_values, 1.0)
    nonfinite_p_value_mask = ~np.isfinite(deflated_p_values)

    node_depths = compute_node_depths(tree)
    child_depths = np.array([node_depths.get(child_id, 0) for child_id in raw_edge_data.child_ids])

    reject_null_hypothesis, corrected_p_values = apply_multiple_testing_correction(
        p_values=p_values_for_correction,
        child_ids=raw_edge_data.child_ids,
        child_depths=child_depths,
        alpha=alpha,
        method=fdr_method,
        tree=tree,
    )
    reject_null_hypothesis = np.where(nonfinite_p_value_mask, False, reject_null_hypothesis)
    return reject_null_hypothesis, corrected_p_values


def _write_edge_calibration_results(
    annotations_df: pd.DataFrame,
    child_ids: list[str],
    calibrated_p_values: np.ndarray,
    corrected_p_values: np.ndarray,
    reject_null_hypothesis: np.ndarray,
) -> None:
    """Write calibrated p-values and significance calls back to DataFrame."""
    annotations_df.loc[child_ids, "Child_Parent_Divergence_P_Value"] = calibrated_p_values
    annotations_df.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = corrected_p_values
    annotations_df.loc[child_ids, "Child_Parent_Divergence_Significant"] = reject_null_hypothesis


def _attach_edge_calibration_metadata(
    annotations_df: pd.DataFrame,
    calibration_model: EdgeCalibrationModel,
    calibration_factor: float | None = None,
) -> None:
    """Attach calibration model and audit metadata to DataFrame attrs."""
    annotations_df.attrs["edge_calibration_model"] = calibration_model
    if calibration_factor is None:
        annotations_df.attrs["edge_calibration_audit"] = calibration_model.diagnostics
        return
    annotations_df.attrs["edge_calibration_audit"] = {
        "c_hat": calibration_factor,
        **calibration_model.diagnostics,
    }


# =============================================================================
# Public API
# =============================================================================


def calibrate_edges_from_sibling_neighborhood(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
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
    annotations_df
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
    raw_edge_data = _extract_raw_edge_calibration_data(annotations_df)
    if raw_edge_data is None:
        return annotations_df

    if raw_edge_data.edge_count == 0:
        return annotations_df

    parent_children_leaf_count_map = _build_parent_children_leaf_count_map(raw_edge_data)
    parent_sibling_difference_map = _extract_parent_sibling_difference_map(annotations_df)
    records = _build_edge_records_for_calibration(
        raw_edge_data,
        parent_children_leaf_count_map,
        parent_sibling_difference_map,
    )

    if not records:
        logger.warning("Edge calibration: no valid edges to calibrate.")
        return annotations_df

    model = _fit_edge_calibration_model(records)

    if model.method == "none":
        logger.info("Edge calibration: model is 'none' — no deflation applied.")
        _attach_edge_calibration_metadata(annotations_df, model)
        return annotations_df

    calibration_factor = predict_edge_inflation_factor(model)
    logger.info("Edge calibration: ĉ = %.3f (method=%s).", calibration_factor, model.method)

    deflated_p_values = _deflate_edge_p_values(raw_edge_data, calibration_factor)
    reject_null_hypothesis, corrected_p_values = _apply_edge_multiple_testing_correction(
        tree=tree,
        raw_edge_data=raw_edge_data,
        deflated_p_values=deflated_p_values,
        alpha=alpha,
        fdr_method=fdr_method,
    )
    _write_edge_calibration_results(
        annotations_df=annotations_df,
        child_ids=raw_edge_data.child_ids,
        calibrated_p_values=deflated_p_values,
        corrected_p_values=corrected_p_values,
        reject_null_hypothesis=reject_null_hypothesis,
    )
    _attach_edge_calibration_metadata(
        annotations_df=annotations_df,
        calibration_model=model,
        calibration_factor=calibration_factor,
    )

    logger.info(
        "Edge calibration applied: method=%s, ĉ=%.3f, %d edges, " "%d null-like.",
        model.method,
        calibration_factor,
        model.n_calibration,
        model.diagnostics.get("n_null_like", 0),
    )

    return annotations_df
