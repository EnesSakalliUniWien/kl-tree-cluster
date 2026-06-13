"""Context-weighted empirical-null inflation estimation."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2

from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .external_selected_tail_calibration import ExternalSelectedTailCalibrationModel
from .types.inflation_model import (
    DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
    CalibrationDecision,
    CalibrationSupportThresholds,
    EmpiricalNullInflationModel,
)

# =============================================================================
# Context-weighted empirical-null inflation estimation
# =============================================================================


def _validate_calibration_record(record: SiblingPairRecord) -> None:
    """Validate one candidate record for empirical-null inflation estimation."""
    if not np.isfinite(record.stat):
        raise ValueError(
            "Sibling inflation calibration requires finite statistics; "
            f"parent={record.parent!r}."
        )
    if record.degrees_of_freedom < 0:
        raise ValueError(
            "Sibling inflation calibration requires non-negative degrees of freedom; "
            f"parent={record.parent!r}."
        )
    if not np.isfinite(record.reference_scale) or record.reference_scale <= 0:
        raise ValueError(
            "Sibling inflation calibration requires finite positive reference_scale; "
            f"parent={record.parent!r}."
        )
    if record.stat < 0:
        raise ValueError(
            "Sibling inflation calibration requires non-negative statistics; "
            f"parent={record.parent!r}."
        )
    if not np.isfinite(record.sibling_null_weight) or not (
        0.0 <= record.sibling_null_weight <= 1.0
    ):
        raise ValueError(
            "Sibling inflation calibration requires finite sibling_null_weight in [0, 1]; "
            f"parent={record.parent!r}."
        )
    if record.degrees_of_freedom > 0 and (
        not np.isfinite(record.sibling_projection_dimension)
        or record.sibling_projection_dimension <= 0
    ):
        raise ValueError(
            "Positive-degree sibling calibration records require positive "
            f"sibling_projection_dimension; parent={record.parent!r}."
        )
    if record.degrees_of_freedom > 0 and record.n_parent <= 0:
        raise ValueError(
            "Positive-degree sibling calibration records require positive "
            f"parent sample size; parent={record.parent!r}."
        )
    if record.feature_family not in {"bernoulli", "categorical", "continuous", "mixed"}:
        raise ValueError(
            "Sibling inflation calibration requires feature_family to be "
            f"'bernoulli', 'categorical', 'continuous', or 'mixed'; "
            f"parent={record.parent!r}, feature_family={record.feature_family!r}."
        )


def _weighted_mean_columns(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[:, None], axis=0) / np.sum(weights)


def _weighted_std_columns(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    center = _weighted_mean_columns(values, weights)
    variances = np.sum(((values - center) ** 2) * weights[:, None], axis=0) / np.sum(
        weights
    )
    return np.sqrt(np.maximum(variances, 0.0))


def _inflation_mle(
    statistics: np.ndarray,
    reference_expectations: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(np.sum(weights * statistics) / np.sum(weights * reference_expectations))


def _has_internal_empirical_null_support(record: SiblingPairRecord) -> bool:
    """Return whether a record is admissible empirical-null calibration."""
    return bool(record.is_null_like or record.is_edge_blocked)


def _effective_sample_size(weights: np.ndarray) -> float:
    positive_weights = weights[weights > 0.0]
    if positive_weights.size == 0:
        raise ValueError("Effective sample size requires positive weights.")
    log_weights = np.log(positive_weights)
    return float(np.exp(2.0 * logsumexp(log_weights) - logsumexp(2.0 * log_weights)))


def _max_weight_share(weights: np.ndarray) -> float:
    positive_weights = weights[weights > 0.0]
    if positive_weights.size == 0:
        return 0.0
    return float(np.max(positive_weights) / np.sum(positive_weights))


def _leave_one_record_max_delta_log_c(
    statistics: np.ndarray,
    reference_expectations: np.ndarray,
    weights: np.ndarray,
    baseline_c_hat: float,
) -> float:
    if weights.size <= 1:
        return float("inf")
    deltas: list[float] = []
    baseline_log_c = float(np.log(max(baseline_c_hat, 1e-300)))
    for index in range(weights.size):
        keep = np.ones(weights.size, dtype=bool)
        keep[index] = False
        if float(np.sum(weights[keep])) <= 0.0:
            return float("inf")
        c_hat = max(
            _inflation_mle(
                statistics[keep],
                reference_expectations[keep],
                weights[keep],
            ),
            1.0,
        )
        deltas.append(abs(float(np.log(max(c_hat, 1e-300))) - baseline_log_c))
    return float(max(deltas)) if deltas else float("inf")


def fit_empirical_null_inflation_model(
    records: list[SiblingPairRecord],
) -> EmpiricalNullInflationModel:
    """Fit the context-weighted empirical-null inflation model."""
    if not records:
        raise ValueError(
            "Cannot fit sibling inflation model: no sibling calibration records."
        )
    for record in records:
        _validate_calibration_record(record)

    positive_df_records = [record for record in records if record.degrees_of_freedom > 0]
    if not positive_df_records:
        raise ValueError(
            "Cannot fit sibling inflation model: no positive-degree calibration records."
        )

    positive_weight_records = [
        record for record in positive_df_records if record.sibling_null_weight > 0.0
    ]
    if not positive_weight_records:
        raise ValueError(
            "Cannot fit sibling inflation model: no positive sibling-null calibration weight."
        )

    supported_records = [
        record
        for record in positive_weight_records
        if _has_internal_empirical_null_support(record)
    ]
    if not supported_records:
        selected_nonnull_count = sum(
            not _has_internal_empirical_null_support(record)
            for record in positive_weight_records
        )
        raise ValueError(
            "Cannot fit sibling inflation model: no strict-null or stopped-edge "
            "empirical-null calibration records with positive weight. "
            f"Found {selected_nonnull_count} selected non-null positive-weight "
            "record(s), which are not valid empirical-null calibration support."
        )

    statistics = np.array([record.stat for record in supported_records], dtype=float)
    reference_scales = np.array(
        [record.reference_scale for record in supported_records],
        dtype=float,
    )
    degrees_of_freedom = np.array(
        [record.degrees_of_freedom for record in supported_records],
        dtype=float,
    )
    null_weights = np.array(
        [record.sibling_null_weight for record in supported_records],
        dtype=float,
    )
    is_strict_null = np.array(
        [record.is_null_like for record in supported_records],
        dtype=bool,
    )
    is_edge_blocked = np.array(
        [record.is_edge_blocked for record in supported_records],
        dtype=bool,
    )
    calibration_scales = np.array(
        [record.sibling_projection_dimension for record in supported_records],
        dtype=float,
    )
    calibration_parent_sample_sizes = np.array(
        [record.n_parent for record in supported_records],
        dtype=float,
    )
    calibration_feature_families = tuple(
        record.feature_family for record in supported_records
    )
    calibration_parent_ids = tuple(record.parent for record in supported_records)
    reference_expectations = reference_scales * degrees_of_freedom

    baseline_empirical_inflation_factor = _inflation_mle(
        statistics,
        reference_expectations,
        null_weights,
    )

    # One-sided post-selection correction: never increase the sibling statistic.
    baseline_empirical_inflation_factor = max(baseline_empirical_inflation_factor, 1.0)

    sample_contexts = np.column_stack(
        [
            np.log(calibration_scales),
            np.log(calibration_parent_sample_sizes),
        ]
    )
    context_center = _weighted_mean_columns(sample_contexts, null_weights)
    context_bandwidth = _weighted_std_columns(sample_contexts, null_weights)
    context_bandwidth = np.where(context_bandwidth <= 1e-12, 0.0, context_bandwidth)

    effective_sample_size = _effective_sample_size(null_weights)

    return EmpiricalNullInflationModel(
        method="context_weighted_supported_empirical_null_inflation",
        n_calibration=int(len(statistics)),
        n_positive_weight_records=int(len(positive_weight_records)),
        n_selected_nonnull_positive_weight_records=sum(
            not _has_internal_empirical_null_support(record)
            for record in positive_weight_records
        ),
        n_strict_null_calibration=sum(record.is_null_like for record in supported_records),
        n_edge_blocked_calibration=sum(
            record.is_edge_blocked for record in supported_records
        ),
        n_stopped_or_null_calibration=len(supported_records),
        baseline_empirical_inflation_factor=baseline_empirical_inflation_factor,
        effective_sample_size=effective_sample_size,
        context_center=context_center,
        context_bandwidth=context_bandwidth,
        sample_contexts=sample_contexts,
        sample_feature_families=calibration_feature_families,
        sample_weights=null_weights,
        sample_parent_ids=calibration_parent_ids,
        sample_is_strict_null=is_strict_null,
        sample_is_edge_blocked=is_edge_blocked,
        sample_statistics=statistics,
        sample_reference_scales=reference_scales,
        sample_degrees_of_freedom=degrees_of_freedom,
    )


def _decision_support(
    *,
    model: EmpiricalNullInflationModel,
    record: SiblingPairRecord,
    family_mask: np.ndarray | None = None,
    local_weights: np.ndarray | None = None,
) -> dict[str, float | int | str | bool]:
    support: dict[str, float | int | str | bool] = {
        "n_positive_weight_records": int(model.n_positive_weight_records),
        "n_supported_records": int(model.n_calibration),
        "n_selected_nonnull_positive_weight_records": int(
            model.n_selected_nonnull_positive_weight_records
        ),
        "n_strict_null_records": int(model.n_strict_null_calibration),
        "n_edge_blocked_records": int(model.n_edge_blocked_calibration),
        "n_stopped_or_null_records": int(model.n_stopped_or_null_calibration),
        "model_effective_sample_size": float(model.effective_sample_size),
        "model_max_weight_share": _max_weight_share(model.sample_weights),
        "feature_family": str(record.feature_family),
    }
    if family_mask is not None:
        support["n_family_supported_records"] = int(np.sum(family_mask))
        if np.any(family_mask):
            family_weights = model.sample_weights[family_mask]
            support["n_family_strict_null_records"] = int(
                np.sum(model.sample_is_strict_null[family_mask])
            )
            support["n_family_edge_blocked_records"] = int(
                np.sum(model.sample_is_edge_blocked[family_mask])
            )
            support["family_effective_sample_size"] = _effective_sample_size(
                family_weights
            )
            support["family_max_weight_share"] = _max_weight_share(family_weights)
            support["leave_one_record_max_delta_log_c"] = (
                _leave_one_record_max_delta_log_c(
                    model.sample_statistics[family_mask],
                    (
                        model.sample_reference_scales[family_mask]
                        * model.sample_degrees_of_freedom[family_mask]
                    ),
                    family_weights,
                    max(
                        _inflation_mle(
                            model.sample_statistics[family_mask],
                            (
                                model.sample_reference_scales[family_mask]
                                * model.sample_degrees_of_freedom[family_mask]
                            ),
                            family_weights,
                        ),
                        1.0,
                    ),
                )
            )
    if local_weights is not None:
        support["local_weight_sum"] = float(np.sum(local_weights))
        support["local_effective_sample_size"] = (
            _effective_sample_size(local_weights)
            if float(np.sum(local_weights)) > 0.0
            else 0.0
        )
        support["local_max_weight_share"] = _max_weight_share(local_weights)
    return support


def _support_contract_failures(
    support: dict[str, float | int | str | bool],
    *,
    thresholds: CalibrationSupportThresholds,
) -> tuple[str, ...]:
    failures: list[str] = []
    if int(support.get("n_supported_records", 0)) < thresholds.min_supported_records:
        failures.append("supported_records_below_threshold")
    if int(support.get("n_family_supported_records", 0)) < (
        thresholds.min_family_supported_records
    ):
        failures.append("family_supported_records_below_threshold")
    if int(support.get("n_stopped_or_null_records", 0)) < (
        thresholds.min_stopped_or_null_records
    ):
        failures.append("stopped_or_null_records_below_threshold")
    if float(support.get("family_effective_sample_size", 0.0)) < (
        thresholds.min_family_effective_sample_size
    ):
        failures.append("family_effective_sample_size_below_threshold")
    if float(support.get("local_effective_sample_size", 0.0)) < (
        thresholds.min_local_effective_sample_size
    ):
        failures.append("local_effective_sample_size_below_threshold")
    if float(support.get("local_max_weight_share", 1.0)) > thresholds.max_weight_share:
        failures.append("local_max_weight_share_above_threshold")
    if float(support.get("leave_one_record_max_delta_log_c", float("inf"))) > (
        thresholds.max_leave_one_record_delta_log_c
    ):
        failures.append("leave_one_record_delta_log_c_above_threshold")
    return tuple(failures)


def _with_support_contract(
    support: dict[str, float | int | str | bool],
    *,
    thresholds: CalibrationSupportThresholds,
) -> tuple[dict[str, float | int | str | bool], tuple[str, ...]]:
    failures = _support_contract_failures(support, thresholds=thresholds)
    annotated_support = dict(support)
    annotated_support["support_contract_status"] = (
        "passes_internal_support_thresholds" if not failures else "below_internal_support_thresholds"
    )
    annotated_support["support_contract_failure_reasons"] = ";".join(failures)
    annotated_support["support_threshold_min_supported_records"] = int(
        thresholds.min_supported_records
    )
    annotated_support["support_threshold_min_family_supported_records"] = int(
        thresholds.min_family_supported_records
    )
    annotated_support["support_threshold_min_stopped_or_null_records"] = int(
        thresholds.min_stopped_or_null_records
    )
    annotated_support["support_threshold_min_family_effective_sample_size"] = float(
        thresholds.min_family_effective_sample_size
    )
    annotated_support["support_threshold_min_local_effective_sample_size"] = float(
        thresholds.min_local_effective_sample_size
    )
    annotated_support["support_threshold_max_weight_share"] = float(
        thresholds.max_weight_share
    )
    annotated_support["support_threshold_max_leave_one_record_delta_log_c"] = float(
        thresholds.max_leave_one_record_delta_log_c
    )
    return annotated_support, failures


def _decision_context(record: SiblingPairRecord) -> dict[str, object]:
    return {
        "feature_family": record.feature_family,
        "sibling_projection_dimension": float(record.sibling_projection_dimension),
        "n_parent": int(record.n_parent),
    }


def _adjusted_p_value(record: SiblingPairRecord, inflation_factor: float) -> float:
    if record.degrees_of_freedom == 0:
        return 1.0
    if not np.isfinite(record.reference_scale) or record.reference_scale <= 0.0:
        raise ValueError(
            "Sibling reference_scale must be finite and positive before adjustment; "
            f"parent={record.parent!r}."
        )
    adjusted_statistic = float(record.stat / (record.reference_scale * inflation_factor))
    return float(chi2.sf(adjusted_statistic, df=float(record.degrees_of_freedom)))


def _with_external_selected_tail_fallback(
    internal_decision: CalibrationDecision,
    record: SiblingPairRecord,
    *,
    external_selected_tail_model: ExternalSelectedTailCalibrationModel | None,
    external_selected_tail_context: dict[str, object] | None,
) -> CalibrationDecision:
    if external_selected_tail_model is None:
        return internal_decision
    return external_selected_tail_model.decision_for(
        record,
        external_context=external_selected_tail_context,
        internal_decision=internal_decision,
    )


def decide_empirical_null_calibration(
    model: EmpiricalNullInflationModel,
    record: SiblingPairRecord,
    *,
    enforce_support_thresholds: bool = False,
    support_thresholds: CalibrationSupportThresholds = DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
    external_selected_tail_model: ExternalSelectedTailCalibrationModel | None = None,
    external_selected_tail_context: dict[str, object] | None = None,
) -> CalibrationDecision:
    """Return the focal empirical-null calibration decision for one sibling record."""
    exact_context = _decision_context(record)
    if record.degrees_of_freedom == 0:
        return CalibrationDecision(
            status="internal_admissible",
            c_hat=1.0,
            p_value=1.0,
            estimator="zero_dimensional_sibling_record",
            support=_decision_support(model=model, record=record),
            exact_context=exact_context,
            descriptive_strata={"degrees_of_freedom": 0.0},
        )
    if (
        not np.isfinite(record.sibling_projection_dimension)
        or record.sibling_projection_dimension <= 0
    ):
        raise ValueError(
            "Positive-degree sibling records require positive sibling_projection_dimension "
            f"for empirical-null inflation prediction; parent={record.parent!r}."
        )
    if model.n_calibration == 0:
        decision = CalibrationDecision(
            status="undefined_no_internal_support",
            c_hat=None,
            p_value=None,
            estimator=model.method,
            support=_decision_support(model=model, record=record),
            exact_context=exact_context,
            descriptive_strata={"reason": "empty_calibration_model"},
        )
        return _with_external_selected_tail_fallback(
            decision,
            record,
            external_selected_tail_model=external_selected_tail_model,
            external_selected_tail_context=external_selected_tail_context,
        )
    if record.feature_family not in {"bernoulli", "categorical", "continuous", "mixed"}:
        raise ValueError(
            "Positive-degree sibling records require feature_family to be "
            f"'bernoulli', 'categorical', 'continuous', or 'mixed'; "
            f"parent={record.parent!r}, feature_family={record.feature_family!r}."
        )

    family_mask = np.array(
        [
            feature_family == record.feature_family
            for feature_family in model.sample_feature_families
        ],
        dtype=bool,
    )
    if not np.any(family_mask):
        decision = CalibrationDecision(
            status="undefined_no_family_support",
            c_hat=None,
            p_value=None,
            estimator=model.method,
            support=_decision_support(
                model=model,
                record=record,
                family_mask=family_mask,
            ),
            exact_context=exact_context,
            descriptive_strata={"reason": "no_matching_feature_family"},
        )
        return _with_external_selected_tail_fallback(
            decision,
            record,
            external_selected_tail_model=external_selected_tail_model,
            external_selected_tail_context=external_selected_tail_context,
        )
    family_reference_expectations = (
        model.sample_reference_scales[family_mask] * model.sample_degrees_of_freedom[family_mask]
    )
    family_baseline_inflation_factor = max(
        _inflation_mle(
            model.sample_statistics[family_mask],
            family_reference_expectations,
            model.sample_weights[family_mask],
        ),
        1.0,
    )

    active_context_axes = np.array([True, record.feature_family != "bernoulli"])
    active_context_axes = active_context_axes & (model.context_bandwidth > 0.0)
    descriptive_strata: dict[str, object] = {
        "active_projection_dimension_axis": bool(active_context_axes[0]),
        "active_parent_size_axis": bool(active_context_axes[1]),
        "context_bandwidth_projection_dimension": float(model.context_bandwidth[0])
        if model.context_bandwidth.size > 0
        else 0.0,
        "context_bandwidth_parent_size": float(model.context_bandwidth[1])
        if model.context_bandwidth.size > 1
        else 0.0,
    }
    dropped_axes = []
    if not bool(active_context_axes[0]):
        dropped_axes.append("sibling_projection_dimension")
    if not bool(active_context_axes[1]):
        dropped_axes.append("parent_sample_size")
    descriptive_strata["dropped_context_axes"] = ",".join(dropped_axes)
    if not np.any(active_context_axes):
        c_hat = float(max(family_baseline_inflation_factor, 1.0))
        local_weights = model.sample_weights[family_mask]
        support, failures = _with_support_contract(
            _decision_support(
                model=model,
                record=record,
                family_mask=family_mask,
                local_weights=local_weights,
            ),
            thresholds=support_thresholds,
        )
        if enforce_support_thresholds and failures:
            decision = CalibrationDecision(
                status="undefined_sparse_context",
                c_hat=None,
                p_value=None,
                estimator=f"{model.method}:family_baseline",
                support=support,
                exact_context=exact_context,
                descriptive_strata={
                    **descriptive_strata,
                    "reason": "internal_support_thresholds_failed",
                },
            )
            return _with_external_selected_tail_fallback(
                decision,
                record,
                external_selected_tail_model=external_selected_tail_model,
                external_selected_tail_context=external_selected_tail_context,
            )
        return CalibrationDecision(
            status="internal_admissible",
            c_hat=c_hat,
            p_value=_adjusted_p_value(record, c_hat),
            estimator=f"{model.method}:family_baseline",
            support=support,
            exact_context=exact_context,
            descriptive_strata=descriptive_strata,
        )

    if record.n_parent <= 0:
        raise ValueError(
            "Positive-degree sibling records require positive parent sample size "
            f"for empirical-null inflation prediction; parent={record.parent!r}."
        )
    target_context = np.array(
        [
            np.log(record.sibling_projection_dimension),
            np.log(float(record.n_parent)),
        ],
        dtype=float,
    )
    family_contexts = model.sample_contexts[family_mask]
    family_weights = model.sample_weights[family_mask]
    family_statistics = model.sample_statistics[family_mask]
    scaled_offsets = (
        family_contexts[:, active_context_axes] - target_context[active_context_axes]
    ) / model.context_bandwidth[active_context_axes]
    log_kernel_weights = -0.5 * np.sum(scaled_offsets**2, axis=1)
    log_kernel_weights = log_kernel_weights - float(np.max(log_kernel_weights))
    local_weights = family_weights * np.exp(log_kernel_weights)
    if float(np.sum(local_weights)) <= 0.0:
        decision = CalibrationDecision(
            status="undefined_sparse_context",
            c_hat=None,
            p_value=None,
            estimator=model.method,
            support=_decision_support(
                model=model,
                record=record,
                family_mask=family_mask,
                local_weights=local_weights,
            ),
            exact_context=exact_context,
            descriptive_strata={
                **descriptive_strata,
                "reason": "zero_local_calibration_weight",
            },
        )
        return _with_external_selected_tail_fallback(
            decision,
            record,
            external_selected_tail_model=external_selected_tail_model,
            external_selected_tail_context=external_selected_tail_context,
        )

    reference_expectations = family_reference_expectations
    local_inflation_factor = _inflation_mle(
        family_statistics,
        reference_expectations,
        local_weights,
    )
    c_hat = float(max(local_inflation_factor, 1.0))
    support, failures = _with_support_contract(
        _decision_support(
            model=model,
            record=record,
            family_mask=family_mask,
            local_weights=local_weights,
        ),
        thresholds=support_thresholds,
    )
    if enforce_support_thresholds and failures:
        decision = CalibrationDecision(
            status="undefined_sparse_context",
            c_hat=None,
            p_value=None,
            estimator=f"{model.method}:local_kernel",
            support=support,
            exact_context=exact_context,
            descriptive_strata={
                **descriptive_strata,
                "reason": "internal_support_thresholds_failed",
            },
        )
        return _with_external_selected_tail_fallback(
            decision,
            record,
            external_selected_tail_model=external_selected_tail_model,
            external_selected_tail_context=external_selected_tail_context,
        )
    return CalibrationDecision(
        status="internal_admissible",
        c_hat=c_hat,
        p_value=_adjusted_p_value(record, c_hat),
        estimator=f"{model.method}:local_kernel",
        support=support,
        exact_context=exact_context,
        descriptive_strata=descriptive_strata,
    )


def predict_empirical_inflation_factor(
    model: EmpiricalNullInflationModel,
    record: SiblingPairRecord,
    *,
    enforce_support_thresholds: bool = False,
    support_thresholds: CalibrationSupportThresholds = DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
) -> float:
    """Predict the empirical post-selection inflation for one sibling record."""
    decision = decide_empirical_null_calibration(
        model,
        record,
        enforce_support_thresholds=enforce_support_thresholds,
        support_thresholds=support_thresholds,
    )
    if decision.status != "internal_admissible" or decision.c_hat is None:
        reason = decision.descriptive_strata.get("reason", decision.status)
        raise ValueError(
            "Empirical-null inflation prediction is not internally admissible: "
            f"{decision.status}; parent={record.parent!r}; reason={reason!r}."
        )
    return decision.c_hat


__all__ = [
    "CalibrationDecision",
    "CalibrationSupportThresholds",
    "EmpiricalNullInflationModel",
    "DEFAULT_INTERNAL_SUPPORT_THRESHOLDS",
    "decide_empirical_null_calibration",
    "fit_empirical_null_inflation_model",
    "predict_empirical_inflation_factor",
]
