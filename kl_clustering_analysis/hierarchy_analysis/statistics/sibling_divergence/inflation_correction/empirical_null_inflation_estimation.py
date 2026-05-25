"""Context-weighted empirical-null inflation estimation."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .types.inflation_model import EmpiricalNullInflationModel

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
        n_strict_null_calibration=sum(record.is_null_like for record in supported_records),
        n_stopped_or_null_calibration=len(supported_records),
        baseline_empirical_inflation_factor=baseline_empirical_inflation_factor,
        effective_sample_size=effective_sample_size,
        context_center=context_center,
        context_bandwidth=context_bandwidth,
        sample_contexts=sample_contexts,
        sample_feature_families=calibration_feature_families,
        sample_weights=null_weights,
        sample_statistics=statistics,
        sample_reference_scales=reference_scales,
        sample_degrees_of_freedom=degrees_of_freedom,
    )


def predict_empirical_inflation_factor(
    model: EmpiricalNullInflationModel,
    record: SiblingPairRecord,
) -> float:
    """Predict the empirical post-selection inflation for one sibling record."""
    if record.degrees_of_freedom == 0:
        return 1.0
    if (
        not np.isfinite(record.sibling_projection_dimension)
        or record.sibling_projection_dimension <= 0
    ):
        raise ValueError(
            "Positive-degree sibling records require positive sibling_projection_dimension "
            f"for empirical-null inflation prediction; parent={record.parent!r}."
        )
    if model.n_calibration == 0:
        raise ValueError("Cannot predict sibling inflation from an empty calibration model.")
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
        raise ValueError(
            "Empirical-null inflation prediction has no calibration records for "
            f"feature_family={record.feature_family!r}."
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
    if not np.any(active_context_axes):
        return float(max(family_baseline_inflation_factor, 1.0))

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
        raise ValueError(
            "Empirical-null inflation prediction has zero effective calibration "
            f"weight; parent={record.parent!r}."
        )

    reference_expectations = family_reference_expectations
    local_inflation_factor = _inflation_mle(
        family_statistics,
        reference_expectations,
        local_weights,
    )
    return float(max(local_inflation_factor, 1.0))


__all__ = [
    "EmpiricalNullInflationModel",
    "fit_empirical_null_inflation_model",
    "predict_empirical_inflation_factor",
]
