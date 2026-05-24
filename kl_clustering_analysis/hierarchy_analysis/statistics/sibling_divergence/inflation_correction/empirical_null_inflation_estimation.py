"""Context-weighted empirical-null inflation estimation."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .types.inflation_model import EmpiricalNullInflationModel

# =============================================================================
# Context-weighted empirical-null inflation estimation
# =============================================================================

EFFECTIVE_SAMPLE_LOG_PENALTY_DIVISOR = 6.0
EFFECTIVE_SAMPLE_LOG_PENALTY_REFERENCE_ALPHA = 0.01


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


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    center = _weighted_mean(values, weights)
    return float(np.sqrt(max(_weighted_mean((values - center) ** 2, weights), 0.0)))


def _inflation_mle(
    statistics: np.ndarray,
    reference_expectations: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(np.sum(weights * statistics) / np.sum(weights * reference_expectations))


def _effective_sample_size(weights: np.ndarray) -> float:
    positive_weights = weights[weights > 0.0]
    if positive_weights.size == 0:
        raise ValueError("Effective sample size requires positive weights.")
    log_weights = np.log(positive_weights)
    return float(np.exp(2.0 * logsumexp(log_weights) - logsumexp(2.0 * log_weights)))


def _effective_sample_degeneracy_penalty(
    *,
    n_calibration: int,
    effective_sample_size: float,
    significance_level_alpha: float,
) -> float:
    if n_calibration <= 0:
        raise ValueError("Effective-sample penalty requires positive n_calibration.")
    if not np.isfinite(effective_sample_size) or effective_sample_size <= 0.0:
        raise ValueError(
            "Effective-sample penalty requires finite positive effective_sample_size."
        )
    if not np.isfinite(significance_level_alpha) or not (
        0.0 < significance_level_alpha <= 1.0
    ):
        raise ValueError("Effective-sample penalty requires alpha in (0, 1].")
    alpha_scale = min(
        1.0,
        EFFECTIVE_SAMPLE_LOG_PENALTY_REFERENCE_ALPHA / float(significance_level_alpha),
    )
    return float(
        1.0
        + (
            alpha_scale
            * np.log(float(n_calibration) / effective_sample_size)
            / EFFECTIVE_SAMPLE_LOG_PENALTY_DIVISOR
        )
    )


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

    statistics = np.array([record.stat for record in positive_df_records], dtype=float)
    reference_scales = np.array(
        [record.reference_scale for record in positive_df_records],
        dtype=float,
    )
    degrees_of_freedom = np.array(
        [record.degrees_of_freedom for record in positive_df_records],
        dtype=float,
    )
    null_weights = np.array(
        [record.sibling_null_weight for record in positive_df_records],
        dtype=float,
    )
    positive_weight_mask = null_weights > 0.0
    if not np.any(positive_weight_mask):
        raise ValueError(
            "Cannot fit sibling inflation model: no positive sibling-null calibration weight."
        )

    statistics = statistics[positive_weight_mask]
    reference_scales = reference_scales[positive_weight_mask]
    degrees_of_freedom = degrees_of_freedom[positive_weight_mask]
    null_weights = null_weights[positive_weight_mask]
    calibration_scales = np.array(
        [
            record.sibling_projection_dimension
            for record in positive_df_records
            if record.sibling_null_weight > 0.0
        ],
        dtype=float,
    )
    reference_expectations = reference_scales * degrees_of_freedom

    baseline_empirical_inflation_factor = _inflation_mle(
        statistics,
        reference_expectations,
        null_weights,
    )

    # One-sided post-selection correction: never increase the sibling statistic.
    baseline_empirical_inflation_factor = max(baseline_empirical_inflation_factor, 1.0)

    sample_contexts = np.log(calibration_scales)
    context_center = _weighted_mean(sample_contexts, null_weights)
    context_bandwidth = _weighted_std(sample_contexts, null_weights)
    if context_bandwidth <= 1e-12:
        context_bandwidth = 0.0

    effective_sample_size = _effective_sample_size(null_weights)

    return EmpiricalNullInflationModel(
        method="context_weighted_empirical_null_inflation",
        n_calibration=int(len(statistics)),
        baseline_empirical_inflation_factor=baseline_empirical_inflation_factor,
        effective_sample_size=effective_sample_size,
        context_center=context_center,
        context_bandwidth=context_bandwidth,
        sample_contexts=sample_contexts,
        sample_weights=null_weights,
        sample_statistics=statistics,
        sample_reference_scales=reference_scales,
        sample_degrees_of_freedom=degrees_of_freedom,
    )


def predict_empirical_inflation_factor(
    model: EmpiricalNullInflationModel,
    record: SiblingPairRecord,
    *,
    significance_level_alpha: float,
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

    effective_sample_degeneracy_penalty = _effective_sample_degeneracy_penalty(
        n_calibration=model.n_calibration,
        effective_sample_size=model.effective_sample_size,
        significance_level_alpha=significance_level_alpha,
    )
    if model.context_bandwidth == 0.0:
        return float(
            max(
                model.baseline_empirical_inflation_factor
                * effective_sample_degeneracy_penalty,
                1.0,
            )
        )

    log_target = float(np.log(record.sibling_projection_dimension))
    scaled_offsets = (model.sample_contexts - log_target) / model.context_bandwidth
    log_kernel_weights = -0.5 * scaled_offsets**2
    log_kernel_weights = log_kernel_weights - float(np.max(log_kernel_weights))
    local_weights = model.sample_weights * np.exp(log_kernel_weights)
    if float(np.sum(local_weights)) <= 0.0:
        raise ValueError(
            "Empirical-null inflation prediction has zero effective calibration "
            f"weight; parent={record.parent!r}."
        )

    reference_expectations = model.sample_reference_scales * model.sample_degrees_of_freedom
    local_inflation_factor = _inflation_mle(
        model.sample_statistics,
        reference_expectations,
        local_weights,
    )
    return float(max(local_inflation_factor * effective_sample_degeneracy_penalty, 1.0))


__all__ = [
    "EmpiricalNullInflationModel",
    "EFFECTIVE_SAMPLE_LOG_PENALTY_DIVISOR",
    "EFFECTIVE_SAMPLE_LOG_PENALTY_REFERENCE_ALPHA",
    "fit_empirical_null_inflation_model",
    "predict_empirical_inflation_factor",
]
