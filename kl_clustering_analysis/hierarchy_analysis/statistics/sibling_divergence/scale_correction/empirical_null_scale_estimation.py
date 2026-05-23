"""Context-weighted empirical-null scale estimation."""

from __future__ import annotations

import numpy as np

from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .types.calibration_model import EmpiricalNullScaleModel


# =============================================================================
# Context-weighted empirical-null scale estimation
# =============================================================================


def _validate_calibration_record(record: SiblingPairRecord) -> None:
    """Validate one candidate record for empirical-null scale estimation."""
    if not np.isfinite(record.stat):
        raise ValueError(
            "Sibling scale calibration requires finite statistics; "
            f"parent={record.parent!r}."
        )
    if record.degrees_of_freedom < 0:
        raise ValueError(
            "Sibling scale calibration requires non-negative degrees of freedom; "
            f"parent={record.parent!r}."
        )
    if record.stat < 0:
        raise ValueError(
            "Sibling scale calibration requires non-negative statistics; "
            f"parent={record.parent!r}."
        )
    if not np.isfinite(record.sibling_null_weight) or not (
        0.0 <= record.sibling_null_weight <= 1.0
    ):
        raise ValueError(
            "Sibling scale calibration requires finite sibling_null_weight in [0, 1]; "
            f"parent={record.parent!r}."
        )
    if record.degrees_of_freedom > 0 and (
        not np.isfinite(record.sibling_calibration_scale)
        or record.sibling_calibration_scale <= 0
    ):
        raise ValueError(
            "Positive-degree sibling calibration records require positive "
            f"sibling_calibration_scale; parent={record.parent!r}."
        )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    center = _weighted_mean(values, weights)
    return float(np.sqrt(max(_weighted_mean((values - center) ** 2, weights), 0.0)))


def _scale_mle(
    statistics: np.ndarray,
    degrees_of_freedom: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(np.sum(weights * statistics) / np.sum(weights * degrees_of_freedom))


def fit_empirical_null_scale_model(
    records: list[SiblingPairRecord],
) -> EmpiricalNullScaleModel:
    """Fit the context-weighted empirical-null scale model."""
    if not records:
        raise ValueError(
            "Cannot fit sibling scale model: no sibling calibration records."
        )
    for record in records:
        _validate_calibration_record(record)

    positive_df_records = [record for record in records if record.degrees_of_freedom > 0]
    if not positive_df_records:
        raise ValueError(
            "Cannot fit sibling scale model: no positive-degree calibration records."
        )

    statistics = np.array([record.stat for record in positive_df_records], dtype=float)
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
            "Cannot fit sibling scale model: no positive sibling-null calibration weight."
        )

    statistics = statistics[positive_weight_mask]
    degrees_of_freedom = degrees_of_freedom[positive_weight_mask]
    null_weights = null_weights[positive_weight_mask]
    calibration_scales = np.array(
        [
            record.sibling_calibration_scale
            for record in positive_df_records
            if record.sibling_null_weight > 0.0
        ],
        dtype=float,
    )
    stat_df_ratios = statistics / degrees_of_freedom
    max_observed_ratio = float(np.max(stat_df_ratios))

    baseline_scale_factor = _scale_mle(
        statistics,
        degrees_of_freedom,
        null_weights,
    )

    # One-sided post-selection correction: never increase the sibling statistic.
    baseline_scale_factor = max(baseline_scale_factor, 1.0)

    sample_contexts = np.log(calibration_scales)
    context_center = _weighted_mean(sample_contexts, null_weights)
    context_bandwidth = _weighted_std(sample_contexts, null_weights)
    if context_bandwidth <= 1e-12:
        context_bandwidth = 0.0

    diagnostics = {
        "fit_status": "context_weighted_empirical_null_scale",
        "n_candidate_records": len(records),
        "n_positive_degree_records": len(positive_df_records),
        "n_contributing": int(len(statistics)),
        "effective_sample_size": float(
            np.sum(null_weights) ** 2 / np.sum(null_weights**2)
        ),
        "max_observed_ratio": max_observed_ratio,
        "median_ratio": float(np.median(stat_df_ratios)),
        "mean_ratio": float(np.mean(stat_df_ratios)),
        "weighted_sum_statistic": float(np.sum(null_weights * statistics)),
        "weighted_sum_degrees_of_freedom": float(
            np.sum(null_weights * degrees_of_freedom)
        ),
        "mean_sibling_null_weight": float(np.mean(null_weights)),
        "min_sibling_null_weight": float(np.min(null_weights)),
        "max_sibling_null_weight": float(np.max(null_weights)),
        "context_center": context_center,
        "context_bandwidth": context_bandwidth,
    }

    return EmpiricalNullScaleModel(
        method="context_weighted_empirical_null_scale",
        n_calibration=int(len(statistics)),
        baseline_scale_factor=baseline_scale_factor,
        max_observed_ratio=max_observed_ratio,
        context_center=context_center,
        context_bandwidth=context_bandwidth,
        sample_contexts=sample_contexts,
        sample_weights=null_weights,
        sample_statistics=statistics,
        sample_degrees_of_freedom=degrees_of_freedom,
        diagnostics=diagnostics,
    )


def predict_scale_factor(
    model: EmpiricalNullScaleModel,
    record: SiblingPairRecord,
) -> float:
    """Predict the post-selection scale factor for one sibling record."""
    if record.degrees_of_freedom == 0:
        return 1.0
    if (
        not np.isfinite(record.sibling_calibration_scale)
        or record.sibling_calibration_scale <= 0
    ):
        raise ValueError(
            "Positive-degree sibling records require positive sibling_calibration_scale "
            f"for empirical-null scale prediction; parent={record.parent!r}."
        )
    if model.n_calibration == 0:
        raise ValueError("Cannot predict sibling scale from an empty calibration model.")

    if model.context_bandwidth == 0.0:
        return float(max(model.baseline_scale_factor, 1.0))

    log_target = float(np.log(record.sibling_calibration_scale))
    scaled_offsets = (model.sample_contexts - log_target) / model.context_bandwidth
    log_kernel_weights = -0.5 * scaled_offsets**2
    log_kernel_weights = log_kernel_weights - float(np.max(log_kernel_weights))
    local_weights = model.sample_weights * np.exp(log_kernel_weights)
    if float(np.sum(local_weights)) <= 0.0:
        raise ValueError(
            "Empirical-null scale prediction has zero effective calibration "
            f"weight; parent={record.parent!r}."
        )

    local_scale_factor = _scale_mle(
        model.sample_statistics,
        model.sample_degrees_of_freedom,
        local_weights,
    )
    return float(max(local_scale_factor, 1.0))


__all__ = [
    "EmpiricalNullScaleModel",
    "fit_empirical_null_scale_model",
    "predict_scale_factor",
]
