"""Inflation-adjusted sibling-test summaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .empirical_null_inflation_estimation import (
    DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
    decide_empirical_null_calibration,
)
from .types.inflation_model import (
    CalibrationSupportThresholds,
    EmpiricalNullInflationModel,
)

InflationAdjustedSiblingTestSummary = tuple[float, float, float]


def _compute_inflation_adjusted_sibling_test(
    sibling_test_record: SiblingPairRecord,
    *,
    model: EmpiricalNullInflationModel | None,
    enforce_support_thresholds: bool = False,
    support_thresholds: CalibrationSupportThresholds = DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
) -> tuple[InflationAdjustedSiblingTestSummary, str]:
    """Return one inflation-adjusted sibling-test summary."""
    if not np.isfinite(sibling_test_record.stat):
        raise ValueError(
            "Sibling record must have a finite statistic before "
            f"adjustment; parent={sibling_test_record.parent!r}."
        )
    if sibling_test_record.degrees_of_freedom == 0:
        if sibling_test_record.stat != 0.0 or sibling_test_record.p_value != 1.0:
            raise ValueError(
                "Zero-dimensional sibling records must carry statistic=0 and p_value=1; "
                f"parent={sibling_test_record.parent!r}."
            )
        method = "zero_dimensional_sibling_record" if model is None else model.method
        return (0.0, 0.0, 1.0), method
    if sibling_test_record.degrees_of_freedom < 0:
        raise ValueError(
            "Sibling record must have non-negative degrees of freedom before "
            f"adjustment; parent={sibling_test_record.parent!r}."
        )

    if model is None:
        raise ValueError(
            "Sibling empirical calibration has no internal model; "
            f"parent={sibling_test_record.parent!r}."
        )
    decision = decide_empirical_null_calibration(
        model,
        sibling_test_record,
        enforce_support_thresholds=enforce_support_thresholds,
        support_thresholds=support_thresholds,
    )
    admissible_statuses = {
        "internal_admissible",
    }
    if decision.status not in admissible_statuses or decision.c_hat is None:
        raise ValueError(
            "Sibling empirical calibration is not internally admissible; "
            f"status={decision.status!r}, parent={sibling_test_record.parent!r}."
        )
    empirical_inflation_factor = decision.c_hat
    if not np.isfinite(empirical_inflation_factor) or empirical_inflation_factor < 1.0:
        raise ValueError(
            "Sibling empirical inflation factor must be finite and >= 1.0; "
            f"got {empirical_inflation_factor!r}."
        )

    if (
        not np.isfinite(sibling_test_record.reference_scale)
        or sibling_test_record.reference_scale <= 0.0
    ):
        raise ValueError(
            "Sibling reference_scale must be finite and positive before adjustment; "
            f"parent={sibling_test_record.parent!r}."
        )

    inflation_adjusted_statistic = sibling_test_record.stat / (
        sibling_test_record.reference_scale * empirical_inflation_factor
    )
    inflation_adjusted_degrees_of_freedom = float(sibling_test_record.degrees_of_freedom)
    if decision.p_value is None:
        raise ValueError(
            "Sibling empirical calibration decision did not include an adjusted "
            f"p-value; parent={sibling_test_record.parent!r}."
        )
    inflation_adjusted_p_value = decision.p_value
    return (
        inflation_adjusted_statistic,
        inflation_adjusted_degrees_of_freedom,
        inflation_adjusted_p_value,
    ), decision.estimator


def compute_inflation_adjusted_sibling_tests(
    sibling_test_records: Iterable[SiblingPairRecord],
    *,
    model: EmpiricalNullInflationModel | None,
    enforce_support_thresholds: bool = False,
    support_thresholds: CalibrationSupportThresholds = DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
) -> tuple[list[str], list[InflationAdjustedSiblingTestSummary], list[str]]:
    """Return inflation-adjusted sibling-test summaries for tested parents."""
    tested_parent_ids: list[str] = []
    inflation_adjusted_test_summaries: list[InflationAdjustedSiblingTestSummary] = []
    inflation_adjustment_method_labels: list[str] = []

    for sibling_test_record in sibling_test_records:
        if sibling_test_record.is_null_like:
            continue

        test_summary, method_label = _compute_inflation_adjusted_sibling_test(
            sibling_test_record,
            model=model,
            enforce_support_thresholds=enforce_support_thresholds,
            support_thresholds=support_thresholds,
        )
        tested_parent_ids.append(sibling_test_record.parent)
        inflation_adjusted_test_summaries.append(test_summary)
        inflation_adjustment_method_labels.append(method_label)

    return (
        tested_parent_ids,
        inflation_adjusted_test_summaries,
        inflation_adjustment_method_labels,
    )


__all__ = [
    "InflationAdjustedSiblingTestSummary",
    "compute_inflation_adjusted_sibling_tests",
]
