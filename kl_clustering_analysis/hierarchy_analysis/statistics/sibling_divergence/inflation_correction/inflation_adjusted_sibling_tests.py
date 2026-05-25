"""Inflation-adjusted sibling-test summaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import chi2

from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .empirical_null_inflation_estimation import predict_empirical_inflation_factor
from .types.inflation_model import EmpiricalNullInflationModel

InflationAdjustedSiblingTestSummary = tuple[float, float, float]


def _compute_inflation_adjusted_sibling_test(
    sibling_test_record: SiblingPairRecord,
    *,
    model: EmpiricalNullInflationModel,
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
        return (0.0, 0.0, 1.0), model.method
    if sibling_test_record.degrees_of_freedom < 0:
        raise ValueError(
            "Sibling record must have non-negative degrees of freedom before "
            f"adjustment; parent={sibling_test_record.parent!r}."
        )

    empirical_inflation_factor = predict_empirical_inflation_factor(
        model,
        sibling_test_record,
    )
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
    inflation_adjusted_p_value = float(
        chi2.sf(
            inflation_adjusted_statistic,
            df=inflation_adjusted_degrees_of_freedom,
        )
    )
    return (
        inflation_adjusted_statistic,
        inflation_adjusted_degrees_of_freedom,
        inflation_adjusted_p_value,
    ), model.method


def compute_inflation_adjusted_sibling_tests(
    sibling_test_records: Iterable[SiblingPairRecord],
    *,
    model: EmpiricalNullInflationModel,
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
