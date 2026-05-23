"""Adjusted sibling-test summaries under estimated inflation correction."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from scipy.stats import chi2

from ..pair_testing.types.sibling_pair_record import SiblingPairRecord

AdjustedSiblingTestSummary = tuple[float, float, float]


def _compute_adjusted_sibling_test(
    sibling_test_record: SiblingPairRecord,
    *,
    resolve_inflation_adjustment: Callable[[SiblingPairRecord], tuple[float, str]],
) -> tuple[AdjustedSiblingTestSummary, str] | None:
    """Return one adjusted sibling-test summary, or ``None`` for null-like records."""
    if sibling_test_record.is_null_like:
        return None

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
        return (0.0, 0.0, 1.0), "zero_dimensional_spectral_context"
    if sibling_test_record.degrees_of_freedom < 0:
        raise ValueError(
            "Sibling record must have non-negative degrees of freedom before "
            f"adjustment; parent={sibling_test_record.parent!r}."
        )

    estimated_inflation_factor, adjustment_method_label = resolve_inflation_adjustment(
        sibling_test_record
    )
    adjusted_statistic = sibling_test_record.stat / estimated_inflation_factor
    adjusted_degrees_of_freedom = float(sibling_test_record.degrees_of_freedom)
    adjusted_p_value = float(chi2.sf(adjusted_statistic, df=adjusted_degrees_of_freedom))
    return (
        adjusted_statistic,
        adjusted_degrees_of_freedom,
        adjusted_p_value,
    ), adjustment_method_label


def compute_adjusted_sibling_tests(
    sibling_test_records: Iterable[SiblingPairRecord],
    *,
    resolve_inflation_adjustment: Callable[[SiblingPairRecord], tuple[float, str]],
) -> tuple[list[str], list[AdjustedSiblingTestSummary], list[str]]:
    """Return adjusted sibling-test summaries and adjustment labels for tested parents."""
    tested_parent_ids: list[str] = []
    adjusted_test_summaries: list[AdjustedSiblingTestSummary] = []
    adjustment_method_labels: list[str] = []

    for sibling_test_record in sibling_test_records:
        adjusted_test_summary = _compute_adjusted_sibling_test(
            sibling_test_record,
            resolve_inflation_adjustment=resolve_inflation_adjustment,
        )
        if adjusted_test_summary is None:
            continue

        test_summary, adjustment_method_label = adjusted_test_summary
        tested_parent_ids.append(sibling_test_record.parent)
        adjusted_test_summaries.append(test_summary)
        adjustment_method_labels.append(adjustment_method_label)

    return tested_parent_ids, adjusted_test_summaries, adjustment_method_labels


def count_null_focal_pairs(records: Iterable[SiblingPairRecord]) -> tuple[int, int, int]:
    """Count (null-like, focal, gate2-blocked) record totals."""
    n_null = 0
    n_focal = 0
    n_blocked = 0
    for record in records:
        if record.is_null_like:
            n_null += 1
        else:
            n_focal += 1
        if record.is_gate2_blocked:
            n_blocked += 1
    return n_null, n_focal, n_blocked


__all__ = [
    "AdjustedSiblingTestSummary",
    "compute_adjusted_sibling_tests",
    "count_null_focal_pairs",
]
