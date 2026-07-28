"""Canonical support roles for guarded neighborhood bandwidth evidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping


class SupportRole(StrEnum):
    """Role of a selected row in neighborhood support accounting."""

    NULL_ANCHOR = "null_anchor"
    STOPPED_EDGE_NULL_ANCHOR = "stopped_edge_null_anchor"
    ALGORITHM_SELECTED_SIGNAL_LIKE_EXCLUDED = "algorithm_selected_signal_like_excluded"
    TRUTH_SIGNAL = "truth_signal"
    UNKNOWN = "unknown"


NULL_CALIBRATION_ROLES = frozenset(
    {
        SupportRole.NULL_ANCHOR,
        SupportRole.STOPPED_EDGE_NULL_ANCHOR,
    }
)


def role_allows_empirical_null_calibration(role: SupportRole | str) -> bool:
    """Return whether a row role may calibrate empirical-null behavior."""

    try:
        support_role = SupportRole(str(role))
    except ValueError:
        return False
    return support_role in NULL_CALIBRATION_ROLES


def classify_support_role(row: Mapping[str, object]) -> SupportRole:
    """Classify a benchmark/diagnostic row into the canonical support-role contract.

    The old wording "selected non-null support" mixed two ideas. Rows selected
    by the algorithm as signal-like may be useful locality evidence, but they
    are excluded from empirical-null calibration.
    """

    for explicit_column in (
        "support_role",
        "canonical_support_role",
        "neighborhood_support_role",
    ):
        explicit_value = row.get(explicit_column)
        if explicit_value is None:
            continue
        try:
            return SupportRole(str(explicit_value))
        except ValueError:
            pass

    if _truthy(row.get("is_null_like")) or str(row.get("data_role", "")).lower() == "null":
        return SupportRole.NULL_ANCHOR
    if _truthy(row.get("is_edge_blocked")) or _truthy(row.get("edge_blocked")):
        return SupportRole.STOPPED_EDGE_NULL_ANCHOR
    if (
        _truthy(row.get("is_truth_signal"))
        or str(row.get("topology_signal_role", "")).lower() == "signal"
    ):
        return SupportRole.TRUTH_SIGNAL
    if _truthy(row.get("is_selected_nonnull")) or _truthy(
        row.get("algorithm_selected_signal_like")
    ):
        return SupportRole.ALGORITHM_SELECTED_SIGNAL_LIKE_EXCLUDED
    if str(row.get("topology_support_role", "")).lower() == "selected_nonnull":
        return SupportRole.ALGORITHM_SELECTED_SIGNAL_LIKE_EXCLUDED
    return SupportRole.UNKNOWN


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


__all__ = [
    "NULL_CALIBRATION_ROLES",
    "SupportRole",
    "classify_support_role",
    "role_allows_empirical_null_calibration",
]
