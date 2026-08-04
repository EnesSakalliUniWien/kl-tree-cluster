"""Shared value parsing and root-tail coordinates for root diagnostics."""

from __future__ import annotations

import math

import pandas as pd

CALIBRATION_SUPPORT_ROLES = {
    "selected_null",
    "selected_null_candidate_support",
    "external_selected_null",
    "external_null_support",
    "calibration_null",
}


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    """Validate that a diagnostic input table has the required columns."""
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def finite_float(value: object) -> float:
    """Return a finite float or ``nan`` for invalid diagnostic values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def finite_int(value: object) -> int:
    """Return a finite integer or zero for invalid diagnostic counts."""
    numeric = finite_float(value)
    return int(numeric) if math.isfinite(numeric) else 0


def string_value(
    row: pd.Series | dict[str, object],
    column: str,
    default: str = "",
) -> str:
    """Return a non-missing string field from a diagnostic row."""
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def safe_log1p(value: object) -> float:
    """Return log1p of the nonnegative part of a numeric diagnostic value."""
    numeric = finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    return float(math.log1p(max(numeric, 0.0)))


def spectral_excess_log(value: object) -> float:
    """Return the nonnegative log spectral excess used by root diagnostics."""
    numeric = finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    return float(max(math.log(max(numeric, 1e-12)), 0.0))


def lookup_numeric_by_key(
    frame: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
) -> dict[str, float]:
    """Build a finite numeric lookup keyed by a diagnostic row identifier."""
    if frame.empty or key_column not in frame.columns or value_column not in frame.columns:
        return {}
    lookup: dict[str, float] = {}
    for _, row in frame.iterrows():
        key = string_value(row, key_column)
        if not key:
            continue
        value = finite_float(row.get(value_column, math.nan))
        if math.isfinite(value):
            lookup[key] = value
    return lookup


def is_observed_target(row: pd.Series) -> bool:
    """Return whether a root diagnostic row is the observed target."""
    return (
        string_value(row, "proposal_family") == "observed_target"
        or string_value(row, "calibration_role") == "observed_target_not_null_support"
        or string_value(row, "data_role") == "observed_target"
    )


def is_calibration_support(row: pd.Series) -> bool:
    """Return whether a root diagnostic row is admissible null support."""
    return (
        string_value(row, "data_role") in CALIBRATION_SUPPORT_ROLES
        or string_value(row, "calibration_role") in CALIBRATION_SUPPORT_ROLES
    )


def bandwidth_gap_status(*, target_band: str, generated_band: str) -> str:
    """Classify whether a generated row has measured root-bandwidth support."""
    if target_band == generated_band:
        return "bandwidth_band_match"
    if generated_band == "bandwidth_reopen_missing":
        return "generated_bandwidth_unmeasured"
    if target_band == "bandwidth_reopen_missing":
        return "target_bandwidth_unmeasured"
    return "bandwidth_band_mismatch"


def tie_band(value: float) -> str:
    """Return the coarse selected tie-rank band."""
    if not math.isfinite(value):
        return "tie_missing"
    if value < 0.70:
        return "tie_low_lt_0_70"
    if value < 0.85:
        return "tie_mid_0_70_0_85"
    return "tie_high_ge_0_85"


def action_band(value: float) -> str:
    """Return the coarse log-action band used by root-tail strata."""
    if not math.isfinite(value):
        return "action_missing"
    if value < 5.0:
        return "action_log_low_lt_5"
    if value < 7.0:
        return "action_log_mid_5_7"
    return "action_log_high_ge_7"


def root_tail_stratum_key(
    *,
    target: pd.Series,
    h_u_population_law_status: str,
) -> str:
    """Return the shared selected-root T,A,E,B,H_u stratum key."""
    tie = finite_float(target.get("root_tie_rank_median_fraction", math.nan))
    action = safe_log1p(target.get("root_sibling_selected_ratio", math.nan))
    edge = safe_log1p(target.get("root_edge_path_statistic_margin", math.nan))
    bandwidth = string_value(target, "root_bandwidth_reopen_band", "")
    return "|".join(
        [
            string_value(target, "root_mixed_region_component", "root_component_missing"),
            tie_band(tie),
            action_band(action),
            action_band(edge),
            bandwidth or "bandwidth_missing",
            str(h_u_population_law_status),
        ]
    )


def tail_excess_for_case(
    row: pd.Series,
    *,
    deformed_excess_by_case: dict[str, float],
) -> tuple[float, float, float, str]:
    """Return selected-root tail excess under identity or measured deformed H_u."""
    case_id = string_value(row, "case_id")
    identity_excess = spectral_excess_log(
        row.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
    )
    deformed_excess = deformed_excess_by_case.get(case_id, math.nan)
    if math.isfinite(deformed_excess):
        return (
            float(deformed_excess),
            identity_excess,
            float(deformed_excess),
            "deformed_mp_s_h_u",
        )
    return identity_excess, identity_excess, math.nan, "identity_mp_s_root"


__all__ = [
    "CALIBRATION_SUPPORT_ROLES",
    "action_band",
    "bandwidth_gap_status",
    "finite_float",
    "finite_int",
    "is_calibration_support",
    "is_observed_target",
    "lookup_numeric_by_key",
    "require_columns",
    "root_tail_stratum_key",
    "safe_log1p",
    "spectral_excess_log",
    "string_value",
    "tail_excess_for_case",
    "tie_band",
]
