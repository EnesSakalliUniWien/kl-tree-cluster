"""Support-threshold auditing for traceable benchmark rows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SupportThresholds:
    min_supported_records: int = 50
    min_strict_null_records: int = 20
    min_stopped_edge_records: int = 20
    min_family_supported_records: int = 20
    min_n_eff_global: float = 30.0
    min_n_eff_family: float = 20.0
    min_n_eff_local_kernel: float = 10.0
    max_weight_share: float = 0.2
    max_leave_one_sim_delta_log_c: float = 0.1


def _column_or_default(table: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in table.columns:
        return pd.to_numeric(table[column], errors="coerce")
    return pd.Series(default, index=table.index, dtype=float)


def audit_support_table(
    table: pd.DataFrame,
    thresholds: SupportThresholds = SupportThresholds(),
) -> pd.DataFrame:
    """Evaluate fail-closed support thresholds for each trace row."""
    result = table[[column for column in ("case_id", "replicate_id", "node_id") if column in table]]
    result = result.copy()
    checks = {
        "min_supported_records": (
            _column_or_default(table, "n_supported_records", 0.0)
            >= thresholds.min_supported_records
        ),
        "min_strict_null_records": (
            _column_or_default(table, "n_strict_null_records", 0.0)
            >= thresholds.min_strict_null_records
        ),
        "min_stopped_edge_records": (
            _column_or_default(table, "n_stopped_records", 0.0)
            >= thresholds.min_stopped_edge_records
        ),
        "min_family_supported_records": (
            _column_or_default(
                table, "n_family_supported_records", thresholds.min_family_supported_records
            )
            >= thresholds.min_family_supported_records
        ),
        "min_n_eff_global": (
            _column_or_default(table, "n_eff_global", thresholds.min_n_eff_global)
            >= thresholds.min_n_eff_global
        ),
        "min_n_eff_family": (
            _column_or_default(table, "n_eff_family", 0.0) >= thresholds.min_n_eff_family
        ),
        "min_n_eff_local_kernel": (
            _column_or_default(table, "n_eff_local_kernel", thresholds.min_n_eff_local_kernel)
            >= thresholds.min_n_eff_local_kernel
        ),
        "max_weight_share": (
            _column_or_default(table, "max_weight_share", 0.0) <= thresholds.max_weight_share
        ),
        "max_leave_one_sim_delta_log_c": (
            _column_or_default(table, "leave_one_sim_delta_log_c", 0.0).abs()
            <= thresholds.max_leave_one_sim_delta_log_c
        ),
    }
    for name, passed in checks.items():
        result[name] = passed
    check_names = list(checks)
    result["support_status"] = (
        result[check_names].all(axis=1).map({True: "supported", False: "unsupported"})
    )
    result["failure_reasons"] = [
        ";".join(name for name in check_names if not bool(row[name]))
        for _index, row in result.iterrows()
    ]
    return result
