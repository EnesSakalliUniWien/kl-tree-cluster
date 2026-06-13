"""Deterministic failure attribution for traceable benchmark rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

FAILURE_LABELS = [
    "tree_metric_unrecoverable",
    "covariance_failure",
    "projected_wald_failure",
    "selected_MP_projection_failure",
    "calibration_support_undefined",
    "external_selected_tail_undefined",
    "sibling_FDR_failure",
    "traversal_failure",
    "solved_or_unattributed",
]


def _truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "failed", "failure", "invalid", "high"}


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_failure(row: Mapping[str, Any]) -> str:
    """Classify a trace row using the predeclared mathematical layer order."""
    if _truthy(row.get("tree_oracle_unrecoverable", False)):
        return "tree_metric_unrecoverable"
    if _truthy(row.get("covariance_null_invalid", False)):
        return "covariance_failure"
    if _truthy(row.get("fixed_projection_pvalues_not_uniform", False)):
        return "projected_wald_failure"
    if _truthy(row.get("selected_mp_calibration_failed", False)):
        return "selected_MP_projection_failure"

    support_status = _text(row.get("internal_support_status", ""))
    if support_status and support_status not in {"supported", "ok", "pass", "passed"}:
        return "calibration_support_undefined"
    supported_records = _number(row.get("n_supported_records", float("nan")))
    strict_records = _number(row.get("n_strict_null_records", float("nan")))
    n_eff_family = _number(row.get("n_eff_family", float("nan")))
    if (
        pd.notna(supported_records)
        and pd.notna(strict_records)
        and pd.notna(n_eff_family)
        and min(supported_records, strict_records, n_eff_family) <= 0
    ):
        return "calibration_support_undefined"

    if _truthy(row.get("selected_tail_context_not_admissible", False)):
        return "external_selected_tail_undefined"
    if _truthy(row.get("sibling_BH_false_split_rate_high", False)):
        return "sibling_FDR_failure"
    if _truthy(row.get("traversal_overrides_local_evidence_badly", False)):
        return "traversal_failure"

    existing = _text(row.get("failure_label", ""))
    if existing and existing not in {"nan", "none", "solved", "ok"}:
        return str(row.get("failure_label"))
    return "solved_or_unattributed"


def classify_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a row-level failure-attribution table with primary labels."""
    columns = [
        column
        for column in ("case_id", "replicate_id", "node_id", "parent_id")
        if column in table.columns
    ]
    result = table.loc[:, columns].copy()
    result["primary_failure_label"] = [
        classify_failure(row) for row in table.to_dict(orient="records")
    ]
    return result
