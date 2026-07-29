"""Shared threshold scanning for binary diagnostic roles."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

THRESHOLD_SCAN_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "direction",
    "threshold",
    "truth_selected_count",
    "negative_selected_count",
    "truth_total",
    "negative_total",
    "truth_retention",
    "negative_selection_rate",
)


def scan_binary_role_thresholds(
    rows: pd.DataFrame,
    *,
    metric: str,
    schema_version: str,
    study_role: str,
    role_column: str = "guard_truth_role",
    truth_role: str = "truth_recovery",
) -> pd.DataFrame:
    """Scan both threshold directions for one numeric diagnostic metric."""
    truth = rows[role_column].astype(str).eq(truth_role)
    negative = ~truth
    values = pd.to_numeric(rows[metric], errors="coerce")
    finite = values[np.isfinite(values)]
    thresholds = np.sort(finite.unique())
    records: list[dict[str, object]] = []
    truth_total = int(truth.sum())
    negative_total = int(negative.sum())

    for direction in ("greater_equal", "less_equal"):
        for threshold in thresholds:
            selected = values.ge(float(threshold))
            if direction == "less_equal":
                selected = values.le(float(threshold))
            truth_selected = int((selected & truth).sum())
            negative_selected = int((selected & negative).sum())
            records.append(
                {
                    "schema_version": schema_version,
                    "study_role": study_role,
                    "metric": metric,
                    "direction": direction,
                    "threshold": float(threshold),
                    "truth_selected_count": truth_selected,
                    "negative_selected_count": negative_selected,
                    "truth_total": truth_total,
                    "negative_total": negative_total,
                    "truth_retention": (
                        float(truth_selected / truth_total) if truth_total else math.nan
                    ),
                    "negative_selection_rate": (
                        float(negative_selected / negative_total) if negative_total else math.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=THRESHOLD_SCAN_COLUMNS)


__all__ = ["THRESHOLD_SCAN_COLUMNS", "scan_binary_role_thresholds"]
