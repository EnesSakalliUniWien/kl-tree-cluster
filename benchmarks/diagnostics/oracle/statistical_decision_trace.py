"""Canonical statistical decision trace derived from gate-path diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATISTICAL_DECISION_TRACE_COLUMNS: tuple[str, ...] = (
    "case_id",
    "node_id",
    "node_depth",
    "edge_raw_p",
    "edge_bh_p",
    "edge_tested",
    "edge_rejected",
    "sibling_raw_p",
    "sibling_adjusted_p",
    "sibling_bh_p",
    "sibling_rejected",
    "inflation_factor",
    "calibration_support_status",
    "traversal_decision",
)


_REQUIRED_GATE_TRACE_COLUMNS = frozenset(
    {
        "case_id",
        "node_id",
        "depth",
        "left_edge_p_value",
        "right_edge_p_value",
        "left_edge_p_value_bh",
        "right_edge_p_value_bh",
        "left_edge_tested",
        "right_edge_tested",
        "left_edge_significant",
        "right_edge_significant",
        "raw_sibling_p_value",
        "sibling_adjusted_p_value",
        "sibling_corrected_p_value",
        "sibling_bh_different",
        "empirical_inflation_factor",
        "inflation_applied",
        "sibling_is_null_like",
        "sibling_is_edge_blocked",
        "actual_decision",
    }
)


def _finite_min(values: list[object]) -> float:
    numeric = np.asarray(values, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def _bool_any(values: list[object]) -> bool:
    return bool(any(bool(value) for value in values))


def _calibration_support_status(row: pd.Series) -> str:
    if bool(row["sibling_is_null_like"]):
        return "strict_empirical_null_supported"
    if bool(row["sibling_is_edge_blocked"]):
        return "stopped_or_strict_empirical_null_supported"
    if bool(row["inflation_applied"]):
        return "focal_test_adjusted"
    if np.isfinite(float(row["raw_sibling_p_value"])):
        return "focal_test_unadjusted"
    return "not_tested"


def statistical_decision_trace_from_gate_path_trace(
    gate_path_trace: pd.DataFrame,
) -> pd.DataFrame:
    """Project the wide gate-path trace to the stable statistical-test contract."""
    missing = sorted(_REQUIRED_GATE_TRACE_COLUMNS - set(gate_path_trace.columns))
    if missing:
        raise ValueError(
            "Gate-path trace is missing columns required for statistical decision tracing: "
            f"{missing!r}."
        )

    rows: list[dict[str, object]] = []
    for row in gate_path_trace.itertuples(index=False):
        series = pd.Series(row._asdict())
        rows.append(
            {
                "case_id": str(series["case_id"]),
                "node_id": series["node_id"],
                "node_depth": int(series["depth"]),
                "edge_raw_p": _finite_min(
                    [series["left_edge_p_value"], series["right_edge_p_value"]]
                ),
                "edge_bh_p": _finite_min(
                    [series["left_edge_p_value_bh"], series["right_edge_p_value_bh"]]
                ),
                "edge_tested": _bool_any(
                    [series["left_edge_tested"], series["right_edge_tested"]]
                ),
                "edge_rejected": _bool_any(
                    [series["left_edge_significant"], series["right_edge_significant"]]
                ),
                "sibling_raw_p": float(series["raw_sibling_p_value"]),
                "sibling_adjusted_p": float(series["sibling_adjusted_p_value"]),
                "sibling_bh_p": float(series["sibling_corrected_p_value"]),
                "sibling_rejected": bool(series["sibling_bh_different"]),
                "inflation_factor": float(series["empirical_inflation_factor"]),
                "calibration_support_status": _calibration_support_status(series),
                "traversal_decision": str(series["actual_decision"]),
            }
        )

    return pd.DataFrame.from_records(rows, columns=STATISTICAL_DECISION_TRACE_COLUMNS)


__all__ = [
    "STATISTICAL_DECISION_TRACE_COLUMNS",
    "statistical_decision_trace_from_gate_path_trace",
]
