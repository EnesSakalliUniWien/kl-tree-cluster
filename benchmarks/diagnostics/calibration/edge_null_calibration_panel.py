"""Edge-null calibration panel diagnostic.

This module scores precomputed child-parent edge-test rows across fixed-tree
null, selected-tree null, and selected-tree signal roles. It is diagnostic-only:
passing this panel does not install a production edge calibration rule.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_edge_null_calibration_panel_not_calibration"
SCHEMA_VERSION = "edge_null_calibration_panel/v1"
P_VALUE_FLOOR = 1e-300
EDGE_CONTEXT_ROLES = (
    "fixed_tree_null",
    "selected_tree_null",
    "selected_tree_signal",
)
NULL_EDGE_CONTEXT_ROLES = frozenset({"fixed_tree_null", "selected_tree_null"})
SIGNAL_EDGE_CONTEXT_ROLES = frozenset({"selected_tree_signal"})
REQUIRED_COLUMNS = {
    "edge_context_role",
    "edge_p_value",
}


def _numeric_p_value(table: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(table[column], errors="raise").astype(float)
    invalid = ~np.isfinite(values) | (values < 0.0) | (values > 1.0)
    if bool(invalid.any()):
        bad_index = invalid[invalid].index[0]
        raise ValueError(
            f"{column} must contain finite p-values in [0, 1]; "
            f"row={int(bad_index)}, value={float(values.loc[bad_index])!r}."
        )
    return values


def _optional_bool(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(False, index=table.index, dtype=bool)
    if table[column].dtype == bool:
        return table[column].fillna(False)
    lowered = table[column].astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes"})


def _context_id(table: pd.DataFrame) -> pd.Series:
    if "edge_context_id" in table.columns:
        return table["edge_context_id"].astype(str)
    if "context_id" in table.columns:
        return table["context_id"].astype(str)
    return pd.Series([f"edge_context_{index}" for index in table.index], index=table.index)


def _edge_id(table: pd.DataFrame) -> pd.Series:
    if "edge_id" in table.columns:
        return table["edge_id"].astype(str)
    if {"parent_id", "child_id"} <= set(table.columns):
        return table["parent_id"].astype(str) + "->" + table["child_id"].astype(str)
    if "node_id" in table.columns:
        return table["node_id"].astype(str)
    return pd.Series([f"edge_{index}" for index in table.index], index=table.index)


def evaluate_edge_null_calibration_rows(
    panel: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Return normalized edge-calibration decision rows."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    missing = REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(
            f"Edge-null calibration panel is missing columns: {sorted(missing)!r}."
        )

    table = panel.copy()
    role = table["edge_context_role"].astype(str)
    unknown_roles = sorted(set(role) - set(EDGE_CONTEXT_ROLES))
    if unknown_roles:
        raise ValueError(
            "edge_context_role contains unknown roles: "
            f"{unknown_roles!r}; expected {list(EDGE_CONTEXT_ROLES)!r}."
        )

    p_values = _numeric_p_value(table, "edge_p_value")
    if "edge_bh_p_value" in table.columns:
        bh_p_values = _numeric_p_value(table, "edge_bh_p_value")
    elif "edge_p_value_bh" in table.columns:
        bh_p_values = _numeric_p_value(table, "edge_p_value_bh")
    else:
        bh_p_values = pd.Series(np.nan, index=table.index, dtype=float)

    rows = pd.DataFrame(
        {
            "edge_context_id": _context_id(table),
            "edge_id": _edge_id(table),
            "edge_context_role": role,
            "edge_p_value": p_values,
            "edge_bh_p_value": bh_p_values,
            "edge_rejected_at_alpha": p_values <= float(alpha),
            "edge_bh_rejected_at_alpha": bh_p_values <= float(alpha),
            "edge_neg_log10_p_value": -np.log10(np.maximum(p_values, P_VALUE_FLOOR)),
            "is_null_edge_context": role.isin(NULL_EDGE_CONTEXT_ROLES),
            "is_signal_edge_context": role.isin(SIGNAL_EDGE_CONTEXT_ROLES),
            "is_selected_tree_context": role.str.startswith("selected_tree_"),
            "edge_selected": _optional_bool(table, "edge_selected"),
            "edge_path_open": _optional_bool(table, "edge_path_open"),
            "alpha": float(alpha),
            "study_role": STUDY_ROLE,
        }
    )
    passthrough_columns = [
        column
        for column in (
            "case_id",
            "replicate_id",
            "parent_id",
            "child_id",
            "node_id",
            "edge_test_statistic",
            "edge_degrees_of_freedom",
            "edge_projection_dimension",
            "parent_sample_size",
            "child_sample_size",
            "feature_family",
        )
        if column in table.columns
    ]
    for column in passthrough_columns:
        rows[column] = table[column].to_numpy()
    return rows


def _role_status(
    *,
    role: str,
    rejection_rate: float,
    alpha: float,
    tolerance: float,
    n_edges: int,
    min_rows: int,
) -> str:
    if n_edges < min_rows:
        return "insufficient_rows"
    if role in SIGNAL_EDGE_CONTEXT_ROLES:
        return "signal_retention_descriptive"
    delta = rejection_rate - alpha
    if abs(delta) <= tolerance:
        return "within_nominal_tolerance"
    if delta > 0:
        return "above_nominal_tolerance"
    return "below_nominal_tolerance"


def summarize_edge_null_calibration_rows(
    rows: pd.DataFrame,
    *,
    alpha: float = 0.05,
    tolerance: float = 0.02,
    min_rows: int = 30,
) -> pd.DataFrame:
    """Summarize edge-calibration rows by context role."""
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "edge_context_role",
                "n_edges",
                "edge_rejection_rate",
                "edge_rejection_standard_error",
                "edge_rejection_delta_from_alpha",
                "edge_calibration_status",
                "study_role",
            ]
        )
    if not 0.0 <= float(tolerance) < 1.0:
        raise ValueError(f"tolerance must lie in [0, 1); got {tolerance!r}.")
    if int(min_rows) <= 0:
        raise ValueError(f"min_rows must be positive; got {min_rows!r}.")

    summaries: list[dict[str, object]] = []
    for role in EDGE_CONTEXT_ROLES:
        group = rows[rows["edge_context_role"].eq(role)]
        rejected = group["edge_rejected_at_alpha"].astype(bool)
        n_edges = int(group.shape[0])
        rejection_rate = float(rejected.mean()) if n_edges else math.nan
        standard_error = (
            float(math.sqrt(rejection_rate * (1.0 - rejection_rate) / n_edges))
            if n_edges and math.isfinite(rejection_rate)
            else math.nan
        )
        neglog = pd.to_numeric(
            group.get("edge_neg_log10_p_value", pd.Series(dtype=float)),
            errors="coerce",
        )
        summaries.append(
            {
                "edge_context_role": role,
                "n_edges": n_edges,
                "edge_rejection_rate": rejection_rate,
                "edge_rejection_standard_error": standard_error,
                "edge_rejection_delta_from_alpha": (
                    float(rejection_rate - float(alpha))
                    if math.isfinite(rejection_rate)
                    else math.nan
                ),
                "edge_neg_log10_p_q50": (
                    float(neglog.quantile(0.50)) if not neglog.empty else math.nan
                ),
                "edge_neg_log10_p_q90": (
                    float(neglog.quantile(0.90)) if not neglog.empty else math.nan
                ),
                "edge_path_open_rate": (
                    float(group["edge_path_open"].astype(bool).mean())
                    if "edge_path_open" in group.columns and n_edges
                    else math.nan
                ),
                "edge_calibration_status": _role_status(
                    role=role,
                    rejection_rate=rejection_rate,
                    alpha=float(alpha),
                    tolerance=float(tolerance),
                    n_edges=n_edges,
                    min_rows=int(min_rows),
                ),
                "alpha": float(alpha),
                "tolerance": float(tolerance),
                "min_rows": int(min_rows),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(summaries)


def run_edge_null_calibration_panel(
    *,
    panel_path: Path,
    output_dir: Path,
    alpha: float = 0.05,
    tolerance: float = 0.02,
    min_rows: int = 30,
) -> dict[str, Path]:
    """Run the edge-null calibration diagnostic from a CSV panel."""
    panel = pd.read_csv(panel_path)
    rows = evaluate_edge_null_calibration_rows(panel, alpha=alpha)
    summary = summarize_edge_null_calibration_rows(
        rows,
        alpha=alpha,
        tolerance=tolerance,
        min_rows=min_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "edge_null_calibration_rows.csv"
    summary_path = output_dir / "edge_null_calibration_summary.csv"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "panel_path": str(panel_path),
        "alpha": float(alpha),
        "tolerance": float(tolerance),
        "min_rows": int(min_rows),
        "outputs": {
            "rows": str(rows_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic edge-null role panel. Fixed-tree and selected-tree null "
            "rows must be interpreted separately; signal rows report retention "
            "only and do not validate null calibration."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "rows": rows_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--min-rows", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_edge_null_calibration_panel(
        panel_path=args.panel,
        output_dir=args.output_dir,
        alpha=args.alpha,
        tolerance=args.tolerance,
        min_rows=args.min_rows,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "EDGE_CONTEXT_ROLES",
    "STUDY_ROLE",
    "evaluate_edge_null_calibration_rows",
    "run_edge_null_calibration_panel",
    "summarize_edge_null_calibration_rows",
]
