"""Sibling-null calibration panel diagnostic.

This module scores precomputed sibling-test rows across strict-null,
stopped-edge null, selected-nonnull-only, and external selected-tail context
roles. It is diagnostic-only and does not install a production sibling
calibration rule.
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

STUDY_ROLE = "diagnostic_sibling_null_calibration_panel_not_calibration"
SCHEMA_VERSION = "sibling_null_calibration_panel/v1"
P_VALUE_FLOOR = 1e-300
SIBLING_CONTEXT_ROLES = (
    "strict_null",
    "stopped_edge_null",
    "selected_nonnull_only",
    "external_selected_tail_context",
)
NULL_SIBLING_CONTEXT_ROLES = frozenset({"strict_null", "stopped_edge_null"})
SIGNAL_SIBLING_CONTEXT_ROLES = frozenset({"selected_nonnull_only"})
EXTERNAL_SIBLING_CONTEXT_ROLES = frozenset({"external_selected_tail_context"})
REQUIRED_COLUMNS = {
    "sibling_context_role",
    "sibling_p_value",
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
    if "sibling_context_id" in table.columns:
        return table["sibling_context_id"].astype(str)
    if "context_id" in table.columns:
        return table["context_id"].astype(str)
    return pd.Series(
        [f"sibling_context_{index}" for index in table.index],
        index=table.index,
    )


def evaluate_sibling_null_calibration_rows(
    panel: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Return normalized sibling-calibration diagnostic rows."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    missing = REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(
            f"Sibling-null calibration panel is missing columns: {sorted(missing)!r}."
        )

    table = panel.copy()
    role = table["sibling_context_role"].astype(str)
    unknown_roles = sorted(set(role) - set(SIBLING_CONTEXT_ROLES))
    if unknown_roles:
        raise ValueError(
            "sibling_context_role contains unknown roles: "
            f"{unknown_roles!r}; expected {list(SIBLING_CONTEXT_ROLES)!r}."
        )

    p_values = _numeric_p_value(table, "sibling_p_value")
    if "sibling_adjusted_p_value" in table.columns:
        adjusted = _numeric_p_value(table, "sibling_adjusted_p_value")
    elif "sibling_bh_p_value" in table.columns:
        adjusted = _numeric_p_value(table, "sibling_bh_p_value")
    else:
        adjusted = pd.Series(np.nan, index=table.index, dtype=float)

    rows = pd.DataFrame(
        {
            "sibling_context_id": _context_id(table),
            "sibling_context_role": role,
            "sibling_p_value": p_values,
            "sibling_adjusted_p_value": adjusted,
            "sibling_rejected_at_alpha": p_values <= float(alpha),
            "sibling_adjusted_rejected_at_alpha": adjusted <= float(alpha),
            "sibling_neg_log10_p_value": -np.log10(
                np.maximum(p_values, P_VALUE_FLOOR)
            ),
            "is_null_sibling_context": role.isin(NULL_SIBLING_CONTEXT_ROLES),
            "is_signal_sibling_context": role.isin(SIGNAL_SIBLING_CONTEXT_ROLES),
            "is_external_selected_tail_context": role.isin(
                EXTERNAL_SIBLING_CONTEXT_ROLES
            ),
            "internal_support_admissible": _optional_bool(
                table,
                "internal_support_admissible",
            ),
            "external_rule_admissible": _optional_bool(
                table,
                "external_rule_admissible",
            ),
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
            "sibling_test_statistic",
            "sibling_degrees_of_freedom",
            "sibling_projection_dimension",
            "parent_sample_size",
            "feature_family",
            "calibration_decision_status",
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
    n_rows: int,
    min_rows: int,
    external_admissible_rate: float,
) -> str:
    if n_rows < min_rows:
        return "insufficient_rows"
    if role in SIGNAL_SIBLING_CONTEXT_ROLES:
        return "selected_nonnull_retention_descriptive"
    if role in EXTERNAL_SIBLING_CONTEXT_ROLES:
        return (
            "external_selected_tail_candidate_descriptive"
            if external_admissible_rate > 0.0
            else "external_selected_tail_fail_closed"
        )
    delta = rejection_rate - alpha
    if abs(delta) <= tolerance:
        return "within_nominal_tolerance"
    if delta > 0:
        return "above_nominal_tolerance"
    return "below_nominal_tolerance"


def summarize_sibling_null_calibration_rows(
    rows: pd.DataFrame,
    *,
    alpha: float = 0.05,
    tolerance: float = 0.02,
    min_rows: int = 30,
) -> pd.DataFrame:
    """Summarize sibling-calibration rows by context role."""
    if rows.empty:
        return pd.DataFrame()
    if not 0.0 <= float(tolerance) < 1.0:
        raise ValueError(f"tolerance must lie in [0, 1); got {tolerance!r}.")
    if int(min_rows) <= 0:
        raise ValueError(f"min_rows must be positive; got {min_rows!r}.")

    summaries: list[dict[str, object]] = []
    for role in SIBLING_CONTEXT_ROLES:
        group = rows[rows["sibling_context_role"].eq(role)]
        rejected = group["sibling_rejected_at_alpha"].astype(bool)
        n_rows = int(group.shape[0])
        rejection_rate = float(rejected.mean()) if n_rows else math.nan
        standard_error = (
            float(math.sqrt(rejection_rate * (1.0 - rejection_rate) / n_rows))
            if n_rows and math.isfinite(rejection_rate)
            else math.nan
        )
        neglog = pd.to_numeric(
            group.get("sibling_neg_log10_p_value", pd.Series(dtype=float)),
            errors="coerce",
        )
        external_rate = (
            float(group["external_rule_admissible"].astype(bool).mean())
            if n_rows and "external_rule_admissible" in group.columns
            else 0.0
        )
        summaries.append(
            {
                "sibling_context_role": role,
                "n_rows": n_rows,
                "sibling_rejection_rate": rejection_rate,
                "sibling_rejection_standard_error": standard_error,
                "sibling_rejection_delta_from_alpha": (
                    float(rejection_rate - float(alpha))
                    if math.isfinite(rejection_rate)
                    else math.nan
                ),
                "sibling_neg_log10_p_q50": (
                    float(neglog.quantile(0.50)) if not neglog.empty else math.nan
                ),
                "sibling_neg_log10_p_q90": (
                    float(neglog.quantile(0.90)) if not neglog.empty else math.nan
                ),
                "internal_support_admissible_rate": (
                    float(group["internal_support_admissible"].astype(bool).mean())
                    if n_rows and "internal_support_admissible" in group.columns
                    else math.nan
                ),
                "external_rule_admissible_rate": external_rate,
                "sibling_calibration_status": _role_status(
                    role=role,
                    rejection_rate=rejection_rate,
                    alpha=float(alpha),
                    tolerance=float(tolerance),
                    n_rows=n_rows,
                    min_rows=int(min_rows),
                    external_admissible_rate=external_rate,
                ),
                "alpha": float(alpha),
                "tolerance": float(tolerance),
                "min_rows": int(min_rows),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(summaries)


def run_sibling_null_calibration_panel(
    *,
    panel_path: Path,
    output_dir: Path,
    alpha: float = 0.05,
    tolerance: float = 0.02,
    min_rows: int = 30,
) -> dict[str, Path]:
    """Run the sibling-null calibration diagnostic from a CSV panel."""
    panel = pd.read_csv(panel_path)
    rows = evaluate_sibling_null_calibration_rows(panel, alpha=alpha)
    summary = summarize_sibling_null_calibration_rows(
        rows,
        alpha=alpha,
        tolerance=tolerance,
        min_rows=min_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "sibling_null_calibration_rows.csv"
    summary_path = output_dir / "sibling_null_calibration_summary.csv"
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
            "Diagnostic sibling-null role panel. Strict-null, stopped-edge, "
            "selected-nonnull, and external selected-tail rows are not pooled; "
            "unsupported external rows remain fail-closed."
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
    outputs = run_sibling_null_calibration_panel(
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
    "SIBLING_CONTEXT_ROLES",
    "STUDY_ROLE",
    "evaluate_sibling_null_calibration_rows",
    "run_sibling_null_calibration_panel",
    "summarize_sibling_null_calibration_rows",
]
