"""Selected edge+sibling null equation diagnostic.

This module turns the barycentric relationship between edge and sibling tests
into an explicit conditional empirical-null equation:

    p = P(T_sib >= t | edge action bin, barycentric balance bin,
          sibling projection dimension, feature family, edge path open).

The implementation is diagnostic-only. It returns adjusted p-values only when
matched null support is present; unsupported contexts fail closed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.reporting import print_diagnostic_output_paths
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_selected_edge_sibling_null_equation_not_calibration"
SCHEMA_VERSION = "selected_edge_sibling_null_equation/v1"
EDGE_ACTION_BINS = (0.0, 2.0, 4.0, 6.0, 8.0, math.inf)
EDGE_ACTION_BIN_LABELS = (
    "edge_action_0_2",
    "edge_action_2_4",
    "edge_action_4_6",
    "edge_action_6_8",
    "edge_action_ge8",
)
BALANCE_BINS = (0.0, 0.1, 0.25, 0.4, 0.5000000001)
BALANCE_BIN_LABELS = (
    "balance_0_0.1",
    "balance_0.1_0.25",
    "balance_0.25_0.4",
    "balance_0.4_0.5",
)
CONTEXT_COLUMNS = (
    "feature_family",
    "sibling_projection_dimension",
    "edge_action_bin",
    "barycentric_balance_bin",
    "edge_path_open",
)
REQUIRED_COLUMNS = {
    "sibling_test_statistic",
    "left_edge_p_value",
    "right_edge_p_value",
    "left_child_sample_size",
    "right_child_sample_size",
    "sibling_projection_dimension",
    "feature_family",
    "edge_path_open",
    "is_null_context",
}


def _positive_numeric(values: pd.Series, *, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    invalid = ~np.isfinite(numeric) | (numeric <= 0.0)
    if bool(invalid.any()):
        bad_index = invalid[invalid].index[0]
        raise ValueError(
            f"{column_name} must contain finite positive values; "
            f"row={int(bad_index)}, value={float(numeric.loc[bad_index])!r}."
        )
    return numeric


def _probability(values: pd.Series, *, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    invalid = ~np.isfinite(numeric) | (numeric < 0.0) | (numeric > 1.0)
    if bool(invalid.any()):
        bad_index = invalid[invalid].index[0]
        raise ValueError(
            f"{column_name} must contain finite probabilities in [0, 1]; "
            f"row={int(bad_index)}, value={float(numeric.loc[bad_index])!r}."
        )
    return numeric


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    lowered = values.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes"})


def _bin_value(value: float, bins: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for lower, upper, label in zip(bins, bins[1:], labels):
        if lower <= value < upper:
            return label
    if value == bins[-1]:
        return labels[-1]
    raise ValueError(f"value did not match a predeclared bin: {value!r}.")


def _balance_bin(balance: float) -> str:
    if not np.isfinite(balance) or balance <= 0.0 or balance > 0.5:
        raise ValueError(f"barycentric balance must lie in (0, 0.5]; got {balance!r}.")
    for lower, upper, label in zip(BALANCE_BINS, BALANCE_BINS[1:], BALANCE_BIN_LABELS):
        if lower < balance <= upper or (lower == 0.0 and lower < balance < upper):
            return label
    raise ValueError(f"barycentric balance did not match a bin: {balance!r}.")


def _context_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(row[column] for column in CONTEXT_COLUMNS)


def prepare_selected_edge_sibling_equation_records(records: pd.DataFrame) -> pd.DataFrame:
    """Add barycentric and selected-edge context variables to sibling rows."""
    missing = REQUIRED_COLUMNS - set(records.columns)
    if missing:
        raise ValueError(
            f"Selected edge+sibling equation records are missing columns: {sorted(missing)!r}."
        )
    table = records.copy()
    stat = pd.to_numeric(table["sibling_test_statistic"], errors="raise").astype(float)
    invalid_stat = ~np.isfinite(stat) | (stat < 0.0)
    if bool(invalid_stat.any()):
        bad_index = invalid_stat[invalid_stat].index[0]
        raise ValueError(
            "sibling_test_statistic must contain finite non-negative values; "
            f"row={int(bad_index)}, value={float(stat.loc[bad_index])!r}."
        )
    table["sibling_test_statistic"] = stat
    left_size = _positive_numeric(
        table["left_child_sample_size"],
        column_name="left_child_sample_size",
    )
    right_size = _positive_numeric(
        table["right_child_sample_size"],
        column_name="right_child_sample_size",
    )
    parent_size = left_size + right_size
    beta_left = left_size / parent_size
    balance = np.minimum(beta_left, 1.0 - beta_left)
    table["left_barycentric_weight"] = beta_left
    table["barycentric_balance"] = balance
    table["log_barycentric_leverage"] = np.log(np.maximum(beta_left, 1.0 - beta_left) / balance)
    table["sampling_variance_scale"] = (1.0 / left_size) + (1.0 / right_size)
    table["barycentric_balance_bin"] = [_balance_bin(float(value)) for value in balance]

    left_p = _probability(table["left_edge_p_value"], column_name="left_edge_p_value")
    right_p = _probability(table["right_edge_p_value"], column_name="right_edge_p_value")
    edge_action = -np.log10(np.maximum(np.minimum(left_p, right_p), 1e-300))
    table["edge_action"] = edge_action
    table["edge_action_bin"] = [
        _bin_value(float(value), EDGE_ACTION_BINS, EDGE_ACTION_BIN_LABELS) for value in edge_action
    ]
    table["sibling_projection_dimension"] = pd.to_numeric(
        table["sibling_projection_dimension"],
        errors="raise",
    ).astype(int)
    table["feature_family"] = table["feature_family"].astype(str)
    table["edge_path_open"] = _bool_series(table["edge_path_open"])
    table["is_null_context"] = _bool_series(table["is_null_context"])
    if "is_signal_context" in table.columns:
        table["is_signal_context"] = _bool_series(table["is_signal_context"])
    else:
        table["is_signal_context"] = ~table["is_null_context"]
    if "record_id" not in table.columns:
        table["record_id"] = [f"record_{index}" for index in table.index]
    return table


def fit_selected_edge_sibling_null_contexts(
    records: pd.DataFrame,
    *,
    min_null_records: int = 30,
) -> pd.DataFrame:
    """Fit matched empirical-null context summaries."""
    if int(min_null_records) <= 0:
        raise ValueError(f"min_null_records must be positive; got {min_null_records!r}.")
    table = prepare_selected_edge_sibling_equation_records(records)
    nulls = table[table["is_null_context"].astype(bool)].copy()
    rows: list[dict[str, object]] = []
    for context_key, group in nulls.groupby(list(CONTEXT_COLUMNS), sort=True):
        stats = group["sibling_test_statistic"].to_numpy(dtype=float)
        n_null = int(stats.size)
        row = dict(zip(CONTEXT_COLUMNS, context_key))
        row.update(
            {
                "n_null_records": n_null,
                "min_null_records": int(min_null_records),
                "context_support_status": (
                    "supported_context"
                    if n_null >= int(min_null_records)
                    else "insufficient_null_support"
                ),
                "null_stat_q50": float(np.quantile(stats, 0.50)) if n_null else math.nan,
                "null_stat_q90": float(np.quantile(stats, 0.90)) if n_null else math.nan,
                "null_stat_q95": float(np.quantile(stats, 0.95)) if n_null else math.nan,
                "study_role": STUDY_ROLE,
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _context_lookup(contexts: pd.DataFrame) -> dict[tuple[object, ...], Mapping[str, object]]:
    lookup: dict[tuple[object, ...], Mapping[str, object]] = {}
    if contexts.empty:
        return lookup
    for _, row in contexts.iterrows():
        lookup[_context_key(row)] = row.to_dict()
    return lookup


def evaluate_selected_edge_sibling_null_equation(
    records: pd.DataFrame,
    *,
    min_null_records: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-record conditional empirical p-values and context summaries."""
    table = prepare_selected_edge_sibling_equation_records(records)
    contexts = fit_selected_edge_sibling_null_contexts(
        table,
        min_null_records=min_null_records,
    )
    lookup = _context_lookup(contexts)
    grouped_null_stats = {
        key: group["sibling_test_statistic"].to_numpy(dtype=float)
        for key, group in table[table["is_null_context"].astype(bool)].groupby(
            list(CONTEXT_COLUMNS),
            sort=True,
        )
    }

    rows: list[dict[str, object]] = []
    for _, row in table.iterrows():
        key = _context_key(row)
        context = lookup.get(key)
        null_stats = grouped_null_stats.get(key, np.array([], dtype=float))
        status = "undefined_no_matched_null_context"
        p_value = math.nan
        n_null = int(null_stats.size)
        if context is not None:
            status = str(context["context_support_status"])
            if status == "supported_context":
                # Add-one finite-sample tail probability keeps p > 0.
                p_value = float(
                    (np.sum(null_stats >= float(row["sibling_test_statistic"])) + 1.0)
                    / (n_null + 1.0)
                )
        rows.append(
            {
                "record_id": str(row["record_id"]),
                "selected_edge_sibling_status": (
                    "conditional_empirical_p_value" if status == "supported_context" else status
                ),
                "selected_edge_sibling_p_value": p_value,
                "n_matched_null_records": n_null,
                "min_null_records": int(min_null_records),
                "sibling_test_statistic": float(row["sibling_test_statistic"]),
                "is_null_context": bool(row["is_null_context"]),
                "is_signal_context": bool(row["is_signal_context"]),
                "left_barycentric_weight": float(row["left_barycentric_weight"]),
                "barycentric_balance": float(row["barycentric_balance"]),
                "log_barycentric_leverage": float(row["log_barycentric_leverage"]),
                "sampling_variance_scale": float(row["sampling_variance_scale"]),
                "edge_action": float(row["edge_action"]),
                **{column: row[column] for column in CONTEXT_COLUMNS},
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows), contexts


def summarize_selected_edge_sibling_null_equation(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize equation decisions by status and signal/null role."""
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []
    for status, group in rows.groupby("selected_edge_sibling_status", sort=True):
        p_values = pd.to_numeric(group["selected_edge_sibling_p_value"], errors="coerce")
        summaries.append(
            {
                "selected_edge_sibling_status": status,
                "n_records": int(group.shape[0]),
                "n_null_records": int(group["is_null_context"].astype(bool).sum()),
                "n_signal_records": int(group["is_signal_context"].astype(bool).sum()),
                "finite_p_value_fraction": float(p_values.notna().mean()),
                "p_value_q50": float(p_values.quantile(0.50))
                if p_values.notna().any()
                else math.nan,
                "p_value_q10": float(p_values.quantile(0.10))
                if p_values.notna().any()
                else math.nan,
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(summaries)


def run_selected_edge_sibling_null_equation(
    *,
    records_path: Path,
    output_dir: Path,
    min_null_records: int = 30,
) -> dict[str, Path]:
    """Run the selected edge+sibling null equation diagnostic from CSV records."""
    records = pd.read_csv(records_path)
    rows, contexts = evaluate_selected_edge_sibling_null_equation(
        records,
        min_null_records=min_null_records,
    )
    summary = summarize_selected_edge_sibling_null_equation(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "selected_edge_sibling_null_equation_rows.csv"
    contexts_path = output_dir / "selected_edge_sibling_null_equation_contexts.csv"
    summary_path = output_dir / "selected_edge_sibling_null_equation_summary.csv"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    contexts.to_csv(contexts_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "records_path": str(records_path),
        "min_null_records": int(min_null_records),
        "context_columns": list(CONTEXT_COLUMNS),
        "outputs": {
            "rows": str(rows_path),
            "contexts": str(contexts_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic conditional empirical selected edge+sibling null equation. "
            "Unsupported matched contexts fail closed and do not produce p-values."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "rows": rows_path,
        "contexts": contexts_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-null-records", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_selected_edge_sibling_null_equation(
        records_path=args.records,
        output_dir=args.output_dir,
        min_null_records=args.min_null_records,
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()


__all__ = [
    "CONTEXT_COLUMNS",
    "STUDY_ROLE",
    "evaluate_selected_edge_sibling_null_equation",
    "fit_selected_edge_sibling_null_contexts",
    "prepare_selected_edge_sibling_equation_records",
    "run_selected_edge_sibling_null_equation",
    "summarize_selected_edge_sibling_null_equation",
]
