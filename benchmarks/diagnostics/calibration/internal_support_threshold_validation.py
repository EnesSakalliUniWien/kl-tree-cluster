"""Internal calibration support-threshold validation panel.

This module scores candidate fail-closed support-threshold profiles on labeled
mixed null/signal calibration contexts. It is diagnostic-only: passing a panel
does not install a production threshold tuple.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.types.inflation_model import (
    DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
    CalibrationSupportThresholds,
)

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_internal_support_threshold_validation_not_calibration"
SCHEMA_VERSION = "internal_support_threshold_validation/v1"

REQUIRED_SUPPORT_COLUMNS = {
    "n_supported_records",
    "n_family_supported_records",
    "n_stopped_or_null_records",
    "family_effective_sample_size",
    "local_effective_sample_size",
    "local_max_weight_share",
    "leave_one_record_max_delta_log_c",
}

OUTCOME_COLUMNS = {
    "is_null_context",
    "is_signal_context",
    "split_rejected_at_alpha",
}


def default_threshold_profiles() -> dict[str, CalibrationSupportThresholds]:
    """Return predeclared profiles for mixed-panel threshold diagnostics."""
    return {
        "permissive": replace(
            DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
            min_supported_records=10,
            min_family_supported_records=5,
            min_stopped_or_null_records=5,
            min_family_effective_sample_size=5.0,
            min_local_effective_sample_size=4.0,
            max_weight_share=0.5,
            max_leave_one_record_delta_log_c=float(np.log(1.5)),
        ),
        "current_default": DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
        "strict": replace(
            DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
            min_supported_records=60,
            min_family_supported_records=30,
            min_stopped_or_null_records=20,
            min_family_effective_sample_size=20.0,
            min_local_effective_sample_size=15.0,
            max_weight_share=0.15,
            max_leave_one_record_delta_log_c=float(np.log(1.1)),
        ),
    }


def _numeric_column(
    table: pd.DataFrame,
    column: str,
    *,
    allow_inf: bool = False,
) -> pd.Series:
    values = pd.to_numeric(table[column], errors="raise").astype(float)
    invalid = values.isna() if allow_inf else ~np.isfinite(values)
    if bool(invalid.any()):
        bad_index = values[invalid].index[0]
        raise ValueError(
            f"{column} must contain {'numeric' if allow_inf else 'finite'} values; "
            f"row={int(bad_index)}, value={float(values.loc[bad_index])!r}."
        )
    return values


def _bool_column(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(False, index=table.index, dtype=bool)
    return table[column].astype(bool)


def _threshold_failures(
    row: pd.Series,
    *,
    thresholds: CalibrationSupportThresholds,
) -> tuple[str, ...]:
    failures: list[str] = []
    if int(row["n_supported_records"]) < thresholds.min_supported_records:
        failures.append("supported_records_below_threshold")
    if int(row["n_family_supported_records"]) < thresholds.min_family_supported_records:
        failures.append("family_supported_records_below_threshold")
    if int(row["n_stopped_or_null_records"]) < thresholds.min_stopped_or_null_records:
        failures.append("stopped_or_null_records_below_threshold")
    if (
        float(row["family_effective_sample_size"])
        < thresholds.min_family_effective_sample_size
    ):
        failures.append("family_effective_sample_size_below_threshold")
    if (
        float(row["local_effective_sample_size"])
        < thresholds.min_local_effective_sample_size
    ):
        failures.append("local_effective_sample_size_below_threshold")
    if float(row["local_max_weight_share"]) > thresholds.max_weight_share:
        failures.append("local_max_weight_share_above_threshold")
    if (
        float(row["leave_one_record_max_delta_log_c"])
        > thresholds.max_leave_one_record_delta_log_c
    ):
        failures.append("leave_one_record_delta_log_c_above_threshold")
    return tuple(failures)


def _context_id(table: pd.DataFrame) -> pd.Series:
    if "context_id" in table.columns:
        return table["context_id"].astype(str)
    return pd.Series([f"context_{index}" for index in table.index], index=table.index)


def evaluate_internal_support_threshold_rows(
    contexts: pd.DataFrame,
    *,
    threshold_profiles: Mapping[str, CalibrationSupportThresholds] | None = None,
) -> pd.DataFrame:
    """Return one threshold-decision row per context/profile pair."""
    missing = REQUIRED_SUPPORT_COLUMNS - set(contexts.columns)
    if missing:
        raise ValueError(
            f"Support-threshold panel is missing columns: {sorted(missing)!r}."
        )
    table = contexts.copy()
    for column in REQUIRED_SUPPORT_COLUMNS:
        table[column] = _numeric_column(
            table,
            column,
            allow_inf=column == "leave_one_record_max_delta_log_c",
        )

    profiles = dict(threshold_profiles or default_threshold_profiles())
    context_ids = _context_id(table)
    is_null_context = _bool_column(table, "is_null_context")
    is_signal_context = _bool_column(table, "is_signal_context")
    split_rejected = _bool_column(table, "split_rejected_at_alpha")
    has_outcome_labels = OUTCOME_COLUMNS <= set(table.columns)

    rows: list[dict[str, object]] = []
    for profile_id, thresholds in profiles.items():
        for index, row in table.iterrows():
            failures = _threshold_failures(row, thresholds=thresholds)
            threshold_admissible = not failures
            rows.append(
                {
                    "context_id": str(context_ids.loc[index]),
                    "threshold_profile_id": profile_id,
                    "threshold_admissible": threshold_admissible,
                    "threshold_status": (
                        "passes_internal_support_thresholds"
                        if threshold_admissible
                        else "below_internal_support_thresholds"
                    ),
                    "failure_reasons": ";".join(failures),
                    "has_outcome_labels": has_outcome_labels,
                    "is_null_context": bool(is_null_context.loc[index]),
                    "is_signal_context": bool(is_signal_context.loc[index]),
                    "split_rejected_at_alpha": bool(split_rejected.loc[index]),
                    "selected_nonnull_leakage_count": int(
                        row.get("n_selected_nonnull_positive_weight_records", 0)
                    ),
                    **{
                        f"threshold_{key}": value
                        for key, value in asdict(thresholds).items()
                    },
                    "study_role": STUDY_ROLE,
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_internal_support_threshold_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize threshold-panel decisions by profile."""
    summaries: list[dict[str, object]] = []
    for profile_id, group in rows.groupby("threshold_profile_id", sort=False):
        admissible = group["threshold_admissible"].astype(bool)
        has_outcome_labels = bool(group["has_outcome_labels"].all())
        null_mask = group["is_null_context"].astype(bool)
        signal_mask = group["is_signal_context"].astype(bool)
        split_rejected = group["split_rejected_at_alpha"].astype(bool)
        admissible_null = admissible & null_mask
        admissible_signal = admissible & signal_mask
        has_mixed_outcomes = (
            has_outcome_labels
            and bool(null_mask.any())
            and bool(signal_mask.any())
        )
        summaries.append(
            {
                "threshold_profile_id": profile_id,
                "n_contexts": int(group.shape[0]),
                "n_admissible_contexts": int(admissible.sum()),
                "admissible_fraction": float(admissible.mean()),
                "n_null_contexts": int(null_mask.sum()),
                "n_signal_contexts": int(signal_mask.sum()),
                "n_admissible_null_contexts": int(admissible_null.sum()),
                "n_admissible_signal_contexts": int(admissible_signal.sum()),
                "null_false_split_rate_among_admissible": (
                    float((split_rejected & admissible_null).sum() / admissible_null.sum())
                    if has_outcome_labels and bool(admissible_null.any())
                    else np.nan
                ),
                "signal_retention_rate_among_admissible": (
                    float(
                        (split_rejected & admissible_signal).sum()
                        / admissible_signal.sum()
                    )
                    if has_outcome_labels and bool(admissible_signal.any())
                    else np.nan
                ),
                "outcome_status": (
                    "has_mixed_null_signal_outcomes"
                    if has_mixed_outcomes
                    else (
                        "outcome_labels_not_mixed"
                        if has_outcome_labels
                        else "outcome_labels_unavailable"
                    )
                ),
                "max_selected_nonnull_leakage_count": int(
                    group["selected_nonnull_leakage_count"].max()
                ),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(summaries)


def run_internal_support_threshold_validation(
    *,
    panel_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run the internal support-threshold diagnostic from a CSV panel."""
    contexts = pd.read_csv(panel_path)
    rows = evaluate_internal_support_threshold_rows(contexts)
    summary = summarize_internal_support_threshold_rows(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "internal_support_threshold_decisions.csv"
    summary_path = output_dir / "internal_support_threshold_summary.csv"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "panel_path": str(panel_path),
        "outputs": {
            "decisions": str(rows_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic threshold-profile scoring. Threshold profiles remain "
            "non-production until mixed null/signal outcomes validate false-split "
            "control and planted-signal retention."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "decisions": rows_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_internal_support_threshold_validation(
        panel_path=args.panel,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STUDY_ROLE",
    "default_threshold_profiles",
    "evaluate_internal_support_threshold_rows",
    "run_internal_support_threshold_validation",
    "summarize_internal_support_threshold_rows",
]
