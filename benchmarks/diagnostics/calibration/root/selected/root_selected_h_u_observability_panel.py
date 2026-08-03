"""Observability audit for selected-root local population-law inference.

The selected-root spectral-tail law conditions on

    (T, A, E, B, H_u)

where H_u is the local null-whitened spectral population law. Existing root
tail artifacts carry the selected root spectral excess and a few compressed
MP diagnostics. This panel asks whether those captured fields are sufficient
to estimate a deformed Marchenko--Pastur reference edge for H_u.

It is diagnostic only. It does not compute a production p-value and it does
not promote nearest or action-dominating support to calibration support.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.reporting import print_diagnostic_output_paths
from benchmarks.diagnostics.calibration.root.root_tail_values import finite_float
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
)

SCHEMA_VERSION = "root_selected_h_u_observability_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_h_u_observability_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.selected.root_selected_h_u_observability_panel"
)

DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_importance_external_null_topology_join_mild_accumulated"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)
DEFAULT_POPULATION_LAW_REQUIREMENT_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_population_law_requirement_mild_accumulated"
    / "root_selected_population_law_requirement_rows.csv"
)

ROWS_OUTPUT = "root_selected_h_u_observability_rows.csv"
SUMMARY_OUTPUT = "root_selected_h_u_observability_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

SPECTRUM_COLUMNS = (
    "root_full_component_eigenvalues_json",
    "root_principal_component_eigenvalues_json",
    "root_pca_eigenvalues_json",
    "root_eigenvalues_json",
    "root_eigenvalues",
)
FEATURE_COUNT_COLUMNS = (
    "root_active_feature_count",
    "root_feature_count",
    "active_feature_count",
    "feature_count",
)
DEFORMED_EDGE_COLUMNS = (
    "root_deformed_mp_upper_edge",
    "root_h_u_deformed_mp_edge",
)

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "required_population_law_status",
    "required_mp_edge_multiplier",
    "has_root_top_eigenvalue_ratio",
    "has_root_eigenvalue_mass_fraction",
    "has_root_raw_mp_signal_count",
    "has_root_mp_threshold_rows",
    "has_root_active_feature_count",
    "has_root_eigenvalue_spectrum",
    "root_eigenvalue_spectrum_count",
    "has_deformed_mp_edge",
    "h_u_observability_status",
    "missing_h_u_fields",
    "production_inference_status",
    "next_mathematical_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "h_u_estimable_target_count",
    "missing_spectrum_target_count",
    "missing_feature_count_target_count",
    "support_missing_target_count",
    "exact_tail_support_target_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedHUObservabilityConfig:
    """Input/output contract for selected-root H_u observability diagnostics."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    population_law_requirement_rows_path: Path = DEFAULT_POPULATION_LAW_REQUIREMENT_ROWS
    minimum_bulk_eigenvalue_count: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joined-feasibility-rows-path",
        type=Path,
        default=DEFAULT_JOINED_FEASIBILITY_ROWS,
    )
    parser.add_argument(
        "--population-law-requirement-rows-path",
        type=Path,
        default=DEFAULT_POPULATION_LAW_REQUIREMENT_ROWS,
    )
    parser.add_argument("--minimum-bulk-eigenvalue-count", type=int, default=3)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedHUObservabilityConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def string_value(
    row: pd.Series | dict[str, object],
    column: str,
    default: str = "",
) -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def is_observed_target(row: pd.Series) -> bool:
    return (
        string_value(row, "proposal_family") == "observed_target"
        or string_value(row, "calibration_role") == "observed_target_not_null_support"
        or string_value(row, "data_role") == "observed_target"
    )


def _has_finite(row: pd.Series, column: str, *, positive: bool = False) -> bool:
    if column not in row:
        return False
    value = finite_float(row[column])
    if not math.isfinite(value):
        return False
    return value > 0.0 if positive else True


def _first_finite(row: pd.Series, columns: Iterable[str], *, positive: bool = False) -> bool:
    return any(_has_finite(row, column, positive=positive) for column in columns)


def _parse_spectrum_count(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    if isinstance(value, np.ndarray):
        return int(np.asarray(value).size)
    if isinstance(value, (list, tuple)):
        return int(len(value))
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return int(len(parsed))
    if parsed is not None:
        return 1
    separators = [",", ";", " "]
    parts = [text]
    for separator in separators:
        if separator in text:
            parts = [part for part in text.replace("[", "").replace("]", "").split(separator)]
            break
    count = 0
    for part in parts:
        if math.isfinite(finite_float(part.strip())):
            count += 1
    return count


def _spectrum_count(row: pd.Series) -> int:
    for column in SPECTRUM_COLUMNS:
        if column in row:
            count = _parse_spectrum_count(row[column])
            if count > 0:
                return count
    return 0


def _observability_status(
    *,
    required_status: str,
    production_status: str,
    has_feature_count: bool,
    has_spectrum: bool,
    has_mp_rows: bool,
    has_deformed_edge: bool,
) -> str:
    if production_status == "exact_support_available_defer_to_root_tail_panel":
        return "exact_tail_support_available_h_u_optional"
    if required_status == "exact_tail_support_available_no_population_law_requirement":
        return "exact_tail_support_available_h_u_optional"
    if required_status == "population_law_requirement_missing_support":
        return "support_missing_before_h_u_estimation"
    if has_deformed_edge:
        return "deformed_mp_edge_already_captured"
    if has_feature_count and has_spectrum and has_mp_rows:
        return "h_u_estimation_inputs_available_deformed_edge_missing"
    return "h_u_estimation_blocked_missing_capture"


def _missing_fields(
    *,
    has_feature_count: bool,
    has_spectrum: bool,
    has_mp_rows: bool,
    has_deformed_edge: bool,
) -> str:
    missing: list[str] = []
    if not has_spectrum:
        missing.append("root_eigenvalue_spectrum")
    if not has_feature_count:
        missing.append("root_active_feature_count")
    if not has_mp_rows:
        missing.append("root_mp_threshold_rows")
    if not has_deformed_edge:
        missing.append("root_deformed_mp_edge")
    return ";".join(missing)


def _next_step(status: str) -> str:
    if status == "exact_tail_support_available_h_u_optional":
        return "use_exact_root_tail_panel"
    if status == "support_missing_before_h_u_estimation":
        return "generate_support_or_external_law_before_tail_calibration"
    if status == "deformed_mp_edge_already_captured":
        return "rerun_selected_root_tail_with_captured_h_u_edge"
    if status == "h_u_estimation_inputs_available_deformed_edge_missing":
        return "compute_deformed_mp_edge_from_captured_root_bulk_spectrum"
    return "capture_root_eigenvalue_spectrum_and_active_feature_count"


def build_root_selected_h_u_observability_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    population_law_requirement_rows: pd.DataFrame,
    minimum_bulk_eigenvalue_count: int = 3,
) -> pd.DataFrame:
    """Build target-level rows describing whether H_u can be estimated."""
    _require_columns(
        joined_feasibility_rows,
        {
            "case_id",
            "data_role",
            "calibration_role",
            "proposal_family",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        "joined feasibility rows",
    )
    _require_columns(
        population_law_requirement_rows,
        {
            "target_case_id",
            "required_population_law_status",
            "required_mp_edge_multiplier",
            "production_inference_status",
        },
        "population law requirement rows",
    )
    requirement_by_case = {
        string_value(row, "target_case_id"): row
        for _, row in population_law_requirement_rows.iterrows()
    }
    targets = joined_feasibility_rows[
        joined_feasibility_rows.apply(is_observed_target, axis=1)
    ].copy()
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        case_id = string_value(target, "case_id")
        requirement = requirement_by_case.get(case_id)
        required_status = (
            string_value(requirement, "required_population_law_status")
            if requirement is not None
            else "population_law_requirement_missing"
        )
        multiplier = (
            finite_float(requirement["required_mp_edge_multiplier"])
            if requirement is not None
            else math.nan
        )
        production_status = (
            string_value(requirement, "production_inference_status")
            if requirement is not None
            else "fail_closed_h_u_requirement_missing"
        )
        spectrum_count = _spectrum_count(target)
        has_spectrum = int(spectrum_count) >= int(minimum_bulk_eigenvalue_count)
        has_feature_count = _first_finite(target, FEATURE_COUNT_COLUMNS, positive=True)
        has_mp_rows = _has_finite(target, "root_mp_threshold_rows", positive=True)
        has_deformed_edge = _first_finite(target, DEFORMED_EDGE_COLUMNS, positive=True)
        status = _observability_status(
            required_status=required_status,
            production_status=production_status,
            has_feature_count=has_feature_count,
            has_spectrum=has_spectrum,
            has_mp_rows=has_mp_rows,
            has_deformed_edge=has_deformed_edge,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": case_id,
                "required_population_law_status": required_status,
                "required_mp_edge_multiplier": multiplier,
                "has_root_top_eigenvalue_ratio": _has_finite(
                    target,
                    "root_selected_eigenvalue_over_mp_upper_bound",
                    positive=True,
                ),
                "has_root_eigenvalue_mass_fraction": _has_finite(
                    target,
                    "root_selected_eigenvalue_mass_fraction",
                ),
                "has_root_raw_mp_signal_count": _has_finite(
                    target,
                    "root_raw_mp_signal_count",
                ),
                "has_root_mp_threshold_rows": has_mp_rows,
                "has_root_active_feature_count": has_feature_count,
                "has_root_eigenvalue_spectrum": has_spectrum,
                "root_eigenvalue_spectrum_count": int(spectrum_count),
                "has_deformed_mp_edge": has_deformed_edge,
                "h_u_observability_status": status,
                "missing_h_u_fields": _missing_fields(
                    has_feature_count=has_feature_count,
                    has_spectrum=has_spectrum,
                    has_mp_rows=has_mp_rows,
                    has_deformed_edge=has_deformed_edge,
                ),
                "production_inference_status": production_status,
                "next_mathematical_step": _next_step(status),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_h_u_observability_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-root H_u observability rows."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    status = rows["h_u_observability_status"].astype(str)
    missing_fields = rows["missing_h_u_fields"].astype(str)
    estimable = int(
        status.isin(
            [
                "h_u_estimation_inputs_available_deformed_edge_missing",
                "deformed_mp_edge_already_captured",
            ]
        ).sum()
    )
    missing_spectrum = int(
        missing_fields.str.contains("root_eigenvalue_spectrum", regex=False).sum()
    )
    missing_feature_count = int(
        missing_fields.str.contains("root_active_feature_count", regex=False).sum()
    )
    if estimable == int(rows.shape[0]):
        summary_status = "h_u_estimable_for_all_required_roots"
    elif estimable > 0 and missing_spectrum == 0 and missing_feature_count == 0:
        summary_status = "deformed_mp_edge_missing_for_estimable_roots"
    else:
        summary_status = "h_u_capture_missing_for_some_required_roots"
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "h_u_estimable_target_count": estimable,
                "missing_spectrum_target_count": missing_spectrum,
                "missing_feature_count_target_count": missing_feature_count,
                "support_missing_target_count": int(
                    status.eq("support_missing_before_h_u_estimation").sum()
                ),
                "exact_tail_support_target_count": int(
                    status.eq("exact_tail_support_available_h_u_optional").sum()
                ),
                "summary_status": summary_status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_h_u_observability_panel(
    config: RootSelectedHUObservabilityConfig,
) -> dict[str, pd.DataFrame]:
    joined = pd.read_csv(config.joined_feasibility_rows_path)
    requirements = pd.read_csv(config.population_law_requirement_rows_path)
    rows = build_root_selected_h_u_observability_rows(
        joined_feasibility_rows=joined,
        population_law_requirement_rows=requirements,
        minimum_bulk_eigenvalue_count=config.minimum_bulk_eigenvalue_count,
    )
    summary = summarize_root_selected_h_u_observability_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_h_u_observability_panel(
    config: RootSelectedHUObservabilityConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_h_u_observability_panel(config)
    rows_path = config.output_dir / ROWS_OUTPUT
    summary_path = config.output_dir / SUMMARY_OUTPUT
    manifest_path = config.output_dir / MANIFEST_OUTPUT
    tables["rows"].to_csv(rows_path, index=False)
    tables["summary"].to_csv(summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "outputs": {
            "rows": str(rows_path),
            "summary": str(summary_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return {"rows": rows_path, "summary": summary_path, "manifest": manifest_path}


def main() -> None:
    args = parse_args()
    config = RootSelectedHUObservabilityConfig(
        output_dir=args.output_dir,
        joined_feasibility_rows_path=args.joined_feasibility_rows_path,
        population_law_requirement_rows_path=args.population_law_requirement_rows_path,
        minimum_bulk_eigenvalue_count=args.minimum_bulk_eigenvalue_count,
    )
    outputs = run_root_selected_h_u_observability_panel(config)
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
