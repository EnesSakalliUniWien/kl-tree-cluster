"""Equation targets for the selected-root external spectral-tail law.

The conditional-tilt feasibility panel proves that current external-null rows
cannot be reweighted into the hard selected-root targets. This panel turns that
negative result into explicit equations for the next generator:

1. condition on the selected root event and the requested T,A,E,B,H_u stratum;
2. match target moments for phi_root=(T,A,E,S_Hu), or the required active axes;
3. put actual support in the tail event S_Hu >= S_Hu,target;
4. keep production fail-closed until conservative empirical support exists.

Rows are diagnostic-only. They specify what a selected-null or external law must
produce; they do not create p-values.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.root.root_tail_values import finite_float, string_value
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
)

SCHEMA_VERSION = "root_selected_external_law_equation_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_external_law_equation_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.selected.root_selected_external_law_equation_panel"
)

DEFAULT_CONDITIONAL_TILT_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_conditional_tilt_feasibility_mild_replay_v3_smoke"
    / "root_selected_conditional_tilt_feasibility_rows.csv"
)
DEFAULT_EXTERNAL_LAW_TARGET_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_deformed_external_law_target_mild_replay_v3_smoke"
    / "root_selected_deformed_external_law_target_rows.csv"
)
DEFAULT_SUPPORT_GAP_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_deformed_tail_support_gap_mild_replay_v3_smoke"
    / "root_selected_deformed_tail_support_gap_rows.csv"
)
DEFAULT_DEFORMED_EDGE_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_deformed_mp_edge_mild_hu_replay_v3_smoke"
    / "root_selected_deformed_mp_edge_rows.csv"
)

ROWS_OUTPUT = "root_selected_external_law_equation_rows.csv"
SUMMARY_OUTPUT = "root_selected_external_law_equation_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_status",
    "conditioning_event",
    "target_root_tail_stratum_key",
    "required_tilt_axes",
    "law_moment_equation",
    "tail_support_equation",
    "target_moment_vector_json",
    "required_axes_residual_vector_json",
    "full_phi_residual_vector_json",
    "target_s_h_u_excess_log",
    "support_hull_max_s_h_u_excess_log",
    "required_s_h_u_gap_to_support_hull",
    "required_deformed_ratio_lower_bound",
    "target_selected_root_deformed_ratio",
    "target_deformed_mp_upper_edge",
    "target_selected_root_eigenvalue",
    "minimum_support_count_for_alpha_resolution",
    "minimum_tail_exceedance_support_count",
    "current_required_axes_feasibility_status",
    "external_law_equation_status",
    "next_generator_requirement",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "new_spectral_support_required_count",
    "moment_only_reweighting_possible_count",
    "existing_tail_support_count",
    "max_required_s_h_u_gap_to_support_hull",
    "max_required_deformed_ratio_lower_bound",
    "minimum_support_count_for_alpha_resolution",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedExternalLawEquationConfig:
    """Input/output contract for selected-root external-law equations."""

    output_dir: Path
    conditional_tilt_rows_path: Path = DEFAULT_CONDITIONAL_TILT_ROWS
    external_law_target_rows_path: Path = DEFAULT_EXTERNAL_LAW_TARGET_ROWS
    support_gap_rows_path: Path = DEFAULT_SUPPORT_GAP_ROWS
    deformed_edge_rows_path: Path = DEFAULT_DEFORMED_EDGE_ROWS
    target_alpha: float = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditional-tilt-rows-path",
        type=Path,
        default=DEFAULT_CONDITIONAL_TILT_ROWS,
    )
    parser.add_argument(
        "--external-law-target-rows-path",
        type=Path,
        default=DEFAULT_EXTERNAL_LAW_TARGET_ROWS,
    )
    parser.add_argument(
        "--support-gap-rows-path",
        type=Path,
        default=DEFAULT_SUPPORT_GAP_ROWS,
    )
    parser.add_argument(
        "--deformed-edge-rows-path",
        type=Path,
        default=DEFAULT_DEFORMED_EDGE_ROWS,
    )
    parser.add_argument("--target-alpha", type=float, default=0.01)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedExternalLawEquationConfig):
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


def _finite_json(values: dict[str, float]) -> str:
    clean = {
        key: (float(value) if math.isfinite(float(value)) else None)
        for key, value in values.items()
    }
    return json.dumps(clean, sort_keys=True)


def _parse_json_dict(value: object) -> dict[str, float]:
    if isinstance(value, str) and value:
        raw = json.loads(value)
    elif isinstance(value, dict):
        raw = value
    else:
        raw = {}
    return {str(key): finite_float(val) for key, val in raw.items()}


def _lookup_by_case(rows: pd.DataFrame, case_column: str) -> dict[str, pd.Series]:
    if rows.empty:
        return {}
    _require_columns(rows, {case_column}, "lookup rows")
    return {str(row[case_column]): row for _, row in rows.iterrows()}


def _tilt_lookup(rows: pd.DataFrame, *, axes: str) -> dict[str, pd.Series]:
    if rows.empty:
        return {}
    _require_columns(
        rows,
        {"target_case_id", "support_pool_scope", "checked_moment_axes"},
        "conditional tilt rows",
    )
    selected = rows.loc[
        rows["support_pool_scope"].astype(str).eq("same_B_Hu")
        & rows["checked_moment_axes"].astype(str).eq(str(axes))
    ]
    return {str(row["target_case_id"]): row for _, row in selected.iterrows()}


def _minimum_support_count(target_alpha: float) -> int:
    if not (math.isfinite(target_alpha) and 0.0 < target_alpha < 1.0):
        raise ValueError("target_alpha must be in (0, 1).")
    return int(math.ceil(1.0 / float(target_alpha)) - 1)


def build_root_selected_external_law_equation_rows(
    *,
    conditional_tilt_rows: pd.DataFrame,
    external_law_target_rows: pd.DataFrame,
    support_gap_rows: pd.DataFrame,
    deformed_edge_rows: pd.DataFrame,
    target_alpha: float = 0.01,
) -> pd.DataFrame:
    """Return selected-root external-law equation rows."""
    _require_columns(
        external_law_target_rows,
        {
            "target_case_id",
            "target_status",
            "target_root_tail_stratum_key",
            "target_moment_vector_json",
            "required_tilt_axes",
            "conditioning_event",
        },
        "external law target rows",
    )
    full_lookup = _tilt_lookup(conditional_tilt_rows, axes="T,A,E,S_Hu")
    gap_lookup = _lookup_by_case(support_gap_rows, "target_case_id")
    edge_lookup = _lookup_by_case(deformed_edge_rows, "target_case_id")
    min_support = _minimum_support_count(target_alpha)
    records: list[dict[str, object]] = []
    for _, target_row in external_law_target_rows.sort_values("target_case_id").iterrows():
        case_id = string_value(target_row, "target_case_id")
        target_status = string_value(target_row, "target_status")
        required_axes = string_value(target_row, "required_tilt_axes")
        if required_axes == "none":
            required_lookup = full_lookup
            checked_axes = "T,A,E,S_Hu"
        else:
            required_lookup = _tilt_lookup(conditional_tilt_rows, axes=required_axes)
            checked_axes = required_axes
        required_feasibility = required_lookup.get(case_id)
        full_feasibility = full_lookup.get(case_id)
        support_max = math.nan
        if required_feasibility is not None:
            support_max = _parse_json_dict(
                required_feasibility.get("support_max_moment_vector_json", "")
            ).get("S_Hu", math.nan)
        target_moments = _parse_json_dict(target_row["target_moment_vector_json"])
        target_s = target_moments.get("S_Hu", math.nan)
        gap_to_hull = (
            max(float(target_s) - float(support_max), 0.0)
            if math.isfinite(target_s) and math.isfinite(support_max)
            else math.nan
        )
        required_ratio = math.exp(target_s) if math.isfinite(target_s) else math.nan
        gap_row = gap_lookup.get(case_id)
        edge_row = edge_lookup.get(case_id)
        current_status = (
            string_value(required_feasibility, "tilt_feasibility_status")
            if required_feasibility is not None
            else "missing_required_axes_feasibility_row"
        )
        needs_new_spectral_support = bool(math.isfinite(gap_to_hull) and gap_to_hull > 1e-9)
        moment_feasible = (
            bool(required_feasibility.get("convex_hull_moment_feasible", False))
            if required_feasibility is not None
            else False
        )
        if target_status == "exact_tail_support_available":
            equation_status = "existing_tail_support_defer_to_tail_panel"
            next_step = "use_existing_conservative_deformed_tail_panel"
        elif needs_new_spectral_support:
            equation_status = "requires_new_same_stratum_nonzero_s_h_u_support"
            next_step = "sample_or_derive_external_law_with_positive_mass_on_tail_event"
        elif moment_feasible:
            equation_status = "moment_equation_feasible_but_still_diagnostic_only"
            next_step = "estimate_tail_with_sufficient_same_stratum_support"
        else:
            equation_status = "requires_new_same_stratum_geometry_support"
            next_step = "generate_roots_matching_selected_geometry_moments"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": case_id,
                "target_status": target_status,
                "conditioning_event": string_value(target_row, "conditioning_event"),
                "target_root_tail_stratum_key": string_value(
                    target_row,
                    "target_root_tail_stratum_key",
                ),
                "required_tilt_axes": required_axes,
                "law_moment_equation": (
                    f"E_Q[{checked_axes} | R_root_selected,T/A/E/B/H_u stratum] "
                    f"= target_{checked_axes}"
                ),
                "tail_support_equation": (
                    "P_Q(S_Hu >= target_S_Hu | R_root_selected,T/A/E/B/H_u stratum) > 0"
                ),
                "target_moment_vector_json": _finite_json(target_moments),
                "required_axes_residual_vector_json": (
                    string_value(
                        required_feasibility,
                        "moment_residual_vector_json",
                    )
                    if required_feasibility is not None
                    else "{}"
                ),
                "full_phi_residual_vector_json": (
                    string_value(full_feasibility, "moment_residual_vector_json")
                    if full_feasibility is not None
                    else "{}"
                ),
                "target_s_h_u_excess_log": target_s,
                "support_hull_max_s_h_u_excess_log": support_max,
                "required_s_h_u_gap_to_support_hull": gap_to_hull,
                "required_deformed_ratio_lower_bound": required_ratio,
                "target_selected_root_deformed_ratio": finite_float(
                    edge_row.get("selected_root_deformed_ratio", math.nan)
                    if edge_row is not None
                    else math.nan
                ),
                "target_deformed_mp_upper_edge": finite_float(
                    edge_row.get("deformed_mp_upper_edge", math.nan)
                    if edge_row is not None
                    else math.nan
                ),
                "target_selected_root_eigenvalue": finite_float(
                    edge_row.get("selected_root_eigenvalue", math.nan)
                    if edge_row is not None
                    else math.nan
                ),
                "minimum_support_count_for_alpha_resolution": min_support,
                "minimum_tail_exceedance_support_count": 1,
                "current_required_axes_feasibility_status": current_status,
                "external_law_equation_status": equation_status,
                "next_generator_requirement": next_step
                if gap_row is None
                else string_value(gap_row, "required_external_law", next_step),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_external_law_equation_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected-root external-law equations."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    status = rows["external_law_equation_status"].astype(str)
    gaps = pd.to_numeric(rows["required_s_h_u_gap_to_support_hull"], errors="coerce")
    ratios = pd.to_numeric(rows["required_deformed_ratio_lower_bound"], errors="coerce")
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "new_spectral_support_required_count": int(
                    status.eq("requires_new_same_stratum_nonzero_s_h_u_support").sum()
                ),
                "moment_only_reweighting_possible_count": int(
                    status.eq("moment_equation_feasible_but_still_diagnostic_only").sum()
                ),
                "existing_tail_support_count": int(
                    status.eq("existing_tail_support_defer_to_tail_panel").sum()
                ),
                "max_required_s_h_u_gap_to_support_hull": float(gaps.max())
                if gaps.notna().any()
                else math.nan,
                "max_required_deformed_ratio_lower_bound": float(ratios.max())
                if ratios.notna().any()
                else math.nan,
                "minimum_support_count_for_alpha_resolution": int(
                    rows["minimum_support_count_for_alpha_resolution"].max()
                ),
                "summary_status": (
                    "new_selected_root_spectral_support_required"
                    if status.str.contains("requires_new").any()
                    else "equation_targets_have_current_support_diagnostic_only"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_external_law_equation_panel(
    config: RootSelectedExternalLawEquationConfig,
) -> dict[str, pd.DataFrame]:
    tilt = pd.read_csv(config.conditional_tilt_rows_path, low_memory=False)
    targets = pd.read_csv(config.external_law_target_rows_path, low_memory=False)
    gaps = pd.read_csv(config.support_gap_rows_path, low_memory=False)
    edges = pd.read_csv(config.deformed_edge_rows_path, low_memory=False)
    rows = build_root_selected_external_law_equation_rows(
        conditional_tilt_rows=tilt,
        external_law_target_rows=targets,
        support_gap_rows=gaps,
        deformed_edge_rows=edges,
        target_alpha=config.target_alpha,
    )
    summary = summarize_root_selected_external_law_equation_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_external_law_equation_panel(
    config: RootSelectedExternalLawEquationConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_external_law_equation_panel(config)
    paths = {
        "rows": config.output_dir / ROWS_OUTPUT,
        "summary": config.output_dir / SUMMARY_OUTPUT,
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
    manifest_path = config.output_dir / MANIFEST_OUTPUT
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "row_counts": {key: int(table.shape[0]) for key, table in tables.items()},
        "outputs": paths,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    args = parse_args()
    outputs = run_root_selected_external_law_equation_panel(
        RootSelectedExternalLawEquationConfig(
            output_dir=args.output_dir,
            conditional_tilt_rows_path=args.conditional_tilt_rows_path,
            external_law_target_rows_path=args.external_law_target_rows_path,
            support_gap_rows_path=args.support_gap_rows_path,
            deformed_edge_rows_path=args.deformed_edge_rows_path,
            target_alpha=args.target_alpha,
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
