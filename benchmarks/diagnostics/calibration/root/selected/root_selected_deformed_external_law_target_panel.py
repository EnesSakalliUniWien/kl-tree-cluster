"""External-law targets for deformed selected-root spectral tails.

The support-gap panel says which selected-root strata are missing. This panel
turns those gaps into an explicit external-null law target. It does not sample
new roots and does not calibrate p-values.

The proposed law form is a conditional exponential tilt:

    dQ_theta(x) proportional to dP0(x) exp(theta^T phi_root(x))
    conditioned on the selected root event and the requested B,H_u stratum.

Rows specify the required sufficient-statistic moments phi_root=(T,A,E,S_Hu)
for every fail-closed target.
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

from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
    _finite_float,
)

SCHEMA_VERSION = "root_selected_deformed_external_law_target_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_deformed_external_law_target_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.root.selected.root_selected_deformed_external_law_target_panel"

DEFAULT_SUPPORT_GAP_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_deformed_tail_support_gap_mild_replay_v3_smoke"
    / "root_selected_deformed_tail_support_gap_rows.csv"
)

ROWS_OUTPUT = "root_selected_deformed_external_law_target_rows.csv"
SUMMARY_OUTPUT = "root_selected_deformed_external_law_target_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_root_tail_stratum_key",
    "target_status",
    "conditional_law_family",
    "conditioning_event",
    "sufficient_statistics",
    "target_moment_vector_json",
    "nearest_support_moment_vector_json",
    "required_moment_gap_json",
    "required_tilt_axes",
    "required_s_h_u_lift_multiplier",
    "dominant_conditioning_gap",
    "likelihood_ratio_formula",
    "admissibility_status",
    "next_generator_requirement",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "external_law_required_count",
    "existing_support_count",
    "selected_ratio_axis_count",
    "edge_axis_count",
    "max_required_s_h_u_lift_multiplier",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedDeformedExternalLawTargetConfig:
    """Input/output contract for deformed external-law target rows."""

    output_dir: Path
    support_gap_rows_path: Path = DEFAULT_SUPPORT_GAP_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--support-gap-rows-path",
        type=Path,
        default=DEFAULT_SUPPORT_GAP_ROWS,
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedDeformedExternalLawTargetConfig):
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


def _string_value(row: pd.Series | dict[str, object], column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _moment_vector(row: pd.Series, *, prefix: str) -> dict[str, float]:
    return {
        "T": _finite_float(row.get(f"{prefix}_tie_fraction", math.nan)),
        "A": _finite_float(row.get(f"{prefix}_action_log1p", math.nan)),
        "E": _finite_float(row.get(f"{prefix}_edge_log1p", math.nan)),
        "S_Hu": _finite_float(row.get(f"{prefix}_s_h_u_excess_log", math.nan)),
    }


def _nearest_moment_vector(row: pd.Series) -> dict[str, float]:
    return {
        "T": math.nan,
        "A": math.nan,
        "E": math.nan,
        "S_Hu": _finite_float(row.get("nearest_support_s_h_u_excess_log", math.nan)),
    }


def _absolute_moment_gap(row: pd.Series) -> dict[str, float]:
    return {
        "T": _finite_float(row.get("nearest_support_tie_gap", math.nan)),
        "A": _finite_float(row.get("nearest_support_action_gap", math.nan)),
        "E": _finite_float(row.get("nearest_support_edge_gap", math.nan)),
        "S_Hu": _finite_float(row.get("required_s_h_u_gap_to_nearest_support", math.nan)),
    }


def _finite_json(values: dict[str, float]) -> str:
    clean = {
        key: (float(value) if math.isfinite(float(value)) else None)
        for key, value in values.items()
    }
    return json.dumps(clean, sort_keys=True)


def _required_axes(row: pd.Series) -> str:
    dominant = _string_value(row, "dominant_conditioning_gap")
    axes = ["S_Hu"]
    if dominant == "selected_ratio_action":
        axes.insert(0, "A")
    elif dominant == "edge_action":
        axes.insert(0, "E")
    elif dominant == "tie_rank":
        axes.insert(0, "T")
    elif dominant == "bandwidth_topology":
        axes.insert(0, "B")
    return ",".join(axes)


def build_root_selected_deformed_external_law_target_rows(
    support_gap_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return law target rows from deformed support-gap rows."""
    _require_columns(
        support_gap_rows,
        {
            "target_case_id",
            "target_root_tail_stratum_key",
            "target_tie_fraction",
            "target_action_log1p",
            "target_edge_log1p",
            "target_s_h_u_excess_log",
            "target_bandwidth_topology_status",
            "support_gap_status",
            "required_s_h_u_lift_multiplier",
            "dominant_conditioning_gap",
        },
        "support gap rows",
    )
    records: list[dict[str, object]] = []
    for _, row in support_gap_rows.sort_values("target_case_id").iterrows():
        support_status = _string_value(row, "support_gap_status")
        law_required = support_status != "exact_tail_support_available"
        target = _moment_vector(row, prefix="target")
        nearest = _nearest_moment_vector(row)
        gap = _absolute_moment_gap(row)
        conditioning_event = (
            f"R_root_selected AND stratum={_string_value(row, 'target_root_tail_stratum_key')}"
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": _string_value(row, "target_case_id"),
                "target_root_tail_stratum_key": _string_value(
                    row,
                    "target_root_tail_stratum_key",
                ),
                "target_status": support_status,
                "conditional_law_family": "selected_root_conditional_exponential_tilt",
                "conditioning_event": conditioning_event,
                "sufficient_statistics": "phi_root=(T,A,E,S_Hu); hard_condition_on=(R,B,H_u)",
                "target_moment_vector_json": _finite_json(target),
                "nearest_support_moment_vector_json": _finite_json(nearest),
                "required_moment_gap_json": _finite_json(gap),
                "required_tilt_axes": _required_axes(row) if law_required else "none",
                "required_s_h_u_lift_multiplier": _finite_float(
                    row.get("required_s_h_u_lift_multiplier", math.nan)
                ),
                "dominant_conditioning_gap": _string_value(
                    row,
                    "dominant_conditioning_gap",
                ),
                "likelihood_ratio_formula": (
                    "log_w=log_dP0_minus_log_dQtheta; "
                    "Qtheta proportional P0*exp(theta_T*T+theta_A*A+theta_E*E+theta_S*S_Hu) "
                    "conditioned_on_R_B_Hu"
                ),
                "admissibility_status": (
                    "external_law_required_fail_closed_until_support_generated"
                    if law_required
                    else "existing_tail_support_available"
                ),
                "next_generator_requirement": (
                    "fit_or_sample_conditional_tilt_matching_target_moments"
                    if law_required
                    else "defer_to_existing_tail_panel"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_deformed_external_law_target_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize external-law target requirements."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    status = rows["admissibility_status"].astype(str)
    required = status.eq("external_law_required_fail_closed_until_support_generated")
    axes = rows["required_tilt_axes"].astype(str)
    lifts = pd.to_numeric(rows["required_s_h_u_lift_multiplier"], errors="coerce")
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "external_law_required_count": int(required.sum()),
                "existing_support_count": int(status.eq("existing_tail_support_available").sum()),
                "selected_ratio_axis_count": int(axes.str.contains("A").sum()),
                "edge_axis_count": int(axes.str.contains("E").sum()),
                "max_required_s_h_u_lift_multiplier": float(lifts.max())
                if lifts.notna().any()
                else math.nan,
                "summary_status": (
                    "external_law_targets_required"
                    if bool(required.any())
                    else "all_targets_have_existing_tail_support"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_deformed_external_law_target_panel(
    config: RootSelectedDeformedExternalLawTargetConfig,
) -> dict[str, pd.DataFrame]:
    gaps = pd.read_csv(config.support_gap_rows_path, low_memory=False)
    rows = build_root_selected_deformed_external_law_target_rows(gaps)
    summary = summarize_root_selected_deformed_external_law_target_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_deformed_external_law_target_panel(
    config: RootSelectedDeformedExternalLawTargetConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_deformed_external_law_target_panel(config)
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
    outputs = run_root_selected_deformed_external_law_target_panel(
        RootSelectedDeformedExternalLawTargetConfig(
            output_dir=args.output_dir,
            support_gap_rows_path=args.support_gap_rows_path,
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
