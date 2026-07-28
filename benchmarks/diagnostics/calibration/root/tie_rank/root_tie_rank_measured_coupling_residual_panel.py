"""Target-level residual panel for measured root tie-rank coupling.

After generated proposal matrices have been replayed through the
selected-neighborhood stack, bandwidth is no longer merely missing. This panel
answers the next question: for each observed root, which measured proposal gets
closest, and which term in the selected spectral-action/tie-rank equation still
blocks the hard roots?
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

SCHEMA_VERSION = "root_tie_rank_measured_coupling_residual_panel/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_measured_coupling_residual_panel_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_measured_coupling_residual_panel"

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_COUPLING_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_coupling_equation_after_generated_replay"
    / "root_tie_rank_coupling_equation_rows.csv"
)

ROWS_OUTPUT = "root_tie_rank_measured_coupling_residual_rows.csv"
SUMMARY_OUTPUT = "root_tie_rank_measured_coupling_residual_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "best_proposal_family",
    "best_generated_case_id",
    "proposal_support_role",
    "target_bottleneck_coupling",
    "best_generated_bottleneck_coupling",
    "target_measured_neighborhood_coupling",
    "best_generated_measured_neighborhood_coupling",
    "best_bottleneck_coupling_ratio",
    "best_measured_neighborhood_coupling_ratio",
    "measured_neighborhood_coupling_deficit",
    "tie_fraction_deficit",
    "tie_fraction_relative_deficit",
    "action_log_deficit",
    "edge_log_deficit",
    "action_edge_bottleneck_deficit",
    "action_edge_bottleneck_relative_deficit",
    "spectral_excess_log_deficit",
    "spectral_excess_relative_deficit",
    "dominant_action_edge_axis",
    "dominant_residual_axis",
    "bandwidth_gap_status",
    "bandwidth_conditioning_status",
    "best_coupling_pattern",
    "measured_neighborhood_coupling_reached",
    "target_resolution_status",
    "next_mathematical_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "dominant_residual_axis",
    "target_resolution_status",
    "row_count",
    "measured_coupling_reached_count",
    "diagnostic_proposal_best_count",
    "calibration_candidate_best_count",
    "median_best_measured_neighborhood_coupling_ratio",
    "median_action_edge_relative_deficit",
    "median_spectral_relative_deficit",
    "summary_status",
)


@dataclass(frozen=True)
class RootTieRankMeasuredCouplingResidualConfig:
    """Input/output paths for measured coupling residual diagnostics."""

    output_dir: Path
    coupling_rows_path: Path = DEFAULT_COUPLING_ROWS
    partial_ratio_floor: float = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--coupling-rows-path",
        type=Path,
        default=DEFAULT_COUPLING_ROWS,
    )
    parser.add_argument("--partial-ratio-floor", type=float, default=0.50)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootTieRankMeasuredCouplingResidualConfig):
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


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _string_value(
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


def _positive_deficit(target: object, generated: object) -> float:
    target_value = _finite_float(target)
    generated_value = _finite_float(generated)
    if not (math.isfinite(target_value) and math.isfinite(generated_value)):
        return math.nan
    return float(max(target_value - generated_value, 0.0))


def _relative_deficit(deficit: float, target: object) -> float:
    target_value = _finite_float(target)
    if not (math.isfinite(deficit) and math.isfinite(target_value)):
        return math.nan
    if abs(target_value) <= 1e-12:
        return 0.0 if deficit <= 0.0 else math.inf
    return float(deficit / abs(target_value))


def _finite_or_negative_inf(value: object) -> float:
    numeric = _finite_float(value)
    return numeric if math.isfinite(numeric) else -math.inf


def _finite_or_positive_inf(value: object) -> float:
    numeric = _finite_float(value)
    return numeric if math.isfinite(numeric) else math.inf


def _proposal_support_role(family: str) -> str:
    if family == "iid_marginal_bernoulli":
        return "selected_null_candidate_support"
    return "diagnostic_proposal_not_calibration"


def _dominant_action_edge_axis(action_deficit: float, edge_deficit: float) -> str:
    action = _finite_float(action_deficit)
    edge = _finite_float(edge_deficit)
    if not (math.isfinite(action) and math.isfinite(edge)):
        return "action_edge_unavailable"
    if action <= 0.0 and edge <= 0.0:
        return "action_edge_satisfied"
    if action > 0.0 and edge > 0.0:
        if abs(action - edge) <= 1e-9:
            return "action_and_edge_joint_deficit"
        return "action_deficit_larger" if action > edge else "edge_deficit_larger"
    return "action_deficit" if action > 0.0 else "edge_deficit"


def _bandwidth_conditioning_status(row: pd.Series) -> str:
    if not _bool_value(row.get("generated_neighborhood_measured", False)):
        return "generated_bandwidth_unmeasured"
    status = _string_value(row, "bandwidth_gap_status")
    if status == "bandwidth_band_match":
        return "bandwidth_stratum_match"
    return "bandwidth_measured_but_stratum_mismatch"


def _dominant_residual_axis(
    *,
    measured_reached: bool,
    bandwidth_status: str,
    tie_relative_deficit: float,
    action_edge_relative_deficit: float,
    spectral_relative_deficit: float,
) -> str:
    if measured_reached:
        return "none_measured_coupling_reached"
    if bandwidth_status == "generated_bandwidth_unmeasured":
        return "bandwidth_unmeasured"
    candidates = {
        "tie_rank_fraction": _finite_float(tie_relative_deficit),
        "action_edge_bottleneck": _finite_float(action_edge_relative_deficit),
        "spectral_excess": _finite_float(spectral_relative_deficit),
    }
    finite = {key: value for key, value in candidates.items() if math.isfinite(value)}
    if not finite:
        return "residual_unlocalized"
    key, value = max(finite.items(), key=lambda item: (item[1], item[0]))
    if value <= 1e-12:
        return "residual_unlocalized"
    return key


def _target_resolution_status(
    *,
    measured_reached: bool,
    support_role: str,
    measured_ratio: float,
    partial_ratio_floor: float,
) -> str:
    if measured_reached:
        if support_role == "selected_null_candidate_support":
            return "calibration_candidate_reaches_target"
        return "diagnostic_measured_coupling_reaches_target_not_calibration"
    if math.isfinite(measured_ratio) and measured_ratio >= float(partial_ratio_floor):
        return "partial_measured_coupling_residual"
    return "hard_measured_coupling_residual"


def _next_mathematical_step(
    *,
    target_status: str,
    dominant_axis: str,
    support_role: str,
    bandwidth_status: str,
) -> str:
    if target_status == "calibration_candidate_reaches_target":
        return "estimate_selected_root_tail_in_reached_null_stratum"
    if target_status == "diagnostic_measured_coupling_reaches_target_not_calibration":
        return "convert_diagnostic_family_to_external_null_support_or_reject"
    if bandwidth_status == "generated_bandwidth_unmeasured":
        return "run_or_join_selected_neighborhood_replay"
    if dominant_axis == "spectral_excess":
        return "derive_selected_spectral_excess_given_high_action_edge_tie_rank"
    if dominant_axis == "action_edge_bottleneck":
        return "derive_joint_action_edge_generator_with_spectral_persistence"
    if dominant_axis == "tie_rank_fraction":
        return "calibrate_discrete_tie_rank_multiplicity_law"
    if support_role != "selected_null_candidate_support":
        return "find_external_null_support_for_diagnostic_coupling_stratum"
    return "increase_joint_measured_coupling_support"


def _best_row_for_target(group: pd.DataFrame) -> pd.Series:
    ranked = group.copy()
    ranked["_measured_dominates_rank"] = ranked["measured_neighborhood_coupling_dominates"].map(
        _bool_value
    )
    ranked["_measured_ratio_rank"] = ranked["measured_neighborhood_coupling_ratio"].map(
        _finite_or_negative_inf
    )
    ranked["_bottleneck_ratio_rank"] = ranked["bottleneck_coupling_ratio"].map(
        _finite_or_negative_inf
    )
    ranked["_measured_deficit_rank"] = ranked["measured_neighborhood_coupling_deficit"].map(
        _finite_or_positive_inf
    )
    ranked["_score_rank"] = ranked["coupling_deficit_score"].map(_finite_or_positive_inf)
    ranked = ranked.sort_values(
        [
            "_measured_dominates_rank",
            "_measured_ratio_rank",
            "_bottleneck_ratio_rank",
            "_measured_deficit_rank",
            "_score_rank",
            "proposal_family",
            "best_generated_case_id",
        ],
        ascending=[False, False, False, True, True, True, True],
    )
    return ranked.iloc[0]


def _residual_record(row: pd.Series, *, partial_ratio_floor: float) -> dict[str, object]:
    family = _string_value(row, "proposal_family")
    support_role = _proposal_support_role(family)
    measured_reached = _bool_value(row.get("measured_neighborhood_coupling_dominates", False))
    tie_deficit = _positive_deficit(
        row.get("target_tie_fraction", math.nan),
        row.get("generated_tie_fraction", math.nan),
    )
    action_deficit = _positive_deficit(
        row.get("target_action_log", math.nan),
        row.get("generated_action_log", math.nan),
    )
    edge_deficit = _positive_deficit(
        row.get("target_edge_log", math.nan),
        row.get("generated_edge_log", math.nan),
    )
    action_edge_deficit = _positive_deficit(
        row.get("target_action_edge_bottleneck", math.nan),
        row.get("generated_action_edge_bottleneck", math.nan),
    )
    spectral_deficit = _positive_deficit(
        row.get("target_spectral_excess_log", math.nan),
        row.get("generated_spectral_excess_log", math.nan),
    )
    tie_relative = _relative_deficit(tie_deficit, row.get("target_tie_fraction"))
    action_edge_relative = _relative_deficit(
        action_edge_deficit,
        row.get("target_action_edge_bottleneck"),
    )
    spectral_relative = _relative_deficit(
        spectral_deficit,
        row.get("target_spectral_excess_log"),
    )
    bandwidth_status = _bandwidth_conditioning_status(row)
    dominant_axis = _dominant_residual_axis(
        measured_reached=measured_reached,
        bandwidth_status=bandwidth_status,
        tie_relative_deficit=tie_relative,
        action_edge_relative_deficit=action_edge_relative,
        spectral_relative_deficit=spectral_relative,
    )
    measured_ratio = _finite_float(row.get("measured_neighborhood_coupling_ratio"))
    target_status = _target_resolution_status(
        measured_reached=measured_reached,
        support_role=support_role,
        measured_ratio=measured_ratio,
        partial_ratio_floor=float(partial_ratio_floor),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "target_case_id": _string_value(row, "target_case_id"),
        "best_proposal_family": family,
        "best_generated_case_id": _string_value(row, "best_generated_case_id"),
        "proposal_support_role": support_role,
        "target_bottleneck_coupling": _finite_float(
            row.get("target_bottleneck_coupling", math.nan)
        ),
        "best_generated_bottleneck_coupling": _finite_float(
            row.get("generated_bottleneck_coupling", math.nan)
        ),
        "target_measured_neighborhood_coupling": _finite_float(
            row.get("target_measured_neighborhood_coupling", math.nan)
        ),
        "best_generated_measured_neighborhood_coupling": _finite_float(
            row.get("generated_measured_neighborhood_coupling", math.nan)
        ),
        "best_bottleneck_coupling_ratio": _finite_float(
            row.get("bottleneck_coupling_ratio", math.nan)
        ),
        "best_measured_neighborhood_coupling_ratio": measured_ratio,
        "measured_neighborhood_coupling_deficit": _finite_float(
            row.get("measured_neighborhood_coupling_deficit", math.nan)
        ),
        "tie_fraction_deficit": tie_deficit,
        "tie_fraction_relative_deficit": tie_relative,
        "action_log_deficit": action_deficit,
        "edge_log_deficit": edge_deficit,
        "action_edge_bottleneck_deficit": action_edge_deficit,
        "action_edge_bottleneck_relative_deficit": action_edge_relative,
        "spectral_excess_log_deficit": spectral_deficit,
        "spectral_excess_relative_deficit": spectral_relative,
        "dominant_action_edge_axis": _dominant_action_edge_axis(
            action_deficit,
            edge_deficit,
        ),
        "dominant_residual_axis": dominant_axis,
        "bandwidth_gap_status": _string_value(row, "bandwidth_gap_status"),
        "bandwidth_conditioning_status": bandwidth_status,
        "best_coupling_pattern": _string_value(row, "coupling_pattern"),
        "measured_neighborhood_coupling_reached": measured_reached,
        "target_resolution_status": target_status,
        "next_mathematical_step": _next_mathematical_step(
            target_status=target_status,
            dominant_axis=dominant_axis,
            support_role=support_role,
            bandwidth_status=bandwidth_status,
        ),
    }


def build_measured_coupling_residual_rows(
    coupling_rows: pd.DataFrame,
    *,
    partial_ratio_floor: float = 0.50,
) -> pd.DataFrame:
    """Return the best measured-coupling residual row for each target."""
    _require_columns(
        coupling_rows,
        {
            "target_case_id",
            "proposal_family",
            "best_generated_case_id",
            "target_tie_fraction",
            "generated_tie_fraction",
            "target_action_log",
            "generated_action_log",
            "target_edge_log",
            "generated_edge_log",
            "target_spectral_excess_log",
            "generated_spectral_excess_log",
            "target_action_edge_bottleneck",
            "generated_action_edge_bottleneck",
            "target_bottleneck_coupling",
            "generated_bottleneck_coupling",
            "bottleneck_coupling_ratio",
            "target_measured_neighborhood_coupling",
            "generated_measured_neighborhood_coupling",
            "measured_neighborhood_coupling_deficit",
            "measured_neighborhood_coupling_ratio",
            "measured_neighborhood_coupling_dominates",
            "generated_neighborhood_measured",
            "bandwidth_gap_status",
            "coupling_deficit_score",
            "coupling_pattern",
        },
        "coupling rows",
    )
    if coupling_rows.empty:
        return pd.DataFrame(columns=ROW_COLUMNS)
    records = [
        _residual_record(best, partial_ratio_floor=float(partial_ratio_floor))
        for _, group in coupling_rows.groupby("target_case_id", sort=True)
        for best in [_best_row_for_target(group)]
    ]
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _safe_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return float(finite.median()) if not finite.empty else math.nan


def summarize_measured_coupling_residual_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize target-level measured-coupling residuals."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    group_columns = ["dominant_residual_axis", "target_resolution_status"]
    for keys, group in rows.groupby(group_columns, dropna=False, sort=True):
        reached_count = int(group["measured_neighborhood_coupling_reached"].sum())
        diagnostic_count = int(
            group["proposal_support_role"]
            .astype(str)
            .eq("diagnostic_proposal_not_calibration")
            .sum()
        )
        calibration_count = int(
            group["proposal_support_role"].astype(str).eq("selected_null_candidate_support").sum()
        )
        if str(keys[1]).startswith("hard"):
            status = "hard_roots_require_new_selected_coupling_law"
        elif str(keys[1]).startswith("partial"):
            status = "partial_residual_requires_targeted_generator_or_law"
        elif diagnostic_count > 0:
            status = "diagnostic_reach_requires_external_null_support"
        else:
            status = "calibration_candidate_reaches_some_targets"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "dominant_residual_axis": keys[0],
                "target_resolution_status": keys[1],
                "row_count": int(group.shape[0]),
                "measured_coupling_reached_count": reached_count,
                "diagnostic_proposal_best_count": diagnostic_count,
                "calibration_candidate_best_count": calibration_count,
                "median_best_measured_neighborhood_coupling_ratio": _safe_median(
                    group["best_measured_neighborhood_coupling_ratio"]
                ),
                "median_action_edge_relative_deficit": _safe_median(
                    group["action_edge_bottleneck_relative_deficit"]
                ),
                "median_spectral_relative_deficit": _safe_median(
                    group["spectral_excess_relative_deficit"]
                ),
                "summary_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def evaluate_measured_coupling_residual_panel(
    config: RootTieRankMeasuredCouplingResidualConfig,
) -> dict[str, pd.DataFrame]:
    """Read coupling rows and return residual panel tables."""
    coupling_rows = pd.read_csv(config.coupling_rows_path)
    rows = build_measured_coupling_residual_rows(
        coupling_rows,
        partial_ratio_floor=float(config.partial_ratio_floor),
    )
    summary = summarize_measured_coupling_residual_rows(rows)
    return {"rows": rows, "summary": summary}


def run_measured_coupling_residual_panel(
    config: RootTieRankMeasuredCouplingResidualConfig,
) -> dict[str, Path]:
    """Run the measured-coupling residual panel and write outputs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_measured_coupling_residual_panel(config)
    paths = {
        "rows": output_dir / ROWS_OUTPUT,
        "summary": output_dir / SUMMARY_OUTPUT,
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
    manifest_path = output_dir / MANIFEST_OUTPUT
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
    outputs = run_measured_coupling_residual_panel(
        RootTieRankMeasuredCouplingResidualConfig(
            output_dir=args.output_dir,
            coupling_rows_path=args.coupling_rows_path,
            partial_ratio_floor=float(args.partial_ratio_floor),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
