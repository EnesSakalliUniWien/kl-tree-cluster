"""Continuous dominance panel for root spectral-action proposal coupling.

Band-level proposal gaps show that action/edge and spectral coordinates are
separable. This panel checks the stronger continuous condition: whether a
generated proposal row dominates each observed root in tie rank, selected
ratio, edge margin, and spectral ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.calibration.reporting import (
    print_diagnostic_output_paths,
    write_diagnostic_bundle,
)
from benchmarks.diagnostics.calibration.root.tie_rank.cli import parse_proposal_panel_args
from benchmarks.diagnostics.calibration.root.tie_rank.comparison import (
    partition_target_and_generated_rows,
    row_bandwidth_gap_status,
)

SCHEMA_VERSION = "root_tie_rank_spectral_action_dominance_panel/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_spectral_action_dominance_panel_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_spectral_action_dominance_panel"
)

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_PROPOSAL_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_null_proposal_frontier_two_case_smoke"
    / "root_tie_rank_null_proposal_combined_feasibility_rows.csv"
)

ROWS_OUTPUT = "root_tie_rank_spectral_action_dominance_rows.csv"
SUMMARY_OUTPUT = "root_tie_rank_spectral_action_dominance_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "proposal_family",
    "best_generated_case_id",
    "target_tie_fraction",
    "generated_tie_fraction",
    "tie_fraction_deficit",
    "target_selected_ratio_log",
    "generated_selected_ratio_log",
    "selected_ratio_log_deficit",
    "target_edge_log",
    "generated_edge_log",
    "edge_log_deficit",
    "target_spectral_log",
    "generated_spectral_log",
    "spectral_log_deficit",
    "tie_dominates",
    "selected_ratio_dominates",
    "edge_dominates",
    "spectral_dominates",
    "action_edge_dominates",
    "spectral_action_dominates",
    "full_continuous_dominates",
    "bandwidth_measured_match",
    "bandwidth_gap_status",
    "dominance_deficit_score",
    "dominance_pattern",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "proposal_family",
    "target_count",
    "full_continuous_dominance_count",
    "spectral_action_dominance_count",
    "action_edge_dominance_count",
    "spectral_dominance_count",
    "selected_ratio_dominance_count",
    "edge_dominance_count",
    "tie_dominance_count",
    "bandwidth_measured_match_count",
    "median_dominance_deficit_score",
    "min_dominance_deficit_score",
    "median_spectral_log_deficit",
    "median_selected_ratio_log_deficit",
    "summary_status",
)


@dataclass(frozen=True)
class RootTieRankSpectralActionDominanceConfig:
    """Input/output paths for the continuous dominance panel."""

    output_dir: Path
    proposal_feasibility_rows_path: Path = DEFAULT_PROPOSAL_FEASIBILITY_ROWS


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _safe_log1p(value: object) -> float:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    return float(math.log1p(max(numeric, 0.0)))


def _safe_log(value: object) -> float:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    return float(math.log(max(numeric, 1e-12)))


def _deficit(*, target: float, generated: float, missing_value: float = 10.0) -> float:
    if not (math.isfinite(target) and math.isfinite(generated)):
        return float(missing_value)
    return float(max(target - generated, 0.0))


def _string_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _dominance_pattern(
    *,
    full: bool,
    spectral_action: bool,
    action_edge: bool,
    spectral: bool,
    selected_ratio: bool,
    edge: bool,
    bandwidth_status: str,
) -> str:
    if full and bandwidth_status == "bandwidth_band_match":
        return "full_continuous_and_bandwidth_dominance"
    if full:
        return "full_continuous_dominance_bandwidth_unmeasured_or_mismatch"
    if action_edge and not spectral:
        return "action_edge_dominance_spectral_deficit"
    if spectral and not (selected_ratio and edge):
        return "spectral_dominance_action_edge_deficit"
    if spectral_action:
        return "spectral_action_partial_tie_or_edge_deficit"
    if selected_ratio or edge or spectral:
        return "single_axis_dominance_only"
    return "no_continuous_coordinate_dominance"


def _dominance_record(target: pd.Series, generated: pd.Series) -> dict[str, object]:
    target_tie = _finite_float(target.get("root_tie_rank_median_fraction", math.nan))
    generated_tie = _finite_float(generated.get("root_tie_rank_median_fraction", math.nan))
    target_action = _safe_log1p(target.get("root_sibling_selected_ratio", math.nan))
    generated_action = _safe_log1p(generated.get("root_sibling_selected_ratio", math.nan))
    target_edge = _safe_log1p(target.get("root_edge_path_statistic_margin", math.nan))
    generated_edge = _safe_log1p(generated.get("root_edge_path_statistic_margin", math.nan))
    target_spectral = _safe_log(
        target.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
    )
    generated_spectral = _safe_log(
        generated.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
    )

    tie_deficit = _deficit(target=target_tie, generated=generated_tie, missing_value=1.0)
    action_deficit = _deficit(target=target_action, generated=generated_action)
    edge_deficit = _deficit(target=target_edge, generated=generated_edge)
    spectral_deficit = _deficit(target=target_spectral, generated=generated_spectral)

    tie_dominates = tie_deficit <= 0.0
    selected_ratio_dominates = action_deficit <= 0.0
    edge_dominates = edge_deficit <= 0.0
    spectral_dominates = spectral_deficit <= 0.0
    action_edge_dominates = bool(selected_ratio_dominates and edge_dominates)
    spectral_action_dominates = bool(selected_ratio_dominates and spectral_dominates)
    full_dominates = bool(
        tie_dominates and selected_ratio_dominates and edge_dominates and spectral_dominates
    )
    bandwidth_status = row_bandwidth_gap_status(target, generated)
    bandwidth_measured_match = bandwidth_status == "bandwidth_band_match"
    score = (
        tie_deficit
        + min(action_deficit, 10.0)
        + min(edge_deficit, 10.0)
        + min(spectral_deficit, 10.0)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "target_case_id": _string_value(target, "case_id"),
        "proposal_family": _string_value(generated, "proposal_family"),
        "best_generated_case_id": _string_value(generated, "case_id"),
        "target_tie_fraction": target_tie,
        "generated_tie_fraction": generated_tie,
        "tie_fraction_deficit": tie_deficit,
        "target_selected_ratio_log": target_action,
        "generated_selected_ratio_log": generated_action,
        "selected_ratio_log_deficit": action_deficit,
        "target_edge_log": target_edge,
        "generated_edge_log": generated_edge,
        "edge_log_deficit": edge_deficit,
        "target_spectral_log": target_spectral,
        "generated_spectral_log": generated_spectral,
        "spectral_log_deficit": spectral_deficit,
        "tie_dominates": tie_dominates,
        "selected_ratio_dominates": selected_ratio_dominates,
        "edge_dominates": edge_dominates,
        "spectral_dominates": spectral_dominates,
        "action_edge_dominates": action_edge_dominates,
        "spectral_action_dominates": spectral_action_dominates,
        "full_continuous_dominates": full_dominates,
        "bandwidth_measured_match": bandwidth_measured_match,
        "bandwidth_gap_status": bandwidth_status,
        "dominance_deficit_score": float(score),
        "dominance_pattern": _dominance_pattern(
            full=full_dominates,
            spectral_action=spectral_action_dominates,
            action_edge=action_edge_dominates,
            spectral=spectral_dominates,
            selected_ratio=selected_ratio_dominates,
            edge=edge_dominates,
            bandwidth_status=bandwidth_status,
        ),
    }


def build_root_tie_rank_spectral_action_dominance_rows(
    combined_feasibility_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return best continuous-dominance row for each target and proposal family."""
    targets, generated = partition_target_and_generated_rows(
        combined_feasibility_rows,
        required_columns={
            "case_id",
            "calibration_role",
            "proposal_family",
            "root_bandwidth_reopen_band",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        label="combined feasibility rows",
    )
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        for family, family_rows in generated.groupby("proposal_family", sort=True):
            candidate_records = [
                _dominance_record(target, generated_row)
                for _, generated_row in family_rows.iterrows()
            ]
            if not candidate_records:
                continue
            best = min(
                candidate_records,
                key=lambda record: (
                    float(record["dominance_deficit_score"]),
                    str(record["best_generated_case_id"]),
                ),
            )
            best["proposal_family"] = str(family)
            records.append(best)
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_tie_rank_spectral_action_dominance_rows(
    dominance_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return family summary over continuous dominance rows."""
    if dominance_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for family, group in dominance_rows.groupby("proposal_family", sort=True):
        full_count = int(group["full_continuous_dominates"].sum())
        spectral_action_count = int(group["spectral_action_dominates"].sum())
        action_edge_count = int(group["action_edge_dominates"].sum())
        spectral_count = int(group["spectral_dominates"].sum())
        action_count = int(group["selected_ratio_dominates"].sum())
        edge_count = int(group["edge_dominates"].sum())
        bandwidth_match = int(group["bandwidth_measured_match"].sum())
        if full_count > 0 and bandwidth_match > 0:
            status = "full_continuous_and_bandwidth_dominance_observed"
        elif full_count > 0:
            status = "full_continuous_dominance_bandwidth_unmeasured"
        elif action_edge_count > 0 and spectral_count > 0 and spectral_action_count == 0:
            status = "separable_action_edge_and_spectral_no_joint_dominance"
        elif action_edge_count > 0:
            status = "action_edge_dominance_spectral_deficit"
        elif spectral_count > 0:
            status = "spectral_dominance_action_edge_deficit"
        else:
            status = "no_continuous_coordinate_dominance"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "proposal_family": str(family),
                "target_count": int(group.shape[0]),
                "full_continuous_dominance_count": full_count,
                "spectral_action_dominance_count": spectral_action_count,
                "action_edge_dominance_count": action_edge_count,
                "spectral_dominance_count": spectral_count,
                "selected_ratio_dominance_count": action_count,
                "edge_dominance_count": edge_count,
                "tie_dominance_count": int(group["tie_dominates"].sum()),
                "bandwidth_measured_match_count": bandwidth_match,
                "median_dominance_deficit_score": float(group["dominance_deficit_score"].median()),
                "min_dominance_deficit_score": float(group["dominance_deficit_score"].min()),
                "median_spectral_log_deficit": float(group["spectral_log_deficit"].median()),
                "median_selected_ratio_log_deficit": float(
                    group["selected_ratio_log_deficit"].median()
                ),
                "summary_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def evaluate_root_tie_rank_spectral_action_dominance_panel(
    config: RootTieRankSpectralActionDominanceConfig,
) -> dict[str, pd.DataFrame]:
    """Read proposal frontier rows and return continuous dominance tables."""
    combined = pd.read_csv(config.proposal_feasibility_rows_path)
    rows = build_root_tie_rank_spectral_action_dominance_rows(combined)
    summary = summarize_root_tie_rank_spectral_action_dominance_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_tie_rank_spectral_action_dominance_panel(
    config: RootTieRankSpectralActionDominanceConfig,
) -> dict[str, Path]:
    """Run the continuous dominance panel and write outputs."""
    tables = evaluate_root_tie_rank_spectral_action_dominance_panel(config)
    return write_diagnostic_bundle(
        output_dir=config.output_dir,
        tables=tables,
        filenames={
            "rows": ROWS_OUTPUT,
            "summary": SUMMARY_OUTPUT,
        },
        manifest={
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "generated_by": GENERATED_BY,
            "config": config,
        },
        manifest_filename=MANIFEST_OUTPUT,
    )


def main() -> None:
    args = parse_proposal_panel_args(
        description=__doc__,
        default_rows_path=DEFAULT_PROPOSAL_FEASIBILITY_ROWS,
    )
    outputs = run_root_tie_rank_spectral_action_dominance_panel(
        RootTieRankSpectralActionDominanceConfig(
            output_dir=args.output_dir,
            proposal_feasibility_rows_path=args.proposal_feasibility_rows_path,
        )
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
