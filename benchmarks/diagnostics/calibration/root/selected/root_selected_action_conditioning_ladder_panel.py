"""Conditioning ladder for selected-root action gaps.

The root spectral-tail law is exact-stratum only: conservative p-values are
valid only when selected-null or external-null support matches
T, A, E, B, and H_u. This diagnostic asks a narrower question after nearest
support localized the hard roots to selected-ratio action:

How much support appears if we relax A while keeping the other root-tail
conditioning coordinates fixed?

Relaxed rows are diagnostic only and never produce production p-values.
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
from benchmarks.diagnostics.calibration.root.root_tail_values import finite_float
from benchmarks.diagnostics.calibration.root.selected.cli import (
    parse_action_support_panel_args,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
)
from benchmarks.diagnostics.calibration.root.selected.root_tail_action_support import (
    prepare_root_tail_action_support,
    root_tail_coordinates,
    string_value,
)

SCHEMA_VERSION = "root_selected_action_conditioning_ladder_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_action_conditioning_ladder_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.root.selected.root_selected_action_conditioning_ladder_panel"

DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_importance_external_null_topology_join_mild_accumulated"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)
ROWS_OUTPUT = "root_selected_action_conditioning_ladder_rows.csv"
SUMMARY_OUTPUT = "root_selected_action_conditioning_ladder_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

LADDER_LEVELS = (
    "exact_T_A_E_B_H",
    "relax_A_keep_T_E_B_H",
    "relax_A_B_keep_T_E_H",
    "relax_A_E_keep_T_B_H",
)

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "ladder_level",
    "target_root_tail_stratum_key",
    "target_tie_band",
    "target_action_band",
    "target_edge_band",
    "target_bandwidth_topology_status",
    "target_s_root_spectral_excess_log",
    "target_a_selected_ratio_action_log1p",
    "support_count",
    "spectral_exceedance_count",
    "action_ge_target_count",
    "min_abs_action_gap",
    "median_abs_action_gap",
    "best_support_case_id",
    "best_support_proposal_family",
    "best_support_action_log1p",
    "best_support_s_root_spectral_excess_log",
    "conservative_spectral_tail_p_value",
    "p_value_status",
    "production_inference_status",
    "conditioning_gap_interpretation",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "exact_supported_target_count",
    "action_relaxed_supported_target_count",
    "action_only_gap_target_count",
    "fail_closed_target_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedActionConditioningLadderConfig:
    """Input/output contract for selected-root action conditioning ladder."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated"


def _level_mask(
    support: pd.DataFrame,
    *,
    target: dict[str, object],
    level: str,
) -> pd.Series:
    base_tie = support["_tie_band"].eq(str(target["tie_band"]))
    base_edge = support["_edge_band"].eq(str(target["edge_band"]))
    base_bandwidth = support["_bandwidth"].eq(str(target["bandwidth"]))
    base_h = support["_h_u"].eq(str(target["h_u"]))
    base_action = support["_action_band"].eq(str(target["action_band"]))
    if level == "exact_T_A_E_B_H":
        return base_tie & base_action & base_edge & base_bandwidth & base_h
    if level == "relax_A_keep_T_E_B_H":
        return base_tie & base_edge & base_bandwidth & base_h
    if level == "relax_A_B_keep_T_E_H":
        return base_tie & base_edge & base_h
    if level == "relax_A_E_keep_T_B_H":
        return base_tie & base_bandwidth & base_h
    raise ValueError(f"Unsupported ladder level: {level!r}.")


def _conservative_p(exceedance_count: int, support_count: int) -> float:
    if support_count <= 0:
        return math.nan
    return float((int(exceedance_count) + 1) / (int(support_count) + 1))


def _interpret_level(level: str, support_count: int, exact_count: int) -> str:
    if level == "exact_T_A_E_B_H":
        return "exact_tail_support_available" if support_count else "exact_tail_support_missing"
    if level == "relax_A_keep_T_E_B_H" and support_count and not exact_count:
        return "selected_ratio_action_is_minimal_missing_coordinate"
    if support_count:
        return "relaxed_support_diagnostic_only"
    return "relaxed_support_missing"


def _build_level_row(
    *,
    target_case_id: str,
    target: dict[str, object],
    level: str,
    level_support: pd.DataFrame,
    exact_count: int,
) -> dict[str, object]:
    support_count = int(level_support.shape[0])
    if support_count:
        spectral_exceed = int(level_support["_s_root"].ge(float(target["s_root"])).sum())
        action_ge = int(level_support["_action_log1p"].ge(float(target["action"])).sum())
        action_gaps = (level_support["_action_log1p"] - float(target["action"])).abs()
        ranked = level_support.assign(_action_gap=action_gaps).sort_values(
            ["_action_gap", "case_id"],
            ascending=[True, True],
        )
        best = ranked.iloc[0]
        min_gap = float(action_gaps.min())
        median_gap = float(action_gaps.median())
    else:
        spectral_exceed = 0
        action_ge = 0
        best = {}
        min_gap = math.nan
        median_gap = math.nan
    exact_level = level == "exact_T_A_E_B_H"
    p_value = _conservative_p(spectral_exceed, support_count) if exact_level else math.nan
    p_status = (
        "exact_conservative_empirical_tail"
        if exact_level and support_count
        else "exact_tail_support_missing"
        if exact_level
        else "relaxed_conditioning_not_calibration"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "target_case_id": target_case_id,
        "ladder_level": level,
        "target_root_tail_stratum_key": target["stratum"],
        "target_tie_band": target["tie_band"],
        "target_action_band": target["action_band"],
        "target_edge_band": target["edge_band"],
        "target_bandwidth_topology_status": target["bandwidth"],
        "target_s_root_spectral_excess_log": target["s_root"],
        "target_a_selected_ratio_action_log1p": target["action"],
        "support_count": support_count,
        "spectral_exceedance_count": spectral_exceed,
        "action_ge_target_count": action_ge,
        "min_abs_action_gap": min_gap,
        "median_abs_action_gap": median_gap,
        "best_support_case_id": string_value(best, "case_id"),
        "best_support_proposal_family": string_value(best, "proposal_family"),
        "best_support_action_log1p": finite_float(
            best.get("_action_log1p", math.nan) if support_count else math.nan
        ),
        "best_support_s_root_spectral_excess_log": finite_float(
            best.get("_s_root", math.nan) if support_count else math.nan
        ),
        "conservative_spectral_tail_p_value": p_value,
        "p_value_status": p_status,
        "production_inference_status": (
            "exact_support_available_defer_to_root_tail_panel"
            if exact_count
            else "fail_closed_action_ladder_diagnostic_only"
        ),
        "conditioning_gap_interpretation": _interpret_level(
            level,
            support_count,
            exact_count,
        ),
    }


def build_root_selected_action_conditioning_ladder_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated",
) -> pd.DataFrame:
    """Build exact and relaxed action-conditioning support rows."""
    targets, support = prepare_root_tail_action_support(
        joined_feasibility_rows,
        h_u_population_law_status=h_u_population_law_status,
    )
    records: list[dict[str, object]] = []
    for _, target_row in targets.sort_values("case_id").iterrows():
        target_case_id = string_value(target_row, "case_id")
        target = root_tail_coordinates(
            target_row,
            h_u_population_law_status=h_u_population_law_status,
        )
        exact_support = support.loc[_level_mask(support, target=target, level="exact_T_A_E_B_H")]
        exact_count = int(exact_support.shape[0])
        for level in LADDER_LEVELS:
            level_support = support.loc[_level_mask(support, target=target, level=level)]
            records.append(
                _build_level_row(
                    target_case_id=target_case_id,
                    target=target,
                    level=level,
                    level_support=level_support,
                    exact_count=exact_count,
                )
            )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_action_conditioning_ladder_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize target-level exact versus action-relaxed support."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    exact = rows.loc[rows["ladder_level"].eq("exact_T_A_E_B_H")]
    relaxed_a = rows.loc[rows["ladder_level"].eq("relax_A_keep_T_E_B_H")]
    exact_supported = int(exact["support_count"].gt(0).sum())
    relaxed_supported = int(relaxed_a["support_count"].gt(0).sum())
    exact_by_target = exact.set_index("target_case_id")["support_count"].gt(0)
    relaxed_by_target = relaxed_a.set_index("target_case_id")["support_count"].gt(0)
    action_only = int((~exact_by_target & relaxed_by_target).sum())
    fail_closed = int(exact["production_inference_status"].str.contains("fail_closed").sum())
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(exact.shape[0]),
                "exact_supported_target_count": exact_supported,
                "action_relaxed_supported_target_count": relaxed_supported,
                "action_only_gap_target_count": action_only,
                "fail_closed_target_count": fail_closed,
                "summary_status": (
                    "selected_ratio_action_conditioning_gap_localized"
                    if action_only
                    else "action_relaxation_does_not_restore_support"
                    if fail_closed
                    else "all_targets_exact_supported"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_action_conditioning_ladder_panel(
    config: RootSelectedActionConditioningLadderConfig,
) -> dict[str, pd.DataFrame]:
    joined = pd.read_csv(config.joined_feasibility_rows_path)
    rows = build_root_selected_action_conditioning_ladder_rows(
        joined_feasibility_rows=joined,
        h_u_population_law_status=config.h_u_population_law_status,
    )
    summary = summarize_root_selected_action_conditioning_ladder_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_action_conditioning_ladder_panel(
    config: RootSelectedActionConditioningLadderConfig,
) -> dict[str, Path]:
    tables = evaluate_root_selected_action_conditioning_ladder_panel(config)
    return write_diagnostic_bundle(
        output_dir=config.output_dir,
        tables=tables,
        filenames={"rows": ROWS_OUTPUT, "summary": SUMMARY_OUTPUT},
        manifest={
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "generated_by": GENERATED_BY,
            "config": config,
        },
        manifest_filename=MANIFEST_OUTPUT,
    )


def main() -> None:
    args = parse_action_support_panel_args(
        description=__doc__,
        default_rows_path=DEFAULT_JOINED_FEASIBILITY_ROWS,
        default_population_law_status="identity_mp_assumed_deformed_mp_unestimated",
    )
    outputs = run_root_selected_action_conditioning_ladder_panel(
        RootSelectedActionConditioningLadderConfig(
            output_dir=args.output_dir,
            joined_feasibility_rows_path=args.joined_feasibility_rows_path,
            h_u_population_law_status=str(args.h_u_population_law_status),
        )
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
