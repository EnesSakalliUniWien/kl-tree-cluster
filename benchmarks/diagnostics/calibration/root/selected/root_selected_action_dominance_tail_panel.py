"""One-sided action-dominance diagnostic for selected-root spectral tails.

Exact selected-root spectral-tail calibration still requires support in the
same T,A,E,B,H_u stratum. This diagnostic tests a narrower possible analytic
path: if support matches T,E,B,H_u and has selected-ratio action
A_support >= A_target, can it upper-bound or approximate the target spectral
tail?

The answer is diagnostic only. A one-sided action monotonicity theorem would be
needed before these rows could become a calibrated rescue rule.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.calibration.reporting import write_diagnostic_bundle
from benchmarks.diagnostics.calibration.root.root_tail_values import (
    finite_float,
    is_observed_target,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
)
from benchmarks.diagnostics.calibration.root.selected.root_tail_action_support import (
    annotate_root_tail_support,
    calibration_support_rows,
    root_tail_coordinates,
    string_value,
)

SCHEMA_VERSION = "root_selected_action_dominance_tail_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_action_dominance_tail_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.selected.root_selected_action_dominance_tail_panel"
)

DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_importance_external_null_topology_join_mild_accumulated"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)
ROWS_OUTPUT = "root_selected_action_dominance_tail_rows.csv"
SUMMARY_OUTPUT = "root_selected_action_dominance_tail_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_root_tail_stratum_key",
    "target_tie_band",
    "target_action_band",
    "target_edge_band",
    "target_bandwidth_topology_status",
    "target_s_root_spectral_excess_log",
    "target_a_selected_ratio_action_log1p",
    "exact_support_count",
    "action_dominating_support_count",
    "action_dominating_spectral_exceedance_count",
    "action_dominating_min_action_gap",
    "action_dominating_min_spectral_gap",
    "action_dominating_conservative_tail_p_value",
    "action_dominating_p_value_status",
    "best_action_dominating_support_case_id",
    "best_action_dominating_support_proposal_family",
    "best_action_dominating_support_action_log1p",
    "best_action_dominating_support_s_root_log",
    "production_inference_status",
    "next_mathematical_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "exact_supported_target_count",
    "action_dominating_supported_target_count",
    "action_dominating_no_spectral_exceedance_count",
    "fail_closed_target_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedActionDominanceTailConfig:
    """Input/output contract for one-sided action-dominance diagnostics."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joined-feasibility-rows-path",
        type=Path,
        default=DEFAULT_JOINED_FEASIBILITY_ROWS,
    )
    parser.add_argument(
        "--h-u-population-law-status",
        default="identity_mp_assumed_deformed_mp_unestimated",
    )
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def _exact_support_mask(support: pd.DataFrame, target: dict[str, object]) -> pd.Series:
    return (
        support["_tie_band"].eq(str(target["tie_band"]))
        & support["_action_band"].eq(str(target["action_band"]))
        & support["_edge_band"].eq(str(target["edge_band"]))
        & support["_bandwidth"].eq(str(target["bandwidth"]))
        & support["_h_u"].eq(str(target["h_u"]))
    )


def _action_dominance_mask(
    support: pd.DataFrame,
    target: dict[str, object],
) -> pd.Series:
    return (
        support["_tie_band"].eq(str(target["tie_band"]))
        & support["_edge_band"].eq(str(target["edge_band"]))
        & support["_bandwidth"].eq(str(target["bandwidth"]))
        & support["_h_u"].eq(str(target["h_u"]))
        & support["_action_log1p"].ge(float(target["action"]))
    )


def _conservative_p(exceedance_count: int, support_count: int) -> float:
    if support_count <= 0:
        return math.nan
    return float((int(exceedance_count) + 1) / (int(support_count) + 1))


def build_root_selected_action_dominance_tail_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated",
) -> pd.DataFrame:
    """Build one-sided action-dominance tail diagnostic rows."""
    _require_columns(
        joined_feasibility_rows,
        {
            "case_id",
            "data_role",
            "calibration_role",
            "proposal_family",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        "joined feasibility rows",
    )
    rows = joined_feasibility_rows.copy()
    if "root_bandwidth_reopen_band" not in rows.columns:
        rows["root_bandwidth_reopen_band"] = ""
    if "root_mixed_region_component" not in rows.columns:
        rows["root_mixed_region_component"] = "root_component_missing"
    targets = rows.loc[rows.apply(is_observed_target, axis=1)].copy()
    support = annotate_root_tail_support(
        calibration_support_rows(rows),
        h_u_population_law_status=h_u_population_law_status,
    )
    records: list[dict[str, object]] = []
    for _, target_row in targets.sort_values("case_id").iterrows():
        target_case_id = string_value(target_row, "case_id")
        target = root_tail_coordinates(
            target_row,
            h_u_population_law_status=h_u_population_law_status,
        )
        if support.empty:
            exact = support.copy()
            dominated = support.copy()
        else:
            exact = support.loc[_exact_support_mask(support, target)].copy()
            dominated = support.loc[_action_dominance_mask(support, target)].copy()
        exact_count = int(exact.shape[0])
        dominated_count = int(dominated.shape[0])
        if dominated_count:
            spectral_gaps = dominated["_s_root"] - float(target["s_root"])
            action_gaps = dominated["_action_log1p"] - float(target["action"])
            exceedances = int(spectral_gaps.ge(0.0).sum())
            ranked = dominated.assign(
                _spectral_gap=spectral_gaps,
                _action_gap=action_gaps,
            ).sort_values(
                ["_spectral_gap", "_action_gap", "case_id"], ascending=[False, True, True]
            )
            best = ranked.iloc[0]
            min_action_gap = float(action_gaps.min())
            min_spectral_gap = float(spectral_gaps.min())
            diagnostic_p = _conservative_p(exceedances, dominated_count)
        else:
            exceedances = 0
            best = {}
            min_action_gap = math.nan
            min_spectral_gap = math.nan
            diagnostic_p = math.nan
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": target_case_id,
                "target_root_tail_stratum_key": target["stratum"],
                "target_tie_band": target["tie_band"],
                "target_action_band": target["action_band"],
                "target_edge_band": target["edge_band"],
                "target_bandwidth_topology_status": target["bandwidth"],
                "target_s_root_spectral_excess_log": target["s_root"],
                "target_a_selected_ratio_action_log1p": target["action"],
                "exact_support_count": exact_count,
                "action_dominating_support_count": dominated_count,
                "action_dominating_spectral_exceedance_count": exceedances,
                "action_dominating_min_action_gap": min_action_gap,
                "action_dominating_min_spectral_gap": min_spectral_gap,
                "action_dominating_conservative_tail_p_value": diagnostic_p,
                "action_dominating_p_value_status": (
                    "diagnostic_one_sided_action_monotonicity_required"
                    if dominated_count
                    else "action_dominating_support_missing"
                ),
                "best_action_dominating_support_case_id": string_value(best, "case_id"),
                "best_action_dominating_support_proposal_family": string_value(
                    best,
                    "proposal_family",
                ),
                "best_action_dominating_support_action_log1p": finite_float(
                    best.get("_action_log1p", math.nan) if dominated_count else math.nan
                ),
                "best_action_dominating_support_s_root_log": finite_float(
                    best.get("_s_root", math.nan) if dominated_count else math.nan
                ),
                "production_inference_status": (
                    "exact_support_available_defer_to_root_tail_panel"
                    if exact_count
                    else "fail_closed_action_dominance_diagnostic_only"
                ),
                "next_mathematical_step": (
                    "prove_or_reject_one_sided_action_spectral_tail_monotonicity"
                    if dominated_count and not exact_count
                    else "use_exact_root_tail_panel"
                    if exact_count
                    else "generate_action_dominating_support_in_same_T_E_B_H"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_action_dominance_tail_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize one-sided action-dominance diagnostics."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    exact_supported = int(rows["exact_support_count"].gt(0).sum())
    dominated_supported = int(rows["action_dominating_support_count"].gt(0).sum())
    dominated_no_exceed = int(
        rows["action_dominating_support_count"]
        .gt(0)
        .mul(rows["action_dominating_spectral_exceedance_count"].eq(0))
        .sum()
    )
    fail_closed = int(
        rows["production_inference_status"]
        .astype(str)
        .eq("fail_closed_action_dominance_diagnostic_only")
        .sum()
    )
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "exact_supported_target_count": exact_supported,
                "action_dominating_supported_target_count": dominated_supported,
                "action_dominating_no_spectral_exceedance_count": dominated_no_exceed,
                "fail_closed_target_count": fail_closed,
                "summary_status": (
                    "one_sided_action_dominance_diagnostic_support_available"
                    if dominated_supported > exact_supported
                    else "one_sided_action_dominance_no_new_support"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_action_dominance_tail_panel(
    config: RootSelectedActionDominanceTailConfig,
) -> dict[str, pd.DataFrame]:
    joined = pd.read_csv(config.joined_feasibility_rows_path)
    rows = build_root_selected_action_dominance_tail_rows(
        joined_feasibility_rows=joined,
        h_u_population_law_status=config.h_u_population_law_status,
    )
    summary = summarize_root_selected_action_dominance_tail_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_action_dominance_tail_panel(
    config: RootSelectedActionDominanceTailConfig,
) -> dict[str, Path]:
    tables = evaluate_root_selected_action_dominance_tail_panel(config)
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
    args = parse_args()
    outputs = run_root_selected_action_dominance_tail_panel(
        RootSelectedActionDominanceTailConfig(
            output_dir=args.output_dir,
            joined_feasibility_rows_path=args.joined_feasibility_rows_path,
            h_u_population_law_status=str(args.h_u_population_law_status),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
