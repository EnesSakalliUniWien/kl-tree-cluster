"""Attempt same-geometry external support for selected-root spectral tails.

The selected-root spectral tail law can only produce a conservative p-value
when selected/external-null roots exist in the same root-tail stratum
``(C, T, A, E, B, H_u)``. This diagnostic runs the missing operational loop:

1. generate likelihood-ratio external-null roots targeted by selected
   tie-rank/action-edge geometry;
2. replay those generated matrices through selected-neighborhood topology;
3. join measured bandwidth/topology and deformed-MP ``H_u`` support back into
   the spectral-tail panel;
4. fail closed unless the existing tail-law code finds support in the same
   full stratum.

Rows remain diagnostic. This module does not create a production rescue rule.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.root.root_tail_values import (
    is_calibration_support,
    is_observed_target,
    lookup_numeric_by_key,
    root_tail_stratum_key,
    tail_excess_for_case,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_deformed_mp_edge_panel import (
    ROWS_OUTPUT as DEFORMED_ROWS_OUTPUT,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_deformed_mp_edge_panel import (
    SUMMARY_OUTPUT as DEFORMED_SUMMARY_OUTPUT,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_deformed_mp_edge_panel import (
    SUPPORT_ROWS_OUTPUT as DEFORMED_SUPPORT_ROWS_OUTPUT,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_deformed_mp_edge_panel import (
    build_root_selected_deformed_mp_edge_rows,
    build_root_support_deformed_mp_edge_rows,
    summarize_root_selected_deformed_mp_edge_rows,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    ROWS_OUTPUT as TAIL_ROWS_OUTPUT,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    SUMMARY_OUTPUT as TAIL_SUMMARY_OUTPUT,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    build_root_selected_spectral_tail_law_rows,
    summarize_root_selected_spectral_tail_law_rows,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_conditioned_coherent_topology_join import (
    build_conditioned_coherent_joined_feasibility_rows,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_generated_neighborhood_replay import (
    DISTRIBUTION_ROWS_OUTPUT,
    MEASURABILITY_ROWS_OUTPUT,
    MEASURABILITY_SUMMARY_OUTPUT,
    NODE_DECISIONS_OUTPUT,
    PVALUE_CASE_SUMMARY_OUTPUT,
    PVALUE_ROWS_OUTPUT,
    PVALUE_SUMMARY_OUTPUT,
    PVALUE_TAU_S_OUTPUT,
    RUN_ROWS_OUTPUT,
    TOPOLOGY_ROWS_OUTPUT,
    TOPOLOGY_SUMMARY_OUTPUT,
    build_generated_neighborhood_replay_tables,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_selected_null_simulation_pilot import (
    DEFAULT_OBSERVED_MIXED_ROWS,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_target_conditioned_importance_frontier import (
    FAILURES_OUTPUT as FRONTIER_FAILURES_OUTPUT,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_target_conditioned_importance_frontier import (
    GENERATED_ROWS_OUTPUT as FRONTIER_GENERATED_ROWS_OUTPUT,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_target_conditioned_importance_frontier import (
    IMPORTANCE_CORRELATED_TWO_FACTOR_EXTERNAL_NULL,
    TargetConditionedImportanceFrontierConfig,
    evaluate_target_conditioned_importance_frontier,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_target_conditioned_importance_frontier import (
    SUMMARY_OUTPUT as FRONTIER_SUMMARY_OUTPUT,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_target_conditioned_importance_frontier import (
    TARGET_ROWS_OUTPUT as FRONTIER_TARGET_ROWS_OUTPUT,
)

SCHEMA_VERSION = "root_selected_same_geometry_external_support_attempt/v1"
STUDY_ROLE = "diagnostic_root_selected_same_geometry_external_support_attempt"
GENERATED_BY = "benchmarks.diagnostics.calibration.root.selected.root_selected_same_geometry_external_support_attempt"

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_EXTERNAL_LAW_EQUATION_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_external_law_equation_mild_replay_v3_smoke"
    / "root_selected_external_law_equation_rows.csv"
)
DEFAULT_BASE_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_importance_external_null_topology_join_mild_hu_replay_v3_smoke"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)
DEFAULT_OBSERVED_ROOT_SUMMARY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_region_margins_overlap_case_family"
    / "root_selected_region_summary.csv"
)
DEFAULT_H_U_OBSERVABILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_h_u_observability_mild_accumulated"
    / "root_selected_h_u_observability_rows.csv"
)

DEFAULT_PROPOSAL_FAMILIES = (IMPORTANCE_CORRELATED_TWO_FACTOR_EXTERNAL_NULL,)
DEFAULT_TWO_BLOCK_GRID = (0.118, 0.122)
DEFAULT_SPIKE_FRACTION_GRID = (0.20,)
DEFAULT_SPIKE_DELTA_GRID = (0.08, 0.14)
DEFAULT_BLOCK_FRACTION_GRID = (0.25, 0.50)
DEFAULT_SEED_OFFSET = 1_960_000
DEFAULT_H_U_STATUS = "deformed_mp_edge_measured_support_side"
TAIL_ADMISSIBLE_DEFORMED_STATUS = "support_deformed_mp_edge_computed_diagnostic_only"

JOINED_ROWS_OUTPUT = "same_geometry_external_joined_feasibility_rows.csv"
ATTEMPT_ROWS_OUTPUT = "root_selected_same_geometry_external_support_attempt_rows.csv"
ATTEMPT_SUMMARY_OUTPUT = "root_selected_same_geometry_external_support_attempt_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

EXTERNAL_LAW_TARGET_STATUSES = {
    "requires_new_same_stratum_nonzero_s_h_u_support",
    "moment_equation_feasible_but_still_diagnostic_only",
}
EXTERNAL_LAW_NEXT_GENERATOR_REQUIREMENT = (
    "generate_same_T_A_E_B_Hu_roots_with_nonzero_deformed_spectral_excess"
)

ATTEMPT_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_root_tail_stratum_key",
    "external_law_equation_status",
    "target_s_h_u_excess_log",
    "generated_candidate_count",
    "pre_topology_stratum_hit_count",
    "replay_completed_candidate_count",
    "same_tail_stratum_support_count",
    "new_same_tail_stratum_support_count",
    "same_tail_positive_s_h_u_support_count",
    "new_same_tail_positive_s_h_u_support_count",
    "same_tail_exceedance_count",
    "new_same_tail_exceedance_count",
    "best_new_same_tail_s_h_u_excess_log",
    "tail_panel_support_count",
    "tail_panel_exceedance_count",
    "tail_panel_effective_sample_size",
    "tail_panel_conservative_p_value",
    "tail_panel_inference_status",
    "attempt_status",
    "next_mathematical_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "generated_candidate_count",
    "replay_completed_candidate_count",
    "pre_topology_supported_target_count",
    "same_tail_supported_target_count",
    "new_same_tail_supported_target_count",
    "new_nonzero_s_h_u_supported_target_count",
    "new_tail_exceedance_supported_target_count",
    "calibrated_tail_count",
    "fail_closed_tail_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedSameGeometryExternalSupportAttemptConfig:
    """Input/output contract for the same-geometry support attempt."""

    output_dir: Path
    external_law_equation_rows_path: Path = DEFAULT_EXTERNAL_LAW_EQUATION_ROWS
    observed_mixed_region_rows_path: Path = DEFAULT_OBSERVED_MIXED_ROWS
    base_feasibility_rows_path: Path = DEFAULT_BASE_FEASIBILITY_ROWS
    observed_root_summary_rows_path: Path | None = DEFAULT_OBSERVED_ROOT_SUMMARY_ROWS
    h_u_observability_rows_path: Path = DEFAULT_H_U_OBSERVABILITY_ROWS
    suite: str = "full"
    target_case_ids: tuple[str, ...] = ()
    proposal_families: tuple[str, ...] = DEFAULT_PROPOSAL_FAMILIES
    two_block_delta_grid: tuple[float, ...] = DEFAULT_TWO_BLOCK_GRID
    spike_feature_fraction_grid: tuple[float, ...] = DEFAULT_SPIKE_FRACTION_GRID
    spike_delta_grid: tuple[float, ...] = DEFAULT_SPIKE_DELTA_GRID
    block_fraction_grid: tuple[float, ...] = DEFAULT_BLOCK_FRACTION_GRID
    replicates_per_setting: int = 1
    attempts_per_setting: int = 1
    accept_target_pre_topology_stratum: bool = False
    max_replay_cases: int | None = None
    candidate_only: bool = False
    reference_tau_s: float = 20.0
    require_spectral_flow: bool = False
    sibling_alpha: float = 0.01
    edge_alpha: float = 0.001
    tree_linkage_method: str = "average"
    method_id: str = "fixed_coordinate_conditional_topology_diagnostic_v1"
    h_u_population_law_status: str = DEFAULT_H_U_STATUS
    seed_offset: int = DEFAULT_SEED_OFFSET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--external-law-equation-rows-path",
        type=Path,
        default=DEFAULT_EXTERNAL_LAW_EQUATION_ROWS,
    )
    parser.add_argument(
        "--observed-mixed-region-rows-path",
        type=Path,
        default=DEFAULT_OBSERVED_MIXED_ROWS,
    )
    parser.add_argument(
        "--base-feasibility-rows-path",
        type=Path,
        default=DEFAULT_BASE_FEASIBILITY_ROWS,
    )
    parser.add_argument(
        "--observed-root-summary-rows-path",
        type=Path,
        default=DEFAULT_OBSERVED_ROOT_SUMMARY_ROWS,
    )
    parser.add_argument(
        "--h-u-observability-rows-path",
        type=Path,
        default=DEFAULT_H_U_OBSERVABILITY_ROWS,
    )
    parser.add_argument("--suite", default="full")
    parser.add_argument("--target-case-ids", default=None)
    parser.add_argument(
        "--proposal-families",
        default=",".join(DEFAULT_PROPOSAL_FAMILIES),
    )
    parser.add_argument(
        "--two-block-delta-grid",
        default=",".join(str(value) for value in DEFAULT_TWO_BLOCK_GRID),
    )
    parser.add_argument(
        "--spike-feature-fraction-grid",
        default=",".join(str(value) for value in DEFAULT_SPIKE_FRACTION_GRID),
    )
    parser.add_argument(
        "--spike-delta-grid",
        default=",".join(str(value) for value in DEFAULT_SPIKE_DELTA_GRID),
    )
    parser.add_argument(
        "--block-fraction-grid",
        default=",".join(str(value) for value in DEFAULT_BLOCK_FRACTION_GRID),
    )
    parser.add_argument("--replicates-per-setting", type=int, default=1)
    parser.add_argument("--attempts-per-setting", type=int, default=1)
    parser.add_argument("--accept-target-pre-topology-stratum", action="store_true")
    parser.add_argument("--max-replay-cases", type=int, default=None)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--reference-tau-s", type=float, default=20.0)
    parser.add_argument("--require-spectral-flow", action="store_true")
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument("--tree-linkage-method", default="average")
    parser.add_argument(
        "--method-id",
        default="fixed_coordinate_conditional_topology_diagnostic_v1",
    )
    parser.add_argument(
        "--h-u-population-law-status",
        default=DEFAULT_H_U_STATUS,
    )
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedSameGeometryExternalSupportAttemptConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_csv_list(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_float_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("grid must contain at least one value.")
    return values


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


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


def _select_external_law_target_case_ids(
    external_law_equation_rows: pd.DataFrame,
    *,
    explicit_target_case_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return target roots that still need an external same-geometry law."""
    if explicit_target_case_ids:
        return explicit_target_case_ids
    _require_columns(
        external_law_equation_rows,
        {"target_case_id", "external_law_equation_status"},
        "external-law equation rows",
    )
    rows = external_law_equation_rows.copy()
    status_mask = (
        rows["external_law_equation_status"].astype(str).isin(EXTERNAL_LAW_TARGET_STATUSES)
    )
    if "next_generator_requirement" in rows.columns:
        status_mask = status_mask | rows["next_generator_requirement"].astype(str).eq(
            EXTERNAL_LAW_NEXT_GENERATOR_REQUIREMENT
        )
    selected = rows.loc[status_mask, "target_case_id"].astype(str).drop_duplicates()
    return tuple(selected.tolist())


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _deformed_excess_by_case(
    *,
    deformed_rows: pd.DataFrame,
    deformed_support_rows: pd.DataFrame,
) -> dict[str, float]:
    return {
        **lookup_numeric_by_key(
            deformed_rows,
            key_column="target_case_id",
            value_column="s_root_deformed_excess_log",
        ),
        **lookup_numeric_by_key(
            deformed_support_rows,
            key_column="case_id",
            value_column="s_root_deformed_excess_log",
        ),
    }


def _tail_admissible_joined_rows(
    *,
    joined_rows: pd.DataFrame,
    deformed_support_rows: pd.DataFrame,
    h_u_population_law_status: str,
) -> pd.DataFrame:
    """Exclude calibration-support rows whose deformed ``H_u`` was not computed."""
    if str(h_u_population_law_status) != DEFAULT_H_U_STATUS:
        return joined_rows.copy()
    if joined_rows.empty:
        return joined_rows.copy()
    if "case_id" not in joined_rows.columns:
        return joined_rows.copy()
    computed_cases = set()
    if (
        not deformed_support_rows.empty
        and "case_id" in deformed_support_rows.columns
        and "support_deformed_mp_edge_status" in deformed_support_rows.columns
    ):
        computed_cases = set(
            deformed_support_rows.loc[
                deformed_support_rows["support_deformed_mp_edge_status"]
                .astype(str)
                .eq(TAIL_ADMISSIBLE_DEFORMED_STATUS),
                "case_id",
            ].astype(str)
        )
    rows = joined_rows.copy()
    if "tail_support_exclusion_reason" not in rows.columns:
        rows["tail_support_exclusion_reason"] = ""
    for index, row in rows.iterrows():
        if is_observed_target(row) or not is_calibration_support(row):
            continue
        case_id = string_value(row, "case_id")
        if case_id in computed_cases:
            continue
        old_data_role = string_value(row, "data_role")
        old_calibration_role = string_value(row, "calibration_role")
        rows.at[index, "data_role"] = f"{old_data_role}_h_u_missing_not_tail_support"
        rows.at[index, "calibration_role"] = f"{old_calibration_role}_h_u_missing_not_tail_support"
        rows.at[index, "tail_support_exclusion_reason"] = (
            "deformed_h_u_edge_missing_for_support_row"
        )
    return rows


def _same_tail_support_rows(
    *,
    target: pd.Series,
    joined_rows: pd.DataFrame,
    h_u_population_law_status: str,
) -> pd.DataFrame:
    if joined_rows.empty:
        return pd.DataFrame()
    target_key = string_value(target, "root_tail_stratum_key")
    if not target_key:
        target_key = root_tail_stratum_key(
            target=target,
            h_u_population_law_status=h_u_population_law_status,
        )
    candidates = joined_rows.loc[~joined_rows.apply(is_observed_target, axis=1)].copy()
    if candidates.empty:
        return candidates
    candidates["root_tail_stratum_key"] = candidates.apply(
        lambda row: root_tail_stratum_key(
            target=row,
            h_u_population_law_status=h_u_population_law_status,
        ),
        axis=1,
    )
    candidates = candidates.loc[candidates["root_tail_stratum_key"].eq(target_key)]
    return candidates.loc[candidates.apply(is_calibration_support, axis=1)].copy()


def _generated_count_by_target(generated_rows: pd.DataFrame) -> dict[str, int]:
    if generated_rows.empty or "conditioning_target_case_id" not in generated_rows.columns:
        return {}
    counts = generated_rows.groupby("conditioning_target_case_id", dropna=False).size().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _replay_completed_count_by_target(
    *,
    replay_rows: pd.DataFrame,
    generated_rows: pd.DataFrame,
) -> dict[str, int]:
    if replay_rows.empty or generated_rows.empty:
        return {}
    if "case_id" not in replay_rows.columns or "case_id" not in generated_rows.columns:
        return {}
    lookup = generated_rows[["case_id", "conditioning_target_case_id"]].drop_duplicates("case_id")
    joined = replay_rows.merge(lookup, on="case_id", how="left")
    if "run_status" not in joined.columns:
        return {}
    completed = joined.loc[joined["run_status"].astype(str).eq("ok")]
    counts = completed.groupby("conditioning_target_case_id", dropna=False).size().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _pre_topology_hit_count_by_target(target_rows: pd.DataFrame) -> dict[str, int]:
    if target_rows.empty:
        return {}
    if "target_case_id" not in target_rows.columns:
        return {}
    if "pre_topology_stratum_hit_count" not in target_rows.columns:
        return {}
    grouped = target_rows.groupby("target_case_id")["pre_topology_stratum_hit_count"].sum()
    return {str(key): int(value) for key, value in grouped.to_dict().items()}


def _target_row_by_case(frame: pd.DataFrame, key_column: str) -> dict[str, pd.Series]:
    if frame.empty or key_column not in frame.columns:
        return {}
    return {str(row[key_column]): row for _, row in frame.iterrows()}


def build_same_geometry_external_support_attempt_rows(
    *,
    external_law_equation_rows: pd.DataFrame,
    generated_rows: pd.DataFrame,
    target_conditioning_rows: pd.DataFrame,
    replay_rows: pd.DataFrame,
    joined_rows: pd.DataFrame,
    deformed_rows: pd.DataFrame,
    deformed_support_rows: pd.DataFrame,
    tail_rows: pd.DataFrame,
    target_case_ids: tuple[str, ...],
    h_u_population_law_status: str = DEFAULT_H_U_STATUS,
) -> pd.DataFrame:
    """Summarize whether the attempt created admissible same-stratum support."""
    equation_by_case = _target_row_by_case(
        external_law_equation_rows,
        "target_case_id",
    )
    tail_by_case = _target_row_by_case(tail_rows, "target_case_id")
    generated_counts = _generated_count_by_target(generated_rows)
    replay_counts = _replay_completed_count_by_target(
        replay_rows=replay_rows,
        generated_rows=generated_rows,
    )
    pre_topology_hits = _pre_topology_hit_count_by_target(target_conditioning_rows)
    deformed_lookup = _deformed_excess_by_case(
        deformed_rows=deformed_rows,
        deformed_support_rows=deformed_support_rows,
    )
    new_generated_case_ids = (
        set(generated_rows["case_id"].astype(str))
        if not generated_rows.empty and "case_id" in generated_rows.columns
        else set()
    )
    records: list[dict[str, object]] = []
    for target_id in target_case_ids:
        target_id = str(target_id)
        tail = tail_by_case.get(target_id)
        equation = equation_by_case.get(target_id)
        if tail is None:
            target_s = (
                finite_float(equation.get("target_s_h_u_excess_log", math.nan))
                if equation is not None
                else math.nan
            )
            support = pd.DataFrame()
            target_key = (
                string_value(equation, "target_root_tail_stratum_key")
                if equation is not None
                else ""
            )
        else:
            target_s = finite_float(tail.get("s_root_deformed_excess_log", math.nan))
            if not math.isfinite(target_s):
                target_s = finite_float(tail.get("s_root_spectral_excess_log", math.nan))
            support = _same_tail_support_rows(
                target=tail,
                joined_rows=joined_rows,
                h_u_population_law_status=h_u_population_law_status,
            )
            target_key = string_value(tail, "root_tail_stratum_key")
        support_s = (
            support.apply(
                lambda row: tail_excess_for_case(
                    row,
                    deformed_excess_by_case=deformed_lookup,
                )[0],
                axis=1,
            )
            if not support.empty
            else pd.Series(dtype=float)
        )
        new_mask = (
            support["case_id"].astype(str).isin(new_generated_case_ids).to_numpy()
            if not support.empty and "case_id" in support.columns
            else np.asarray([], dtype=bool)
        )
        positive_mask = support_s.gt(0.0).to_numpy(dtype=bool)
        exceedance_mask = (
            support_s.ge(target_s).to_numpy(dtype=bool)
            if math.isfinite(target_s)
            else np.zeros(int(support_s.shape[0]), dtype=bool)
        )
        new_s = support_s[new_mask] if support_s.shape[0] else pd.Series(dtype=float)
        new_positive = int(np.sum(new_mask & positive_mask))
        new_exceedance = int(np.sum(new_mask & exceedance_mask))
        same_tail_support_count = int(support.shape[0])
        new_support_count = int(np.sum(new_mask))
        tail_status = (
            string_value(tail, "root_tail_inference_status")
            if tail is not None
            else "root_tail_panel_missing"
        )
        if new_exceedance > 0:
            attempt_status = "new_same_geometry_tail_exceedance_support_found"
            next_step = "increase_same_stratum_external_null_support_for_precision"
        elif new_positive > 0:
            attempt_status = "new_same_geometry_nonzero_s_h_u_support_found"
            next_step = "extend_tail_support_until_target_exceedances_are_resolved"
        elif new_support_count > 0:
            attempt_status = "new_same_geometry_support_but_zero_s_h_u"
            next_step = "generate_same_geometry_roots_with_positive_s_h_u"
        elif same_tail_support_count > 0:
            attempt_status = "existing_same_geometry_support_only"
            next_step = "use_existing_tail_panel_or_add_external_support"
        elif int(pre_topology_hits.get(target_id, 0)) > 0:
            attempt_status = "pre_topology_hits_lost_after_b_h_u_join"
            next_step = "condition_generator_on_measured_b_and_h_u_not_only_pre_topology"
        elif int(generated_counts.get(target_id, 0)) > 0:
            attempt_status = "same_geometry_support_missing_after_attempt"
            next_step = "sharpen_selected_root_external_proposal_geometry"
        else:
            attempt_status = "no_external_candidates_generated"
            next_step = "generate_target_conditioned_external_null_candidates"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": target_id,
                "target_root_tail_stratum_key": target_key,
                "external_law_equation_status": (
                    string_value(equation, "external_law_equation_status")
                    if equation is not None
                    else "external_law_equation_missing"
                ),
                "target_s_h_u_excess_log": target_s,
                "generated_candidate_count": int(generated_counts.get(target_id, 0)),
                "pre_topology_stratum_hit_count": int(pre_topology_hits.get(target_id, 0)),
                "replay_completed_candidate_count": int(replay_counts.get(target_id, 0)),
                "same_tail_stratum_support_count": same_tail_support_count,
                "new_same_tail_stratum_support_count": new_support_count,
                "same_tail_positive_s_h_u_support_count": int(np.sum(positive_mask)),
                "new_same_tail_positive_s_h_u_support_count": new_positive,
                "same_tail_exceedance_count": int(np.sum(exceedance_mask)),
                "new_same_tail_exceedance_count": new_exceedance,
                "best_new_same_tail_s_h_u_excess_log": (
                    float(new_s.max()) if not new_s.empty else math.nan
                ),
                "tail_panel_support_count": int(
                    finite_float(tail.get("selected_null_support_count", 0))
                    if tail is not None
                    else 0
                ),
                "tail_panel_exceedance_count": int(
                    finite_float(tail.get("selected_null_exceedance_count", 0))
                    if tail is not None
                    else 0
                ),
                "tail_panel_effective_sample_size": (
                    finite_float(
                        tail.get("selected_null_importance_effective_sample_size", math.nan)
                    )
                    if tail is not None
                    else math.nan
                ),
                "tail_panel_conservative_p_value": (
                    finite_float(tail.get("conservative_spectral_tail_p_value", math.nan))
                    if tail is not None
                    else math.nan
                ),
                "tail_panel_inference_status": tail_status,
                "attempt_status": attempt_status,
                "next_mathematical_step": next_step,
            }
        )
    return pd.DataFrame.from_records(records, columns=ATTEMPT_ROW_COLUMNS)


def summarize_same_geometry_external_support_attempt_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return compact support-attempt counts."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    calibrated = (
        rows["tail_panel_inference_status"]
        .astype(str)
        .eq("calibrated_selected_root_spectral_tail_available")
    )
    fail_closed = (
        rows["tail_panel_inference_status"]
        .astype(str)
        .eq("fail_closed_selected_root_spectral_tail_support_missing")
    )
    new_nonzero = rows["new_same_tail_positive_s_h_u_support_count"].astype(int).gt(0)
    new_exceedance = rows["new_same_tail_exceedance_count"].astype(int).gt(0)
    if bool(new_exceedance.any()):
        status = "new_same_geometry_tail_exceedance_support_observed"
    elif bool(new_nonzero.any()):
        status = "new_same_geometry_nonzero_s_h_u_support_observed"
    elif bool(fail_closed.any()):
        status = "same_geometry_nonzero_s_h_u_support_missing_fail_closed"
    else:
        status = "same_geometry_tail_support_available_without_new_support"
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "generated_candidate_count": int(rows["generated_candidate_count"].sum()),
                "replay_completed_candidate_count": int(
                    rows["replay_completed_candidate_count"].sum()
                ),
                "pre_topology_supported_target_count": int(
                    rows["pre_topology_stratum_hit_count"].astype(int).gt(0).sum()
                ),
                "same_tail_supported_target_count": int(
                    rows["same_tail_stratum_support_count"].astype(int).gt(0).sum()
                ),
                "new_same_tail_supported_target_count": int(
                    rows["new_same_tail_stratum_support_count"].astype(int).gt(0).sum()
                ),
                "new_nonzero_s_h_u_supported_target_count": int(new_nonzero.sum()),
                "new_tail_exceedance_supported_target_count": int(new_exceedance.sum()),
                "calibrated_tail_count": int(calibrated.sum()),
                "fail_closed_tail_count": int(fail_closed.sum()),
                "summary_status": status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_same_geometry_external_support_attempt(
    config: RootSelectedSameGeometryExternalSupportAttemptConfig,
) -> dict[str, pd.DataFrame]:
    """Run generation/replay/join in memory and return diagnostic tables."""
    external_law_rows = pd.read_csv(
        config.external_law_equation_rows_path,
        low_memory=False,
    )
    target_case_ids = _select_external_law_target_case_ids(
        external_law_rows,
        explicit_target_case_ids=tuple(config.target_case_ids),
    )
    frontier_tables = evaluate_target_conditioned_importance_frontier(
        TargetConditionedImportanceFrontierConfig(
            output_dir=config.output_dir,
            observed_mixed_region_rows_path=config.observed_mixed_region_rows_path,
            tail_rows_path=None,
            suite=str(config.suite),
            target_case_ids=target_case_ids,
            proposal_families=tuple(config.proposal_families),
            two_block_delta_grid=tuple(config.two_block_delta_grid),
            spike_feature_fraction_grid=tuple(config.spike_feature_fraction_grid),
            spike_delta_grid=tuple(config.spike_delta_grid),
            block_fraction_grid=tuple(config.block_fraction_grid),
            replicates_per_setting=int(config.replicates_per_setting),
            attempts_per_setting=int(config.attempts_per_setting),
            accept_target_pre_topology_stratum=bool(config.accept_target_pre_topology_stratum),
            seed_offset=int(config.seed_offset),
        )
    )
    generated_rows = frontier_tables["generated_rows"]
    replay_tables = build_generated_neighborhood_replay_tables(
        proposal_rows=generated_rows,
        generated_matrix_dir=config.output_dir / "generated_target_conditioned_matrices",
        method_id=str(config.method_id),
        sibling_alpha=float(config.sibling_alpha),
        edge_alpha=float(config.edge_alpha),
        tree_linkage_method=str(config.tree_linkage_method),
        max_cases=config.max_replay_cases,
        candidate_only=bool(config.candidate_only),
        reference_tau_s=float(config.reference_tau_s),
        require_spectral_flow=bool(config.require_spectral_flow),
    )
    base_rows = pd.read_csv(config.base_feasibility_rows_path, low_memory=False)
    observed_summary = _read_optional_csv(config.observed_root_summary_rows_path)
    if generated_rows.empty:
        joined_rows = base_rows.copy()
    else:
        joined_rows = build_conditioned_coherent_joined_feasibility_rows(
            base_feasibility_rows=base_rows,
            conditioned_generated_rows=generated_rows,
            conditioned_topology_rows=replay_tables["topology_rows"],
            observed_root_summary_rows=observed_summary,
        )
    h_u_rows = pd.read_csv(config.h_u_observability_rows_path, low_memory=False)
    deformed_rows = build_root_selected_deformed_mp_edge_rows(
        joined_feasibility_rows=joined_rows,
        h_u_observability_rows=h_u_rows,
    )
    deformed_support_rows = build_root_support_deformed_mp_edge_rows(
        joined_feasibility_rows=joined_rows,
    )
    joined_tail_rows = _tail_admissible_joined_rows(
        joined_rows=joined_rows,
        deformed_support_rows=deformed_support_rows,
        h_u_population_law_status=str(config.h_u_population_law_status),
    )
    tail_rows = build_root_selected_spectral_tail_law_rows(
        joined_feasibility_rows=joined_tail_rows,
        deformed_mp_edge_rows=deformed_rows,
        deformed_mp_edge_support_rows=deformed_support_rows,
        h_u_population_law_status=str(config.h_u_population_law_status),
    )
    attempt_rows = build_same_geometry_external_support_attempt_rows(
        external_law_equation_rows=external_law_rows,
        generated_rows=generated_rows,
        target_conditioning_rows=frontier_tables["target_rows"],
        replay_rows=replay_tables["run_rows"],
        joined_rows=joined_tail_rows,
        deformed_rows=deformed_rows,
        deformed_support_rows=deformed_support_rows,
        tail_rows=tail_rows,
        target_case_ids=target_case_ids,
        h_u_population_law_status=str(config.h_u_population_law_status),
    )
    attempt_summary = summarize_same_geometry_external_support_attempt_rows(attempt_rows)
    deformed_summary = summarize_root_selected_deformed_mp_edge_rows(deformed_rows)
    tail_summary = summarize_root_selected_spectral_tail_law_rows(tail_rows)
    return {
        "frontier_generated_rows": generated_rows,
        "frontier_target_rows": frontier_tables["target_rows"],
        "frontier_summary": frontier_tables["summary"],
        "frontier_failures": frontier_tables["failures"],
        "replay_run_rows": replay_tables["run_rows"],
        "replay_node_decisions": replay_tables["node_decisions"],
        "replay_distribution_rows": replay_tables["distribution_rows"],
        "replay_pvalue_rows": replay_tables["pvalue_rows"],
        "replay_pvalue_summary": replay_tables["pvalue_summary"],
        "replay_pvalue_case_summary": replay_tables["pvalue_case_summary"],
        "replay_pvalue_tau_s_sensitivity": replay_tables["pvalue_tau_s_sensitivity"],
        "replay_measurability_rows": replay_tables["measurability_rows"],
        "replay_measurability_summary": replay_tables["measurability_summary"],
        "replay_topology_rows": replay_tables["topology_rows"],
        "replay_topology_summary": replay_tables["topology_summary"],
        "joined_rows": joined_tail_rows,
        "deformed_rows": deformed_rows,
        "deformed_support_rows": deformed_support_rows,
        "deformed_summary": deformed_summary,
        "tail_rows": tail_rows,
        "tail_summary": tail_summary,
        "attempt_rows": attempt_rows,
        "attempt_summary": attempt_summary,
    }


def run_same_geometry_external_support_attempt(
    config: RootSelectedSameGeometryExternalSupportAttemptConfig,
) -> dict[str, Path]:
    """Run the same-geometry external support attempt and write artifacts."""
    start = perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_same_geometry_external_support_attempt(config)
    outputs = {
        "frontier_generated_rows": config.output_dir / FRONTIER_GENERATED_ROWS_OUTPUT,
        "frontier_target_rows": config.output_dir / FRONTIER_TARGET_ROWS_OUTPUT,
        "frontier_summary": config.output_dir / FRONTIER_SUMMARY_OUTPUT,
        "frontier_failures": config.output_dir / FRONTIER_FAILURES_OUTPUT,
        "replay_run_rows": config.output_dir / RUN_ROWS_OUTPUT,
        "replay_node_decisions": config.output_dir / NODE_DECISIONS_OUTPUT,
        "replay_distribution_rows": config.output_dir / DISTRIBUTION_ROWS_OUTPUT,
        "replay_pvalue_rows": config.output_dir / PVALUE_ROWS_OUTPUT,
        "replay_pvalue_summary": config.output_dir / PVALUE_SUMMARY_OUTPUT,
        "replay_pvalue_case_summary": config.output_dir / PVALUE_CASE_SUMMARY_OUTPUT,
        "replay_pvalue_tau_s_sensitivity": config.output_dir / PVALUE_TAU_S_OUTPUT,
        "replay_measurability_rows": config.output_dir / MEASURABILITY_ROWS_OUTPUT,
        "replay_measurability_summary": config.output_dir / MEASURABILITY_SUMMARY_OUTPUT,
        "replay_topology_rows": config.output_dir / TOPOLOGY_ROWS_OUTPUT,
        "replay_topology_summary": config.output_dir / TOPOLOGY_SUMMARY_OUTPUT,
        "joined_rows": config.output_dir / JOINED_ROWS_OUTPUT,
        "deformed_rows": config.output_dir / DEFORMED_ROWS_OUTPUT,
        "deformed_support_rows": config.output_dir / DEFORMED_SUPPORT_ROWS_OUTPUT,
        "deformed_summary": config.output_dir / DEFORMED_SUMMARY_OUTPUT,
        "tail_rows": config.output_dir / TAIL_ROWS_OUTPUT,
        "tail_summary": config.output_dir / TAIL_SUMMARY_OUTPUT,
        "attempt_rows": config.output_dir / ATTEMPT_ROWS_OUTPUT,
        "attempt_summary": config.output_dir / ATTEMPT_SUMMARY_OUTPUT,
    }
    for key, path in outputs.items():
        tables[key].to_csv(path, index=False)
    manifest_path = config.output_dir / MANIFEST_OUTPUT
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": float(perf_counter() - start),
        "config": config,
        "row_counts": {key: int(table.shape[0]) for key, table in tables.items()},
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    outputs["manifest"] = manifest_path
    return outputs


def main() -> None:
    args = parse_args()
    outputs = run_same_geometry_external_support_attempt(
        RootSelectedSameGeometryExternalSupportAttemptConfig(
            output_dir=args.output_dir,
            external_law_equation_rows_path=args.external_law_equation_rows_path,
            observed_mixed_region_rows_path=args.observed_mixed_region_rows_path,
            base_feasibility_rows_path=args.base_feasibility_rows_path,
            observed_root_summary_rows_path=args.observed_root_summary_rows_path,
            h_u_observability_rows_path=args.h_u_observability_rows_path,
            suite=str(args.suite),
            target_case_ids=_parse_csv_list(args.target_case_ids),
            proposal_families=_parse_csv_list(args.proposal_families) or DEFAULT_PROPOSAL_FAMILIES,
            two_block_delta_grid=_parse_float_grid(args.two_block_delta_grid),
            spike_feature_fraction_grid=_parse_float_grid(args.spike_feature_fraction_grid),
            spike_delta_grid=_parse_float_grid(args.spike_delta_grid),
            block_fraction_grid=_parse_float_grid(args.block_fraction_grid),
            replicates_per_setting=int(args.replicates_per_setting),
            attempts_per_setting=int(args.attempts_per_setting),
            accept_target_pre_topology_stratum=bool(args.accept_target_pre_topology_stratum),
            max_replay_cases=args.max_replay_cases,
            candidate_only=bool(args.candidate_only),
            reference_tau_s=float(args.reference_tau_s),
            require_spectral_flow=bool(args.require_spectral_flow),
            sibling_alpha=float(args.sibling_alpha),
            edge_alpha=float(args.edge_alpha),
            tree_linkage_method=str(args.tree_linkage_method),
            method_id=str(args.method_id),
            h_u_population_law_status=str(args.h_u_population_law_status),
            seed_offset=int(args.seed_offset),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
