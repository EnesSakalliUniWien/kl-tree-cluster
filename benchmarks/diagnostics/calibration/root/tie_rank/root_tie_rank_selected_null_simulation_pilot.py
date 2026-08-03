"""Selected-null simulation pilot for root tie-rank calibration strata.

This diagnostic generates iid Bernoulli selected-null roots matched to the
scale of the binary overlap cases, sends each replicate through the existing
root selected-region extractor, and reports whether those null roots populate
the observed mixed-law tie-rank strata.

The output is diagnostic-only. It does not promote a root rescue rule.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.reporting import (
    print_diagnostic_output_paths,
    write_diagnostic_bundle,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_mixed_region_law import (
    build_root_selected_mixed_region_law_rows,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_region_margins import (
    SCHEMA_VERSION as ROOT_MARGIN_SCHEMA_VERSION,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_region_margins import (
    collect_observed_root_selected_region_row,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_tie_cell_burden import (
    build_root_selected_tie_cell_burden_rows,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_calibration_feasibility import (
    build_root_tie_rank_calibration_feasibility_rows,
    summarize_root_tie_rank_calibration_feasibility,
    summarize_root_tie_rank_calibration_strata,
)
from benchmarks.shared.cases import get_test_cases_by_suite

SCHEMA_VERSION = "root_tie_rank_selected_null_simulation_pilot/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_selected_null_simulation_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_selected_null_simulation_pilot"
)

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_OBSERVED_MIXED_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_mixed_region_law_overlap_case_family"
    / "root_selected_mixed_region_law_rows.csv"
)
DEFAULT_TARGET_ALPHA = 0.01
DEFAULT_RELATIVE_SE_TARGET = 0.25
DEFAULT_REPLICATES_PER_CASE = 1
DEFAULT_SEED_OFFSET = 920_000

ROOT_ROWS_OUTPUT = "root_tie_rank_selected_null_root_rows.csv"
MERGE_MARGINS_OUTPUT = "root_tie_rank_selected_null_merge_margins.csv"
TIE_ROWS_OUTPUT = "root_tie_rank_selected_null_tie_rows.csv"
MIXED_ROWS_OUTPUT = "root_tie_rank_selected_null_mixed_rows.csv"
COMBINED_FEASIBILITY_ROWS_OUTPUT = "root_tie_rank_selected_null_combined_feasibility_rows.csv"
COMBINED_FEASIBILITY_STRATA_OUTPUT = "root_tie_rank_selected_null_combined_feasibility_strata.csv"
COMBINED_FEASIBILITY_SUMMARY_OUTPUT = "root_tie_rank_selected_null_combined_feasibility_summary.csv"
TARGET_SUPPORT_OUTPUT = "root_tie_rank_selected_null_target_support.csv"
SUMMARY_OUTPUT = "root_tie_rank_selected_null_simulation_summary.csv"
FAILURES_OUTPUT = "root_tie_rank_selected_null_failures.csv"
MANIFEST_OUTPUT = "manifest.json"

TARGET_SUPPORT_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "root_conditioning_stratum_key",
    "root_sibling_selected_ratio",
    "root_tie_rank_median_fraction",
    "root_edge_path_statistic_margin",
    "root_selected_eigenvalue_over_mp_upper_bound",
    "root_bandwidth_reopen_count",
    "stratum_observed_count",
    "stratum_calibration_null_support_count",
    "stratum_calibration_null_exceedance_count",
    "empirical_conservative_tail_p_value",
    "additional_null_count_for_alpha_resolution",
    "additional_null_count_for_tail_precision",
    "calibration_feasibility_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "base_case_count",
    "replicates_per_case",
    "attempted_null_replicates",
    "successful_null_replicates",
    "failed_null_replicates",
    "observed_target_case_count",
    "observed_target_stratum_count",
    "null_generated_stratum_count",
    "target_strata_with_null_support_count",
    "target_strata_alpha_resolution_ready_count",
    "target_strata_tail_precision_ready_count",
    "missing_alpha_resolution_null_count_total",
    "missing_tail_precision_null_count_total",
    "summary_status",
)

FAILURE_COLUMNS = (
    "schema_version",
    "study_role",
    "base_case_id",
    "replicate",
    "simulation_case_id",
    "seed",
    "failure_type",
    "failure_message",
)


@dataclass(frozen=True)
class RootTieRankSelectedNullSimulationConfig:
    """Configuration for selected-null root tie-rank simulation."""

    output_dir: Path
    observed_mixed_region_rows_path: Path = DEFAULT_OBSERVED_MIXED_ROWS
    suite: str = "full"
    case_names: tuple[str, ...] | None = None
    replicates_per_case: int = DEFAULT_REPLICATES_PER_CASE
    seed_offset: int = DEFAULT_SEED_OFFSET
    target_alpha: float = DEFAULT_TARGET_ALPHA
    relative_se_target: float = DEFAULT_RELATIVE_SE_TARGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--observed-mixed-region-rows-path",
        type=Path,
        default=DEFAULT_OBSERVED_MIXED_ROWS,
    )
    parser.add_argument("--suite", default="full")
    parser.add_argument(
        "--case-names",
        default=None,
        help=(
            "Optional comma-separated base case names. Defaults to case_id values "
            "from the observed mixed-law rows."
        ),
    )
    parser.add_argument("--replicates-per-case", type=int, default=DEFAULT_REPLICATES_PER_CASE)
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    parser.add_argument("--target-alpha", type=float, default=DEFAULT_TARGET_ALPHA)
    parser.add_argument(
        "--relative-se-target",
        type=float,
        default=DEFAULT_RELATIVE_SE_TARGET,
    )
    return parser.parse_args()


def _parse_csv_list(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def _select_cases(
    *,
    suite: str,
    case_names: Sequence[str],
) -> list[dict[str, object]]:
    cases = get_test_cases_by_suite(suite)
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing}.")
    return [by_name[name].copy() for name in case_names]


def binary_null_probability_for_case(case: dict[str, object]) -> float:
    """Return the iid Bernoulli null probability matched to a binary case."""
    if "null_feature_probability" in case:
        probability = float(case["null_feature_probability"])
    elif "feature_sparsity" in case and "n_clusters" in case:
        sparsity = float(case["feature_sparsity"])
        n_clusters = max(int(case["n_clusters"]), 1)
        probability = (1.0 + sparsity * float(n_clusters - 2)) / float(n_clusters)
    else:
        probability = 0.5
    return float(np.clip(probability, 1e-3, 1.0 - 1e-3))


def _write_iid_binary_null_case(
    *,
    base_case: dict[str, object],
    simulation_case_id: str,
    seed: int,
    matrix_dir: Path,
) -> dict[str, object]:
    n_samples = int(base_case["n_samples"])
    n_features = int(base_case["n_features"])
    probability = binary_null_probability_for_case(base_case)
    rng = np.random.default_rng(seed)
    matrix = rng.binomial(1, probability, size=(n_samples, n_features)).astype(int)
    for feature_index in np.where(matrix.sum(axis=0) == 0)[0]:
        matrix[int(rng.integers(0, n_samples)), int(feature_index)] = 1
    sample_names = [f"L{i + 1}" for i in range(n_samples)]
    feature_names = [f"F{j}" for j in range(n_features)]
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / f"{simulation_case_id}.csv"
    pd.DataFrame(matrix, index=sample_names, columns=feature_names).to_csv(matrix_path)
    return {
        "name": simulation_case_id,
        "generator": "preloaded",
        "file_path": str(matrix_path),
        "sep": ",",
        "n_clusters": 1,
        "category": "selected_null_root_tie_rank_calibration",
        "baseline_case_name": str(base_case["name"]),
        "null_generation": "iid_bernoulli_marginal_mean",
        "null_feature_probability": probability,
        "seed": seed,
    }


def _metadata_by_case(root_rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_id",
        "base_case_id",
        "data_role",
        "calibration_role",
        "replicate",
        "null_seed",
        "null_feature_probability",
    ]
    present = [column for column in columns if column in root_rows.columns]
    return root_rows[present].copy()


def _attach_metadata(rows: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or metadata.empty:
        return rows
    return rows.merge(metadata, on="case_id", how="left")


def collect_selected_null_root_region_rows(
    *,
    base_cases: Sequence[dict[str, object]],
    output_dir: Path,
    replicates_per_case: int = DEFAULT_REPLICATES_PER_CASE,
    seed_offset: int = DEFAULT_SEED_OFFSET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate selected-null roots and return root rows, margins, and failures."""
    root_records: list[dict[str, object]] = []
    margin_tables: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    matrix_dir = Path(output_dir) / "generated_null_matrices"
    if int(replicates_per_case) <= 0:
        raise ValueError("replicates_per_case must be positive.")
    for base_index, base_case in enumerate(base_cases):
        base_case_id = str(base_case["name"])
        for replicate in range(int(replicates_per_case)):
            seed = int(seed_offset) + base_index * 100_000 + replicate
            simulation_case_id = f"{base_case_id}__selected_null_r{replicate:04d}"
            try:
                null_case = _write_iid_binary_null_case(
                    base_case=base_case,
                    simulation_case_id=simulation_case_id,
                    seed=seed,
                    matrix_dir=matrix_dir,
                )
                row, margins = collect_observed_root_selected_region_row(null_case)
                row.update(
                    {
                        "base_case_id": base_case_id,
                        "data_role": "selected_null",
                        "calibration_role": "selected_null_candidate_support",
                        "replicate": int(replicate),
                        "null_seed": int(seed),
                        "null_feature_probability": float(null_case["null_feature_probability"]),
                    }
                )
                root_records.append(row)
                case_margins = margins.copy()
                case_margins.insert(0, "case_id", row["case_id"])
                case_margins.insert(1, "schema_version", ROOT_MARGIN_SCHEMA_VERSION)
                case_margins.insert(2, "study_role", STUDY_ROLE)
                margin_tables.append(case_margins)
            except Exception as exc:  # pragma: no cover - exercised by integration runs
                failures.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "base_case_id": base_case_id,
                        "replicate": int(replicate),
                        "simulation_case_id": simulation_case_id,
                        "seed": int(seed),
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    }
                )
    root_rows = pd.DataFrame.from_records(root_records)
    merge_margins = pd.concat(margin_tables, ignore_index=True) if margin_tables else pd.DataFrame()
    failure_rows = pd.DataFrame.from_records(failures, columns=FAILURE_COLUMNS)
    return root_rows, merge_margins, failure_rows


def build_selected_null_mixed_rows(
    *,
    root_rows: pd.DataFrame,
    merge_margins: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return null tie-burden rows and mixed-law rows from root-margin output."""
    if root_rows.empty or merge_margins.empty:
        return pd.DataFrame(), pd.DataFrame()
    tie_rows = build_root_selected_tie_cell_burden_rows(
        root_summary=root_rows,
        merge_margins=merge_margins,
    )
    mixed_rows = build_root_selected_mixed_region_law_rows(
        root_summary=root_rows,
        tie_cell_burden_rows=tie_rows,
        topology_frontier_rows=None,
    )
    metadata = _metadata_by_case(root_rows)
    mixed_rows = _attach_metadata(mixed_rows, metadata)
    tie_rows = _attach_metadata(tie_rows, metadata)
    return tie_rows, mixed_rows


def _observed_target_rows(observed_mixed_rows: pd.DataFrame) -> pd.DataFrame:
    observed = observed_mixed_rows.copy()
    observed["data_role"] = "observed_target"
    observed["calibration_role"] = "observed_target_not_null_support"
    if "base_case_id" not in observed.columns:
        observed["base_case_id"] = observed["case_id"].astype(str)
    if "replicate" not in observed.columns:
        observed["replicate"] = -1
    return observed


def build_target_support_rows(
    *,
    observed_case_ids: Sequence[str],
    combined_feasibility_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return observed target rows with null support counts by stratum."""
    if combined_feasibility_rows.empty:
        return pd.DataFrame(columns=TARGET_SUPPORT_COLUMNS)
    targets = combined_feasibility_rows[
        combined_feasibility_rows["case_id"]
        .astype(str)
        .isin({str(case_id) for case_id in observed_case_ids})
    ].copy()
    records: list[dict[str, object]] = []
    for _, row in targets.sort_values("case_id").iterrows():
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(row["case_id"]),
                "root_conditioning_stratum_key": str(row["root_conditioning_stratum_key"]),
                "root_sibling_selected_ratio": float(row["root_sibling_selected_ratio"]),
                "root_tie_rank_median_fraction": float(row["root_tie_rank_median_fraction"]),
                "root_edge_path_statistic_margin": float(row["root_edge_path_statistic_margin"]),
                "root_selected_eigenvalue_over_mp_upper_bound": float(
                    row["root_selected_eigenvalue_over_mp_upper_bound"]
                ),
                "root_bandwidth_reopen_count": float(row["root_bandwidth_reopen_count"]),
                "stratum_observed_count": int(row["stratum_observed_count"]),
                "stratum_calibration_null_support_count": int(
                    row["stratum_calibration_null_support_count"]
                ),
                "stratum_calibration_null_exceedance_count": int(
                    row["stratum_calibration_null_exceedance_count"]
                ),
                "empirical_conservative_tail_p_value": float(
                    row["empirical_conservative_tail_p_value"]
                )
                if pd.notna(row["empirical_conservative_tail_p_value"])
                else math.nan,
                "additional_null_count_for_alpha_resolution": int(
                    row["additional_null_count_for_alpha_resolution"]
                ),
                "additional_null_count_for_tail_precision": int(
                    row["additional_null_count_for_tail_precision"]
                ),
                "calibration_feasibility_status": str(row["calibration_feasibility_status"]),
            }
        )
    return pd.DataFrame.from_records(records, columns=TARGET_SUPPORT_COLUMNS)


def summarize_selected_null_simulation(
    *,
    base_case_count: int,
    replicates_per_case: int,
    root_rows: pd.DataFrame,
    failures: pd.DataFrame,
    target_support: pd.DataFrame,
    null_mixed_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return one global simulation pilot summary row."""
    attempted = int(base_case_count) * int(replicates_per_case)
    successful = int(root_rows.shape[0])
    failed = int(failures.shape[0])
    target_strata = (
        int(target_support["root_conditioning_stratum_key"].nunique())
        if not target_support.empty
        else 0
    )
    null_strata = (
        int(null_mixed_rows["root_conditioning_stratum_key"].nunique())
        if "root_conditioning_stratum_key" in null_mixed_rows
        else 0
    )
    with_support = (
        int((target_support["stratum_calibration_null_support_count"] > 0).sum())
        if not target_support.empty
        else 0
    )
    alpha_ready = (
        int(
            target_support["calibration_feasibility_status"]
            .astype(str)
            .isin(
                {
                    "alpha_resolution_ready_precision_missing",
                    "tail_precision_ready_diagnostic_only",
                }
            )
            .sum()
        )
        if not target_support.empty
        else 0
    )
    precision_ready = (
        int(
            target_support["calibration_feasibility_status"]
            .astype(str)
            .eq("tail_precision_ready_diagnostic_only")
            .sum()
        )
        if not target_support.empty
        else 0
    )
    missing_resolution = (
        int(target_support["additional_null_count_for_alpha_resolution"].sum())
        if not target_support.empty
        else 0
    )
    missing_precision = (
        int(target_support["additional_null_count_for_tail_precision"].sum())
        if not target_support.empty
        else 0
    )
    if precision_ready == target_strata and target_strata > 0:
        status = "all_target_strata_tail_precision_ready_diagnostic_only"
    elif alpha_ready > 0:
        status = "partial_target_alpha_resolution_support"
    elif with_support > 0:
        status = "target_null_support_observed_below_alpha_resolution"
    else:
        status = "no_observed_target_stratum_support_yet"
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "base_case_count": int(base_case_count),
                "replicates_per_case": int(replicates_per_case),
                "attempted_null_replicates": attempted,
                "successful_null_replicates": successful,
                "failed_null_replicates": failed,
                "observed_target_case_count": int(target_support.shape[0]),
                "observed_target_stratum_count": target_strata,
                "null_generated_stratum_count": null_strata,
                "target_strata_with_null_support_count": with_support,
                "target_strata_alpha_resolution_ready_count": alpha_ready,
                "target_strata_tail_precision_ready_count": precision_ready,
                "missing_alpha_resolution_null_count_total": missing_resolution,
                "missing_tail_precision_null_count_total": missing_precision,
                "summary_status": status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_tie_rank_selected_null_simulation(
    config: RootTieRankSelectedNullSimulationConfig,
) -> dict[str, pd.DataFrame]:
    """Run selected-null simulation and return output tables."""
    observed_mixed = pd.read_csv(config.observed_mixed_region_rows_path)
    _require_columns(observed_mixed, {"case_id"}, "observed mixed-law rows")
    observed_case_ids = tuple(observed_mixed["case_id"].astype(str).tolist())
    case_names = tuple(config.case_names) if config.case_names else observed_case_ids
    base_cases = _select_cases(suite=config.suite, case_names=case_names)
    root_rows, merge_margins, failures = collect_selected_null_root_region_rows(
        base_cases=base_cases,
        output_dir=config.output_dir,
        replicates_per_case=int(config.replicates_per_case),
        seed_offset=int(config.seed_offset),
    )
    tie_rows, null_mixed_rows = build_selected_null_mixed_rows(
        root_rows=root_rows,
        merge_margins=merge_margins,
    )
    observed_targets = _observed_target_rows(observed_mixed)
    combined_mixed = pd.concat(
        [observed_targets, null_mixed_rows],
        ignore_index=True,
        sort=False,
    )
    feasibility_rows = build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=combined_mixed,
        target_alpha=float(config.target_alpha),
        relative_se_target=float(config.relative_se_target),
    )
    feasibility_strata = summarize_root_tie_rank_calibration_strata(feasibility_rows)
    feasibility_summary = summarize_root_tie_rank_calibration_feasibility(
        feasibility_rows,
        feasibility_strata,
    )
    target_support = build_target_support_rows(
        observed_case_ids=observed_case_ids,
        combined_feasibility_rows=feasibility_rows,
    )
    null_mixed_with_strata = feasibility_rows[
        feasibility_rows["calibration_role"].astype(str).eq("selected_null_candidate_support")
    ][["case_id", "root_conditioning_stratum_key"]].merge(
        null_mixed_rows,
        on="case_id",
        how="right",
    )
    summary = summarize_selected_null_simulation(
        base_case_count=len(base_cases),
        replicates_per_case=int(config.replicates_per_case),
        root_rows=root_rows,
        failures=failures,
        target_support=target_support,
        null_mixed_rows=null_mixed_with_strata,
    )
    return {
        "root_rows": root_rows,
        "merge_margins": merge_margins,
        "tie_rows": tie_rows,
        "mixed_rows": null_mixed_with_strata,
        "combined_feasibility_rows": feasibility_rows,
        "combined_feasibility_strata": feasibility_strata,
        "combined_feasibility_summary": feasibility_summary,
        "target_support": target_support,
        "summary": summary,
        "failures": failures,
    }


def run_root_tie_rank_selected_null_simulation(
    config: RootTieRankSelectedNullSimulationConfig,
) -> dict[str, Path]:
    """Run the selected-null root simulation pilot and write outputs."""
    start = perf_counter()
    tables = evaluate_root_tie_rank_selected_null_simulation(config)
    return write_diagnostic_bundle(
        output_dir=config.output_dir,
        tables=tables,
        filenames={
            "root_rows": ROOT_ROWS_OUTPUT,
            "merge_margins": MERGE_MARGINS_OUTPUT,
            "tie_rows": TIE_ROWS_OUTPUT,
            "mixed_rows": MIXED_ROWS_OUTPUT,
            "combined_feasibility_rows": COMBINED_FEASIBILITY_ROWS_OUTPUT,
            "combined_feasibility_strata": COMBINED_FEASIBILITY_STRATA_OUTPUT,
            "combined_feasibility_summary": COMBINED_FEASIBILITY_SUMMARY_OUTPUT,
            "target_support": TARGET_SUPPORT_OUTPUT,
            "summary": SUMMARY_OUTPUT,
            "failures": FAILURES_OUTPUT,
        },
        manifest={
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "generated_by": GENERATED_BY,
            "config": config,
            "elapsed_seconds": float(perf_counter() - start),
        },
        manifest_filename=MANIFEST_OUTPUT,
    )


def main() -> None:
    args = parse_args()
    outputs = run_root_tie_rank_selected_null_simulation(
        RootTieRankSelectedNullSimulationConfig(
            output_dir=args.output_dir,
            observed_mixed_region_rows_path=args.observed_mixed_region_rows_path,
            suite=str(args.suite),
            case_names=_parse_csv_list(args.case_names),
            replicates_per_case=int(args.replicates_per_case),
            seed_offset=int(args.seed_offset),
            target_alpha=float(args.target_alpha),
            relative_se_target=float(args.relative_se_target),
        )
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
