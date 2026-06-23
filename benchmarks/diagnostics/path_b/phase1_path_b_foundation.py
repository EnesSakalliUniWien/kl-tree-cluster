"""Phase-1 Path B foundation sweep.

This diagnostic runs TBS-only benchmark ablations for the MP minimum projection
dimension and pass-through traversal, then summarizes cluster-count behavior.
It can also rerun the existing Q5 selected-tail law diagnostic and report the
predictive gain from barycentric/spectral geometry covariates.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.selected_tail_law_q5_validation import (
    run_q5_selected_tail_law_validation,
)
from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.result_records import benchmark_rows_to_dataframe
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import run_single_method_once
from benchmarks.shared.util.time import format_timestamp_utc
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)

STUDY_ROLE = "diagnostic_path_b_phase1_foundation_not_calibration"
DEFAULT_SELECTED_TAIL_RECORDS = Path(
    "raw/assets/benchmark-results/"
    "selected_hierarchy_topology_refinement_input_20260603_300/"
    "selected_geometry_records.csv"
)


@dataclass(frozen=True)
class Phase1Outputs:
    comparison_csv: Path
    summary_csv: Path
    cluster_count_distribution_csv: Path
    selected_tail_q5_dir: Path | None
    q5_predictive_gain_csv: Path | None
    report_md: Path
    manifest_json: Path


def _parse_int_grid(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Grid must contain at least one integer.")
    if min(values) < 0:
        raise ValueError(f"Grid values must be nonnegative: {values!r}.")
    return values


def _parse_bool_grid(raw: str) -> tuple[bool, ...]:
    mapping = {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}
    values: list[bool] = []
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Invalid bool grid value {part!r}.")
        values.append(mapping[key])
    if not values:
        raise ValueError("Bool grid must contain at least one value.")
    return tuple(dict.fromkeys(values))


def build_phase1_param_grid(
    *,
    k_min_values: Sequence[int],
    passthrough_values: Sequence[bool],
) -> list[dict[str, object]]:
    base = dict(METHOD_SPECS["tbs"].param_grid[0])
    rows: list[dict[str, object]] = []
    for k_min in k_min_values:
        for passthrough in passthrough_values:
            params = dict(base)
            params["spectral_minimum_dimension"] = int(k_min)
            params["passthrough"] = bool(passthrough)
            rows.append(params)
    return rows


def _variant_from_params(params: object) -> tuple[int | None, bool | None]:
    text = str(params)
    k_match = re.search(r"spectral_minimum_dimension=([0-9]+)", text)
    pass_match = re.search(r"passthrough=(True|False)", text)
    k_min = int(k_match.group(1)) if k_match else None
    passthrough = pass_match.group(1) == "True" if pass_match else None
    return k_min, passthrough


def add_phase1_variant_columns(results: pd.DataFrame) -> pd.DataFrame:
    table = results.copy()
    variants = table["params"].apply(_variant_from_params)
    table["spectral_minimum_dimension"] = [variant[0] for variant in variants]
    table["passthrough"] = [variant[1] for variant in variants]
    return table


def _completed_variants(existing: pd.DataFrame) -> set[tuple[int, int, bool]]:
    if existing.empty:
        return set()
    table = add_phase1_variant_columns(existing)
    completed = table[
        table["spectral_minimum_dimension"].notna() & table["passthrough"].notna()
    ]
    return {
        (
            int(row.test_case),
            int(row.spectral_minimum_dimension),
            bool(row.passthrough),
        )
        for row in completed.itertuples(index=False)
    }


def summarize_phase1_benchmark(results: pd.DataFrame) -> pd.DataFrame:
    table = add_phase1_variant_columns(results)
    table["ari_for_selection"] = pd.to_numeric(table["ari"], errors="coerce").fillna(0.0)
    table["ok_flag"] = table["status"].astype(str).eq("ok")
    table["skip_flag"] = table["status"].astype(str).eq("skip")
    table["exact_k"] = np.where(
        table["ok_flag"],
        table["found_clusters"].eq(table["true_clusters"]).astype(float),
        np.nan,
    )
    rows: list[dict[str, object]] = []
    group_cols = ["spectral_minimum_dimension", "passthrough"]
    for (k_min, passthrough), group in table.groupby(group_cols, dropna=False, sort=True):
        ok = group[group["ok_flag"]]
        rows.append(
            {
                "spectral_minimum_dimension": int(k_min),
                "passthrough": bool(passthrough),
                "n_rows": int(group.shape[0]),
                "n_ok": int(group["ok_flag"].sum()),
                "n_skip": int(group["skip_flag"].sum()),
                "skip_rate": float(group["skip_flag"].mean()),
                "mean_ari_ok": float(ok["ari"].mean()) if not ok.empty else np.nan,
                "median_ari_ok": float(ok["ari"].median()) if not ok.empty else np.nan,
                "penalized_mean_ari": float(group["ari_for_selection"].mean()),
                "mean_cluster_count_abs_error_ok": (
                    float(ok["cluster_count_abs_error"].mean()) if not ok.empty else np.nan
                ),
                "exact_k_rate_ok": float(ok["exact_k"].mean()) if not ok.empty else np.nan,
                "over_split_rate_ok": float(ok["over_split"].mean()) if not ok.empty else np.nan,
                "under_split_rate_ok": float(ok["under_split"].mean()) if not ok.empty else np.nan,
                "mean_found_clusters_ok": (
                    float(ok["found_clusters"].mean()) if not ok.empty else np.nan
                ),
                "median_found_clusters_ok": (
                    float(ok["found_clusters"].median()) if not ok.empty else np.nan
                ),
                "study_role": STUDY_ROLE,
            }
        )
    summary = pd.DataFrame.from_records(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        [
            "penalized_mean_ari",
            "mean_cluster_count_abs_error_ok",
            "skip_rate",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def cluster_count_distribution(results: pd.DataFrame) -> pd.DataFrame:
    table = add_phase1_variant_columns(results)
    ok = table[table["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    grouped = (
        ok.groupby(
            ["spectral_minimum_dimension", "passthrough", "true_clusters", "found_clusters"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_rows")
    )
    grouped["study_role"] = STUDY_ROLE
    return grouped


def q5_predictive_gain_summary(q5_summary: pd.DataFrame) -> pd.DataFrame:
    if q5_summary.empty:
        return pd.DataFrame()
    baseline_rows = q5_summary[
        q5_summary["model_id"].eq("q5_without_spectral_geometry")
    ]
    if baseline_rows.empty:
        raise ValueError("Q5 summary is missing q5_without_spectral_geometry baseline.")
    baseline = baseline_rows.iloc[0]
    table = q5_summary.copy()
    table["median_tail_auc_gain_vs_baseline"] = (
        table["median_tail_auc"] - float(baseline["median_tail_auc"])
    )
    table["median_r_squared_gain_vs_baseline"] = (
        table["median_holdout_r_squared"] - float(baseline["median_holdout_r_squared"])
    )
    table["median_tail_error_reduction_vs_baseline"] = (
        float(baseline["median_residual_tail_exceedance_absolute_error"])
        - table["median_residual_tail_exceedance_absolute_error"]
    )
    table["study_role"] = STUDY_ROLE
    return table.sort_values(
        [
            "median_tail_error_reduction_vs_baseline",
            "median_tail_auc_gain_vs_baseline",
            "median_r_squared_gain_vs_baseline",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _write_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    q5_gain: pd.DataFrame,
    selected_tail_records: Path | None,
) -> Path:
    report_path = output_dir / "phase1_path_b_report.md"
    lines = [
        "# Phase 1 Path B Foundation Diagnostic",
        "",
        f"- study_role: `{STUDY_ROLE}`",
        "",
        "## K-Min And Pass-Through",
        "",
    ]
    if summary.empty:
        lines.append("- No benchmark rows available.")
    else:
        best = summary.iloc[0]
        lines.extend(
            [
                (
                    "- selected_optimum: "
                    f"`k_min={int(best.spectral_minimum_dimension)}, "
                    f"passthrough={bool(best.passthrough)}`"
                ),
                f"- penalized_mean_ari: `{best.penalized_mean_ari:.6f}`",
                f"- mean_ari_ok: `{best.mean_ari_ok:.6f}`",
                f"- skip_rate: `{best.skip_rate:.6f}`",
                f"- exact_k_rate_ok: `{best.exact_k_rate_ok:.6f}`",
                "",
            ]
        )
    lines.extend(["## Geometry Covariates", ""])
    if q5_gain.empty:
        lines.append("- Q5 selected-tail geometry gain was not run.")
    else:
        best_q5 = q5_gain.iloc[0]
        lines.extend(
            [
                f"- selected_tail_records: `{selected_tail_records}`",
                f"- best_q5_model_by_gain: `{best_q5.model_id}`",
                (
                    "- median_tail_error_reduction_vs_baseline: "
                    f"`{best_q5.median_tail_error_reduction_vs_baseline:.6f}`"
                ),
                (
                    "- median_tail_auc_gain_vs_baseline: "
                    f"`{best_q5.median_tail_auc_gain_vs_baseline:.6f}`"
                ),
                (
                    "- median_r_squared_gain_vs_baseline: "
                    f"`{best_q5.median_r_squared_gain_vs_baseline:.6f}`"
                ),
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _run_benchmark_sweep(
    *,
    case_suite: str,
    output_dir: Path,
    k_min_values: Sequence[int],
    passthrough_values: Sequence[bool],
    edge_alpha: float,
    sibling_alpha: float,
) -> Path:
    comparison_csv = output_dir / "phase1_path_b_benchmark_comparison.csv"
    existing = pd.read_csv(comparison_csv) if comparison_csv.exists() else pd.DataFrame()
    completed = _completed_variants(existing)
    test_cases = get_test_cases_by_suite(case_suite)
    param_grid = build_phase1_param_grid(
        k_min_values=k_min_values,
        passthrough_values=passthrough_values,
    )
    plots_root = output_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    for case_index, case in enumerate(test_cases, start=1):
        case = dict(case)
        case["test_case_num"] = case_index
        inputs = prepare_case_inputs(case, ["tbs"])
        for params in param_grid:
            key = (
                case_index,
                int(params["spectral_minimum_dimension"]),
                bool(params["passthrough"]),
            )
            if key in completed:
                continue
            row, _computed, _audit = run_single_method_once(
                method_id="tbs",
                spec=METHOD_SPECS["tbs"],
                params=params,
                case_idx=case_index,
                case_name=str(case["name"]),
                tc_seed=case["seed"],
                significance_level=sibling_alpha,
                edge_alpha=edge_alpha,
                data_t=inputs.data,
                y_t=inputs.labels,
                x_original=inputs.original_features,
                meta=inputs.metadata,
                distance_matrix=inputs.distance_matrix,
                distance_condensed=inputs.distance_condensed,
                matrix_audit=False,
            )
            row_df = benchmark_rows_to_dataframe([row])
            row_df.to_csv(
                comparison_csv,
                mode="a",
                header=not comparison_csv.exists(),
                index=False,
            )
            completed.add(key)
    return comparison_csv


def run_phase1_path_b_foundation(
    *,
    case_suite: str,
    output_dir: Path,
    k_min_values: Sequence[int] = (0, 1, 2, 3),
    passthrough_values: Sequence[bool] = (True, False),
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    selected_tail_records: Path | None = DEFAULT_SELECTED_TAIL_RECORDS,
    run_selected_tail_q5: bool = True,
) -> Phase1Outputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = _run_benchmark_sweep(
        case_suite=case_suite,
        output_dir=output_dir,
        k_min_values=k_min_values,
        passthrough_values=passthrough_values,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
    )
    results = pd.read_csv(comparison_csv)
    summary = summarize_phase1_benchmark(results)
    distribution = cluster_count_distribution(results)
    summary_csv = output_dir / "phase1_path_b_summary.csv"
    distribution_csv = output_dir / "phase1_cluster_count_distribution.csv"
    summary.to_csv(summary_csv, index=False)
    distribution.to_csv(distribution_csv, index=False)

    q5_dir: Path | None = None
    q5_gain_csv: Path | None = None
    q5_gain = pd.DataFrame()
    if run_selected_tail_q5 and selected_tail_records is not None:
        records_path = selected_tail_records.expanduser()
        if records_path.exists():
            q5_dir = output_dir / "selected_tail_q5_geometry_covariates"
            q5_outputs = run_q5_selected_tail_law_validation(
                records_path=records_path,
                output_dir=q5_dir,
            )
            q5_summary = pd.read_csv(q5_outputs["summary"])
            q5_gain = q5_predictive_gain_summary(q5_summary)
            q5_gain_csv = output_dir / "q5_geometry_covariate_predictive_gain.csv"
            q5_gain.to_csv(q5_gain_csv, index=False)

    report_md = _write_report(
        output_dir=output_dir,
        summary=summary,
        q5_gain=q5_gain,
        selected_tail_records=selected_tail_records,
    )
    manifest_json = output_dir / "manifest.json"
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "study_role": STUDY_ROLE,
        "case_suite": case_suite,
        "k_min_values": [int(value) for value in k_min_values],
        "passthrough_values": [bool(value) for value in passthrough_values],
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "selected_tail_records": None
        if selected_tail_records is None
        else str(selected_tail_records),
        "outputs": {
            "comparison": str(comparison_csv),
            "summary": str(summary_csv),
            "cluster_count_distribution": str(distribution_csv),
            "q5_predictive_gain": None if q5_gain_csv is None else str(q5_gain_csv),
            "report": str(report_md),
        },
        "interpretation": (
            "Diagnostic Phase-1 Path B ablation. No production calibration or "
            "traversal rule is promoted by these outputs."
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return Phase1Outputs(
        comparison_csv=comparison_csv,
        summary_csv=summary_csv,
        cluster_count_distribution_csv=distribution_csv,
        selected_tail_q5_dir=q5_dir,
        q5_predictive_gain_csv=q5_gain_csv,
        report_md=report_md,
        manifest_json=manifest_json,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-suite", default="full")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-min-grid", default="0,1,2,3")
    parser.add_argument("--passthrough-grid", default="true,false")
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--selected-tail-records", type=Path, default=DEFAULT_SELECTED_TAIL_RECORDS)
    parser.add_argument("--skip-selected-tail-q5", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_phase1_path_b_foundation(
        case_suite=str(args.case_suite),
        output_dir=args.output_dir,
        k_min_values=_parse_int_grid(args.k_min_grid),
        passthrough_values=_parse_bool_grid(args.passthrough_grid),
        edge_alpha=float(args.edge_alpha),
        sibling_alpha=float(args.sibling_alpha),
        selected_tail_records=args.selected_tail_records,
        run_selected_tail_q5=not bool(args.skip_selected_tail_q5),
    )
    print(json.dumps(outputs.__dict__ | {
        key: None if value is None else str(value)
        for key, value in outputs.__dict__.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "Phase1Outputs",
    "add_phase1_variant_columns",
    "build_phase1_param_grid",
    "cluster_count_distribution",
    "q5_predictive_gain_summary",
    "run_phase1_path_b_foundation",
    "summarize_phase1_benchmark",
]
