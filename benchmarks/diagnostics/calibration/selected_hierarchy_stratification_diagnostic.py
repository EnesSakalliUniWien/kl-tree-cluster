"""Descriptive stratification for selected-hierarchy null records.

This diagnostic estimates how selected-hierarchy null scale varies across
parent depth and parent-size strata. It is diagnostic-only: strata describe
heterogeneity in the selected null and must not be used as fallback calibration
borrowing rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.feature_space import FeatureSpace

from benchmarks.diagnostics.calibration.selected_hierarchy_null_audit import (
    DEFAULT_CASE_NAMES,
    _parse_csv_list,
    _run_edge_and_sibling_records,
    _selected_cases,
    _simulate_null_data,
)
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.time import format_timestamp_utc

SIZE_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
SIZE_BIN_LABELS = ("small_0_0.25", "medium_0.25_0.5", "large_0.5_0.75", "root_0.75_1")


def _parent_size_bin(parent_fraction: float) -> str:
    if not 0.0 < parent_fraction <= 1.0:
        raise ValueError(f"Parent-size fraction must lie in (0, 1]; got {parent_fraction!r}.")
    for lower, upper, label in zip(SIZE_BIN_EDGES, SIZE_BIN_EDGES[1:], SIZE_BIN_LABELS):
        if lower < parent_fraction <= upper:
            return label
    raise ValueError(f"Parent-size fraction did not match a bin: {parent_fraction!r}.")


def _selected_record_rows(
    *,
    case_id: str,
    replicate_index: int,
    n_samples: int,
    n_features: int,
    sample,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record_sample in sample.selected_records:
        record = record_sample.record
        reference_expectation = float(record.reference_scale * record.degrees_of_freedom)
        if reference_expectation <= 0.0:
            raise ValueError(
                "Selected-hierarchy stratification requires positive reference "
                f"expectation; case={case_id!r}, parent={record.parent!r}."
            )
        parent_fraction = float(record.n_parent / n_samples)
        rows.append(
            {
                "case_id": case_id,
                "replicate_index": int(replicate_index),
                "parent": record.parent,
                "feature_family": record.feature_family,
                "n_samples": int(n_samples),
                "feature_dimension": int(n_features),
                "projection_dimension": int(record.sibling_projection_dimension),
                "parent_depth": int(record_sample.parent_depth),
                "parent_sample_size": int(record.n_parent),
                "parent_size_fraction": parent_fraction,
                "parent_size_bin": _parent_size_bin(parent_fraction),
                "statistic": float(record.stat),
                "reference_expectation": reference_expectation,
                "selected_hierarchy_ratio": float(record.stat / reference_expectation),
                "raw_p_value": float(record.p_value),
                "stratum_role": "descriptive_not_calibration_borrowing",
            }
        )
    return rows


def _stratum_summary(
    records: pd.DataFrame,
    *,
    group_columns: list[str],
    n_replicates: int,
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for group_values, group in records.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        simulation_means = group.groupby("replicate_index")[
            "selected_hierarchy_ratio"
        ].mean()
        ratios = group["selected_hierarchy_ratio"].to_numpy(dtype=float)
        statistics = group["statistic"].to_numpy(dtype=float)
        reference_expectations = group["reference_expectation"].to_numpy(dtype=float)
        raw_p_values = group["raw_p_value"].to_numpy(dtype=float)
        n_matching_simulations = int(simulation_means.shape[0])
        c_hat = float(np.mean(ratios))
        simulation_se = (
            float(np.std(simulation_means.to_numpy(dtype=float), ddof=1) / np.sqrt(n_matching_simulations))
            if n_matching_simulations > 1
            else np.nan
        )
        row = {
            column: value for column, value in zip(group_columns, group_values)
        }
        row.update(
            {
                "n_records": int(group.shape[0]),
                "n_matching_simulations": n_matching_simulations,
                "simulation_acceptance_rate": float(n_matching_simulations / n_replicates),
                "selected_hierarchy_c_hat": c_hat,
                "selected_hierarchy_c_hat_record_se": (
                    float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
                    if len(ratios) > 1
                    else np.nan
                ),
                "selected_hierarchy_c_hat_simulation_se": simulation_se,
                "selected_hierarchy_c_hat_relative_simulation_se": (
                    float(simulation_se / c_hat)
                    if np.isfinite(simulation_se) and c_hat != 0.0
                    else np.nan
                ),
                "selected_hierarchy_ratio_median": float(np.quantile(ratios, 0.5)),
                "selected_hierarchy_ratio_q10": float(np.quantile(ratios, 0.1)),
                "selected_hierarchy_ratio_q90": float(np.quantile(ratios, 0.9)),
                "selected_hierarchy_ratio_q95": float(np.quantile(ratios, 0.95)),
                "selected_hierarchy_ratio_q99": float(np.quantile(ratios, 0.99)),
                "statistic_mean": float(np.mean(statistics)),
                "statistic_median": float(np.quantile(statistics, 0.5)),
                "statistic_q95": float(np.quantile(statistics, 0.95)),
                "statistic_q99": float(np.quantile(statistics, 0.99)),
                "reference_expectation_mean": float(np.mean(reference_expectations)),
                "reference_expectation_median": float(
                    np.quantile(reference_expectations, 0.5)
                ),
                "raw_p_value_median": float(np.quantile(raw_p_values, 0.5)),
                "raw_p_value_q10": float(np.quantile(raw_p_values, 0.1)),
                "raw_p_value_q90": float(np.quantile(raw_p_values, 0.9)),
                "standard_reference_sibling_alpha_rejection_rate": float(
                    np.mean(raw_p_values <= config.SIBLING_ALPHA)
                ),
                "mean_parent_sample_size": float(group["parent_sample_size"].mean()),
                "mean_parent_size_fraction": float(group["parent_size_fraction"].mean()),
                "stratum_role": "descriptive_not_calibration_borrowing",
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(group_columns).reset_index(drop=True)


def _diagnose_case(
    case: dict[str, object],
    *,
    n_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    inputs = prepare_case_inputs(case, ["kl"])
    feature_space = inputs.metadata.get("feature_space")
    if feature_space is not None and not isinstance(feature_space, FeatureSpace):
        raise ValueError("Prepared feature_space metadata must be a FeatureSpace.")
    if feature_space is not None and feature_space.family_label == "continuous":
        raise ValueError(
            "Continuous selected-hierarchy stratification requires a validated "
            "continuous null covariance generator and an audit-owned continuous "
            "tree-distance contract before use."
        )

    rng = np.random.default_rng(int(seed))
    records: list[dict[str, object]] = []
    candidate_records = 0
    selected_records = 0
    tested_edges = 0
    significant_edges = 0
    tree_metric = ""

    for replicate_index in range(int(n_replicates)):
        simulated = _simulate_null_data(inputs.data, feature_space, rng=rng)
        sample, tree_metric = _run_edge_and_sibling_records(
            simulated,
            inputs.metadata,
            feature_space,
        )
        candidate_records += int(sample.candidate_records)
        selected_records += int(len(sample.selected_records))
        tested_edges += int(sample.tested_edges)
        significant_edges += int(sample.significant_edges)
        records.extend(
            _selected_record_rows(
                case_id=str(inputs.metadata["name"]),
                replicate_index=replicate_index,
                n_samples=int(inputs.data.shape[0]),
                n_features=int(inputs.data.shape[1]),
                sample=sample,
            )
        )

    case_summary = {
        "case_id": str(inputs.metadata["name"]),
        "case_category": str(inputs.metadata["category"]),
        "feature_family": "bernoulli" if feature_space is None else feature_space.family_label,
        "tree_distance_metric": tree_metric,
        "n_samples": int(inputs.data.shape[0]),
        "feature_dimension": int(inputs.data.shape[1]),
        "n_replicates": int(n_replicates),
        "n_candidate_records": int(candidate_records),
        "n_selected_records": int(selected_records),
        "selected_record_rate": (
            float(selected_records / candidate_records) if candidate_records else 0.0
        ),
        "tested_edges": int(tested_edges),
        "significant_edges": int(significant_edges),
        "edge_rejection_rate": (
            float(significant_edges / tested_edges) if tested_edges else 0.0
        ),
        "stratification_role": "descriptive_not_calibration_borrowing",
    }
    return pd.DataFrame.from_records(records), case_summary


def run_selected_hierarchy_stratification_diagnostic(
    *,
    case_names: list[str],
    output_dir: Path,
    n_replicates: int,
    seed: int,
    write_selected_records: bool = False,
) -> dict[str, pd.DataFrame]:
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()

    record_tables: list[pd.DataFrame] = []
    case_summaries: list[dict[str, object]] = []
    for index, case in enumerate(_selected_cases(case_names), start=1):
        print(f"[{index}/{len(case_names)}] {case['name']}", flush=True)
        try:
            records, case_summary = _diagnose_case(
                case,
                n_replicates=int(n_replicates),
                seed=int(seed) + index * 1_000_000,
            )
            case_summary["status"] = "ok"
            case_summary["skip_reason"] = ""
            record_tables.append(records)
        except Exception as exc:
            case_summary = {
                "case_id": str(case["name"]),
                "case_category": str(case["category"]),
                "n_replicates": int(n_replicates),
                "status": "skip",
                "skip_reason": str(exc),
                "stratification_role": "descriptive_not_calibration_borrowing",
            }
        case_summaries.append(case_summary)

    selected_records = (
        pd.concat(record_tables, ignore_index=True)
        if record_tables
        else pd.DataFrame()
    )
    case_summary = pd.DataFrame.from_records(case_summaries)
    by_depth = _stratum_summary(
        selected_records,
        group_columns=[
            "case_id",
            "feature_family",
            "n_samples",
            "feature_dimension",
            "projection_dimension",
            "parent_depth",
        ],
        n_replicates=int(n_replicates),
    )
    by_parent_size = _stratum_summary(
        selected_records,
        group_columns=[
            "case_id",
            "feature_family",
            "n_samples",
            "feature_dimension",
            "projection_dimension",
            "parent_size_bin",
        ],
        n_replicates=int(n_replicates),
    )
    by_depth_parent_size = _stratum_summary(
        selected_records,
        group_columns=[
            "case_id",
            "feature_family",
            "n_samples",
            "feature_dimension",
            "projection_dimension",
            "parent_depth",
            "parent_size_bin",
        ],
        n_replicates=int(n_replicates),
    )

    outputs = {
        "case_summary": case_summary,
        "strata_by_depth": by_depth,
        "strata_by_parent_size": by_parent_size,
        "strata_by_depth_parent_size": by_depth_parent_size,
    }
    if write_selected_records:
        outputs["selected_records"] = selected_records
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    manifest = {
        "seed": int(seed),
        "n_replicates": int(n_replicates),
        "case_names": case_names,
        "edge_alpha": float(config.EDGE_ALPHA),
        "sibling_alpha": float(config.SIBLING_ALPHA),
        "elapsed_sec": round(float(perf_counter() - started_at), 6),
        "write_selected_records": bool(write_selected_records),
        "outputs": {name: str(output_dir / f"{name}.csv") for name in outputs},
        "note": (
            "Diagnostic-only selected-hierarchy stratification. Depth and "
            "parent-size strata describe heterogeneity and are not calibration "
            "fallback or borrowing rules."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Describe selected-hierarchy null scale across parent depth and "
            "parent-size strata. This is not a calibration fallback."
        )
    )
    parser.add_argument("--case-names", default=",".join(DEFAULT_CASE_NAMES))
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument(
        "--write-selected-records",
        action="store_true",
        help="Write the full per-record table. Off by default to keep raw evidence compact.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("benchmarks")
            / "results"
            / f"selected_hierarchy_stratification_{format_timestamp_utc()}"
        )
    outputs = run_selected_hierarchy_stratification_diagnostic(
        case_names=_parse_csv_list(str(args.case_names)),
        output_dir=output_dir,
        n_replicates=int(args.n_replicates),
        seed=int(args.seed),
        write_selected_records=bool(args.write_selected_records),
    )
    print(outputs["case_summary"].to_string(index=False))
    print(f"Wrote selected-hierarchy stratification outputs to {output_dir}")


if __name__ == "__main__":
    main()
