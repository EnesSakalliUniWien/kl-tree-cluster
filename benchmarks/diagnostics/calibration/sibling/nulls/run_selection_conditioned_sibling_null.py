#!/usr/bin/env python3
"""Run a local selection-conditioned sibling-null diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[3]

from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.tree_decomposition import TreeDecomposition

from benchmarks.diagnostics.calibration.sibling.nulls.selection_conditioned_sibling_null import (
    CONTINUOUS_LOCAL_EDGE_SELECTION_SCOPE,
    NONCONTINUOUS_LOCAL_EDGE_Z_PROXY_SCOPE,
    SelectionConditionedSiblingNullContext,
    simulate_local_edge_selection_conditioned_null,
)
from benchmarks.diagnostics.calibration.sibling.nulls.sibling_inflation_diagnostic import (
    build_sibling_inflation_diagnostic_tables,
    collect_sibling_inflation_inputs,
)
from benchmarks.diagnostics.oracle.gate_path_trace import build_gate_path_trace_dataframe
from benchmarks.diagnostics.oracle.oracle_tree_recoverability import (
    FAILURE_CLASS_GATE_UNDER_SPLIT,
    oracle_subtree_cut,
)
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.cases.regression_gate import get_regression_gate_test_cases
from benchmarks.shared.tbs_tree_context import build_tbs_tree_context

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_DEFAULT_GAUSSIAN_BLOCKERS = (
    "gauss_extreme_noise_highd",
    "gauss_extreme_noise_highd_continuous",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the sibling statistic under a fixed-tree, fixed-projection "
            "Gaussian z-null conditioned on the local child-parent edge gate opening."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("regression_gate", "full"),
        default="full",
        help="Benchmark suite containing the classified case names.",
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=None,
        help=(
            "Oracle recoverability CSV with failure_class. Defaults to the latest "
            "oracle_tree_recoverability CSV containing failure_class."
        ),
    )
    parser.add_argument(
        "--failure-classes",
        default=FAILURE_CLASS_GATE_UNDER_SPLIT,
        help="Comma-separated failure classes to diagnose.",
    )
    parser.add_argument(
        "--case-names",
        default=",".join(_DEFAULT_GAUSSIAN_BLOCKERS),
        help="Comma-separated case names after failure-class filtering.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("blockers", "current_blocks", "all_focal"),
        default="blockers",
        help="Which focal sibling tests receive a selection-conditioned diagnostic.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=1_000_000,
        help="Null candidates per sibling context before edge-gate conditioning.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of null candidates simulated per chunk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260524,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "benchmarks/results/selection_conditioned_sibling_null_<timestamp>/."
        ),
    )
    return parser.parse_args()


def _configure_runtime_defaults() -> None:
    for env_var in _THREAD_ENV_VARS:
        os.environ.setdefault(env_var, "1")
    os.environ.setdefault("TBS_N_JOBS", "1")


def _load_cases(suite: str) -> list[dict[str, object]]:
    if suite == "regression_gate":
        return get_regression_gate_test_cases()
    if suite == "full":
        return get_default_test_cases()
    raise ValueError(f"Unknown suite {suite!r}.")


def _latest_classification_csv() -> Path:
    candidates = sorted(
        repo_root.glob(
            "benchmarks/results/oracle_tree_recoverability_*/oracle_tree_recoverability.csv"
        )
    )
    for candidate in reversed(candidates):
        columns = pd.read_csv(candidate, nrows=0).columns
        if "failure_class" in columns:
            return candidate
    raise FileNotFoundError(
        "No oracle_tree_recoverability CSV with a failure_class column was found."
    )


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_classification(path: Path | None) -> tuple[pd.DataFrame, Path]:
    resolved = _latest_classification_csv() if path is None else path
    df = pd.read_csv(resolved)
    required = {
        "case_id",
        "failure_class",
        "tbs_ari",
        "oracle_true_k_subtree_ari",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Classification CSV {resolved} is missing columns: {sorted(missing)}.")
    return df, resolved


def _select_cases(
    cases: list[dict[str, object]],
    classification_df: pd.DataFrame,
    *,
    failure_classes: list[str],
    case_names: list[str],
) -> list[tuple[dict[str, object], pd.Series]]:
    case_by_name = {str(case["name"]): case for case in cases}
    selected = classification_df[
        classification_df["failure_class"].astype(str).isin(failure_classes)
    ].copy()
    if case_names:
        requested = set(case_names)
        selected = selected[selected["case_id"].astype(str).isin(requested)]
    if selected.empty:
        raise ValueError("No cases matched the requested failure class/name filters.")

    missing = [
        case_id for case_id in selected["case_id"].astype(str) if case_id not in case_by_name
    ]
    if missing:
        raise ValueError(
            f"Selected cases are not present in the {len(cases)}-case suite: {missing}."
        )
    return [
        (case_by_name[str(row.case_id)].copy(), row) for row in selected.itertuples(index=False)
    ]


def _make_output_dir(explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        output_dir = explicit_output_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_dir = (
            repo_root / "benchmarks" / "results" / f"selection_conditioned_sibling_null_{stamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _standardized_contrast_dimension(context) -> int:
    if context.feature_space is None:
        return int(context.data.shape[1])
    return int(context.feature_space.contrast_dimension)


def _select_target_rows(targets: pd.DataFrame, *, target_mode: str) -> pd.DataFrame:
    if target_mode == "blockers":
        selected = targets[targets["blocker_candidate"].astype(bool)].copy()
    elif target_mode == "current_blocks":
        selected = targets[targets["current_blocks_at_alpha"].astype(bool)].copy()
    elif target_mode == "all_focal":
        selected = targets.copy()
    else:
        raise ValueError(f"Unknown target_mode={target_mode!r}.")
    if selected.empty:
        raise ValueError(f"No sibling targets matched target_mode={target_mode!r}.")
    return selected


def _target_context(
    *,
    context,
    row: pd.Series,
    edge_projection_dimension: int,
) -> SelectionConditionedSiblingNullContext:
    left_child = row["left_child"]
    right_child = row["right_child"]
    return SelectionConditionedSiblingNullContext(
        case_id=str(row["case_id"]),
        parent=row["parent"],
        feature_family=str(row["feature_family"]),
        standardized_contrast_dimension=_standardized_contrast_dimension(context),
        edge_projection_dimension=int(edge_projection_dimension),
        sibling_projection_dimension=int(row["sibling_projection_dimension"]),
        degrees_of_freedom=float(row["degrees_of_freedom"]),
        reference_scale=float(row["reference_scale"]),
        observed_statistic=float(row["raw_sibling_statistic"]),
        observed_p_value=float(row["raw_sibling_p_value"]),
        edge_alpha=float(DEFAULT_EDGE_ALPHA),
        sibling_alpha=float(DEFAULT_SIBLING_ALPHA),
        parent_sample_size=int(row["parent_sample_size"]),
        left_sample_size=int(context.tree.nodes[left_child]["leaf_count"]),
        right_sample_size=int(context.tree.nodes[right_child]["leaf_count"]),
    )


def _append_selection_conditioned_columns(
    targets: pd.DataFrame,
    results: list[dict[str, object]],
) -> pd.DataFrame:
    result_df = pd.DataFrame.from_records(results)
    if result_df.empty:
        raise ValueError("Selection-conditioned diagnostic produced no result rows.")
    merged = targets.merge(
        result_df,
        on=["case_id", "parent"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_selection"),
    )
    if len(merged) != len(targets):
        raise ValueError(
            "Selection-conditioned diagnostic did not cover every target row: "
            f"{len(merged)} of {len(targets)} rows matched."
        )
    return merged


def _diagnose_case(
    case: dict[str, object],
    classification_row,
    *,
    target_mode: str,
    n_candidates: int,
    chunk_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = build_tbs_tree_context(case, populate_node_distributions=True)
    gate_annotation_bundle = run_gate_annotation_pipeline(
        context.tree,
        context.tree.annotations_df,
        edge_alpha=DEFAULT_EDGE_ALPHA,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
        leaf_data=context.data,
        feature_space=context.feature_space,
    )
    edge_gate_result = gate_annotation_bundle.edge_gate_result

    decomposer = TreeDecomposition(
        tree=context.tree,
        gate_annotation_bundle=gate_annotation_bundle,
        passthrough=True,
        trace_level="full",
    )
    decomposition = decomposer.decompose_tree()

    true_k = int(context.metadata["n_clusters"])
    oracle_any = oracle_subtree_cut(
        context.tree,
        sample_index=context.data.index,
        true_labels=context.true_labels,
    )
    oracle_true_k = oracle_subtree_cut(
        context.tree,
        sample_index=context.data.index,
        true_labels=context.true_labels,
        exact_k=true_k,
    )
    trace_df = build_gate_path_trace_dataframe(
        tree=context.tree,
        annotations_df=gate_annotation_bundle.annotated_df,
        decomposition=decomposition,
        oracle_true_k_boundary_nodes=oracle_true_k.selected_nodes,
        oracle_any_k_boundary_nodes=oracle_any.selected_nodes,
        sibling_inflation_trace_by_parent={},
        case_id=str(context.metadata["name"]),
        failure_class=str(classification_row.failure_class),
        tbs_ari=float(classification_row.tbs_ari),
        oracle_true_k_ari=float(oracle_true_k.ari),
        oracle_any_k_ari=float(oracle_any.ari),
        passthrough=True,
    )

    inputs = collect_sibling_inflation_inputs(
        context.tree,
        gate_annotation_bundle,
        feature_space=context.feature_space,
    )
    if inputs.model is None:
        raise ValueError(
            f"Case {context.metadata['name']!r} has no focal sibling records to diagnose."
        )
    tables = build_sibling_inflation_diagnostic_tables(
        records=inputs.records,
        model=inputs.model,
        trace_df=trace_df,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
        max_contributors=0,
        contributors_for_all_crossings=False,
    )
    targets = _select_target_rows(tables.targets, target_mode=target_mode)

    spectral_dimensions = edge_gate_result.spectral_context.test_projection_dimensions_by_node
    result_rows: list[dict[str, object]] = []
    for row_index, (_target_index, row) in enumerate(targets.iterrows(), start=1):
        parent = row["parent"]
        if parent not in spectral_dimensions:
            raise ValueError(
                f"Missing edge gate test projection dimension for target parent {parent!r}."
            )
        diagnostic_context = _target_context(
            context=context,
            row=row,
            edge_projection_dimension=int(spectral_dimensions[parent]),
        )
        result = simulate_local_edge_selection_conditioned_null(
            diagnostic_context,
            n_candidates=n_candidates,
            seed=seed + row_index,
            chunk_size=chunk_size,
        )
        result_rows.append(result.as_row())

    merged = _append_selection_conditioned_columns(targets, result_rows)
    for frame in (merged, tables.summary):
        if frame.empty:
            continue
        frame.insert(2, "tree_distance_metric", context.tree_distance_metric)
        frame.insert(3, "tree_distance_source", context.tree_distance_source)
        frame.insert(4, "tree_linkage_method", context.tree_linkage_method)
    return merged, tables.summary


def _summary_from_targets(targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_id, group in targets.groupby("case_id", sort=False):
        rows.append(
            {
                "case_id": case_id,
                "n_targets": int(len(group)),
                "n_continuous_local_edge_selection": int(
                    (group["conditioning_scope"] == CONTINUOUS_LOCAL_EDGE_SELECTION_SCOPE).sum()
                ),
                "n_noncontinuous_local_edge_z_proxy": int(
                    (group["conditioning_scope"] == NONCONTINUOUS_LOCAL_EDGE_Z_PROXY_SCOPE).sum()
                ),
                "median_current_empirical_inflation_factor": float(
                    group["current_empirical_inflation_factor"].median()
                ),
                "median_selection_conditioned_c_hat": float(
                    group["selection_conditioned_c_hat"].median()
                ),
                "median_acceptance_rate": float(group["acceptance_rate"].median()),
                "n_selection_blocks_at_alpha": int(group["selection_blocks_at_alpha"].sum()),
                "min_selection_empirical_tail_p_value": float(
                    group["selection_empirical_tail_p_value"].min()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = _parse_args()
    _configure_runtime_defaults()

    classification_df, classification_path = _load_classification(args.classification_csv)
    selected = _select_cases(
        _load_cases(args.suite),
        classification_df,
        failure_classes=_parse_csv_list(args.failure_classes),
        case_names=_parse_csv_list(args.case_names),
    )
    output_dir = _make_output_dir(args.output_dir)
    targets_csv = output_dir / "selection_conditioned_sibling_null_targets.csv"
    case_summary_csv = output_dir / "selection_conditioned_sibling_null_case_summary.csv"
    sibling_summary_csv = output_dir / "sibling_inflation_summary.csv"
    metadata_json = output_dir / "selection_conditioned_sibling_null_metadata.json"

    started_at = time.perf_counter()
    target_frames: list[pd.DataFrame] = []
    sibling_summary_frames: list[pd.DataFrame] = []
    for index, (case, classification_row) in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] {case['name']} ({classification_row.failure_class})",
            flush=True,
        )
        targets, sibling_summary = _diagnose_case(
            case,
            classification_row,
            target_mode=str(args.target_mode),
            n_candidates=int(args.n_candidates),
            chunk_size=int(args.chunk_size),
            seed=int(args.seed) + index * 1_000_000,
        )
        target_frames.append(targets)
        sibling_summary_frames.append(sibling_summary)

    targets_df = pd.concat(target_frames, ignore_index=True)
    case_summary_df = _summary_from_targets(targets_df)
    sibling_summary_df = pd.concat(sibling_summary_frames, ignore_index=True)
    targets_df.to_csv(targets_csv, index=False)
    case_summary_df.to_csv(case_summary_csv, index=False)
    sibling_summary_df.to_csv(sibling_summary_csv, index=False)

    elapsed_sec = time.perf_counter() - started_at
    metadata = {
        "suite": args.suite,
        "classification_csv": str(classification_path),
        "failure_classes": _parse_csv_list(args.failure_classes),
        "case_names": [str(case["name"]) for case, _row in selected],
        "target_mode": str(args.target_mode),
        "n_cases": len(selected),
        "n_candidates": int(args.n_candidates),
        "chunk_size": int(args.chunk_size),
        "seed": int(args.seed),
        "edge_alpha": float(DEFAULT_EDGE_ALPHA),
        "sibling_alpha": float(DEFAULT_SIBLING_ALPHA),
        "conditioning_scope": ("fixed_tree_fixed_projection_local_child_parent_edge_selection"),
        "elapsed_sec": round(elapsed_sec, 6),
        "targets_csv": str(targets_csv),
        "case_summary_csv": str(case_summary_csv),
        "sibling_summary_csv": str(sibling_summary_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Selection-conditioned sibling null complete in {elapsed_sec:.2f}s")
    print(f"Targets: {targets_csv}")
    print(f"Case summary: {case_summary_csv}")
    print(f"Sibling summary: {sibling_summary_csv}")
    print(f"Metadata: {metadata_json}")
    print(case_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
