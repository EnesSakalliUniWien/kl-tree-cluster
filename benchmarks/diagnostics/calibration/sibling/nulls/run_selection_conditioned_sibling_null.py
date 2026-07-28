#!/usr/bin/env python3
"""Run a local selection-conditioned sibling-null diagnostic."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)

from benchmarks.diagnostics.calibration.sibling.nulls.runner_support import (
    prepare_sibling_diagnostic,
    select_sibling_target_rows,
    standardized_contrast_dimension,
)
from benchmarks.diagnostics.calibration.sibling.nulls.selection_conditioned_sibling_null import (
    CONTINUOUS_LOCAL_EDGE_SELECTION_SCOPE,
    NONCONTINUOUS_LOCAL_EDGE_Z_PROXY_SCOPE,
    SelectionConditionedSiblingNullContext,
    simulate_local_edge_selection_conditioned_null,
)
from benchmarks.diagnostics.oracle.oracle_tree_recoverability import (
    FAILURE_CLASS_GATE_UNDER_SPLIT,
)
from benchmarks.diagnostics.runner_support import (
    configure_serial_runtime,
    create_result_directory,
    parse_csv_values,
    resolve_classified_cases,
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
        standardized_contrast_dimension=standardized_contrast_dimension(context),
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
    classification: dict[str, object],
    *,
    target_mode: str,
    n_candidates: int,
    chunk_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_sibling_diagnostic(
        case,
        classification,
        max_contributors=0,
        contributors_for_all_crossings=False,
    )
    context = prepared.context
    targets = select_sibling_target_rows(
        prepared.tables.targets,
        target_mode=target_mode,
    )

    spectral_dimensions = (
        prepared.gate_annotation_bundle.edge_gate_result.spectral_context
        .test_projection_dimensions_by_node
    )
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
    return merged, prepared.tables.summary


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
    configure_serial_runtime()

    selection = resolve_classified_cases(
        suite=args.suite,
        classification_csv=args.classification_csv,
        failure_classes=parse_csv_values(args.failure_classes),
        case_names=parse_csv_values(args.case_names),
        required_columns=("tbs_ari", "oracle_true_k_subtree_ari"),
    )
    output_dir = create_result_directory(
        args.output_dir,
        study_slug="selection_conditioned_sibling_null",
    )
    targets_csv = output_dir / "selection_conditioned_sibling_null_targets.csv"
    case_summary_csv = output_dir / "selection_conditioned_sibling_null_case_summary.csv"
    sibling_summary_csv = output_dir / "sibling_inflation_summary.csv"
    metadata_json = output_dir / "selection_conditioned_sibling_null_metadata.json"

    started_at = time.perf_counter()
    target_frames: list[pd.DataFrame] = []
    sibling_summary_frames: list[pd.DataFrame] = []
    for index, selected_case in enumerate(selection.cases, start=1):
        case = selected_case.case
        classification = selected_case.classification
        print(
            f"[{index}/{len(selection.cases)}] "
            f"{case['name']} ({classification['failure_class']})",
            flush=True,
        )
        targets, sibling_summary = _diagnose_case(
            case,
            dict(classification),
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
        "classification_csv": str(selection.classification_path),
        "failure_classes": list(parse_csv_values(args.failure_classes)),
        "case_names": [str(item.case["name"]) for item in selection.cases],
        "target_mode": str(args.target_mode),
        "n_cases": len(selection.cases),
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
