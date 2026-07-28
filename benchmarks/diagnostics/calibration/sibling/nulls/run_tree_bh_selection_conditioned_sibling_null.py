#!/usr/bin/env python3
"""Run the root Tree-BH selection-conditioned sibling-null diagnostic."""

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
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    fit_empirical_null_inflation_model,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)

from benchmarks.diagnostics.calibration.sibling.nulls.runner_support import (
    standardized_contrast_dimension,
)
from benchmarks.diagnostics.calibration.sibling.nulls.selection_conditioned_sibling_null import (
    SelectionConditionedSiblingNullContext,
    simulate_local_edge_selection_conditioned_null,
)
from benchmarks.diagnostics.oracle.oracle_tree_recoverability import FAILURE_CLASS_GATE_UNDER_SPLIT
from benchmarks.diagnostics.runner_support import (
    configure_serial_runtime,
    create_result_directory,
    parse_csv_values,
    resolve_classified_cases,
)
from benchmarks.shared.tbs_tree_context import build_tbs_tree_context

_DEFAULT_GAUSSIAN_BLOCKERS = (
    "gauss_extreme_noise_highd",
    "gauss_extreme_noise_highd_continuous",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the sibling null conditioned on the fixed observed tree "
            "and the Tree-BH edge path for root blocker contexts. For binary "
            "root blockers this Tree-BH event is equivalent to the local edge "
            "selection event; the output records that equivalence explicitly."
        )
    )
    parser.add_argument("--suite", choices=("regression_gate", "full"), default="full")
    parser.add_argument("--classification-csv", type=Path, default=None)
    parser.add_argument("--failure-classes", default=FAILURE_CLASS_GATE_UNDER_SPLIT)
    parser.add_argument("--case-names", default=",".join(_DEFAULT_GAUSSIAN_BLOCKERS))
    parser.add_argument("--n-candidates", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _production_calibration_status(records) -> tuple[str, str]:
    try:
        fit_empirical_null_inflation_model(list(records))
    except ValueError as exc:
        return "unsupported", str(exc)
    return "supported", ""


def _diagnose_case(
    case: dict[str, object],
    *,
    n_candidates: int,
    chunk_size: int,
    seed: int,
) -> dict[str, object]:
    context = build_tbs_tree_context(case, populate_node_distributions=True)
    edge_df, spectral_context = annotate_child_parent_divergence(
        context.tree,
        context.tree.annotations_df,
        significance_level_alpha=DEFAULT_EDGE_ALPHA,
        leaf_data=context.data,
        feature_space=context.feature_space,
    )
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        context.tree,
        spectral_context=spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=spectral_context,
        )
    )
    records, _non_binary = collect_sibling_pair_records(
        context.tree,
        edge_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=context.feature_space,
    )

    root = context.tree.root()
    target_matches = [record for record in records if record.parent == root]
    if len(target_matches) != 1:
        raise ValueError(
            "Tree-BH selection diagnostic requires exactly one root sibling "
            f"record; found {len(target_matches)} for root={root!r}."
        )
    target = target_matches[0]
    children = list(context.tree.successors(root))
    if len(children) != 2 or {target.left, target.right} != set(children):
        raise ValueError(
            "Root Tree-BH equivalence requires the target sibling pair to be "
            "the binary root sibling group."
        )

    edge_k = int(spectral_context.test_projection_dimensions_by_node[root])
    diagnostic_context = SelectionConditionedSiblingNullContext(
        case_id=str(context.metadata["name"]),
        parent=target.parent,
        feature_family=target.feature_family,
        standardized_contrast_dimension=standardized_contrast_dimension(context),
        edge_projection_dimension=edge_k,
        sibling_projection_dimension=int(target.sibling_projection_dimension),
        degrees_of_freedom=float(target.degrees_of_freedom),
        reference_scale=float(target.reference_scale),
        observed_statistic=float(target.stat),
        observed_p_value=float(target.p_value),
        edge_alpha=float(DEFAULT_EDGE_ALPHA),
        sibling_alpha=float(DEFAULT_SIBLING_ALPHA),
        parent_sample_size=int(target.n_parent),
        left_sample_size=int(context.tree.nodes[target.left]["leaf_count"]),
        right_sample_size=int(context.tree.nodes[target.right]["leaf_count"]),
    )
    result = simulate_local_edge_selection_conditioned_null(
        diagnostic_context,
        n_candidates=n_candidates,
        seed=seed,
        chunk_size=chunk_size,
    )
    calibration_status, calibration_error = _production_calibration_status(records)
    row = result.as_row()
    row.update(
        {
            "tree_distance_metric": context.tree_distance_metric,
            "tree_distance_source": context.tree_distance_source,
            "tree_linkage_method": context.tree_linkage_method,
            "tree_bh_conditioning_scope": (
                "fixed_tree_root_tree_bh_edge_path_and_fixed_blocker_parent"
            ),
            "tree_bh_equivalent_to_local_edge_selection": True,
            "production_calibration_status": calibration_status,
            "production_calibration_error": calibration_error,
        }
    )
    return row


def main() -> None:
    args = _parse_args()
    configure_serial_runtime()
    selection = resolve_classified_cases(
        suite=args.suite,
        classification_csv=args.classification_csv,
        failure_classes=parse_csv_values(args.failure_classes),
        case_names=parse_csv_values(args.case_names),
    )
    output_dir = create_result_directory(
        args.output_dir,
        study_slug="tree_bh_selection_conditioned_sibling_null",
    )
    targets_csv = output_dir / "tree_bh_selection_conditioned_sibling_null_targets.csv"
    metadata_json = output_dir / "tree_bh_selection_conditioned_sibling_null_metadata.json"

    started_at = time.perf_counter()
    rows: list[dict[str, object]] = []
    for index, selected_case in enumerate(selection.cases, start=1):
        case = selected_case.case
        print(f"[{index}/{len(selection.cases)}] {case['name']}", flush=True)
        rows.append(
            _diagnose_case(
                case,
                n_candidates=int(args.n_candidates),
                chunk_size=int(args.chunk_size),
                seed=int(args.seed) + index * 1_000_000,
            )
        )

    targets_df = pd.DataFrame.from_records(rows)
    targets_df.to_csv(targets_csv, index=False)

    elapsed_sec = time.perf_counter() - started_at
    metadata = {
        "suite": args.suite,
        "classification_csv": str(selection.classification_path),
        "failure_classes": list(parse_csv_values(args.failure_classes)),
        "case_names": [str(item.case["name"]) for item in selection.cases],
        "n_cases": len(selection.cases),
        "n_candidates": int(args.n_candidates),
        "chunk_size": int(args.chunk_size),
        "seed": int(args.seed),
        "edge_alpha": float(DEFAULT_EDGE_ALPHA),
        "sibling_alpha": float(DEFAULT_SIBLING_ALPHA),
        "conditioning_scope": ("fixed_tree_root_tree_bh_edge_path_and_fixed_blocker_parent"),
        "elapsed_sec": round(elapsed_sec, 6),
        "targets_csv": str(targets_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Tree-BH selection-conditioned sibling null complete in {elapsed_sec:.2f}s")
    print(f"Targets: {targets_csv}")
    print(f"Metadata: {metadata_json}")
    print(
        targets_df[
            [
                "case_id",
                "acceptance_rate",
                "selection_conditioned_c_hat",
                "selection_empirical_tail_p_value",
                "selection_blocks_at_alpha",
                "production_calibration_status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
