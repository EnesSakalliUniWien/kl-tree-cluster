"""Mixed null/signal sweep for internal calibration thresholds and Q10 leakage.

The sweep builds TBS trees on benchmark cases, collects sibling records with
internal support labels, and emits the labeled panels consumed by the Q9/Q10
diagnostics. It is diagnostic-only and does not promote threshold values or
install a replacement empirical-null weight rule.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    decide_empirical_null_calibration,
    fit_empirical_null_inflation_model,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.child_parent_edge_metadata import (
    extract_child_parent_edge_p_values_by_node,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)

from benchmarks.diagnostics.calibration.reporting import print_diagnostic_output_paths
from benchmarks.diagnostics.calibration.sibling.nulls.sibling_null_weight_rule_validation import (
    evaluate_sibling_null_weight_rules,
)
from benchmarks.diagnostics.calibration.statistics.internal_support_threshold_validation import (
    evaluate_internal_support_threshold_rows,
    summarize_internal_support_threshold_rows,
)
from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.tbs_tree_context import build_tbs_tree_context
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_mixed_internal_calibration_sweep_not_calibration"
SCHEMA_VERSION = "mixed_internal_calibration_sweep/v1"


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _select_cases(*, suite: str, case_names: Sequence[str]) -> list[dict[str, object]]:
    cases = get_test_cases_by_suite(suite)
    if not case_names:
        return [case.copy() for case in cases]
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case name(s) for suite {suite!r}: {missing!r}.")
    return [by_name[name].copy() for name in case_names]


def _case_with_replicate_seed(case: dict[str, object], replicate_index: int) -> dict[str, object]:
    replicated = case.copy()
    if "seed" in replicated:
        replicated["seed"] = int(replicated["seed"]) + int(replicate_index) * 100_003
    return replicated


def _record_selected_ratio(record: SiblingPairRecord) -> float:
    reference_expectation = float(record.reference_scale * record.degrees_of_freedom)
    if reference_expectation <= 0.0:
        return 0.0
    return float(record.stat / reference_expectation)


def _edge_weight_probability(edge_p_by_node: dict[object, float], child: object) -> float:
    value = float(edge_p_by_node[child])
    if not np.isfinite(value):
        return 1.0
    return value


def _true_context_labels(
    *,
    tree,
    true_labels: np.ndarray,
    leaf_index: pd.Index,
    record: SiblingPairRecord,
) -> dict[str, object]:
    descendants = tree.compute_descendant_sets(use_labels=True)
    label_by_leaf = {leaf: true_labels[position] for position, leaf in enumerate(leaf_index)}

    def labels_for(node: object) -> np.ndarray:
        leaves = tuple(descendants[node])
        return np.asarray([label_by_leaf[leaf] for leaf in leaves], dtype=object)

    parent_labels = labels_for(record.parent)
    left_labels = labels_for(record.left)
    right_labels = labels_for(record.right)
    parent_unique = set(parent_labels.tolist())
    left_counts = pd.Series(left_labels).value_counts()
    right_counts = pd.Series(right_labels).value_counts()
    left_majority = left_counts.index[0] if not left_counts.empty else None
    right_majority = right_counts.index[0] if not right_counts.empty else None
    is_null_context = len(parent_unique) <= 1
    is_signal_context = (not is_null_context) and left_majority != right_majority
    return {
        "is_null_context": bool(is_null_context),
        "is_signal_context": bool(is_signal_context),
        "true_parent_label_count": int(len(parent_unique)),
        "true_left_majority_label": "" if left_majority is None else str(left_majority),
        "true_right_majority_label": "" if right_majority is None else str(right_majority),
    }


def _collect_case_replicate(
    *,
    case: dict[str, object],
    replicate_index: int,
    edge_alpha: float,
    sibling_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    context = build_tbs_tree_context(
        _case_with_replicate_seed(case, replicate_index),
        populate_node_distributions=True,
    )
    edge_df, spectral_context = annotate_child_parent_divergence(
        context.tree,
        context.tree.annotations_df,
        significance_level_alpha=edge_alpha,
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
    records, non_binary_nodes = collect_sibling_pair_records(
        context.tree,
        edge_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=context.feature_space,
    )
    edge_p_by_node = extract_child_parent_edge_p_values_by_node(edge_df)

    q10_rows: list[dict[str, object]] = []
    for record in records:
        if record.degrees_of_freedom <= 0.0:
            continue
        true_context = _true_context_labels(
            tree=context.tree,
            true_labels=context.true_labels,
            leaf_index=context.data.index,
            record=record,
        )
        q10_rows.append(
            {
                "case_id": str(context.metadata["name"]),
                "replicate_index": int(replicate_index),
                "parent": str(record.parent),
                "left_child": str(record.left),
                "right_child": str(record.right),
                "left_edge_bh_p_value": _edge_weight_probability(
                    edge_p_by_node,
                    record.left,
                ),
                "right_edge_bh_p_value": _edge_weight_probability(
                    edge_p_by_node,
                    record.right,
                ),
                "selected_hierarchy_ratio": _record_selected_ratio(record),
                "is_null_like": bool(record.is_null_like),
                "is_edge_blocked": bool(record.is_edge_blocked),
                "sibling_null_weight": float(record.sibling_null_weight),
                "feature_family": str(record.feature_family),
                "sibling_projection_dimension": float(record.sibling_projection_dimension),
                "parent_sample_size": int(record.n_parent),
                **true_context,
                "study_role": STUDY_ROLE,
            }
        )

    threshold_rows: list[dict[str, object]] = []
    model_status = "not_fit"
    model_error = ""
    try:
        model = fit_empirical_null_inflation_model(records)
        model_status = "fit"
        for record in records:
            if record.is_null_like or record.degrees_of_freedom <= 0.0:
                continue
            decision = decide_empirical_null_calibration(model, record)
            support = dict(decision.support)
            true_context = _true_context_labels(
                tree=context.tree,
                true_labels=context.true_labels,
                leaf_index=context.data.index,
                record=record,
            )
            threshold_rows.append(
                {
                    "context_id": (
                        f"{context.metadata['name']}__r{replicate_index}__{record.parent}"
                    ),
                    "case_id": str(context.metadata["name"]),
                    "replicate_index": int(replicate_index),
                    "parent": str(record.parent),
                    "calibration_status": decision.status,
                    "calibration_estimator": decision.estimator,
                    "calibration_p_value": (
                        np.nan if decision.p_value is None else float(decision.p_value)
                    ),
                    "split_rejected_at_alpha": bool(
                        decision.status == "internal_admissible"
                        and decision.p_value is not None
                        and decision.p_value <= sibling_alpha
                    ),
                    "n_supported_records": int(support.get("n_supported_records", 0)),
                    "n_family_supported_records": int(support.get("n_family_supported_records", 0)),
                    "n_stopped_or_null_records": int(support.get("n_stopped_or_null_records", 0)),
                    "family_effective_sample_size": float(
                        support.get("family_effective_sample_size", 0.0)
                    ),
                    "local_effective_sample_size": float(
                        support.get("local_effective_sample_size", 0.0)
                    ),
                    "local_max_weight_share": float(support.get("local_max_weight_share", 1.0)),
                    "leave_one_record_max_delta_log_c": float(
                        support.get("leave_one_record_max_delta_log_c", float("inf"))
                    ),
                    "n_selected_nonnull_positive_weight_records": int(
                        support.get("n_selected_nonnull_positive_weight_records", 0)
                    ),
                    **true_context,
                    "study_role": STUDY_ROLE,
                }
            )
    except Exception as exc:
        model_status = "skip"
        model_error = str(exc)

    tested_edges = edge_df["Child_Parent_Divergence_Tested"].astype(bool)
    significant_edges = edge_df["Child_Parent_Divergence_Significant"].astype(bool)
    status = {
        "case_id": str(context.metadata["name"]),
        "replicate_index": int(replicate_index),
        "status": "ok",
        "model_status": model_status,
        "model_error": model_error,
        "n_records": int(len(records)),
        "n_q10_records": int(len(q10_rows)),
        "n_threshold_contexts": int(len(threshold_rows)),
        "n_non_binary_nodes": int(len(non_binary_nodes)),
        "n_tested_edges": int(tested_edges.sum()),
        "n_significant_edges": int(significant_edges.sum()),
        "edge_rejection_rate": float(significant_edges.sum() / max(int(tested_edges.sum()), 1)),
        "study_role": STUDY_ROLE,
    }
    return (
        pd.DataFrame.from_records(q10_rows),
        pd.DataFrame.from_records(threshold_rows),
        status,
    )


def run_mixed_internal_calibration_sweep(
    *,
    suite: str,
    case_names: Sequence[str],
    n_replicates: int,
    output_dir: Path,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    resume: bool = False,
) -> dict[str, Path]:
    """Run the mixed internal-calibration sweep and write diagnostic outputs."""
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    q10_records_path = output_dir / "mixed_q10_sibling_records.csv"
    threshold_contexts_path = output_dir / "mixed_internal_support_contexts.csv"
    status_path = output_dir / "mixed_internal_calibration_status.csv"
    if (
        resume
        and q10_records_path.exists()
        and threshold_contexts_path.exists()
        and status_path.exists()
    ):
        q10_records = pd.read_csv(q10_records_path)
        threshold_contexts = pd.read_csv(threshold_contexts_path)
        status = pd.read_csv(status_path)
    else:
        cases = _select_cases(suite=suite, case_names=case_names)
        q10_tables: list[pd.DataFrame] = []
        threshold_tables: list[pd.DataFrame] = []
        status_rows: list[dict[str, object]] = []
        for case_index, case in enumerate(cases, start=1):
            for replicate_index in range(int(n_replicates)):
                print(
                    f"[{case_index}/{len(cases)}] {case['name']} "
                    f"replicate {replicate_index + 1}/{n_replicates}",
                    flush=True,
                )
                try:
                    q10_table, threshold_table, row = _collect_case_replicate(
                        case=case,
                        replicate_index=replicate_index,
                        edge_alpha=edge_alpha,
                        sibling_alpha=sibling_alpha,
                    )
                    q10_tables.append(q10_table)
                    threshold_tables.append(threshold_table)
                    status_rows.append(row)
                except Exception as exc:
                    status_rows.append(
                        {
                            "case_id": str(case["name"]),
                            "replicate_index": int(replicate_index),
                            "status": "skip",
                            "model_status": "skip",
                            "model_error": str(exc),
                            "n_records": 0,
                            "n_q10_records": 0,
                            "n_threshold_contexts": 0,
                            "n_non_binary_nodes": 0,
                            "n_tested_edges": 0,
                            "n_significant_edges": 0,
                            "edge_rejection_rate": np.nan,
                            "study_role": STUDY_ROLE,
                        }
                    )
        q10_records = pd.concat(q10_tables, ignore_index=True) if q10_tables else pd.DataFrame()
        threshold_contexts = (
            pd.concat(threshold_tables, ignore_index=True) if threshold_tables else pd.DataFrame()
        )
        status = pd.DataFrame.from_records(status_rows)
        q10_records.to_csv(q10_records_path, index=False)
        threshold_contexts.to_csv(threshold_contexts_path, index=False)
        status.to_csv(status_path, index=False)

    weight_summary = evaluate_sibling_null_weight_rules(q10_records)
    threshold_decisions = evaluate_internal_support_threshold_rows(threshold_contexts)
    threshold_summary = summarize_internal_support_threshold_rows(threshold_decisions)

    weight_summary_path = output_dir / "mixed_q10_weight_rule_summary.csv"
    threshold_decisions_path = output_dir / "mixed_internal_support_threshold_decisions.csv"
    threshold_summary_path = output_dir / "mixed_internal_support_threshold_summary.csv"
    manifest_path = output_dir / "manifest.json"
    weight_summary.to_csv(weight_summary_path, index=False)
    threshold_decisions.to_csv(threshold_decisions_path, index=False)
    threshold_summary.to_csv(threshold_summary_path, index=False)

    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "suite": suite,
        "case_names": list(case_names),
        "n_replicates": int(n_replicates),
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "outputs": {
            "q10_records": str(q10_records_path),
            "q10_weight_summary": str(weight_summary_path),
            "threshold_contexts": str(threshold_contexts_path),
            "threshold_decisions": str(threshold_decisions_path),
            "threshold_summary": str(threshold_summary_path),
            "status": str(status_path),
        },
        "n_q10_records": int(q10_records.shape[0]),
        "n_threshold_contexts": int(threshold_contexts.shape[0]),
        "n_status_rows": int(status.shape[0]),
        "elapsed_note": "Elapsed seconds are printed by the caller when needed.",
        "interpretation": (
            "Diagnostic mixed null/signal sweep. These outputs can falsify or "
            "support threshold and weight-rule candidates but do not promote a "
            "production calibration rule by themselves."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "q10_records": q10_records_path,
        "q10_weight_summary": weight_summary_path,
        "threshold_contexts": threshold_contexts_path,
        "threshold_decisions": threshold_decisions_path,
        "threshold_summary": threshold_summary_path,
        "status": status_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="method_proof")
    parser.add_argument("--case-names", default="")
    parser.add_argument("--n-replicates", type=int, default=25)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    started = perf_counter()
    outputs = run_mixed_internal_calibration_sweep(
        suite=str(args.suite),
        case_names=_parse_csv_list(str(args.case_names)),
        n_replicates=int(args.n_replicates),
        output_dir=args.output_dir,
        edge_alpha=float(args.edge_alpha),
        sibling_alpha=float(args.sibling_alpha),
        resume=bool(args.resume),
    )
    print_diagnostic_output_paths(outputs)
    print(f"elapsed_sec={perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA_VERSION",
    "STUDY_ROLE",
    "run_mixed_internal_calibration_sweep",
]
