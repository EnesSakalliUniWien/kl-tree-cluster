"""Behavioral MP/projection-dimension sweep for sibling tests.

This diagnostic reruns raw sibling projected-Wald tests on a fitted TBS tree
under candidate projection-dimension rules. It is diagnostic-only: it does not
alter production MP thresholds, production zero-dimensional semantics, or
calibrated sibling p-values.
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
    annotate_child_parent_divergence_with_context,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.pair_observations import (
    extract_sibling_pair_observations,
    identify_binary_sibling_children,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic.sibling_divergence_test import (
    sibling_divergence_test,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from tree_break_selection.tree.distributions import (
    require_node_continuous_covariance_by_block,
)

from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.tbs_tree_context import build_tbs_tree_context
from benchmarks.shared.util.time import format_timestamp_utc

SCHEMA_VERSION = "mp_projection_dimension_behavior_sweep/v1"
STUDY_ROLE = "diagnostic_mp_projection_dimension_behavior_sweep_not_calibration"
RULE_IDS = (
    "current_edge_derived_rule",
    "parent_test_projection_dimension",
    "raw_mp_parent_signal_count",
    "raw_mp_parent_signal_count_floor1",
    "raw_mp_parent_signal_count_floor2",
)


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


def candidate_projection_dimensions(
    *,
    current_edge_derived_dimension: int,
    parent_test_projection_dimension: int,
    raw_mp_signal_count: int,
) -> dict[str, int]:
    """Return candidate sibling dimensions capped to the parent PCA basis."""
    current = int(current_edge_derived_dimension)
    parent = int(parent_test_projection_dimension)
    raw_mp = int(raw_mp_signal_count)
    if current < 0 or parent < 0 or raw_mp < 0:
        raise ValueError("Projection dimensions and raw MP counts must be non-negative.")
    return {
        "current_edge_derived_rule": current,
        "parent_test_projection_dimension": parent,
        "raw_mp_parent_signal_count": min(raw_mp, parent),
        "raw_mp_parent_signal_count_floor1": min(max(raw_mp, 1), parent),
        "raw_mp_parent_signal_count_floor2": min(max(raw_mp, 2), parent),
    }


def _true_context_labels(
    *,
    tree,
    true_labels: np.ndarray,
    leaf_index: pd.Index,
    parent: object,
    left_child: object,
    right_child: object,
) -> dict[str, object]:
    descendants = tree.compute_descendant_sets(use_labels=True)
    label_by_leaf = {
        leaf: true_labels[position]
        for position, leaf in enumerate(leaf_index)
    }

    def labels_for(node: object) -> np.ndarray:
        leaves = tuple(descendants[node])
        return np.asarray([label_by_leaf[leaf] for leaf in leaves], dtype=object)

    parent_labels = labels_for(parent)
    left_labels = labels_for(left_child)
    right_labels = labels_for(right_child)
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


def _zero_dimension_result() -> tuple[float, float, float, float]:
    """Diagnostic mathematical semantics for a zero-dimensional projection."""
    return 0.0, 1.0, 0.0, 1.0


def _run_sibling_test_for_dimension(
    *,
    projection_dimension: int,
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    left_sample_size: float,
    right_sample_size: float,
    parent_projection: np.ndarray | None,
    parent_eigenvalues: np.ndarray | None,
    feature_space,
    continuous_covariance_by_block,
) -> tuple[float, float, float, float, str, str]:
    if int(projection_dimension) == 0:
        stat, scale, degrees, p_value = _zero_dimension_result()
        return stat, scale, degrees, p_value, "zero_dimensional_projection", ""
    try:
        stat, scale, degrees, p_value = sibling_divergence_test(
            left_distribution,
            right_distribution,
            float(left_sample_size),
            float(right_sample_size),
            projection_dimension_from_edge_comparisons=int(projection_dimension),
            parent_principal_component_projection=parent_projection,
            parent_principal_component_eigenvalues=parent_eigenvalues,
            feature_space=feature_space,
            continuous_covariance_by_block=continuous_covariance_by_block,
        )
    except Exception as exc:
        return np.nan, np.nan, np.nan, np.nan, "error", str(exc)
    return float(stat), float(scale), float(degrees), float(p_value), "ok", ""


def evaluate_projection_behavior_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate-rule behavior against conservative true labels."""
    if records.empty:
        return pd.DataFrame(
            columns=[
                "rule_id",
                "n_records",
                "n_evaluated",
                "n_errors",
                "dimension_mean",
                "frequency_k0",
                "frequency_k1",
                "frequency_k2",
                "frequency_k_ge3",
                "differs_from_current_fraction",
                "rejection_rate",
                "null_false_split_rate",
                "signal_retention_rate",
                "study_role",
            ]
        )
    rows: list[dict[str, object]] = []
    for rule_id, group in records.groupby("rule_id", sort=False):
        dimensions = pd.to_numeric(group["projection_dimension"], errors="raise").to_numpy(
            dtype=int
        )
        status = group["test_status"].astype(str)
        evaluated = status.isin({"ok", "zero_dimensional_projection"})
        rejected = group["raw_p_value"].le(group["sibling_alpha"]) & evaluated
        null_mask = group["is_null_context"].astype(bool) & evaluated
        signal_mask = group["is_signal_context"].astype(bool) & evaluated
        current_dimensions = pd.to_numeric(
            group["current_edge_derived_dimension"],
            errors="raise",
        ).to_numpy(dtype=int)
        rows.append(
            {
                "rule_id": str(rule_id),
                "n_records": int(group.shape[0]),
                "n_evaluated": int(evaluated.sum()),
                "n_errors": int(status.eq("error").sum()),
                "dimension_mean": float(np.mean(dimensions)),
                "dimension_median": float(np.median(dimensions)),
                "dimension_min": int(np.min(dimensions)),
                "dimension_max": int(np.max(dimensions)),
                "frequency_k0": float(np.mean(dimensions == 0)),
                "frequency_k1": float(np.mean(dimensions == 1)),
                "frequency_k2": float(np.mean(dimensions == 2)),
                "frequency_k_ge3": float(np.mean(dimensions >= 3)),
                "differs_from_current_fraction": float(
                    np.mean(dimensions != current_dimensions)
                ),
                "rejection_rate": (
                    float(rejected[evaluated].mean()) if bool(evaluated.any()) else np.nan
                ),
                "null_false_split_rate": (
                    float(rejected[null_mask].mean()) if bool(null_mask.any()) else np.nan
                ),
                "n_null_contexts": int(null_mask.sum()),
                "signal_retention_rate": (
                    float(rejected[signal_mask].mean()) if bool(signal_mask.any()) else np.nan
                ),
                "n_signal_contexts": int(signal_mask.sum()),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows)


def _collect_case_replicate(
    *,
    case: dict[str, object],
    replicate_index: int,
    edge_alpha: float,
    sibling_alpha: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    context = build_tbs_tree_context(
        _case_with_replicate_seed(case, replicate_index),
        populate_node_distributions=True,
    )
    edge_df, spectral_context = annotate_child_parent_divergence_with_context(
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
    rows: list[dict[str, object]] = []
    non_binary_nodes = 0
    for parent in context.tree.nodes:
        sibling_children = identify_binary_sibling_children(context.tree, parent)
        if sibling_children is None:
            non_binary_nodes += 1
            continue
        left_child, right_child = sibling_children
        (
            left_distribution,
            right_distribution,
            left_sample_size,
            right_sample_size,
            _branch_length_left,
            _branch_length_right,
        ) = extract_sibling_pair_observations(
            context.tree,
            parent,
            left_child,
            right_child,
        )
        current_dimension = int(projection_dimensions[parent])
        parent_dimension = int(spectral_context.test_projection_dimensions_by_node[parent])
        raw_mp_count = int(spectral_context.raw_mp_signal_counts_by_node[parent])
        rule_dimensions = candidate_projection_dimensions(
            current_edge_derived_dimension=current_dimension,
            parent_test_projection_dimension=parent_dimension,
            raw_mp_signal_count=raw_mp_count,
        )
        parent_projection = spectral_context.principal_component_projections_by_node.get(parent)
        parent_eigenvalues = spectral_context.principal_component_eigenvalues_by_node.get(parent)
        continuous_covariance_by_block = require_node_continuous_covariance_by_block(
            context.tree,
            parent,
            context.feature_space,
        )
        true_context = _true_context_labels(
            tree=context.tree,
            true_labels=context.true_labels,
            leaf_index=context.data.index,
            parent=parent,
            left_child=left_child,
            right_child=right_child,
        )
        for rule_id in RULE_IDS:
            projection_dimension = int(rule_dimensions[rule_id])
            stat, scale, degrees, p_value, test_status, error = _run_sibling_test_for_dimension(
                projection_dimension=projection_dimension,
                left_distribution=left_distribution,
                right_distribution=right_distribution,
                left_sample_size=float(left_sample_size),
                right_sample_size=float(right_sample_size),
                parent_projection=parent_projection,
                parent_eigenvalues=parent_eigenvalues,
                feature_space=context.feature_space,
                continuous_covariance_by_block=continuous_covariance_by_block,
            )
            reference_expectation = float(scale * degrees)
            selected_ratio = (
                float(stat / reference_expectation)
                if np.isfinite(stat) and reference_expectation > 0.0
                else 0.0
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "case_id": str(context.metadata["name"]),
                    "case_category": str(context.metadata.get("category", "")),
                    "case_generator": str(context.metadata.get("generator", "")),
                    "replicate_index": int(replicate_index),
                    "parent": str(parent),
                    "left_child": str(left_child),
                    "right_child": str(right_child),
                    "rule_id": rule_id,
                    "projection_dimension": projection_dimension,
                    "current_edge_derived_dimension": current_dimension,
                    "parent_test_projection_dimension": parent_dimension,
                    "raw_mp_signal_count": raw_mp_count,
                    "parent_sample_size": int(context.tree.nodes[parent]["leaf_count"]),
                    "left_sample_size": float(left_sample_size),
                    "right_sample_size": float(right_sample_size),
                    "edge_alpha": float(edge_alpha),
                    "sibling_alpha": float(sibling_alpha),
                    "raw_statistic": stat,
                    "reference_scale": scale,
                    "degrees_of_freedom": degrees,
                    "raw_p_value": p_value,
                    "raw_rejected_at_alpha": bool(
                        np.isfinite(p_value) and float(p_value) <= float(sibling_alpha)
                    ),
                    "selected_hierarchy_ratio": selected_ratio,
                    "test_status": test_status,
                    "test_error": error,
                    "feature_family": (
                        "bernoulli"
                        if context.feature_space is None
                        else context.feature_space.family_label
                    ),
                    **true_context,
                    "study_role": STUDY_ROLE,
                }
            )
    status = {
        "case_id": str(context.metadata["name"]),
        "case_category": str(context.metadata.get("category", "")),
        "case_generator": str(context.metadata.get("generator", "")),
        "replicate_index": int(replicate_index),
        "status": "ok",
        "n_rule_records": int(len(rows)),
        "n_binary_sibling_parents": int(len(rows) // len(RULE_IDS)),
        "n_non_binary_nodes": int(non_binary_nodes),
        "study_role": STUDY_ROLE,
    }
    return pd.DataFrame.from_records(rows), status


def run_mp_projection_dimension_behavior_sweep(
    *,
    suite: str,
    case_names: Sequence[str],
    n_replicates: int,
    output_dir: Path,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    resume: bool = False,
) -> dict[str, Path]:
    """Run the behavioral MP/projection-dimension diagnostic sweep."""
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "mp_projection_dimension_behavior_records.csv"
    status_path = output_dir / "mp_projection_dimension_behavior_status.csv"
    if resume and records_path.exists() and status_path.exists():
        records = pd.read_csv(records_path)
        status = pd.read_csv(status_path)
    else:
        cases = _select_cases(suite=suite, case_names=case_names)
        tables: list[pd.DataFrame] = []
        status_rows: list[dict[str, object]] = []
        for case_index, case in enumerate(cases, start=1):
            for replicate_index in range(int(n_replicates)):
                print(
                    f"[{case_index}/{len(cases)}] {case['name']} "
                    f"replicate {replicate_index + 1}/{n_replicates}",
                    flush=True,
                )
                try:
                    table, row = _collect_case_replicate(
                        case=case,
                        replicate_index=replicate_index,
                        edge_alpha=edge_alpha,
                        sibling_alpha=sibling_alpha,
                    )
                    tables.append(table)
                    status_rows.append(row)
                except Exception as exc:
                    status_rows.append(
                        {
                            "case_id": str(case["name"]),
                            "case_category": str(case.get("category", "")),
                            "case_generator": str(case.get("generator", "")),
                            "replicate_index": int(replicate_index),
                            "status": "skip",
                            "error": str(exc),
                            "n_rule_records": 0,
                            "n_binary_sibling_parents": 0,
                            "n_non_binary_nodes": 0,
                            "study_role": STUDY_ROLE,
                        }
                    )
        records = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        status = pd.DataFrame.from_records(status_rows)
        records.to_csv(records_path, index=False)
        status.to_csv(status_path, index=False)

    summary = evaluate_projection_behavior_summary(records)
    case_summary_tables: list[pd.DataFrame] = []
    if not records.empty:
        for case_id, case_records in records.groupby("case_id", sort=False):
            table = evaluate_projection_behavior_summary(case_records)
            table.insert(0, "case_id", str(case_id))
            case_summary_tables.append(table)
    case_summary = (
        pd.concat(case_summary_tables, ignore_index=True)
        if case_summary_tables
        else pd.DataFrame()
    )
    summary_path = output_dir / "mp_projection_dimension_behavior_summary.csv"
    case_summary_path = output_dir / "mp_projection_dimension_behavior_case_summary.csv"
    manifest_path = output_dir / "manifest.json"
    summary.to_csv(summary_path, index=False)
    case_summary.to_csv(case_summary_path, index=False)
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
            "records": str(records_path),
            "summary": str(summary_path),
            "case_summary": str(case_summary_path),
            "status": str(status_path),
        },
        "n_records": int(records.shape[0]),
        "n_status_rows": int(status.shape[0]),
        "interpretation": (
            "Diagnostic raw sibling projected-Wald behavior under candidate "
            "MP/projection-dimension rules. Null false-split and signal-retention "
            "rates are descriptive endpoints, not calibrated production decisions."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "records": records_path,
        "summary": summary_path,
        "case_summary": case_summary_path,
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
    outputs = run_mp_projection_dimension_behavior_sweep(
        suite=str(args.suite),
        case_names=_parse_csv_list(str(args.case_names)),
        n_replicates=int(args.n_replicates),
        output_dir=args.output_dir,
        edge_alpha=float(args.edge_alpha),
        sibling_alpha=float(args.sibling_alpha),
        resume=bool(args.resume),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    print(f"elapsed_sec={perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()


__all__ = [
    "RULE_IDS",
    "SCHEMA_VERSION",
    "STUDY_ROLE",
    "candidate_projection_dimensions",
    "evaluate_projection_behavior_summary",
    "run_mp_projection_dimension_behavior_sweep",
]
