"""Cross-fit selection diagnostics for hierarchy-selected edge tests.

The TBS hierarchy is a sample-leaf tree. A literal sample split cannot
recompute held-out node distributions without an additional assignment model
that maps held-out samples into training-tree nodes. This diagnostic therefore
uses a feature split as the canonical cross-fit regime: build the sample
hierarchy from one independent feature block and test node distributions on a
held-out feature block for the same sample leaves.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from tree_break_selection import config
from tree_break_selection.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence_with_context,
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
from tree_break_selection.hierarchy_analysis.tree_decomposition import TreeDecomposition
from tree_break_selection.tree.feature_space import (
    FeatureBlock,
    FeatureSpace,
    continuous_feature_space_from_columns,
)
from tree_break_selection.tree.poset_tree import PosetTree

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.time import format_timestamp_utc

DEFAULT_CASE_NAMES = (
    "gauss_null_large",
    "gauss_clear_medium",
    "binary_low_noise_4c",
    "cat_clear_3cat_4c",
)


@dataclass(frozen=True)
class FeatureSplit:
    """Feature matrices and contracts for hierarchy selection and testing."""

    selection_data: pd.DataFrame
    test_data: pd.DataFrame
    selection_feature_space: FeatureSpace | None
    test_feature_space: FeatureSpace | None


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _select_cases(case_names: list[str]) -> list[dict[str, object]]:
    case_by_name = {str(case["name"]): case for case in get_default_test_cases()}
    missing = [case_name for case_name in case_names if case_name not in case_by_name]
    if missing:
        raise ValueError(f"Unknown benchmark case name(s): {missing!r}.")
    return [case_by_name[case_name].copy() for case_name in case_names]


def _subset_feature_space(
    feature_space: FeatureSpace,
    selected_block_indices: tuple[int, ...],
) -> tuple[list[int], FeatureSpace]:
    """Return raw column indices and a re-indexed feature-space subset."""
    selected_blocks = [feature_space.blocks[index] for index in selected_block_indices]
    original_column_indices = [
        column_index
        for block in selected_blocks
        for column_index in block.column_indices
    ]
    new_position_by_old = {
        old_index: new_index for new_index, old_index in enumerate(original_column_indices)
    }
    column_names = tuple(feature_space.column_names[index] for index in original_column_indices)
    blocks: list[FeatureBlock] = []
    for block in selected_blocks:
        blocks.append(
            FeatureBlock(
                name=block.name,
                family=block.family,
                column_indices=tuple(
                    new_position_by_old[old_index] for old_index in block.column_indices
                ),
                chart=block.chart,
                covariance=block.covariance,
                contrast_dimension=block.contrast_dimension,
            )
        )
    return original_column_indices, FeatureSpace(column_names=column_names, blocks=tuple(blocks))


def _split_untyped_or_continuous_columns(
    data: pd.DataFrame,
    feature_space: FeatureSpace | None,
    *,
    rng: np.random.Generator,
    selection_fraction: float,
) -> FeatureSplit:
    column_count = int(data.shape[1])
    if column_count < 2:
        raise ValueError("Feature-split audit requires at least two columns.")
    shuffled = rng.permutation(column_count)
    selection_count = int(round(column_count * selection_fraction))
    selection_count = max(1, min(column_count - 1, selection_count))
    selection_indices = tuple(sorted(int(index) for index in shuffled[:selection_count]))
    test_indices = tuple(sorted(int(index) for index in shuffled[selection_count:]))

    selection_data = data.iloc[:, list(selection_indices)].copy()
    test_data = data.iloc[:, list(test_indices)].copy()
    if feature_space is not None and feature_space.has_continuous_blocks:
        return FeatureSplit(
            selection_data=selection_data,
            test_data=test_data,
            selection_feature_space=continuous_feature_space_from_columns(
                selection_data.columns
            ),
            test_feature_space=continuous_feature_space_from_columns(test_data.columns),
        )
    return FeatureSplit(
        selection_data=selection_data,
        test_data=test_data,
        selection_feature_space=None,
        test_feature_space=None,
    )


def _split_feature_blocks(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    rng: np.random.Generator,
    selection_fraction: float,
) -> FeatureSplit:
    block_count = len(feature_space.blocks)
    if block_count < 2:
        return _split_untyped_or_continuous_columns(
            data,
            feature_space,
            rng=rng,
            selection_fraction=selection_fraction,
        )
    shuffled = rng.permutation(block_count)
    selection_count = int(round(block_count * selection_fraction))
    selection_count = max(1, min(block_count - 1, selection_count))
    selection_block_indices = tuple(sorted(int(index) for index in shuffled[:selection_count]))
    test_block_indices = tuple(sorted(int(index) for index in shuffled[selection_count:]))
    selection_columns, selection_feature_space = _subset_feature_space(
        feature_space,
        selection_block_indices,
    )
    test_columns, test_feature_space = _subset_feature_space(
        feature_space,
        test_block_indices,
    )
    return FeatureSplit(
        selection_data=data.iloc[:, selection_columns].copy(),
        test_data=data.iloc[:, test_columns].copy(),
        selection_feature_space=selection_feature_space,
        test_feature_space=test_feature_space,
    )


def _make_feature_split(
    data: pd.DataFrame,
    feature_space: FeatureSpace | None,
    *,
    rng: np.random.Generator,
    selection_fraction: float,
) -> FeatureSplit:
    if feature_space is None:
        return _split_untyped_or_continuous_columns(
            data,
            None,
            rng=rng,
            selection_fraction=selection_fraction,
        )
    return _split_feature_blocks(
        data,
        feature_space,
        rng=rng,
        selection_fraction=selection_fraction,
    )


def _permute_test_features(
    data: pd.DataFrame,
    feature_space: FeatureSpace | None,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    permuted = data.copy()
    if feature_space is None:
        values = permuted.to_numpy(copy=True)
        for column_index in range(values.shape[1]):
            rng.shuffle(values[:, column_index])
        return pd.DataFrame(values, index=data.index, columns=data.columns)

    values = permuted.to_numpy(copy=True)
    for block in feature_space.blocks:
        row_order = rng.permutation(values.shape[0])
        block_columns = list(block.column_indices)
        values[:, block_columns] = values[row_order[:, None], block_columns]
    return pd.DataFrame(values, index=data.index, columns=data.columns)


def _tree_distance(data: pd.DataFrame, metadata: dict[str, object]) -> tuple[np.ndarray, str]:
    params = METHOD_SPECS["tbs"].param_grid[0]
    if bool(metadata["requires_precomputed_tbs_distance"]):
        raise ValueError(
            "Feature-split selection audit does not support cases requiring "
            "precomputed TBS tree distances, because the precomputed distance is "
            "defined on the full feature contract."
        )
    metric = str(params["tree_distance_metric"])
    return pdist(data.values, metric=metric), metric


def _build_tree(selection_data: pd.DataFrame, metadata: dict[str, object]) -> tuple[PosetTree, str]:
    distance_condensed, metric = _tree_distance(selection_data, metadata)
    linkage_method = str(METHOD_SPECS["tbs"].param_grid[0]["tree_linkage_method"])
    tree = PosetTree.from_linkage(
        linkage(distance_condensed, method=linkage_method),
        leaf_names=selection_data.index.tolist(),
    )
    return tree, metric


def _collect_support_metrics(
    tree: PosetTree,
    test_data: pd.DataFrame,
    feature_space: FeatureSpace | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    tree.populate_node_divergences(test_data, feature_space=feature_space)
    edge_df, spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        tree.annotations_df,
        significance_level_alpha=DEFAULT_EDGE_ALPHA,
        leaf_data=test_data,
        feature_space=feature_space,
    )
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        tree,
        spectral_context=spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=spectral_context,
        )
    )
    records, non_binary_nodes = collect_sibling_pair_records(
        tree,
        edge_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=feature_space,
    )

    tested_edges = edge_df["Child_Parent_Divergence_Tested"].astype(bool)
    significant_edges = edge_df["Child_Parent_Divergence_Significant"].astype(bool)
    positive_weight_records = [
        record
        for record in records
        if record.degrees_of_freedom > 0.0 and record.sibling_null_weight > 0.0
    ]
    supported_records = [
        record
        for record in positive_weight_records
        if record.is_null_like or record.is_edge_blocked
    ]
    selected_nonnull_records = [
        record
        for record in positive_weight_records
        if not (record.is_null_like or record.is_edge_blocked)
    ]
    focal_records = [
        record
        for record in records
        if not record.is_null_like and record.degrees_of_freedom > 0.0
    ]

    metrics = {
        "tested_edges": int(tested_edges.sum()),
        "significant_edges": int(significant_edges.sum()),
        "edge_rejection_rate": float(
            significant_edges.sum() / max(int(tested_edges.sum()), 1)
        ),
        "sibling_records": int(len(records)),
        "non_binary_nodes": int(len(non_binary_nodes)),
        "focal_records": int(len(focal_records)),
        "positive_weight_records": int(len(positive_weight_records)),
        "supported_records": int(len(supported_records)),
        "selected_nonnull_records": int(len(selected_nonnull_records)),
    }
    return metrics, edge_df


def _labels_from_decomposition(
    decomposition: dict[str, object],
    sample_index: pd.Index,
) -> np.ndarray:
    assignments = build_sample_cluster_assignments(decomposition)
    if set(assignments.index) != set(sample_index):
        raise ValueError("Decomposition assignments do not cover the input sample index.")
    return assignments.loc[sample_index, "cluster_id"].to_numpy(dtype=int)


def _run_full_decomposition(
    tree: PosetTree,
    test_data: pd.DataFrame,
    true_labels: np.ndarray,
    feature_space: FeatureSpace | None,
) -> dict[str, object]:
    gate_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df,
        edge_alpha=DEFAULT_EDGE_ALPHA,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
        leaf_data=test_data,
        feature_space=feature_space,
    )
    decomposer = TreeDecomposition(
        tree=tree,
        gate_annotation_bundle=gate_bundle,
        edge_alpha=DEFAULT_EDGE_ALPHA,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
        leaf_data=test_data,
        feature_space=feature_space,
        passthrough=config.PASSTHROUGH,
    )
    decomposition = decomposer.decompose_tree()
    labels = _labels_from_decomposition(decomposition, test_data.index)
    return {
        "decomposition_status": "ok",
        "skip_reason": "",
        "found_clusters": int(decomposition["num_clusters"]),
        "ari": float(adjusted_rand_score(true_labels, labels)),
    }


def _evaluate_regime(
    *,
    case_id: str,
    regime: str,
    selection_data: pd.DataFrame,
    test_data: pd.DataFrame,
    true_labels: np.ndarray,
    metadata: dict[str, object],
    feature_space: FeatureSpace | None,
) -> dict[str, object]:
    start_sec = perf_counter()
    row: dict[str, object] = {
        "case_id": case_id,
        "regime": regime,
        "selection_features": int(selection_data.shape[1]),
        "test_features": int(test_data.shape[1]),
        "feature_family": "bernoulli" if feature_space is None else feature_space.family_label,
    }
    try:
        tree, tree_metric = _build_tree(selection_data, metadata)
        row["tree_distance_metric"] = tree_metric
        support_metrics, _edge_df = _collect_support_metrics(
            tree,
            test_data,
            feature_space,
        )
        row.update(support_metrics)

        tree_for_decomposition, _ = _build_tree(selection_data, metadata)
        tree_for_decomposition.populate_node_divergences(
            test_data,
            feature_space=feature_space,
        )
        row.update(
            _run_full_decomposition(
                tree_for_decomposition,
                test_data,
                true_labels,
                feature_space,
            )
        )
        row["status"] = "ok"
    except Exception as exc:
        row.setdefault("tested_edges", np.nan)
        row.setdefault("significant_edges", np.nan)
        row.setdefault("edge_rejection_rate", np.nan)
        row.setdefault("sibling_records", np.nan)
        row.setdefault("focal_records", np.nan)
        row.setdefault("positive_weight_records", np.nan)
        row.setdefault("supported_records", np.nan)
        row.setdefault("selected_nonnull_records", np.nan)
        row.setdefault("decomposition_status", "skip")
        row.setdefault("found_clusters", np.nan)
        row.setdefault("ari", np.nan)
        row["status"] = "skip"
        row["skip_reason"] = str(exc)
    row["runtime_sec"] = float(perf_counter() - start_sec)
    return row


def run_feature_split_selection_audit(
    *,
    case_names: list[str],
    output_dir: Path,
    seed: int,
    selection_fraction: float,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for case in _select_cases(case_names):
        inputs = prepare_case_inputs(case, ["tbs"])
        feature_space = inputs.metadata.get("feature_space")
        if feature_space is not None and not isinstance(feature_space, FeatureSpace):
            raise ValueError("Prepared feature_space metadata must be a FeatureSpace.")

        split = _make_feature_split(
            inputs.data,
            feature_space,
            rng=rng,
            selection_fraction=selection_fraction,
        )
        case_id = str(inputs.metadata["name"])
        rows.append(
            _evaluate_regime(
                case_id=case_id,
                regime="in_sample_full_features",
                selection_data=inputs.data,
                test_data=inputs.data,
                true_labels=inputs.labels,
                metadata=inputs.metadata,
                feature_space=feature_space,
            )
        )
        rows.append(
            _evaluate_regime(
                case_id=case_id,
                regime="feature_split_crossfit",
                selection_data=split.selection_data,
                test_data=split.test_data,
                true_labels=inputs.labels,
                metadata=inputs.metadata,
                feature_space=split.test_feature_space,
            )
        )
        rows.append(
            _evaluate_regime(
                case_id=case_id,
                regime="fixed_tree_feature_permutation",
                selection_data=split.selection_data,
                test_data=_permute_test_features(
                    split.test_data,
                    split.test_feature_space,
                    rng=rng,
                ),
                true_labels=inputs.labels,
                metadata=inputs.metadata,
                feature_space=split.test_feature_space,
            )
        )
    summary = pd.DataFrame.from_records(rows)
    summary_path = output_dir / "sample_split_selection_audit_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "seed": int(seed),
        "split_axis": "feature",
        "case_names": list(case_names),
        "selection_fraction": float(selection_fraction),
        "edge_alpha": float(DEFAULT_EDGE_ALPHA),
        "sibling_alpha": float(DEFAULT_SIBLING_ALPHA),
        "output": str(summary_path),
        "note": (
            "Literal sample splitting is not implemented because Tree-Break Selection uses a "
            "sample-leaf hierarchy. Held-out samples require an explicit node "
            "assignment model before their node distributions can be tested."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether cross-fitting hierarchy selection and gate testing "
            "restores edge calibration and sibling empirical-null support."
        )
    )
    parser.add_argument(
        "--split-axis",
        choices=("feature", "sample"),
        default="feature",
        help=(
            "Feature split is the supported cross-fit regime. Literal sample "
            "split requires an explicit held-out-sample assignment model."
        ),
    )
    parser.add_argument(
        "--case-names",
        default=",".join(DEFAULT_CASE_NAMES),
        help="Comma-separated benchmark case names.",
    )
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "benchmarks/results/sample_split_selection_audit_<timestamp>."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.split_axis == "sample":
        raise ValueError(
            "Literal sample splitting is not a valid Tree-Break Selection diagnostic yet: the "
            "tree leaves are samples, so held-out samples have no canonical "
            "membership in a tree built from training samples. Define an explicit "
            "assignment model before enabling sample-axis cross-fitting."
        )
    if not 0.0 < float(args.selection_fraction) < 1.0:
        raise ValueError("--selection-fraction must lie in (0, 1).")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("benchmarks")
            / "results"
            / f"sample_split_selection_audit_{format_timestamp_utc()}"
        )
    summary = run_feature_split_selection_audit(
        case_names=_parse_csv_list(args.case_names),
        output_dir=output_dir,
        seed=int(args.seed),
        selection_fraction=float(args.selection_fraction),
    )
    print(summary.to_string(index=False))
    print(f"Wrote {output_dir / 'sample_split_selection_audit_summary.csv'}")


if __name__ == "__main__":
    main()
