"""Traversal impact diagnostics for data-independent sibling gates.

This panel tests whether fixed sibling gates that avoid same-sample adaptive
PCA can drive the actual top-down decomposition traversal. It is diagnostic
only and does not change production KL-TE behavior.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    ClusterBoundary,
    build_cluster_assignments,
    build_sample_cluster_assignments,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.gate_evaluator import (
    GateEvaluator,
    TraversalDecision,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence_with_context,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.contrast_covariance import (
    build_contrast_covariance,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflated_projected_wald_annotation.fdr_annotation import (
    apply_traversal_aligned_sibling_bh_results,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.pair_observations import (
    identify_binary_sibling_children,
)
from kl_clustering_analysis.tree.feature_space import (
    FeatureSpace,
    bernoulli_feature_space_from_columns,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import norm, t
from sklearn.metrics import adjusted_rand_score

from benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel import (
    DEFAULT_CANDIDATE_METHODS,
    DEFAULT_DATA_ROLES,
    DEFAULT_SELECTED_TOPOLOGY_PENALTIES,
    data_independent_gate_p_value,
    validate_candidate_methods,
    validate_data_roles,
    validate_selected_topology_penalties,
)
from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.shared.generators.generate_case_data import generate_case_data
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
    regenerate_null_case,
)

STUDY_ROLE = "diagnostic_data_independent_sibling_gate_traversal_not_calibration"
SCHEMA_VERSION = "data_independent_sibling_gate_traversal_panel/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel"
)
CONFIDENCE_LEVEL = 0.95
SELECTED_TREE_DISTANCE_METRIC = "hamming"
SELECTED_TREE_LINKAGE_METHOD = "average"
DEFAULT_ROOT_STABILITY_SENSITIVITY_THRESHOLDS = (
    0.15,
    0.18,
    0.20,
    0.22,
    0.24,
    0.25,
    0.28,
    0.30,
)


@dataclass(frozen=True)
class DataIndependentSiblingGateTraversalConfig:
    """Runtime contract for fixed sibling-gate traversal diagnostics."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    data_roles: tuple[str, ...]
    candidate_methods: tuple[str, ...]
    sibling_alpha: float
    edge_alpha: float
    selected_topology_penalties: tuple[float, ...]
    replicates: int
    base_seed: int
    root_selective_bootstrap_replicates: int = 0
    root_stability_subsample_replicates: int = 0
    root_stability_feature_fraction: float = 0.8
    root_stability_seed: int = 0
    root_stability_guard_threshold: float | None = None
    max_null_split_rate: float = 0.05
    min_signal_mean_ari: float = 0.75

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "data_independent_sibling_gate_traversal_rows.csv"

    @property
    def checkpoint_rows_dir(self) -> Path:
        return self.output_dir / "checkpoint_rows"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "data_independent_sibling_gate_traversal_summary.csv"

    @property
    def transfer_summary_path(self) -> Path:
        return self.output_dir / "data_independent_sibling_gate_traversal_transfer_summary.csv"

    @property
    def production_components_path(self) -> Path:
        return self.output_dir / "production_admissibility_components.csv"

    @property
    def production_summary_path(self) -> Path:
        return self.output_dir / "production_admissibility_summary.csv"

    @property
    def root_stability_threshold_sensitivity_path(self) -> Path:
        return self.output_dir / "root_stability_threshold_sensitivity.csv"

    @property
    def root_selective_guard_sensitivity_path(self) -> Path:
        return self.output_dir / "root_selective_guard_sensitivity.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _generate_data_with_truth(
    *,
    case: dict[str, object],
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None,
    data_role: str,
    seed: int,
) -> tuple[pd.DataFrame, FeatureSpace, np.ndarray, int]:
    if data_role == "null":
        data, metadata = regenerate_null_case(
            case_id=case_id,
            source_family=source_family,
            feature_representation=feature_representation,
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            seed=seed,
        )
        return data, metadata["feature_space"], np.zeros(data.shape[0], dtype=int), 1  # type: ignore[return-value]

    if data_role == "signal":
        signal_case = dict(case)
        signal_case["seed"] = int(seed)
        data, labels, _original, metadata = generate_case_data(signal_case)
        feature_space = metadata.get("feature_space")
        if not isinstance(feature_space, FeatureSpace):
            feature_space = bernoulli_feature_space_from_columns(tuple(data.columns))
        return data, feature_space, np.asarray(labels, dtype=int), int(
            metadata.get("n_clusters", case.get("n_clusters", 1))
        )

    raise ValueError(f"Unknown data_role: {data_role!r}.")


def _build_tree_from_leaf_data(
    data: pd.DataFrame,
    *,
    tree_distance_metric: str = SELECTED_TREE_DISTANCE_METRIC,
    tree_linkage_method: str = SELECTED_TREE_LINKAGE_METHOD,
) -> PosetTree:
    distance = pdist(data.to_numpy(dtype=float), metric=str(tree_distance_metric))
    return PosetTree.from_linkage(
        linkage(distance, method=str(tree_linkage_method)),
        leaf_names=data.index.tolist(),
    )


def selected_tree_oracle_cut_ari(
    data: pd.DataFrame,
    truth_labels: np.ndarray,
    *,
    true_clusters: int,
) -> float:
    """Return ARI for the selected KL tree cut at true cluster count."""
    labels = np.asarray(truth_labels, dtype=int)
    if int(true_clusters) <= 1:
        return float(adjusted_rand_score(labels, np.zeros(labels.shape[0], dtype=int)))
    distance = pdist(data.to_numpy(dtype=float), metric=SELECTED_TREE_DISTANCE_METRIC)
    clusters = fcluster(
        linkage(distance, method=SELECTED_TREE_LINKAGE_METHOD),
        t=int(true_clusters),
        criterion="maxclust",
    )
    return float(adjusted_rand_score(labels, clusters))


def _root_split_labels(
    data: pd.DataFrame,
    *,
    tree_distance_metric: str = SELECTED_TREE_DISTANCE_METRIC,
    tree_linkage_method: str = SELECTED_TREE_LINKAGE_METHOD,
) -> np.ndarray:
    """Return binary labels induced by the selected root split."""
    tree = _build_tree_from_leaf_data(
        data,
        tree_distance_metric=tree_distance_metric,
        tree_linkage_method=tree_linkage_method,
    )
    children = list(tree.successors(tree.root()))
    if len(children) != 2:
        return np.zeros(data.shape[0], dtype=int)
    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    left = set(descendant_sets[children[0]])
    return np.asarray([0 if index in left else 1 for index in data.index], dtype=int)


def _method_seed_offset(method: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(method))


def _safe_checkpoint_label(value: object) -> str:
    label = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)


def _checkpoint_rows(
    rows: list[dict[str, object]],
    *,
    checkpoint_dir: Path,
    case_id: str,
    data_role: str,
    replicate: int,
) -> Path:
    """Write recoverable rows for one completed case-role-replicate unit."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / (
        f"{_safe_checkpoint_label(case_id)}__"
        f"{_safe_checkpoint_label(data_role)}__"
        f"replicate_{int(replicate):05d}.csv"
    )
    pd.DataFrame.from_records(rows).to_csv(path, index=False)
    return path


def root_feature_subsample_stability(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    subsample_replicates: int,
    feature_fraction: float,
    seed: int,
    tree_distance_metric: str = SELECTED_TREE_DISTANCE_METRIC,
    tree_linkage_method: str = SELECTED_TREE_LINKAGE_METHOD,
) -> dict[str, float]:
    """Measure selected-root stability under feature/block subsampling."""
    if int(subsample_replicates) <= 0:
        return {
            "root_stability_subsample_mean_ari": np.nan,
            "root_stability_subsample_median_ari": np.nan,
            "root_stability_subsample_q10_ari": np.nan,
        }
    fraction = float(feature_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("root_stability_feature_fraction must lie in (0, 1].")
    blocks = tuple(tuple(block.column_indices) for block in feature_space.blocks)
    if not blocks:
        raise ValueError("Feature-space blocks are required for root stability.")
    n_blocks = len(blocks)
    n_selected = max(1, int(round(fraction * n_blocks)))
    rng = np.random.default_rng(int(seed))
    original = _root_split_labels(
        data,
        tree_distance_metric=tree_distance_metric,
        tree_linkage_method=tree_linkage_method,
    )
    scores: list[float] = []
    for _ in range(int(subsample_replicates)):
        selected_blocks = rng.choice(n_blocks, size=n_selected, replace=False)
        columns = [
            column
            for block_index in selected_blocks
            for column in blocks[int(block_index)]
        ]
        subsampled = data.iloc[:, columns]
        selected = _root_split_labels(
            subsampled,
            tree_distance_metric=tree_distance_metric,
            tree_linkage_method=tree_linkage_method,
        )
        scores.append(float(adjusted_rand_score(original, selected)))
    if not scores:
        return {
            "root_stability_subsample_mean_ari": np.nan,
            "root_stability_subsample_median_ari": np.nan,
            "root_stability_subsample_q10_ari": np.nan,
        }
    values = np.asarray(scores, dtype=float)
    return {
        "root_stability_subsample_mean_ari": float(np.mean(values)),
        "root_stability_subsample_median_ari": float(np.median(values)),
        "root_stability_subsample_q10_ari": float(np.quantile(values, 0.10)),
    }


def _block_permutation_null_sample(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a same-margin feature/block permutation null sample."""
    source = data.to_numpy(dtype=int)
    out = np.zeros_like(source)
    n_rows = int(source.shape[0])
    for block in feature_space.blocks:
        indices = np.asarray(block.column_indices, dtype=int)
        if block.family == "categorical":
            categories = np.argmax(source[:, indices], axis=1)
            permuted = rng.permutation(categories)
            out[np.arange(n_rows), indices[permuted]] = 1
            continue
        if block.family == "bernoulli":
            out[:, indices[0]] = rng.permutation(source[:, indices[0]])
            continue
        raise ValueError(
            "Root selective permutation diagnostics support Bernoulli and "
            f"categorical feature blocks only; got {block.family!r}."
        )
    return pd.DataFrame(
        out,
        index=[f"perm_{index}" for index in range(n_rows)],
        columns=data.columns,
    )


def _selected_root_sibling_p_value(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    candidate_method: str,
) -> float:
    tree = _build_tree_from_leaf_data(data)
    tree.populate_node_divergences(data, feature_space=feature_space)
    parents, results_by_method, _skipped = _sibling_results_by_method(
        tree,
        feature_space,
        (candidate_method,),
    )
    root = tree.root()
    if root not in parents:
        return 1.0
    return float(results_by_method[candidate_method][parents.index(root)][2])


def selected_root_permutation_p_value(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    candidate_method: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, float]:
    """Estimate a selected-root null p-value by feature/block permutation."""
    if int(bootstrap_replicates) <= 0:
        return {
            "root_observed_p_value": _selected_root_sibling_p_value(
                data,
                feature_space,
                candidate_method=candidate_method,
            ),
            "root_selective_p_value": np.nan,
            "root_selective_null_min_p_value": np.nan,
            "root_selective_null_q05_p_value": np.nan,
        }
    rng = np.random.default_rng(int(seed))
    observed = _selected_root_sibling_p_value(
        data,
        feature_space,
        candidate_method=candidate_method,
    )
    null_p_values: list[float] = []
    for _ in range(int(bootstrap_replicates)):
        null_sample = _block_permutation_null_sample(data, feature_space, rng)
        null_p_values.append(
            _selected_root_sibling_p_value(
                null_sample,
                feature_space,
                candidate_method=candidate_method,
            )
        )
    null = np.asarray(null_p_values, dtype=float)
    selective = (1.0 + float(np.sum(null <= observed))) / (float(null.size) + 1.0)
    return {
        "root_observed_p_value": float(observed),
        "root_selective_p_value": float(selective),
        "root_selective_null_min_p_value": float(np.min(null)),
        "root_selective_null_q05_p_value": float(np.quantile(null, 0.05)),
    }


def _base_edge_annotations(
    tree: PosetTree,
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    edge_alpha: float,
) -> pd.DataFrame:
    annotations, _spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        tree.annotations_df.copy(),
        significance_level_alpha=float(edge_alpha),
        leaf_data=data,
        feature_space=feature_space,
    )
    return init_sibling_annotation_df(annotations)


def _sibling_results_by_method(
    tree: PosetTree,
    feature_space: FeatureSpace,
    candidate_methods: tuple[str, ...],
) -> tuple[list[object], dict[str, list[tuple[float, float, float]]], list[object]]:
    parents: list[object] = []
    results_by_method = {method: [] for method in candidate_methods}
    skipped: list[object] = []
    for parent in tree.nodes:
        children = identify_binary_sibling_children(tree, parent)
        if children is None:
            skipped.append(parent)
            continue
        left, right = children
        contrast = build_contrast_covariance(
            np.asarray(tree.nodes[left]["distribution"], dtype=float),
            np.asarray(tree.nodes[right]["distribution"], dtype=float),
            float(tree.nodes[left]["leaf_count"]),
            float(tree.nodes[right]["leaf_count"]),
            comparison="sibling",
            feature_space=feature_space,
        )
        z = contrast.whitened_vector()
        statistic = float(np.dot(z, z))
        degrees_of_freedom = float(contrast.degrees_of_freedom)
        parents.append(parent)
        for method in candidate_methods:
            p_value = data_independent_gate_p_value(
                z,
                feature_space,
                candidate_method=method,
            )
            results_by_method[method].append(
                (statistic, degrees_of_freedom, float(p_value))
            )
    return parents, results_by_method, skipped


def _traverse_annotations(
    tree: PosetTree,
    annotations: pd.DataFrame,
    data_index: pd.Index,
    truth_labels: np.ndarray,
    *,
    root_stability_mean_ari: float | None = None,
    root_stability_guard_threshold: float | None = None,
) -> dict[str, object]:
    children = {node: list(tree.successors(node)) for node in tree.nodes}
    gate = GateEvaluator(
        tree,
        annotations["Child_Parent_Divergence_Significant"]
        .fillna(False)
        .astype(bool)
        .to_dict(),
        annotations["Sibling_BH_Different"].fillna(False).astype(bool).to_dict(),
        annotations["Sibling_Divergence_Skipped"].fillna(False).astype(bool).to_dict(),
        children,
        passthrough=False,
    )
    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    stack = [tree.root()]
    boundaries: list[ClusterBoundary] = []
    split_rows: list[dict[str, object]] = []
    split_count = 0
    root_stability_guard_blocked = False
    guard_threshold = (
        float(root_stability_guard_threshold)
        if root_stability_guard_threshold is not None
        else np.nan
    )
    guard_enabled = bool(np.isfinite(guard_threshold))
    while stack:
        node = stack.pop()
        decision = gate.decision(node)
        guard_blocks_split = (
            decision == TraversalDecision.SPLIT
            and guard_enabled
            and node == tree.root()
            and root_stability_mean_ari is not None
            and np.isfinite(float(root_stability_mean_ari))
            and float(root_stability_mean_ari) < guard_threshold
        )
        if decision == TraversalDecision.SPLIT and not guard_blocks_split:
            split_count += 1
            left, right = children[node]
            left_n = int(len(descendant_sets[left]))
            right_n = int(len(descendant_sets[right]))
            parent_n = int(len(descendant_sets[node]))
            split_rows.append(
                {
                    "node": node,
                    "parent_n": parent_n,
                    "left_n": left_n,
                    "right_n": right_n,
                    "min_child_n": min(left_n, right_n),
                    "child_imbalance": (
                        abs(left_n - right_n) / parent_n if parent_n > 0 else np.nan
                    ),
                    "sibling_p_value": float(
                        annotations.at[node, "Sibling_Divergence_P_Value"]
                    )
                    if node in annotations.index
                    and pd.notna(annotations.at[node, "Sibling_Divergence_P_Value"])
                    else np.nan,
                    "is_root": bool(node == tree.root()),
                }
            )
            stack.append(right)
            stack.append(left)
            continue
        if guard_blocks_split:
            root_stability_guard_blocked = True
        boundaries.append(
            ClusterBoundary(root_node=node, leaves=descendant_sets[node])
        )

    decomposition = {"cluster_assignments": build_cluster_assignments(boundaries)}
    assignments = build_sample_cluster_assignments(decomposition).reindex(data_index)
    predicted = assignments["cluster_id"].to_numpy()
    split_p_values = [
        float(row["sibling_p_value"])
        for row in split_rows
        if pd.notna(row["sibling_p_value"])
    ]
    first_split = split_rows[0] if split_rows else {}
    return {
        "found_clusters": int(pd.Series(predicted).nunique()),
        "split_count": int(split_count),
        "ari": float(adjusted_rand_score(truth_labels, predicted)),
        "root_split": bool(any(bool(row["is_root"]) for row in split_rows)),
        "first_split_parent_n": first_split.get("parent_n", np.nan),
        "first_split_min_child_n": first_split.get("min_child_n", np.nan),
        "first_split_child_imbalance": first_split.get("child_imbalance", np.nan),
        "first_split_sibling_p_value": first_split.get("sibling_p_value", np.nan),
        "min_split_sibling_p_value": min(split_p_values) if split_p_values else np.nan,
        "root_stability_guard_blocked": bool(root_stability_guard_blocked),
    }


def _rows_for_replicate(
    *,
    case: dict[str, object],
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None,
    data_role: str,
    replicate: int,
    data_seed: int,
    candidate_methods: tuple[str, ...],
    selected_topology_penalties: tuple[float, ...],
    sibling_alpha: float,
    edge_alpha: float,
    run_id: str,
    root_selective_bootstrap_replicates: int,
    root_stability_subsample_replicates: int,
    root_stability_feature_fraction: float,
    root_stability_seed: int,
    root_stability_guard_threshold: float | None,
) -> list[dict[str, object]]:
    data, feature_space, truth_labels, true_clusters = _generate_data_with_truth(
        case=case,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        data_role=data_role,
        seed=data_seed,
    )
    tree = _build_tree_from_leaf_data(data)
    oracle_ari = selected_tree_oracle_cut_ari(
        data,
        truth_labels,
        true_clusters=int(true_clusters),
    )
    tree.populate_node_divergences(data, feature_space=feature_space)
    base_annotations = _base_edge_annotations(
        tree,
        data,
        feature_space,
        edge_alpha=edge_alpha,
    )
    parents, results_by_method, skipped = _sibling_results_by_method(
        tree,
        feature_space,
        candidate_methods,
    )
    root_stability = root_feature_subsample_stability(
        data,
        feature_space,
        subsample_replicates=int(root_stability_subsample_replicates),
        feature_fraction=float(root_stability_feature_fraction),
        seed=int(root_stability_seed),
        tree_distance_metric=SELECTED_TREE_DISTANCE_METRIC,
        tree_linkage_method=SELECTED_TREE_LINKAGE_METHOD,
    )
    rows: list[dict[str, object]] = []
    for method in candidate_methods:
        root_observed = selected_root_permutation_p_value(
            data,
            feature_space,
            candidate_method=method,
            bootstrap_replicates=0,
            seed=int(data_seed) + _method_seed_offset(method),
        )
        root_selective: dict[str, float] | None = None
        for penalty in selected_topology_penalties:
            effective_alpha = float(sibling_alpha) / float(penalty)
            annotations = base_annotations.copy()
            mark_non_binary_as_skipped(annotations, skipped)
            annotations = apply_traversal_aligned_sibling_bh_results(
                tree,
                annotations,
                parents,
                results_by_method[method],
                effective_alpha,
                method_labels=[method] * len(parents),
                skipped_parents=skipped,
            )
            traversal = _traverse_annotations(
                tree,
                annotations,
                data.index,
                truth_labels,
                root_stability_mean_ari=root_stability[
                    "root_stability_subsample_mean_ari"
                ],
                root_stability_guard_threshold=root_stability_guard_threshold,
            )
            root_was_open = bool(
                traversal["root_split"] or traversal["root_stability_guard_blocked"]
            )
            if int(root_selective_bootstrap_replicates) > 0 and root_was_open:
                if root_selective is None:
                    root_selective = selected_root_permutation_p_value(
                        data,
                        feature_space,
                        candidate_method=method,
                        bootstrap_replicates=int(root_selective_bootstrap_replicates),
                        seed=int(data_seed) + _method_seed_offset(method),
                    )
                root_selective_for_row = root_selective
            else:
                root_selective_for_row = root_observed
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "run_id": run_id,
                    "case_id": case_id,
                    "data_role": data_role,
                    "source_family": source_family,
                    "feature_representation": feature_representation,
                    "candidate_method": method,
                    "selected_topology_penalty": float(penalty),
                    "effective_selected_alpha": effective_alpha,
                    "edge_alpha": float(edge_alpha),
                    "sibling_alpha": float(sibling_alpha),
                    "replicate": int(replicate),
                    "data_seed": int(data_seed),
                    "root_selective_bootstrap_replicates": int(
                        root_selective_bootstrap_replicates
                    ),
                    "root_observed_p_value": root_selective_for_row[
                        "root_observed_p_value"
                    ],
                    "root_selective_p_value": root_selective_for_row[
                        "root_selective_p_value"
                    ],
                    "root_selective_null_min_p_value": root_selective_for_row[
                        "root_selective_null_min_p_value"
                    ],
                    "root_selective_null_q05_p_value": root_selective_for_row[
                        "root_selective_null_q05_p_value"
                    ],
                    "root_stability_subsample_replicates": int(
                        root_stability_subsample_replicates
                    ),
                    "root_stability_feature_fraction": float(
                        root_stability_feature_fraction
                    ),
                    "root_stability_seed": int(root_stability_seed),
                    "root_stability_tree_distance_metric": SELECTED_TREE_DISTANCE_METRIC,
                    "root_stability_tree_linkage_method": SELECTED_TREE_LINKAGE_METHOD,
                    "root_stability_subsample_mean_ari": root_stability[
                        "root_stability_subsample_mean_ari"
                    ],
                    "root_stability_subsample_median_ari": root_stability[
                        "root_stability_subsample_median_ari"
                    ],
                    "root_stability_subsample_q10_ari": root_stability[
                        "root_stability_subsample_q10_ari"
                    ],
                    "root_stability_guard_threshold": (
                        float(root_stability_guard_threshold)
                        if root_stability_guard_threshold is not None
                        else np.nan
                    ),
                    "root_stability_guard_blocked": traversal[
                        "root_stability_guard_blocked"
                    ],
                    "true_clusters": int(true_clusters),
                    "tree_distance_metric": SELECTED_TREE_DISTANCE_METRIC,
                    "tree_linkage_method": SELECTED_TREE_LINKAGE_METHOD,
                    "selected_tree_oracle_ari": float(oracle_ari),
                    "found_clusters": traversal["found_clusters"],
                    "exact_cluster_count": bool(
                        traversal["found_clusters"] == int(true_clusters)
                    ),
                    "split_count": traversal["split_count"],
                    "false_split": bool(
                        data_role == "null" and int(traversal["found_clusters"]) > 1
                    ),
                    "root_split": traversal["root_split"],
                    "false_root_split": bool(
                        data_role == "null" and traversal["root_split"]
                    ),
                    "first_split_parent_n": traversal["first_split_parent_n"],
                    "first_split_min_child_n": traversal["first_split_min_child_n"],
                    "first_split_child_imbalance": traversal[
                        "first_split_child_imbalance"
                    ],
                    "first_split_sibling_p_value": traversal[
                        "first_split_sibling_p_value"
                    ],
                    "min_split_sibling_p_value": traversal[
                        "min_split_sibling_p_value"
                    ],
                    "ari": traversal["ari"],
                    "edge_open_count": int(
                        annotations["Child_Parent_Divergence_Significant"]
                        .fillna(False)
                        .sum()
                    ),
                    "sibling_open_count": int(
                        annotations["Sibling_BH_Different"].fillna(False).sum()
                    ),
                }
            )
    return rows


def _status(
    group: pd.DataFrame,
    *,
    max_null_split_rate: float,
    min_signal_mean_ari: float,
) -> str:
    if group.empty:
        return "data_independent_traversal_insufficient_rows"
    role = str(group["data_role"].iloc[0])
    if role == "null":
        false_split_rate = float(group["false_split"].mean())
        if false_split_rate <= float(max_null_split_rate):
            return "data_independent_traversal_null_candidate"
        return "data_independent_traversal_null_inflated"
    if role == "signal":
        mean_ari = float(group["ari"].mean())
        if mean_ari >= float(min_signal_mean_ari):
            return "data_independent_traversal_signal_retained"
        return "data_independent_traversal_signal_weak"
    raise ValueError(f"Unknown data role: {role!r}.")


def _wilson_upper_bound(
    successes: int,
    n_observations: int,
    *,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> float:
    if int(n_observations) <= 0:
        return math.nan
    z = float(norm.ppf(1.0 - (1.0 - float(confidence_level)) / 2.0))
    n = float(n_observations)
    p_hat = float(successes) / n
    denominator = 1.0 + z * z / n
    center = p_hat + z * z / (2.0 * n)
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z * z / (4.0 * n * n)))
    return float(min(1.0, (center + radius) / denominator))


def _required_zero_false_split_replicates(
    *,
    max_null_split_rate: float,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> int:
    """Return null replicates needed for zero false splits to meet a target."""
    target = float(max_null_split_rate)
    if not 0.0 < target < 1.0:
        raise ValueError("max_null_split_rate must lie in (0, 1).")
    z = float(norm.ppf(1.0 - (1.0 - float(confidence_level)) / 2.0))
    return int(math.ceil((z * z) * (1.0 - target) / target))


def _required_replicates_for_wilson_upper_bound(
    successes: int,
    *,
    max_null_split_rate: float,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> int:
    """Return n needed for a Wilson upper bound to meet the target."""
    count = int(successes)
    if count < 0:
        raise ValueError("successes must be nonnegative.")
    target = float(max_null_split_rate)
    if not 0.0 < target < 1.0:
        raise ValueError("max_null_split_rate must lie in (0, 1).")
    n = max(1, count)
    while _wilson_upper_bound(
        count,
        n,
        confidence_level=float(confidence_level),
    ) > target:
        n += 1
    return int(n)


def _mean_lower_bound(
    values: pd.Series,
    *,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    n = int(numeric.shape[0])
    if n <= 0:
        return math.nan
    if n == 1:
        return float(numeric.iloc[0])
    standard_error = float(numeric.std(ddof=1)) / math.sqrt(float(n))
    return float(numeric.mean() - float(t.ppf(confidence_level, n - 1)) * standard_error)


def summarize_traversal_rows(
    rows: pd.DataFrame,
    *,
    max_null_split_rate: float = 0.05,
    min_signal_mean_ari: float = 0.75,
) -> pd.DataFrame:
    """Summarize traversal impact by role, method, penalty, and case."""
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    group_columns = (
        "case_id",
        "data_role",
        "source_family",
        "candidate_method",
        "selected_topology_penalty",
    )
    for key, group in rows.groupby(list(group_columns), sort=True):
        root_selective_p = pd.to_numeric(
            group["root_selective_p_value"],
            errors="coerce",
        )
        false_split_count = int(group["false_split"].astype(bool).sum())
        false_root_split_count = int(group["false_root_split"].astype(bool).sum())
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_replicates": int(group.shape[0]),
                "mean_ari": float(group["ari"].mean()),
                "mean_selected_tree_oracle_ari": float(
                    group["selected_tree_oracle_ari"].mean()
                ),
                "mean_ari_lower_confidence": _mean_lower_bound(group["ari"]),
                "median_ari": float(group["ari"].median()),
                "mean_found_clusters": float(group["found_clusters"].mean()),
                "exact_cluster_count_rate": float(group["exact_cluster_count"].mean()),
                "false_split_rate": float(group["false_split"].mean()),
                "false_split_rate_upper_confidence": _wilson_upper_bound(
                    false_split_count,
                    int(group.shape[0]),
                ),
                "mean_split_count": float(group["split_count"].mean()),
                "root_split_rate": float(group["root_split"].mean()),
                "false_root_split_rate": float(group["false_root_split"].mean()),
                "false_root_split_rate_upper_confidence": _wilson_upper_bound(
                    false_root_split_count,
                    int(group.shape[0]),
                ),
                "mean_first_split_parent_n": float(
                    group["first_split_parent_n"].mean()
                ),
                "mean_first_split_min_child_n": float(
                    group["first_split_min_child_n"].mean()
                ),
                "mean_first_split_child_imbalance": float(
                    group["first_split_child_imbalance"].mean()
                ),
                "min_observed_split_p_value": float(
                    group["min_split_sibling_p_value"].min()
                ),
                "root_selective_rejection_rate_at_sibling_alpha": (
                    float(
                        (
                            root_selective_p.dropna()
                            <= float(group["sibling_alpha"].iloc[0])
                        ).mean()
                    )
                    if bool(root_selective_p.notna().any())
                    else np.nan
                ),
                "mean_root_stability_subsample_mean_ari": float(
                    group["root_stability_subsample_mean_ari"].mean()
                ),
                "mean_root_stability_subsample_q10_ari": float(
                    group["root_stability_subsample_q10_ari"].mean()
                ),
                "root_stability_guard_block_rate": float(
                    group["root_stability_guard_blocked"].mean()
                ),
                "mean_edge_open_count": float(group["edge_open_count"].mean()),
                "mean_sibling_open_count": float(group["sibling_open_count"].mean()),
                "traversal_status": _status(
                    group,
                    max_null_split_rate=max_null_split_rate,
                    min_signal_mean_ari=min_signal_mean_ari,
                ),
                "study_role": STUDY_ROLE,
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def _transfer_status(group: pd.DataFrame) -> str:
    null_rows = group[group["data_role"].eq("null")]
    signal_rows = group[group["data_role"].eq("signal")]
    if null_rows.empty or signal_rows.empty:
        return "data_independent_traversal_transfer_insufficient_coverage"
    null_ok = null_rows["traversal_status"].eq(
        "data_independent_traversal_null_candidate"
    )
    signal_ok = signal_rows["traversal_status"].eq(
        "data_independent_traversal_signal_retained"
    )
    if not bool(null_ok.all()):
        return "data_independent_traversal_transfer_null_inflated"
    if not bool(signal_ok.all()):
        return "data_independent_traversal_transfer_signal_weak"
    return "data_independent_traversal_transfer_candidate"


def _confidence_transfer_status(
    group: pd.DataFrame,
    *,
    max_null_split_rate: float = 0.05,
    min_signal_mean_ari: float = 0.75,
) -> str:
    null_rows = group[group["data_role"].eq("null")]
    signal_rows = group[group["data_role"].eq("signal")]
    if null_rows.empty or signal_rows.empty:
        return "data_independent_traversal_transfer_confidence_insufficient"
    if (
        "false_split_rate_upper_confidence" not in null_rows
        or "mean_ari_lower_confidence" not in signal_rows
    ):
        return "data_independent_traversal_transfer_confidence_insufficient"
    max_null_upper = pd.to_numeric(
        null_rows["false_split_rate_upper_confidence"],
        errors="coerce",
    ).max()
    min_signal_lower = pd.to_numeric(
        signal_rows["mean_ari_lower_confidence"],
        errors="coerce",
    ).min()
    if pd.isna(max_null_upper) or pd.isna(min_signal_lower):
        return "data_independent_traversal_transfer_confidence_insufficient"
    if float(max_null_upper) > float(max_null_split_rate):
        return "data_independent_traversal_transfer_confidence_null_uncertain"
    if float(min_signal_lower) < float(min_signal_mean_ari):
        return "data_independent_traversal_transfer_confidence_signal_uncertain"
    return "data_independent_traversal_transfer_confidence_candidate"


def summarize_traversal_transfer(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether a traversal penalty transfers across included cases."""
    if summary.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    group_columns = (
        "source_family",
        "candidate_method",
        "selected_topology_penalty",
    )
    for key, group in summary.groupby(list(group_columns), sort=True):
        row = dict(zip(group_columns, key))
        null_rows = group[group["data_role"].eq("null")]
        signal_rows = group[group["data_role"].eq("signal")]
        required_zero_false = _required_zero_false_split_replicates(
            max_null_split_rate=0.05
        )
        min_null_replicates = (
            int(pd.to_numeric(null_rows["n_replicates"], errors="coerce").min())
            if not null_rows.empty and "n_replicates" in null_rows
            else 0
        )
        observed_false_counts: list[int] = []
        required_for_observed_counts: list[int] = []
        additional_for_observed_counts: list[int] = []
        if not null_rows.empty:
            for _, null_row in null_rows.iterrows():
                n_replicates = int(null_row["n_replicates"])
                false_count = int(
                    round(float(null_row["false_split_rate"]) * n_replicates)
                )
                required_for_count = _required_replicates_for_wilson_upper_bound(
                    false_count,
                    max_null_split_rate=0.05,
                )
                observed_false_counts.append(false_count)
                required_for_observed_counts.append(required_for_count)
                additional_for_observed_counts.append(
                    max(0, required_for_count - n_replicates)
                )
        row.update(
            {
                "n_case_role_rows": int(group.shape[0]),
                "n_null_case_rows": int(null_rows.shape[0]),
                "n_signal_case_rows": int(signal_rows.shape[0]),
                "min_null_replicates_per_case": int(min_null_replicates),
                "required_zero_false_split_null_replicates_per_case": int(
                    required_zero_false
                ),
                "additional_zero_false_split_null_replicates_per_case": int(
                    max(0, required_zero_false - min_null_replicates)
                ),
                "max_observed_null_false_split_count": (
                    int(max(observed_false_counts))
                    if observed_false_counts
                    else 0
                ),
                "max_required_null_replicates_given_observed_false_splits": (
                    int(max(required_for_observed_counts))
                    if required_for_observed_counts
                    else 0
                ),
                "max_additional_zero_false_null_replicates_given_observed_false_splits": (
                    int(max(additional_for_observed_counts))
                    if additional_for_observed_counts
                    else 0
                ),
                "max_null_false_split_rate": (
                    float(null_rows["false_split_rate"].max())
                    if not null_rows.empty
                    else np.nan
                ),
                "max_null_false_split_rate_upper_confidence": (
                    float(null_rows["false_split_rate_upper_confidence"].max())
                    if not null_rows.empty
                    and "false_split_rate_upper_confidence" in null_rows
                    else np.nan
                ),
                "max_null_false_root_split_rate": (
                    float(null_rows["false_root_split_rate"].max())
                    if not null_rows.empty
                    else np.nan
                ),
                "max_null_false_root_split_rate_upper_confidence": (
                    float(null_rows["false_root_split_rate_upper_confidence"].max())
                    if not null_rows.empty
                    and "false_root_split_rate_upper_confidence" in null_rows
                    else np.nan
                ),
                "min_signal_mean_ari": (
                    float(signal_rows["mean_ari"].min())
                    if not signal_rows.empty
                    else np.nan
                ),
                "min_signal_mean_ari_lower_confidence": (
                    float(signal_rows["mean_ari_lower_confidence"].min())
                    if not signal_rows.empty
                    and "mean_ari_lower_confidence" in signal_rows
                    else np.nan
                ),
                "n_null_candidates": int(
                    null_rows["traversal_status"]
                    .eq("data_independent_traversal_null_candidate")
                    .sum()
                ),
                "n_signal_retained": int(
                    signal_rows["traversal_status"]
                    .eq("data_independent_traversal_signal_retained")
                    .sum()
                ),
                "data_independent_traversal_transfer_status": _transfer_status(
                    group
                ),
                "data_independent_traversal_transfer_confidence_status": (
                    _confidence_transfer_status(group)
                ),
                "study_role": STUDY_ROLE,
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def build_traversal_production_components(
    transfer_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build conservative production-admissibility rows from transfer evidence."""
    if transfer_summary.empty:
        components = pd.DataFrame(
            [
                {
                    "contract_id": "data_independent_traversal_transfer:empty",
                    "component_id": "data_independent_traversal_transfer:empty",
                    "component_type": "data_independent_sibling_gate_traversal_transfer",
                    "component_status": (
                        "data_independent_traversal_transfer_insufficient_coverage"
                    ),
                    "required_for_production": True,
                    "notes": "No traversal transfer rows were available.",
                }
            ]
        )
        rows = evaluate_production_admissibility_components(components)
        return rows, summarize_production_admissibility_contracts(rows)

    records: list[dict[str, object]] = []
    for _, row in transfer_summary.iterrows():
        family = str(row["source_family"])
        method = str(row["candidate_method"])
        penalty = float(row["selected_topology_penalty"])
        status = str(row["data_independent_traversal_transfer_status"])
        context = f"{family}:{method}:penalty={penalty:g}"
        records.append(
            {
                "contract_id": f"data_independent_traversal_transfer:{context}",
                "component_id": f"data_independent_traversal_transfer:{context}",
                "component_type": "data_independent_sibling_gate_traversal_transfer",
                "component_status": status,
                "required_for_production": True,
                "notes": (
                    "Fixed data-independent sibling gate traversal transfer "
                    "evidence. Candidate rows remain diagnostic-only unless "
                    "promoted by a separate production calibration decision."
                ),
            }
        )
        records.append(
            {
                "contract_id": f"data_independent_traversal_transfer:{context}",
                "component_id": (
                    f"data_independent_traversal_transfer_confidence:{context}"
                ),
                "component_type": (
                    "data_independent_sibling_gate_traversal_transfer_confidence"
                ),
                "component_status": str(
                    row[
                        "data_independent_traversal_transfer_confidence_status"
                    ]
                ),
                "required_for_production": True,
                "notes": (
                    "Confidence-bound transfer evidence. Small smoke runs are "
                    "expected to fail closed until null and signal bounds are "
                    "tight enough for production promotion."
                ),
            }
        )
    components = pd.DataFrame.from_records(records)
    rows = evaluate_production_admissibility_components(components)
    return rows, summarize_production_admissibility_contracts(rows)


def summarize_root_stability_threshold_sensitivity(
    rows: pd.DataFrame,
    *,
    thresholds: Sequence[float] = DEFAULT_ROOT_STABILITY_SENSITIVITY_THRESHOLDS,
    max_null_split_rate: float = 0.05,
    min_signal_mean_ari: float = 0.75,
) -> pd.DataFrame:
    """Summarize point-estimate outcomes under stricter root-stability thresholds."""
    if rows.empty or "root_stability_subsample_mean_ari" not in rows:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    base = rows.copy()
    root_split = base["root_split"].astype(str).str.lower().isin({"true", "1"})
    root_blocked = base["root_stability_guard_blocked"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    root_was_open = root_split | root_blocked
    stability = pd.to_numeric(
        base["root_stability_subsample_mean_ari"],
        errors="coerce",
    )
    original_ari = pd.to_numeric(base["ari"], errors="coerce")
    original_false = base["false_split"].astype(str).str.lower().isin({"true", "1"})
    group_columns = (
        "case_id",
        "source_family",
        "candidate_method",
        "selected_topology_penalty",
        "data_role",
    )
    for threshold in thresholds:
        threshold_value = float(threshold)
        if not -1.0 <= threshold_value <= 1.0:
            raise ValueError("Root-stability sensitivity thresholds must lie in [-1, 1].")
        adjusted = base.copy()
        blocked = root_was_open & stability.notna() & (stability < threshold_value)
        adjusted["sensitivity_threshold"] = threshold_value
        adjusted["sensitivity_root_blocked"] = blocked
        adjusted["sensitivity_false_split"] = original_false & ~blocked
        adjusted["sensitivity_ari"] = original_ari
        adjusted.loc[adjusted["data_role"].eq("null") & blocked, "sensitivity_ari"] = 1.0
        adjusted.loc[
            adjusted["data_role"].eq("signal") & blocked,
            "sensitivity_ari",
        ] = 0.0
        for key, group in adjusted.groupby(list(group_columns), sort=True):
            status_frame = pd.DataFrame(
                {
                    "data_role": group["data_role"],
                    "false_split": group["sensitivity_false_split"],
                    "ari": group["sensitivity_ari"],
                }
            )
            row = dict(zip(group_columns, key))
            row.update(
                {
                    "sensitivity_threshold": threshold_value,
                    "n_replicates": int(group.shape[0]),
                    "false_split_rate": float(
                        group["sensitivity_false_split"].mean()
                    ),
                    "false_split_rate_upper_confidence": _wilson_upper_bound(
                        int(group["sensitivity_false_split"].astype(bool).sum()),
                        int(group.shape[0]),
                    ),
                    "mean_ari": float(group["sensitivity_ari"].mean()),
                    "mean_ari_lower_confidence": _mean_lower_bound(
                        group["sensitivity_ari"]
                    ),
                    "root_block_rate": float(group["sensitivity_root_blocked"].mean()),
                    "sensitivity_status": _status(
                        status_frame,
                        max_null_split_rate=max_null_split_rate,
                        min_signal_mean_ari=min_signal_mean_ari,
                    ),
                    "study_role": STUDY_ROLE,
                }
            )
            records.append(row)
    case_summary = pd.DataFrame.from_records(records)
    if case_summary.empty:
        return case_summary
    transfer_records: list[dict[str, object]] = []
    transfer_columns = (
        "source_family",
        "candidate_method",
        "selected_topology_penalty",
        "sensitivity_threshold",
    )
    for key, group in case_summary.groupby(list(transfer_columns), sort=True):
        null_rows = group[group["data_role"].eq("null")]
        signal_rows = group[group["data_role"].eq("signal")]
        null_ok = null_rows["sensitivity_status"].eq(
            "data_independent_traversal_null_candidate"
        )
        signal_ok = signal_rows["sensitivity_status"].eq(
            "data_independent_traversal_signal_retained"
        )
        if null_rows.empty or signal_rows.empty:
            status = "data_independent_traversal_transfer_insufficient_coverage"
        elif not bool(null_ok.all()):
            status = "data_independent_traversal_transfer_null_inflated"
        elif not bool(signal_ok.all()):
            status = "data_independent_traversal_transfer_signal_weak"
        else:
            status = "data_independent_traversal_transfer_candidate"
        confidence_status = _confidence_transfer_status(
            group.rename(
                columns={
                    "sensitivity_status": "traversal_status",
                }
            )
        )
        row = dict(zip(transfer_columns, key))
        row.update(
            {
                "n_case_role_rows": int(group.shape[0]),
                "max_null_false_split_rate": (
                    float(null_rows["false_split_rate"].max())
                    if not null_rows.empty
                    else np.nan
                ),
                "max_null_false_split_rate_upper_confidence": (
                    float(null_rows["false_split_rate_upper_confidence"].max())
                    if not null_rows.empty
                    else np.nan
                ),
                "min_signal_mean_ari": (
                    float(signal_rows["mean_ari"].min())
                    if not signal_rows.empty
                    else np.nan
                ),
                "min_signal_mean_ari_lower_confidence": (
                    float(signal_rows["mean_ari_lower_confidence"].min())
                    if not signal_rows.empty
                    else np.nan
                ),
                "max_root_block_rate": float(group["root_block_rate"].max()),
                "sensitivity_transfer_status": status,
                "sensitivity_confidence_status": confidence_status,
                "study_role": STUDY_ROLE,
            }
        )
        transfer_records.append(row)
    return pd.DataFrame.from_records(transfer_records)


def summarize_root_selective_guard_sensitivity(
    rows: pd.DataFrame,
    *,
    max_null_split_rate: float = 0.05,
    min_signal_mean_ari: float = 0.75,
) -> pd.DataFrame:
    """Summarize a post-hoc selected-root permutation guard sensitivity."""
    if rows.empty or "root_selective_p_value" not in rows:
        return pd.DataFrame()
    base = rows.copy()
    root_selective_p = pd.to_numeric(
        base["root_selective_p_value"],
        errors="coerce",
    )
    if not bool(root_selective_p.notna().any()):
        return pd.DataFrame()

    root_split = base["root_split"].astype(str).str.lower().isin({"true", "1"})
    root_blocked = base["root_stability_guard_blocked"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    root_was_open = root_split | root_blocked
    sibling_alpha = pd.to_numeric(base["sibling_alpha"], errors="coerce")
    original_false = base["false_split"].astype(str).str.lower().isin({"true", "1"})
    original_ari = pd.to_numeric(base["ari"], errors="coerce")
    selected_root_blocked = (
        root_was_open
        & root_selective_p.notna()
        & sibling_alpha.notna()
        & (root_selective_p > sibling_alpha)
    )
    adjusted = base.copy()
    adjusted["selected_root_guard_blocked"] = selected_root_blocked
    adjusted["selected_root_guard_false_split"] = original_false & ~selected_root_blocked
    adjusted["selected_root_guard_ari"] = original_ari
    adjusted.loc[
        adjusted["data_role"].eq("null") & selected_root_blocked,
        "selected_root_guard_ari",
    ] = 1.0
    adjusted.loc[
        adjusted["data_role"].eq("signal") & selected_root_blocked,
        "selected_root_guard_ari",
    ] = 0.0

    records: list[dict[str, object]] = []
    group_columns = (
        "case_id",
        "source_family",
        "candidate_method",
        "selected_topology_penalty",
        "data_role",
    )
    for key, group in adjusted.groupby(list(group_columns), sort=True):
        status_frame = pd.DataFrame(
            {
                "data_role": group["data_role"],
                "false_split": group["selected_root_guard_false_split"],
                "ari": group["selected_root_guard_ari"],
            }
        )
        row = dict(zip(group_columns, key))
        false_count = int(group["selected_root_guard_false_split"].astype(bool).sum())
        root_p = pd.to_numeric(group["root_selective_p_value"], errors="coerce")
        row.update(
            {
                "n_replicates": int(group.shape[0]),
                "root_selective_bootstrap_replicates": int(
                    pd.to_numeric(
                        group["root_selective_bootstrap_replicates"],
                        errors="coerce",
                    ).max()
                ),
                "selected_root_guard_block_rate": float(
                    group["selected_root_guard_blocked"].mean()
                ),
                "root_selective_p_value_min": float(root_p.min()),
                "root_selective_p_value_median": float(root_p.median()),
                "root_selective_p_value_max": float(root_p.max()),
                "false_split_rate": float(
                    group["selected_root_guard_false_split"].mean()
                ),
                "false_split_rate_upper_confidence": _wilson_upper_bound(
                    false_count,
                    int(group.shape[0]),
                ),
                "mean_ari": float(group["selected_root_guard_ari"].mean()),
                "mean_ari_lower_confidence": _mean_lower_bound(
                    group["selected_root_guard_ari"]
                ),
                "sensitivity_status": _status(
                    status_frame,
                    max_null_split_rate=max_null_split_rate,
                    min_signal_mean_ari=min_signal_mean_ari,
                ),
                "study_role": STUDY_ROLE,
            }
        )
        records.append(row)

    case_summary = pd.DataFrame.from_records(records)
    if case_summary.empty:
        return case_summary

    transfer_records: list[dict[str, object]] = []
    transfer_columns = (
        "source_family",
        "candidate_method",
        "selected_topology_penalty",
    )
    for key, group in case_summary.groupby(list(transfer_columns), sort=True):
        null_rows = group[group["data_role"].eq("null")]
        signal_rows = group[group["data_role"].eq("signal")]
        if null_rows.empty or signal_rows.empty:
            status = "data_independent_traversal_transfer_insufficient_coverage"
        elif not bool(
            null_rows["sensitivity_status"].eq(
                "data_independent_traversal_null_candidate"
            ).all()
        ):
            status = "data_independent_traversal_transfer_null_inflated"
        elif not bool(
            signal_rows["sensitivity_status"].eq(
                "data_independent_traversal_signal_retained"
            ).all()
        ):
            status = "data_independent_traversal_transfer_signal_weak"
        else:
            status = "data_independent_traversal_transfer_candidate"
        confidence_status = _confidence_transfer_status(
            group.rename(columns={"sensitivity_status": "traversal_status"}),
            max_null_split_rate=max_null_split_rate,
            min_signal_mean_ari=min_signal_mean_ari,
        )
        row = dict(zip(transfer_columns, key))
        row.update(
            {
                "n_case_role_rows": int(group.shape[0]),
                "root_selective_bootstrap_replicates": int(
                    group["root_selective_bootstrap_replicates"].max()
                ),
                "max_null_false_split_rate": (
                    float(null_rows["false_split_rate"].max())
                    if not null_rows.empty
                    else np.nan
                ),
                "max_null_false_split_rate_upper_confidence": (
                    float(null_rows["false_split_rate_upper_confidence"].max())
                    if not null_rows.empty
                    else np.nan
                ),
                "min_signal_mean_ari": (
                    float(signal_rows["mean_ari"].min())
                    if not signal_rows.empty
                    else np.nan
                ),
                "min_signal_mean_ari_lower_confidence": (
                    float(signal_rows["mean_ari_lower_confidence"].min())
                    if not signal_rows.empty
                    else np.nan
                ),
                "max_selected_root_guard_block_rate": float(
                    group["selected_root_guard_block_rate"].max()
                ),
                "selected_root_guard_transfer_status": status,
                "selected_root_guard_confidence_status": confidence_status,
                "study_role": STUDY_ROLE,
            }
        )
        transfer_records.append(row)
    return pd.DataFrame.from_records(transfer_records)


def run_data_independent_sibling_gate_traversal_panel(
    config: DataIndependentSiblingGateTraversalConfig,
) -> dict[str, Path]:
    """Run the fixed-gate traversal diagnostic and write outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if int(config.root_selective_bootstrap_replicates) < 0:
        raise ValueError("root_selective_bootstrap_replicates must be non-negative.")
    if int(config.root_stability_subsample_replicates) < 0:
        raise ValueError("root_stability_subsample_replicates must be non-negative.")
    if not 0.0 < float(config.root_stability_feature_fraction) <= 1.0:
        raise ValueError("root_stability_feature_fraction must lie in (0, 1].")
    if config.root_stability_guard_threshold is not None:
        threshold = float(config.root_stability_guard_threshold)
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("root_stability_guard_threshold must lie in [-1, 1].")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        "data_independent_sibling_gate_traversal__"
        f"{config.suite}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    rows: list[dict[str, object]] = []
    checkpoint_paths: list[Path] = []
    for case in _select_cases(suite=config.suite, case_names=config.case_names):
        (
            case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        if source_family not in {"binary_template", "categorical_multinomial"}:
            raise ValueError(
                "Data-independent traversal panel supports binary and direct "
                f"categorical cases only; got {source_family!r}."
            )
        for replicate in range(int(config.replicates)):
            data_seed = int(config.base_seed) + replicate * 1009
            for data_role in config.data_roles:
                replicate_rows = _rows_for_replicate(
                    case=case,
                    case_id=case_id,
                    source_family=source_family,
                    feature_representation=feature_representation,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_categories=n_categories,
                    data_role=data_role,
                    replicate=replicate,
                    data_seed=data_seed,
                    candidate_methods=config.candidate_methods,
                    selected_topology_penalties=(config.selected_topology_penalties),
                    sibling_alpha=float(config.sibling_alpha),
                    edge_alpha=float(config.edge_alpha),
                    run_id=run_id,
                    root_selective_bootstrap_replicates=int(
                        config.root_selective_bootstrap_replicates
                    ),
                    root_stability_subsample_replicates=int(
                        config.root_stability_subsample_replicates
                    ),
                    root_stability_feature_fraction=float(
                        config.root_stability_feature_fraction
                    ),
                    root_stability_seed=int(config.root_stability_seed),
                    root_stability_guard_threshold=(
                        float(config.root_stability_guard_threshold)
                        if config.root_stability_guard_threshold is not None
                        else None
                    ),
                )
                rows.extend(replicate_rows)
                checkpoint_paths.append(
                    _checkpoint_rows(
                        replicate_rows,
                        checkpoint_dir=config.checkpoint_rows_dir,
                        case_id=case_id,
                        data_role=data_role,
                        replicate=replicate,
                    )
                )
    row_table = pd.DataFrame.from_records(rows)
    summary = summarize_traversal_rows(
        row_table,
        max_null_split_rate=float(config.max_null_split_rate),
        min_signal_mean_ari=float(config.min_signal_mean_ari),
    )
    transfer_summary = summarize_traversal_transfer(summary)
    threshold_sensitivity = summarize_root_stability_threshold_sensitivity(
        row_table,
        max_null_split_rate=float(config.max_null_split_rate),
        min_signal_mean_ari=float(config.min_signal_mean_ari),
    )
    root_selective_sensitivity = summarize_root_selective_guard_sensitivity(
        row_table,
        max_null_split_rate=float(config.max_null_split_rate),
        min_signal_mean_ari=float(config.min_signal_mean_ari),
    )
    components, production_summary = build_traversal_production_components(
        transfer_summary
    )
    row_table.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    transfer_summary.to_csv(config.transfer_summary_path, index=False)
    threshold_sensitivity.to_csv(
        config.root_stability_threshold_sensitivity_path,
        index=False,
    )
    root_selective_sensitivity.to_csv(
        config.root_selective_guard_sensitivity_path,
        index=False,
    )
    components.to_csv(config.production_components_path, index=False)
    production_summary.to_csv(config.production_summary_path, index=False)
    config.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "generated_by": GENERATED_BY,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "suite": config.suite,
                "case_names": list(config.case_names),
                "data_roles": list(config.data_roles),
                "candidate_methods": list(config.candidate_methods),
                "selected_topology_penalties": list(config.selected_topology_penalties),
                "selected_tree_distance_metric": SELECTED_TREE_DISTANCE_METRIC,
                "selected_tree_linkage_method": SELECTED_TREE_LINKAGE_METHOD,
                "replicates": int(config.replicates),
                "base_seed": int(config.base_seed),
                "root_selective_bootstrap_replicates": int(
                    config.root_selective_bootstrap_replicates
                ),
                "root_stability_subsample_replicates": int(
                    config.root_stability_subsample_replicates
                ),
                "root_stability_feature_fraction": float(
                    config.root_stability_feature_fraction
                ),
                "root_stability_seed": int(config.root_stability_seed),
                "root_stability_guard_threshold": (
                    float(config.root_stability_guard_threshold)
                    if config.root_stability_guard_threshold is not None
                    else None
                ),
                "outputs": {
                    "rows": str(config.rows_path),
                    "checkpoint_rows_dir": str(config.checkpoint_rows_dir),
                    "checkpoint_rows": [str(path) for path in checkpoint_paths],
                    "summary": str(config.summary_path),
                    "transfer_summary": str(config.transfer_summary_path),
                    "root_stability_threshold_sensitivity": str(
                        config.root_stability_threshold_sensitivity_path
                    ),
                    "root_selective_guard_sensitivity": str(
                        config.root_selective_guard_sensitivity_path
                    ),
                    "production_components": str(config.production_components_path),
                    "production_summary": str(config.production_summary_path),
                },
                "interpretation": (
                    "Diagnostic traversal impact of fixed sibling gates that avoid "
                    "same-sample adaptive sibling PCA."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "rows": config.rows_path,
        "checkpoint_rows_dir": config.checkpoint_rows_dir,
        "summary": config.summary_path,
        "transfer_summary": config.transfer_summary_path,
        "root_stability_threshold_sensitivity": (
            config.root_stability_threshold_sensitivity_path
        ),
        "root_selective_guard_sensitivity": config.root_selective_guard_sensitivity_path,
        "production_components": config.production_components_path,
        "production_summary": config.production_summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names")
    parser.add_argument("--data-roles", default=",".join(DEFAULT_DATA_ROLES))
    parser.add_argument(
        "--candidate-methods",
        default=",".join(DEFAULT_CANDIDATE_METHODS),
    )
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument(
        "--selected-topology-penalties",
        default=",".join(str(value) for value in DEFAULT_SELECTED_TOPOLOGY_PENALTIES),
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--root-selective-bootstrap-replicates", type=int, default=0)
    parser.add_argument("--root-stability-subsample-replicates", type=int, default=0)
    parser.add_argument("--root-stability-feature-fraction", type=float, default=0.8)
    parser.add_argument("--root-stability-seed", type=int, default=0)
    parser.add_argument("--root-stability-guard-threshold", type=float)
    parser.add_argument("--max-null-split-rate", type=float, default=0.05)
    parser.add_argument("--min-signal-mean-ari", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_data_independent_sibling_gate_traversal_panel(
        DataIndependentSiblingGateTraversalConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=parse_names(args.case_names),
            data_roles=validate_data_roles(parse_names(args.data_roles)),
            candidate_methods=validate_candidate_methods(parse_names(args.candidate_methods)),
            sibling_alpha=float(args.sibling_alpha),
            edge_alpha=float(args.edge_alpha),
            selected_topology_penalties=validate_selected_topology_penalties(
                tuple(float(value) for value in parse_names(args.selected_topology_penalties))
            ),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            root_selective_bootstrap_replicates=int(
                args.root_selective_bootstrap_replicates
            ),
            root_stability_subsample_replicates=int(
                args.root_stability_subsample_replicates
            ),
            root_stability_feature_fraction=float(
                args.root_stability_feature_fraction
            ),
            root_stability_seed=int(args.root_stability_seed),
            root_stability_guard_threshold=(
                float(args.root_stability_guard_threshold)
                if args.root_stability_guard_threshold is not None
                else None
            ),
            max_null_split_rate=float(args.max_null_split_rate),
            min_signal_mean_ari=float(args.min_signal_mean_ari),
        )
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DataIndependentSiblingGateTraversalConfig",
    "build_traversal_production_components",
    "root_feature_subsample_stability",
    "selected_root_permutation_p_value",
    "selected_tree_oracle_cut_ari",
    "run_data_independent_sibling_gate_traversal_panel",
    "summarize_root_stability_threshold_sensitivity",
    "summarize_root_selective_guard_sensitivity",
    "summarize_traversal_rows",
    "summarize_traversal_transfer",
]
