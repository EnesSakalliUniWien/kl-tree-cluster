#!/usr/bin/env python3
"""Validate family-specific graph geometries with aligned NNLS branch targets.

This is an evidence-only grid. It does not change the production method
registry, selector, alpha policy, or fallback behavior.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.metrics import adjusted_rand_score
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.optimized_branch_lengths import (
    BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
    BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN,
)

from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.metrics import _calculate_ari_nmi_purity_metrics
from benchmarks.shared.runners.tbs_diffusion_runner import (
    _run_tbs_diffusion_graphtools_method,
)
from benchmarks.shared.tree_consensus import (
    TARGET_TREE_CONSENSUS_GRID,
    TARGET_TREE_CONSENSUS_METHOD,
    build_tree_consensus_tables,
)
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.time import format_timestamp_utc

SCHEMA_VERSION = "family_metric_nnls_grid/v1"
GENERATED_BY = "benchmarks.validation.sweeps.family_metric_nnls_grid"

DEFAULT_OUTPUT_DIR = Path("reports/family_metric_nnls_grid")
DEFAULT_CASE_NAMES = (
    "gauss_clear_medium_continuous",
    "cont_lowrank_pggn_shrinkage",
    "binary_low_noise_4c",
    "overlap_mod_4c_small",
    "sbm_moderate",
    "cat_clear_3cat_4c",
)
TOPOLOGIES = (
    "average",
    "complete",
    "weighted",
    "single",
    "centroid",
    "median",
    "ward",
    "neighbor_joining",
)
BRANCH_MODES = ("none", "normalized_branch_length")


@dataclass(frozen=True)
class FamilyGeometry:
    """One Euclidean embedding shared by graph construction and NNLS."""

    family: str
    profile_id: str
    graph_distance: str
    nnls_target: str
    embedding: pd.DataFrame
    diagnostics: Mapping[str, object]


def infer_metric_family(
    case: Mapping[str, object],
    data: pd.DataFrame,
    feature_space: FeatureSpace | None,
) -> str:
    """Resolve a label-free geometry family from the benchmark data contract."""
    generator = str(case.get("generator", ""))
    category = str(case.get("category", "")).lower()
    name = str(case.get("name", "")).lower()
    if generator == "sbm":
        return "sbm"
    if (
        feature_space is not None
        and feature_space.has_categorical_blocks
        and feature_space.has_continuous_blocks
    ):
        raise ValueError(
            "Family metric validation does not yet define a mixed continuous/categorical "
            "Gower geometry."
        )
    if feature_space is not None and feature_space.has_categorical_blocks:
        return "categorical"
    if feature_space is not None and feature_space.has_continuous_blocks:
        if data.shape[1] >= data.shape[0] or generator in {
            "continuous_low_rank_factor",
            "continuous_spiked_covariance",
        }:
            return "high_dimensional_continuous"
        return "continuous_gaussian"
    if "overlap" in category or "overlap" in name:
        return "sparse_overlap_binary"
    return "balanced_binary"


def _frame(values: np.ndarray, index: pd.Index, prefix: str) -> pd.DataFrame:
    matrix = np.asarray(values, dtype=float)
    return pd.DataFrame(
        matrix,
        index=index,
        columns=[f"{prefix}_{column}" for column in range(matrix.shape[1])],
    )


def _standardized_embedding(data: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    matrix = data.to_numpy(dtype=float, copy=True)
    center = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, ddof=1, keepdims=True)
    constant = ~np.isfinite(scale) | (scale <= 0.0)
    scale[constant] = 1.0
    embedding = (matrix - center) / scale / math.sqrt(matrix.shape[1])
    return embedding, {"constant_columns": int(constant.sum())}


def _shrinkage_mahalanobis_embedding(
    data: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    standardized, diagnostics = _standardized_embedding(data)
    standardized *= math.sqrt(data.shape[1])
    standardized -= standardized.mean(axis=0, keepdims=True)
    shrinkage = float(
        ledoit_wolf_shrinkage(
            standardized,
            assume_centered=True,
            block_size=256,
        )
    )
    left, singular_values, _right = np.linalg.svd(standardized, full_matrices=False)
    covariance_eigenvalues = singular_values * singular_values / len(standardized)
    covariance_mean = float(np.mean(standardized * standardized))
    denominator = np.sqrt((1.0 - shrinkage) * covariance_eigenvalues + shrinkage * covariance_mean)
    keep = (singular_values > np.finfo(float).eps) & (denominator > np.finfo(float).eps)
    embedding = left[:, keep] * (singular_values[keep] / denominator[keep])
    embedding /= math.sqrt(data.shape[1])
    diagnostics.update(
        {
            "ledoit_wolf_shrinkage": shrinkage,
            "embedding_dimension": int(embedding.shape[1]),
            "dimension_rule": "all_nonzero_sample_singular_directions",
        }
    )
    return embedding, diagnostics


def _cosine_embedding(data: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    matrix = data.to_numpy(dtype=float, copy=True)
    norms = np.linalg.norm(matrix, axis=1)
    zero_rows = norms <= np.finfo(float).eps
    embedding = np.zeros((len(matrix), matrix.shape[1] + 1), dtype=float)
    nonzero = ~zero_rows
    embedding[nonzero, :-1] = matrix[nonzero] / norms[nonzero, None]
    embedding[zero_rows, -1] = 1.0
    embedding /= math.sqrt(2.0)
    return embedding, {"zero_rows": int(zero_rows.sum())}


def _regularized_laplacian_embedding(
    data: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    adjacency = data.to_numpy(dtype=float, copy=True)
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("SBM geometry requires a square adjacency matrix.")
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    mean_degree = float(adjacency.sum(axis=1).mean())
    tau = max(mean_degree, np.finfo(float).eps)
    regularized = adjacency + tau / len(adjacency)
    degrees = regularized.sum(axis=1)
    normalized = regularized / np.sqrt(degrees[:, None] * degrees[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    max_rank = min(10, len(eigenvalues) - 1)
    if max_rank < 2:
        rank = max_rank
    else:
        gaps = eigenvalues[:max_rank] - eigenvalues[1 : max_rank + 1]
        rank = max(2, int(np.argmax(gaps)) + 1)
    positive = np.maximum(eigenvalues[:rank], 0.0)
    embedding = eigenvectors[:, :rank] * np.sqrt(positive)[None, :]
    return embedding, {
        "regularization_tau": tau,
        "mean_degree": mean_degree,
        "embedding_dimension": int(rank),
        "dimension_rule": "largest_regularized_laplacian_eigengap_top10_min2",
        "retained_eigenvalues": json.dumps(eigenvalues[:rank].tolist()),
    }


def _categorical_block_embedding(
    data: pd.DataFrame,
    feature_space: FeatureSpace | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if feature_space is None or not feature_space.has_categorical_blocks:
        raise ValueError("Categorical geometry requires explicit categorical feature blocks.")
    blocks = [block for block in feature_space.blocks if block.family == "categorical"]
    matrix = data.to_numpy(dtype=float, copy=False)
    embedding = np.concatenate(
        [matrix[:, block.column_indices] for block in blocks],
        axis=1,
    ) / math.sqrt(2.0 * len(blocks))
    return embedding, {
        "categorical_blocks": int(len(blocks)),
        "block_distance_identity": "squared_euclidean_equals_block_mismatch_fraction",
    }


def build_family_geometry(
    case: Mapping[str, object],
    data: pd.DataFrame,
    feature_space: FeatureSpace | None,
) -> FamilyGeometry:
    """Build the selected family embedding without using benchmark labels."""
    family = infer_metric_family(case, data, feature_space)
    diagnostics: dict[str, object]
    if family == "continuous_gaussian":
        values, diagnostics = _standardized_embedding(data)
        profile = "standardized_euclidean"
    elif family == "high_dimensional_continuous":
        values, diagnostics = _shrinkage_mahalanobis_embedding(data)
        profile = "ledoit_wolf_shrinkage_mahalanobis"
    elif family == "balanced_binary":
        values = data.to_numpy(dtype=float, copy=True) / math.sqrt(data.shape[1])
        diagnostics = {}
        profile = "hamming_via_binary_euclidean_embedding"
    elif family == "sparse_overlap_binary":
        values, diagnostics = _cosine_embedding(data)
        profile = "cosine_via_unit_sphere_embedding"
    elif family == "sbm":
        values, diagnostics = _regularized_laplacian_embedding(data)
        profile = "regularized_laplacian_spectral"
    elif family == "categorical":
        values, diagnostics = _categorical_block_embedding(data, feature_space)
        profile = "block_aware_mismatch_hellinger"
    else:
        raise AssertionError(f"Unhandled family {family!r}.")
    if values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError(f"Geometry {profile!r} produced an invalid embedding.")
    return FamilyGeometry(
        family=family,
        profile_id=profile,
        graph_distance="euclidean_on_selected_embedding",
        nnls_target=BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN,
        embedding=_frame(values, data.index, "geometry"),
        diagnostics=diagnostics,
    )


def _topology_parameters(topology: str) -> dict[str, str]:
    if topology == "neighbor_joining":
        return {
            "tree_builder": "neighbor_joining",
            "tree_rooting": "mad",
            "tree_linkage_method": "average",
        }
    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology {topology!r}.")
    return {
        "tree_builder": "linkage",
        "tree_rooting": "linkage_root",
        "tree_linkage_method": topology,
    }


def _run_id(topology: str) -> str:
    if topology == "neighbor_joining":
        return "family_metric_tree_builder_neighbor_joining"
    return f"family_metric_tree_linkage_method_{topology}"


def _bool_sum(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(
        frame[column].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"}).sum()
    )


def _root_child_fraction(tree: object) -> float:
    if tree is None:
        return math.nan
    root = tree.graph.get("root")
    if root is None:
        roots = [node for node, degree in tree.in_degree() if degree == 0]
        root = roots[0] if len(roots) == 1 else None
    if root is None:
        return math.nan
    children = list(tree.successors(root))
    if len(children) != 2:
        return math.nan
    sizes = []
    for child in children:
        nodes = {child, *nx.descendants(tree, child)}
        sizes.append(sum(bool(tree.nodes[node].get("is_leaf", False)) for node in nodes))
    return float(min(sizes) / sum(sizes)) if sum(sizes) else math.nan


def _normalized_branch_residual(extra: Mapping[str, object]) -> float:
    rmse = float(extra.get("branch_length_optimization_residual_rmse", math.nan))
    target_mean = float(extra.get("branch_length_optimization_target_mean", math.nan))
    if not math.isfinite(rmse) or not math.isfinite(target_mean) or target_mean <= 0.0:
        return math.nan
    return rmse / target_mean


def _branch_k_cut_labels(
    tree: object,
    sample_index: pd.Index,
    cluster_count: int,
) -> np.ndarray | None:
    """Greedily cut the longest fitted clade edges into ``cluster_count`` parts."""
    if tree is None or cluster_count < 1:
        return None
    tree_leaf_labels = {
        str(attrs.get("label", node))
        for node, attrs in tree.nodes(data=True)
        if bool(attrs.get("is_leaf", False))
    }
    sample_labels = [str(value) for value in sample_index]
    if set(sample_labels) != tree_leaf_labels:
        return None
    all_leaves = frozenset(sample_labels)
    candidates: list[tuple[float, frozenset[str]]] = []
    for _parent, child, attrs in tree.edges(data=True):
        nodes = {child, *nx.descendants(tree, child)}
        leaves = frozenset(
            str(tree.nodes[node].get("label", node))
            for node in nodes
            if bool(tree.nodes[node].get("is_leaf", False))
        )
        length = float(attrs.get("branch_length", 0.0))
        if leaves and leaves != all_leaves and math.isfinite(length):
            candidates.append((length, leaves))
    candidates.sort(key=lambda item: (-item[0], sorted(item[1])))
    parts = [all_leaves]
    while len(parts) < cluster_count:
        selected: tuple[int, frozenset[str]] | None = None
        for _length, leaves in candidates:
            for part_index, part in enumerate(parts):
                if leaves < part:
                    selected = part_index, leaves
                    break
            if selected is not None:
                break
        if selected is None:
            return None
        part_index, leaves = selected
        remainder = parts[part_index].difference(leaves)
        parts[part_index : part_index + 1] = [leaves, frozenset(remainder)]
    cluster_by_sample = {
        sample: cluster_id for cluster_id, part in enumerate(parts) for sample in part
    }
    return np.asarray([cluster_by_sample[sample] for sample in sample_labels], dtype=int)


def _finite_min(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else math.nan


def _true_cluster_clade_audit(
    tree: object,
    sample_index: pd.Index,
    true_labels: np.ndarray,
) -> tuple[float, float]:
    """Return exact-clade recovery and size-weighted minimal-clade purity."""
    if tree is None or len(sample_index) != len(true_labels):
        return math.nan, math.nan
    sample_names = [str(value) for value in sample_index]
    leaf_sets: list[frozenset[str]] = []
    for node in tree.nodes:
        nodes = {node, *nx.descendants(tree, node)}
        leaves = frozenset(
            str(tree.nodes[item].get("label", item))
            for item in nodes
            if bool(tree.nodes[item].get("is_leaf", False))
        )
        if leaves:
            leaf_sets.append(leaves)
    exact = 0
    weighted_purity = 0.0
    total = 0
    for label in np.unique(true_labels):
        cluster = frozenset(
            sample
            for sample, observed in zip(sample_names, true_labels, strict=True)
            if observed == label
        )
        supersets = [leaves for leaves in leaf_sets if cluster.issubset(leaves)]
        if not supersets:
            return math.nan, math.nan
        smallest = min(supersets, key=len)
        exact += int(smallest == cluster)
        weighted_purity += len(cluster) * len(cluster) / len(smallest)
        total += len(cluster)
    cluster_count = len(np.unique(true_labels))
    return exact / cluster_count, weighted_purity / total


def _result_row(
    *,
    case_number: int,
    case: Mapping[str, object],
    geometry: FamilyGeometry,
    topology: str,
    branch_mode: str,
    result: object | None,
    elapsed_sec: float,
    error: str = "",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    composite_case_id = f"{case['name']}::{branch_mode}"
    run_id = _run_id(topology)
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "test_case": int(case_number),
        "source_case_id": str(case["name"]),
        "case_id": composite_case_id,
        "case_category": geometry.family,
        "method": TARGET_TREE_CONSENSUS_METHOD,
        "benchmark_grid": TARGET_TREE_CONSENSUS_GRID,
        "run_id": run_id,
        "tree_inference": topology,
        "branch_time_mode": branch_mode,
        "metric_family": geometry.family,
        "geometry_profile": geometry.profile_id,
        "graph_distance": geometry.graph_distance,
        "nnls_target": geometry.nnls_target,
        "geometry_dimension": int(geometry.embedding.shape[1]),
        "elapsed_sec": float(elapsed_sec),
        **{f"geometry_{key}": value for key, value in geometry.diagnostics.items()},
    }
    if result is None or getattr(result, "status", "skip") != "ok":
        base.update(
            {
                "status": "skip",
                "skip_reason": error or str(getattr(result, "skip_reason", "unknown")),
                "true_clusters": math.nan,
                "found_clusters": math.nan,
                "labels_length": 0,
                "ari": math.nan,
                "nmi": math.nan,
                "macro_f1": math.nan,
                "purity": math.nan,
                "silhouette_score": math.nan,
                "davies_bouldin_index": math.nan,
                "calinski_harabasz_index": math.nan,
                "largest_cluster_fraction": math.nan,
            }
        )
        return base, []

    extra = result.extra or {}
    metrics = _calculate_ari_nmi_purity_metrics(
        result.report_df,
        geometry.embedding.index,
        np.asarray(case["_true_labels"]),
        metadata=dict(case.get("_metadata", {})),
        feature_matrix=geometry.embedding,
    )
    trace = pd.DataFrame(extra.get("full_edge_traversal_trace", []))
    labels = np.asarray(result.labels, dtype=int)
    true_labels = np.asarray(case["_true_labels"])
    topology_labels = _branch_k_cut_labels(
        extra.get("tree"),
        geometry.embedding.index,
        len(np.unique(true_labels)),
    )
    topology_metrics = None
    if topology_labels is not None:
        topology_report = pd.DataFrame(
            {"cluster_id": topology_labels},
            index=geometry.embedding.index,
        )
        topology_metrics = _calculate_ari_nmi_purity_metrics(
            topology_report,
            geometry.embedding.index,
            true_labels,
            feature_matrix=geometry.embedding,
        )
    exact_clade_fraction, weighted_clade_purity = _true_cluster_clade_audit(
        extra.get("tree"),
        geometry.embedding.index,
        true_labels,
    )
    root_trace = trace[
        pd.to_numeric(
            trace.get("depth", pd.Series(index=trace.index, dtype=float)),
            errors="coerce",
        ).eq(0)
    ]
    edge_open_trace = trace[
        trace.get("edge_gate_open", pd.Series(False, index=trace.index)).map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
    ]
    base.update(
        {
            "status": "ok",
            "skip_reason": "",
            "true_clusters": int(len(np.unique(case["_true_labels"]))),
            "found_clusters": int(result.found_clusters),
            "labels_length": int(len(labels)),
            **asdict(metrics),
            "root_smaller_child_fraction": _root_child_fraction(extra.get("tree")),
            "nnls_normalized_residual_rmse": _normalized_branch_residual(extra),
            "nnls_residual_rmse": extra.get("branch_length_optimization_residual_rmse", math.nan),
            "nnls_target_mean": extra.get("branch_length_optimization_target_mean", math.nan),
            "nnls_zero_branch_fraction": (
                float(
                    sum(
                        float(attrs.get("branch_length", 0.0)) <= 1e-12
                        for _, _, attrs in extra["tree"].edges(data=True)
                    )
                    / max(extra["tree"].number_of_edges(), 1)
                )
                if extra.get("tree") is not None
                else math.nan
            ),
            "trace_edge_open": _bool_sum(trace, "edge_gate_open"),
            "trace_sibling_open": _bool_sum(trace, "sibling_gate_open"),
            "root_edge_gate_open": _bool_sum(root_trace, "edge_gate_open") > 0,
            "root_sibling_p_value_corrected": _finite_min(
                root_trace.get("sibling_p_value_corrected", pd.Series(dtype=float))
            ),
            "min_edge_open_sibling_p_value_corrected": _finite_min(
                edge_open_trace.get("sibling_p_value_corrected", pd.Series(dtype=float))
            ),
            "min_left_edge_p_value_bh": _finite_min(
                trace.get("left_edge_p_value_bh", pd.Series(dtype=float))
            ),
            "topology_kcut_ari": (
                math.nan
                if topology_labels is None
                else float(adjusted_rand_score(true_labels, topology_labels))
            ),
            "topology_kcut_nmi": (math.nan if topology_metrics is None else topology_metrics.nmi),
            "topology_kcut_macro_f1": (
                math.nan if topology_metrics is None else topology_metrics.macro_f1
            ),
            "topology_kcut_silhouette_score": (
                math.nan if topology_metrics is None else topology_metrics.silhouette_score
            ),
            "topology_kcut_largest_cluster_fraction": (
                math.nan if topology_metrics is None else topology_metrics.largest_cluster_fraction
            ),
            "topology_exact_true_cluster_clade_fraction": exact_clade_fraction,
            "topology_weighted_true_cluster_clade_purity": weighted_clade_purity,
            "adaptive_neighbor_selected_k": (extra.get("graphtools_diffusion") or {}).get(
                "adaptive_neighbor_selected_k", math.nan
            ),
        }
    )
    label_rows = [
        {
            "test_case": int(case_number),
            "case_id": composite_case_id,
            "method": TARGET_TREE_CONSENSUS_METHOD,
            "run_id": run_id,
            "sample_id": str(sample_id),
            "cluster_label": int(label),
        }
        for sample_id, label in zip(geometry.embedding.index, labels, strict=True)
    ]
    return base, label_rows


def _branch_pairs(cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_case_id", "metric_family", "geometry_profile", "tree_inference"]
    none = cells[cells["branch_time_mode"].eq("none")].copy()
    timed = cells[cells["branch_time_mode"].eq("normalized_branch_length")].copy()
    metrics = [
        "found_clusters",
        "ari",
        "nmi",
        "macro_f1",
        "purity",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "largest_cluster_fraction",
        "singleton_fraction",
    ]
    keep = keys + ["status", *metrics]
    paired = none[keep].merge(timed[keep], on=keys, suffixes=("_none", "_branch"))
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[f"{metric}_branch"] - paired[f"{metric}_none"]
    return paired


def _summary(cells: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = [
        "ari",
        "nmi",
        "macro_f1",
        "purity",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "largest_cluster_fraction",
        "singleton_fraction",
        "nnls_normalized_residual_rmse",
        "root_smaller_child_fraction",
        "topology_kcut_ari",
        "topology_kcut_nmi",
        "topology_kcut_macro_f1",
        "topology_kcut_silhouette_score",
        "topology_kcut_largest_cluster_fraction",
        "topology_exact_true_cluster_clade_fraction",
        "topology_weighted_true_cluster_clade_purity",
        "root_sibling_p_value_corrected",
        "min_edge_open_sibling_p_value_corrected",
    ]
    ok = cells[cells["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(columns=group_columns)
    aggregations: dict[str, tuple[str, str]] = {
        "ok_cells": ("status", "size"),
        "mean_found_clusters": ("found_clusters", "mean"),
    }
    for metric in metrics:
        aggregations[f"mean_{metric}"] = (metric, "mean")
        aggregations[f"median_{metric}"] = (metric, "median")
    return ok.groupby(group_columns, dropna=False).agg(**aggregations).reset_index()


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    visible = frame[[column for column in columns if column in frame.columns]].copy()
    if visible.empty:
        return "_No rows._"
    for column in visible.columns:
        if pd.api.types.is_float_dtype(visible[column]):
            visible[column] = visible[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    lines = [
        "| " + " | ".join(visible.columns) + " |",
        "| " + " | ".join("---" for _ in visible.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in visible.itertuples(index=False)
    )
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    cells: pd.DataFrame,
    case_summary: pd.DataFrame,
    branch_pairs: pd.DataFrame,
    selection: pd.DataFrame,
    consensus_status: str,
) -> Path:
    paired_ok = branch_pairs[
        branch_pairs["status_none"].eq("ok") & branch_pairs["status_branch"].eq("ok")
    ]
    branch_summary = pd.DataFrame(
        [
            {
                "paired_cells": len(paired_ok),
                "mean_delta_ari": paired_ok.get("delta_ari", pd.Series(dtype=float)).mean(),
                "mean_delta_nmi": paired_ok.get("delta_nmi", pd.Series(dtype=float)).mean(),
                "mean_delta_macro_f1": paired_ok.get(
                    "delta_macro_f1", pd.Series(dtype=float)
                ).mean(),
                "mean_delta_silhouette": paired_ok.get(
                    "delta_silhouette_score", pd.Series(dtype=float)
                ).mean(),
                "mean_delta_largest_cluster_fraction": paired_ok.get(
                    "delta_largest_cluster_fraction", pd.Series(dtype=float)
                ).mean(),
            }
        ]
    )
    report = "\n".join(
        [
            "# Family-Specific Metric and NNLS Grid",
            "",
            f"- schema_version: `{SCHEMA_VERSION}`",
            f"- generated_at_utc: `{format_timestamp_utc()}`",
            "- decision_scope: `evidence_only_no_production_promotion`",
            f"- selector_audit_status: `{consensus_status}`",
            f"- cells: `{len(cells)}`",
            f"- successful_cells: `{int(cells['status'].eq('ok').sum())}`",
            f"- skipped_cells: `{int(cells['status'].ne('ok').sum())}`",
            "",
            "The graph kernel and NNLS target use the same selected family embedding. "
            "The original observations remain the input to TBS distributional tests.",
            "",
            "## Family Results",
            "",
            _markdown_table(
                case_summary,
                [
                    "source_case_id",
                    "metric_family",
                    "branch_time_mode",
                    "ok_cells",
                    "mean_ari",
                    "mean_nmi",
                    "mean_macro_f1",
                    "mean_silhouette_score",
                    "mean_davies_bouldin_index",
                    "mean_largest_cluster_fraction",
                    "mean_nnls_normalized_residual_rmse",
                    "mean_topology_kcut_ari",
                    "mean_topology_exact_true_cluster_clade_fraction",
                    "mean_topology_weighted_true_cluster_clade_purity",
                    "median_root_sibling_p_value_corrected",
                ],
            ),
            "",
            "## Label-Free Selections",
            "",
            _markdown_table(
                selection,
                [
                    "case_id",
                    "selected_tree_inference",
                    "selected_found_clusters",
                    "selected_mean_partition_agreement",
                    "selected_ari",
                    "selected_nmi",
                    "selected_macro_f1",
                    "selector_status",
                ],
            ),
            "",
            "## Branch-Time Paired Effect",
            "",
            _markdown_table(branch_summary, list(branch_summary.columns)),
            "",
            "## Skipped Cells",
            "",
            _markdown_table(
                cells[cells["status"].ne("ok")],
                [
                    "source_case_id",
                    "branch_time_mode",
                    "tree_inference",
                    "skip_reason",
                ],
            ),
            "",
        ]
    )
    path = output_dir / "family_metric_nnls_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_family_metric_nnls_grid(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    suite: str = "full",
    case_names: Sequence[str] = DEFAULT_CASE_NAMES,
    topologies: Sequence[str] = TOPOLOGIES,
    branch_modes: Sequence[str] = BRANCH_MODES,
) -> dict[str, Path]:
    """Run the focused metric/topology/branch-time validation grid."""
    invalid_topologies = sorted(set(topologies).difference(TOPOLOGIES))
    invalid_modes = sorted(set(branch_modes).difference(BRANCH_MODES))
    if invalid_topologies or invalid_modes:
        raise ValueError(
            f"Invalid topologies={invalid_topologies!r} or branch_modes={invalid_modes!r}."
        )
    all_cases = get_test_cases_by_suite(suite)
    by_name = {str(case["name"]): (index, case) for index, case in enumerate(all_cases, 1)}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown cases in suite {suite!r}: {missing!r}.")

    cell_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for case_name in case_names:
        case_number, raw_case = by_name[case_name]
        inputs = prepare_case_inputs(dict(raw_case), [])
        feature_space = inputs.metadata.get("feature_space")
        if feature_space is not None and not isinstance(feature_space, FeatureSpace):
            raise TypeError("Case feature_space metadata must be a FeatureSpace.")
        geometry = build_family_geometry(raw_case, inputs.data, feature_space)
        case = {
            **raw_case,
            "_true_labels": inputs.labels,
            "_metadata": inputs.metadata,
        }
        for branch_mode in branch_modes:
            for topology in topologies:
                start = perf_counter()
                try:
                    result = _run_tbs_diffusion_graphtools_method(
                        inputs.data,
                        0.01,
                        k_neighbors=10,
                        diffusion_time=3,
                        n_components=30,
                        metric="euclidean",
                        decay=40,
                        anisotropy=0.0,
                        kernel_symm="+",
                        random_state=0,
                        adaptive_neighbor_profile="fragmentation_guard",
                        adaptive_neighbor_grid=(5, 10, 15, 25, 40, 80, 160),
                        feature_space=feature_space,
                        graph_data_df=geometry.embedding,
                        branch_length_data_df=geometry.embedding,
                        branch_length_optimization_method=(
                            BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS
                        ),
                        branch_length_optimization_target_metric=(
                            BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN
                        ),
                        branch_length_optimization_pair_sample_size=50_000,
                        branch_length_optimization_random_state=0,
                        branch_length_optimization_solver_tolerance=1e-5,
                        branch_length_optimization_max_iterations=300,
                        edge_branch_length_variance_policy=branch_mode,
                        **_topology_parameters(topology),
                    )
                    row, labels = _result_row(
                        case_number=case_number,
                        case=case,
                        geometry=geometry,
                        topology=topology,
                        branch_mode=branch_mode,
                        result=result,
                        elapsed_sec=perf_counter() - start,
                    )
                except Exception as exc:
                    row, labels = _result_row(
                        case_number=case_number,
                        case=case,
                        geometry=geometry,
                        topology=topology,
                        branch_mode=branch_mode,
                        result=None,
                        elapsed_sec=perf_counter() - start,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                cell_rows.append(row)
                label_rows.extend(labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame.from_records(cell_rows)
    labels = pd.DataFrame.from_records(label_rows)
    cells_path = output_dir / "family_metric_nnls_cells.csv"
    labels_path = output_dir / "family_metric_nnls_labels.csv"
    cells.to_csv(cells_path, index=False)
    labels.to_csv(labels_path, index=False)

    consensus = build_tree_consensus_tables(cells, labels)
    pairwise_path = output_dir / "family_metric_nnls_pairwise_agreement.csv"
    stability_path = output_dir / "family_metric_nnls_stability.csv"
    rankings_path = output_dir / "family_metric_nnls_rankings.csv"
    selection_path = output_dir / "family_metric_nnls_selection.csv"
    consensus.pairwise_agreement.to_csv(pairwise_path, index=False)
    consensus.stability.to_csv(stability_path, index=False)
    consensus.rankings.to_csv(rankings_path, index=False)
    consensus.selection.to_csv(selection_path, index=False)

    branch_pairs = _branch_pairs(cells)
    branch_pairs_path = output_dir / "family_metric_nnls_branch_time_pairs.csv"
    branch_pairs.to_csv(branch_pairs_path, index=False)
    case_summary = _summary(
        cells,
        ["source_case_id", "metric_family", "geometry_profile", "branch_time_mode"],
    )
    case_summary_path = output_dir / "family_metric_nnls_case_summary.csv"
    case_summary.to_csv(case_summary_path, index=False)
    method_summary = _summary(
        cells,
        ["metric_family", "branch_time_mode", "tree_inference"],
    )
    method_summary_path = output_dir / "family_metric_nnls_method_summary.csv"
    method_summary.to_csv(method_summary_path, index=False)
    report_path = _write_report(
        output_dir,
        cells,
        case_summary,
        branch_pairs,
        consensus.selection,
        str(consensus.summary.iloc[0]["gate_status"]),
    )
    manifest_path = output_dir / "family_metric_nnls_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "generated_by": GENERATED_BY,
                "generated_at_utc": format_timestamp_utc(),
                "decision_scope": "evidence_only_no_production_promotion",
                "suite": suite,
                "case_names": list(case_names),
                "topologies": list(topologies),
                "branch_modes": list(branch_modes),
                "expected_cells": len(case_names) * len(topologies) * len(branch_modes),
                "observed_cells": len(cells),
                "successful_cells": int(cells["status"].eq("ok").sum()),
                "selector_audit": consensus.summary.iloc[0].to_dict(),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "cells": cells_path,
        "labels": labels_path,
        "pairwise_agreement": pairwise_path,
        "stability": stability_path,
        "rankings": rankings_path,
        "selection": selection_path,
        "branch_time_pairs": branch_pairs_path,
        "case_summary": case_summary_path,
        "method_summary": method_summary_path,
        "report": report_path,
        "manifest": manifest_path,
    }


def _comma_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--case-names", type=_comma_values, default=DEFAULT_CASE_NAMES)
    parser.add_argument("--topologies", type=_comma_values, default=TOPOLOGIES)
    parser.add_argument("--branch-modes", type=_comma_values, default=BRANCH_MODES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_family_metric_nnls_grid(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=args.case_names,
        topologies=args.topologies,
        branch_modes=args.branch_modes,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
