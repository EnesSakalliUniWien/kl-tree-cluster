"""
Precompute-and-reuse utils for fast similarity-based operations.

This module provides drop-in style helpers that precompute a full leaf×leaf
Hamming distance matrix once, and then reuse it to compute node similarities
without repeatedly calling pdist in hot loops.

It also includes "fast" versions of two higher-level routines that mirror
the logic in hierarchy_analysis.similarity_analysis and
hierarchy_analysis.adaptive_clustering but take advantage of the precomputed
distances:

- analyze_hierarchy_similarity_patterns_fast
- extract_clusters_from_tree_fast
- calculate_optimal_threshold_fast

These are opt-in; existing modules are not modified. You can import from
misc.similarity_precompute and use the *_fast variants to accelerate your
pipelines without changing core behavior.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform


# ----------------------------
# Core precompute primitives
# ----------------------------

def precompute_leaf_distances(
    sample_features: pd.DataFrame,
    *,
    dtype: np.dtype | str = np.float32,
) -> Tuple[np.ndarray, Dict[str, int], float]:
    """
    Compute full leaf×leaf Hamming distance matrix once and provide helpers.

    Args:
        sample_features: DataFrame with leaf labels as index; binary/int features.
        dtype: dtype for the full distance matrix (float32 is memory-friendly).

    Returns:
        full_dist: (N, N) ndarray of pairwise Hamming distances
        leaf_index: dict mapping leaf label -> row/col index in full_dist
        baseline_similarity: mean(1 - pairwise_hamming) across all leaf pairs
    """
    X = np.asarray(sample_features.values)
    condensed = pdist(X, metric="hamming")
    full = squareform(condensed).astype(dtype, copy=False)
    leaf_index = {label: i for i, label in enumerate(sample_features.index)}
    baseline_similarity = float(np.mean(1.0 - condensed)) if condensed.size else 1.0
    return full, leaf_index, baseline_similarity


def mean_similarity_from_precomputed(
    node_leaves: List[str],
    *,
    full_dist: np.ndarray,
    leaf_index: Dict[str, int],
) -> float:
    """
    Compute mean pairwise similarity (1 - Hamming) using a precomputed matrix.

    Returns 1.0 for single-leaf nodes.
    """
    k = len(node_leaves)
    if k < 2:
        return 1.0
    idx = [leaf_index[lbl] for lbl in node_leaves]
    if k == 2:
        d = float(full_dist[idx[0], idx[1]])
        return 1.0 - d
    sub = full_dist[np.ix_(idx, idx)]
    tri_sum = float(np.sum(np.triu(sub, k=1)))
    num_pairs = k * (k - 1) / 2.0
    mean_d = tri_sum / num_pairs if num_pairs > 0 else 0.0
    return 1.0 - mean_d


def _to_bool_significance(val) -> bool:
    """Robustly interpret significance flags from various encodings."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        # Support multiple conventions used across the codebase
        if v in {"significant", "features dependent", "dependent", "true", "yes"}:
            return True
        return False
    return bool(val)


# ---------------------------------------------------
# Fast variants mirroring higher-level functionality
# ---------------------------------------------------

def analyze_hierarchy_similarity_patterns_fast(
    tree: "nx.DiGraph",
    significance_results: pd.DataFrame,
    sample_features: pd.DataFrame,
    *,
    significance_column: str = "Are_Features_Dependent",
    dtype: np.dtype | str = np.float32,
) -> pd.DataFrame:
    """
    Faster variant of analyze_hierarchy_similarity_patterns that reuses a
    precomputed leaf×leaf Hamming distance matrix.
    """
    full_dist, leaf_index, baseline_similarity = precompute_leaf_distances(
        sample_features, dtype=dtype
    )

    node_significance: Dict[str, bool] = {}
    for node_id, row in significance_results.iterrows():
        node_significance[str(node_id)] = _to_bool_significance(
            row.get(significance_column, False)
        )

    # Analyze each node
    analysis_rows: List[dict] = []
    for node_name in node_significance.keys():
        leaves = tree.get_leaves(node=node_name, return_labels=True)
        node_similarity = mean_similarity_from_precomputed(
            leaves, full_dist=full_dist, leaf_index=leaf_index
        )
        similarity_improvement = float(node_similarity - baseline_similarity)

        parent_name = next(tree.predecessors(node_name), None)
        parent_significant = (
            node_significance.get(parent_name, False) if parent_name else False
        )

        analysis_rows.append(
            {
                "Node": node_name,
                "Is_Significant": node_significance[node_name],
                "Num_Leaves": len(leaves),
                "Similarity": node_similarity,
                "Similarity_Improvement": similarity_improvement,
                "Parent": parent_name,
                "Parent_Significant": parent_significant,
                "Leaves": ", ".join(sorted(leaves)),
            }
        )

    return pd.DataFrame(analysis_rows).sort_values("Num_Leaves")


def extract_clusters_from_tree_fast(
    tree: "nx.DiGraph",
    significance_results: pd.DataFrame,
    sample_features: pd.DataFrame,
    *,
    significance_column: str,
    similarity_threshold: float,
    dtype: np.dtype | str = np.float32,
) -> Tuple[Dict[str, int], set[str], Dict[str, str]]:
    """
    Faster variant of _extract_clusters_from_tree that reuses a precomputed
    distance matrix.
    """
    node_significance: Dict[str, bool] = {}
    for node_id, row in significance_results.iterrows():
        node_significance[str(node_id)] = _to_bool_significance(
            row.get(significance_column, False)
        )

    full_dist, leaf_index, baseline_similarity = precompute_leaf_distances(
        sample_features, dtype=dtype
    )

    cluster_root_nodes: List[str] = []
    boundary_reasons: Dict[str, str] = {}
    similarity_analysis: Dict[str, dict] = {}

    for node_name, is_sig in node_significance.items():
        parent_name = next(tree.predecessors(node_name), None)
        if not is_sig:
            continue

        leaves = tree.get_leaves(node=node_name, return_labels=True)
        node_similarity = mean_similarity_from_precomputed(
            leaves, full_dist=full_dist, leaf_index=leaf_index
        )
        similarity_improvement = float(node_similarity - baseline_similarity)
        similarity_analysis[node_name] = {
            "node_similarity": node_similarity,
            "similarity_improvement": similarity_improvement,
            "num_leaves": len(leaves),
        }

        if parent_name is None:
            parent_significant = False
            parent_similarity_improvement = 0.0
        else:
            parent_significant = node_significance.get(parent_name, False)
            if parent_significant and parent_name in similarity_analysis:
                parent_similarity_improvement = similarity_analysis[parent_name][
                    "similarity_improvement"
                ]
            else:
                parent_similarity_improvement = 0.0

        similarity_advantage = similarity_improvement - parent_similarity_improvement

        is_boundary = False
        reason = ""
        if not parent_significant:
            is_boundary = True
            reason = "significant with non-significant parent"
        elif similarity_advantage > similarity_threshold:
            is_boundary = True
            reason = (
                f"similarity advantage over parent ({similarity_advantage:.4f} > {similarity_threshold})"
            )

        if is_boundary:
            cluster_root_nodes.append(node_name)
            boundary_reasons[node_name] = reason

    # Assign leaves under boundary nodes
    cluster_assignments: Dict[str, int] = {}
    cluster_id = 0
    for root_node in cluster_root_nodes:
        leaves_in_cluster = tree.get_leaves(node=root_node, return_labels=True)
        for leaf in leaves_in_cluster:
            cluster_assignments[leaf] = cluster_id
        cluster_id += 1

    # Any unassigned leaves → singleton clusters
    all_leaves = tree.get_leaves(return_labels=True)
    for leaf in all_leaves:
        if leaf not in cluster_assignments:
            cluster_assignments[leaf] = cluster_id
            cluster_id += 1

    return cluster_assignments, set(cluster_root_nodes), boundary_reasons


def calculate_optimal_threshold_fast(
    tree: "nx.DiGraph",
    significance_results: pd.DataFrame,
    sample_features: pd.DataFrame,
    *,
    significance_column: str = "Are_Features_Dependent",
    dtype: np.dtype | str = np.float32,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """
    Faster variant mirroring _calculate_optimal_threshold from
    hierarchy_analysis.adaptive_clustering but reusing a precomputed
    distance matrix.

    Returns: (optimal_threshold, sorted_improvements, sorted_node_names, elbow_index)
    """
    full_dist, leaf_index, baseline_similarity = precompute_leaf_distances(
        sample_features, dtype=dtype
    )

    node_significance: Dict[str, bool] = {}
    for node_id, row in significance_results.iterrows():
        node_significance[str(node_id)] = _to_bool_significance(
            row.get(significance_column, False)
        )

    improvements: List[float] = []
    node_names: List[str] = []
    tmp: Dict[str, float] = {}

    # First pass: compute each significant node's improvement
    for node_name, is_sig in node_significance.items():
        if not is_sig:
            continue
        leaves = tree.get_leaves(node=node_name, return_labels=True)
        node_similarity = mean_similarity_from_precomputed(
            leaves, full_dist=full_dist, leaf_index=leaf_index
        )
        tmp[node_name] = float(node_similarity - baseline_similarity)

    # Second pass: compute advantage over parent's improvement
    for node_name, node_impr in tmp.items():
        parent = next(tree.predecessors(node_name), None)
        parent_impr = float(tmp.get(parent, 0.0)) if parent else 0.0
        adv = node_impr - parent_impr
        if adv > 0:
            improvements.append(float(adv))
            node_names.append(node_name)

    if len(improvements) < 3:
        opt = float(np.median(improvements)) if improvements else 0.0
        elbow_idx = len(improvements) // 2
        return opt, np.array(improvements, dtype=float), np.array(node_names, dtype=object), int(elbow_idx)

    # Elbow method via second derivative of the sorted curve
    order = np.argsort(improvements)[::-1]
    y = np.asarray(improvements, dtype=float)[order]
    sorted_nodes = np.asarray(node_names, dtype=object)[order]
    x = np.linspace(0.0, 1.0, len(y))
    dy = np.diff(y) / np.diff(x)
    d2y = np.diff(dy) / np.diff(x[:-1])

    abs_d2 = np.abs(d2y)
    max_curv = float(np.max(abs_d2)) if abs_d2.size else 0.0
    thr = 0.05 * max_curv
    sig_idx = np.flatnonzero(abs_d2 > thr)
    if sig_idx.size:
        elbow_idx = int(sig_idx[-1] + 1)  # +1 due to diff offset
    else:
        elbow_idx = int(np.argmax(abs_d2) + 1)

    optimal_threshold = float(y[elbow_idx]) if y.size else 0.0
    return optimal_threshold, y, sorted_nodes, int(elbow_idx)

