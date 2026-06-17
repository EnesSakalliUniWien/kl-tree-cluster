"""Candidate construction for the Gate 2 single-feature subtree policy."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis.legacy_methods.commit_c2ef9a69.kl_clustering_analysis import config

from ...projection.spectral.tree_helpers import build_subtree_data, is_leaf, precompute_descendants


def _is_single_feature_basis(projection_matrix: np.ndarray | None) -> bool:
    if projection_matrix is None:
        return False
    basis = np.asarray(projection_matrix, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != 1:
        return False
    nonzero = np.flatnonzero(np.abs(basis[0]) > 1e-12)
    if nonzero.size != 1:
        return False
    return bool(np.isclose(np.linalg.norm(basis[0]), 1.0))


def _extract_parent_lambda_max(
    tree: nx.DiGraph,
    node_id: str,
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> tuple[object | None, float | None]:
    """Return the parent id and its largest PCA eigenvalue when available."""
    parent_node = next(iter(tree.predecessors(node_id)), None)
    if parent_node is None:
        return None, None

    parent_eigs = node_pca_eigenvalues.get(parent_node)
    if parent_eigs is None:
        return parent_node, None

    parent_eigs = np.asarray(parent_eigs, dtype=np.float64)
    if parent_eigs.size == 0:
        return parent_node, None
    return parent_node, float(np.max(parent_eigs))


def _build_single_feature_basis(feature_index: int, feature_count: int) -> np.ndarray:
    """Build a canonical basis for a subtree with one varying feature."""
    basis = np.zeros((1, feature_count), dtype=np.float64)
    basis[0, feature_index] = 1.0
    return basis


def _build_single_feature_candidate(
    tree: nx.DiGraph,
    node_id: str,
    *,
    full_feature_matrix: np.ndarray,
    descendant_leaf_indices_by_node: dict[str, list],
    descendant_internal_nodes_by_node: dict[str, list],
    feature_count: int,
    include_internal: bool,
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    """Return audit metadata for a single-feature subtree, or None when ineligible."""
    if is_leaf(tree, node_id):
        return None

    existing_projection = node_pca_projections.get(node_id)
    if existing_projection is not None and not _is_single_feature_basis(existing_projection):
        return None

    descendant_feature_matrix = build_subtree_data(
        tree,
        full_feature_matrix,
        descendant_leaf_indices_by_node,
        descendant_internal_nodes_by_node,
        node_id,
        feature_count,
        include_internal,
    )
    if descendant_feature_matrix is None:
        return None

    column_variances = np.var(descendant_feature_matrix, axis=0)
    active_indices = np.flatnonzero(column_variances > 0)
    if active_indices.size != 1:
        return None

    feature_index = int(active_indices[0])

    parent_node, parent_lambda_max = _extract_parent_lambda_max(
        tree,
        node_id,
        node_pca_eigenvalues,
    )
    variance_ratio = (
        float(column_variances[feature_index] / parent_lambda_max)
        if parent_lambda_max is not None and parent_lambda_max > 0
        else None
    )
    internal_rows = max(
        int(descendant_feature_matrix.shape[0] - len(descendant_leaf_indices_by_node[node_id])),
        0,
    )

    return {
        "node_id": node_id,
        "active_feature": feature_index,
        "active_variance": float(column_variances[feature_index]),
        "parent_node": parent_node,
        "parent_lambda_max": parent_lambda_max,
        "variance_ratio": variance_ratio,
        "n_leaves": int(len(descendant_leaf_indices_by_node[node_id])),
        "n_rows": int(descendant_feature_matrix.shape[0]),
        "internal_rows": internal_rows,
        "projection": _build_single_feature_basis(feature_index, feature_count),
    }


def _collect_single_feature_candidates(
    tree: nx.DiGraph,
    full_feature_matrix: np.ndarray,
    descendant_leaf_indices_by_node: dict[str, list],
    descendant_internal_nodes_by_node: dict[str, list],
    feature_count: int,
    include_internal: bool,
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Collect all eligible single-feature subtree candidates."""
    candidates: list[dict[str, Any]] = []
    for node_id in tree.nodes:
        candidate = _build_single_feature_candidate(
            tree,
            node_id,
            full_feature_matrix=full_feature_matrix,
            descendant_leaf_indices_by_node=descendant_leaf_indices_by_node,
            descendant_internal_nodes_by_node=descendant_internal_nodes_by_node,
            feature_count=feature_count,
            include_internal=include_internal,
            node_pca_projections=node_pca_projections,
            node_pca_eigenvalues=node_pca_eigenvalues,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _prepare_single_feature_subtree_context(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Build the candidate list for the single-feature subtree policy."""
    feature_count = int(leaf_data.shape[1])
    leaf_label_to_index = {label: i for i, label in enumerate(leaf_data.index)}
    full_feature_matrix = leaf_data.values.astype(np.float64)
    include_internal = bool(getattr(config, "INCLUDE_INTERNAL_IN_SPECTRAL", True))
    descendant_leaf_indices_by_node, descendant_internal_nodes_by_node = precompute_descendants(
        tree,
        leaf_label_to_index,
    )
    return _collect_single_feature_candidates(
        tree,
        full_feature_matrix,
        descendant_leaf_indices_by_node,
        descendant_internal_nodes_by_node,
        feature_count,
        include_internal,
        node_pca_projections,
        node_pca_eigenvalues,
    )
