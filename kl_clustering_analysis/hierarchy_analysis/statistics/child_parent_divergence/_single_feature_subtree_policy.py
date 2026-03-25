"""Policy helpers for handling single-feature subtrees in Gate 2 spectral setup."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from kl_clustering_analysis import config

from ..projection.spectral.tree_helpers import build_subtree_data, is_leaf, precompute_descendants


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


def _find_low_variance_ratio_threshold(
    variance_ratios: list[float],
) -> tuple[bool, float | None]:
    valid_ratios = [float(r) for r in variance_ratios if np.isfinite(r) and r > 0]
    if len(valid_ratios) < 2:
        return False, None

    log_ratios = np.log(np.asarray(valid_ratios, dtype=np.float64)).reshape(-1, 1)
    one_component = GaussianMixture(n_components=1, random_state=0)
    two_component = GaussianMixture(n_components=2, random_state=0)
    one_component.fit(log_ratios)
    two_component.fit(log_ratios)

    if two_component.bic(log_ratios) + 1e-9 >= one_component.bic(log_ratios):
        return False, None

    means = np.asarray(two_component.means_, dtype=np.float64).ravel()
    low_label = int(np.argmin(means))
    labels = two_component.predict(log_ratios)
    low_values = np.exp(log_ratios[labels == low_label].ravel())
    high_values = np.exp(log_ratios[labels != low_label].ravel())
    if low_values.size == 0 or high_values.size == 0:
        return False, None

    low_max = float(np.max(low_values))
    high_min = float(np.min(high_values))
    if low_max < high_min:
        threshold = float(np.sqrt(low_max * high_min))
    else:
        threshold = float(np.exp(np.mean(np.sort(means))))
    return True, threshold


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


def _classify_low_information_subtrees(
    single_feature_candidates: list[dict[str, Any]],
) -> tuple[bool, float | None, set[str]]:
    """Return the fitted threshold and node ids for low-information subtrees."""
    has_low_group, low_ratio_threshold = _find_low_variance_ratio_threshold(
        [
            candidate["variance_ratio"]
            for candidate in single_feature_candidates
            if candidate["variance_ratio"] is not None
        ]
    )
    if not has_low_group or low_ratio_threshold is None:
        return has_low_group, low_ratio_threshold, set()

    low_node_ids = {
        str(candidate["node_id"])
        for candidate in single_feature_candidates
        if candidate["variance_ratio"] is not None
        and float(candidate["variance_ratio"]) <= float(low_ratio_threshold)
    }
    return has_low_group, low_ratio_threshold, low_node_ids


def _serialize_single_feature_subtree_candidate(
    candidate: dict[str, Any],
    *,
    low_node_ids: set[str],
    blocked_node_ids: set[str],
) -> dict[str, Any]:
    """Convert internal candidate state into the persisted audit schema."""
    node_id = str(candidate["node_id"])
    return {
        "node_id": node_id,
        "active_feature": int(candidate["active_feature"]),
        "active_variance": float(candidate["active_variance"]),
        "parent_node": candidate["parent_node"],
        "parent_lambda_max": (
            float(candidate["parent_lambda_max"])
            if candidate["parent_lambda_max"] is not None
            else None
        ),
        "variance_ratio": (
            float(candidate["variance_ratio"]) if candidate["variance_ratio"] is not None else None
        ),
        "n_leaves": int(candidate["n_leaves"]),
        "n_rows": int(candidate["n_rows"]),
        "internal_rows": int(candidate["internal_rows"]),
        "is_low_leverage": node_id in low_node_ids,
        "allowed_one_active_1d": node_id not in blocked_node_ids,
    }


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


def _summarize_single_feature_candidate_groups(
    single_feature_candidates: list[dict[str, Any]],
    low_node_ids: set[str],
) -> dict[str, Any]:
    """Summarize low-information and retained candidate groups."""
    low_candidates = [
        candidate
        for candidate in single_feature_candidates
        if str(candidate["node_id"]) in low_node_ids
    ]
    high_candidates = [
        candidate
        for candidate in single_feature_candidates
        if str(candidate["node_id"]) not in low_node_ids
    ]

    low_count = int(len(low_candidates))
    high_count = int(len(high_candidates))
    low_rows = int(sum(int(candidate["n_rows"]) for candidate in low_candidates))
    high_rows = int(sum(int(candidate["n_rows"]) for candidate in high_candidates))
    low_internal_rows = int(sum(int(candidate["internal_rows"]) for candidate in low_candidates))
    dangerous_tree = bool(
        low_count > high_count and low_rows > high_rows and low_internal_rows > 0
    )

    blocked_node_id_set = set(low_node_ids) if dangerous_tree else set()
    allowed_node_ids = sorted(
        str(candidate["node_id"])
        for candidate in single_feature_candidates
        if str(candidate["node_id"]) not in blocked_node_id_set
    )
    return {
        "low_count": low_count,
        "high_count": high_count,
        "low_rows": low_rows,
        "high_rows": high_rows,
        "low_internal_rows": low_internal_rows,
        "dangerous_tree": dangerous_tree,
        "blocked_node_id_set": blocked_node_id_set,
        "blocked_node_ids": sorted(blocked_node_id_set),
        "allowed_node_ids": allowed_node_ids,
        "allowed_node_id_set": set(allowed_node_ids),
    }


def _build_single_feature_subtree_audit_payload(
    single_feature_candidates: list[dict[str, Any]],
    *,
    has_low_group: bool,
    low_ratio_threshold: float | None,
    low_node_ids: set[str],
    group_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the persisted audit payload for the policy."""
    blocked_node_id_set = group_summary["blocked_node_id_set"]
    return {
        "mode": str(getattr(config, "SINGLE_FEATURE_SUBTREE_MODE", "off")),
        "candidate_nodes": int(len(single_feature_candidates)),
        "has_low_group": bool(has_low_group),
        "low_ratio_threshold": (
            float(low_ratio_threshold) if low_ratio_threshold is not None else None
        ),
        "low_count": int(group_summary["low_count"]),
        "high_count": int(group_summary["high_count"]),
        "low_rows": int(group_summary["low_rows"]),
        "high_rows": int(group_summary["high_rows"]),
        "low_internal_rows": int(group_summary["low_internal_rows"]),
        "dangerous_tree": bool(group_summary["dangerous_tree"]),
        "allowed_node_ids": group_summary["allowed_node_ids"],
        "blocked_node_ids": group_summary["blocked_node_ids"],
        "nodes": [
            _serialize_single_feature_subtree_candidate(
                candidate,
                low_node_ids=low_node_ids,
                blocked_node_ids=blocked_node_id_set,
            )
            for candidate in single_feature_candidates
        ],
        "allowed_single_feature_subtrees": [
            candidate
            for candidate in single_feature_candidates
            if str(candidate["node_id"]) in group_summary["allowed_node_id_set"]
        ],
    }


def _build_single_feature_subtree_audit(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    node_spectral_dimensions: dict[str, int],
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> dict[str, Any]:
    _ = node_spectral_dimensions

    single_feature_candidates = _prepare_single_feature_subtree_context(
        tree,
        leaf_data,
        node_pca_projections,
        node_pca_eigenvalues,
    )
    has_low_group, low_ratio_threshold, low_node_ids = _classify_low_information_subtrees(
        single_feature_candidates
    )
    group_summary = _summarize_single_feature_candidate_groups(
        single_feature_candidates,
        low_node_ids if has_low_group else set(),
    )
    return _build_single_feature_subtree_audit_payload(
        single_feature_candidates,
        has_low_group=has_low_group,
        low_ratio_threshold=low_ratio_threshold,
        low_node_ids=low_node_ids,
        group_summary=group_summary,
    )


def _apply_single_feature_subtree_policy(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    node_spectral_dimensions: dict[str, int],
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> tuple[dict[str, int], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    audit = _build_single_feature_subtree_audit(
        tree,
        leaf_data,
        node_spectral_dimensions,
        node_pca_projections,
        node_pca_eigenvalues,
    )

    for candidate in audit.pop("allowed_single_feature_subtrees"):
        node_id = str(candidate["node_id"])
        node_spectral_dimensions[node_id] = 1
        node_pca_projections[node_id] = np.asarray(candidate["projection"], dtype=np.float64)
        node_pca_eigenvalues[node_id] = np.array([1.0], dtype=np.float64)

    return node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues, audit


__all__ = [
    "_apply_single_feature_subtree_policy",
    "_build_single_feature_subtree_audit",
    "_find_low_variance_ratio_threshold",
]
