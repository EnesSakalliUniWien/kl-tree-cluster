"""Spectral setup for child-parent divergence tests."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from kl_clustering_analysis import config

from ..projection.spectral import compute_spectral_decomposition
from ..projection.spectral.tree_helpers import build_subtree_data, is_leaf, precompute_descendants


def _canonical_one_feature_basis(projection_matrix: np.ndarray | None) -> bool:
    if projection_matrix is None:
        return False
    basis = np.asarray(projection_matrix, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != 1:
        return False
    nonzero = np.flatnonzero(np.abs(basis[0]) > 1e-12)
    if nonzero.size != 1:
        return False
    return bool(np.isclose(np.linalg.norm(basis[0]), 1.0))


def _fit_low_ratio_split(variance_ratios: list[float]) -> tuple[bool, float | None]:
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

    means = two_component.means_.ravel()
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


def _build_one_active_guard_audit(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    node_spectral_dimensions: dict[str, int],
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> dict[str, object]:
    feature_count = int(leaf_data.shape[1])
    leaf_label_to_index = {label: i for i, label in enumerate(leaf_data.index)}
    full_feature_matrix = leaf_data.values.astype(np.float64)
    include_internal = bool(getattr(config, "INCLUDE_INTERNAL_IN_SPECTRAL", True))
    descendant_leaf_indices_by_node, descendant_internal_nodes_by_node = precompute_descendants(
        tree,
        leaf_label_to_index,
    )

    one_active_candidates: list[dict[str, object]] = []
    for node_id in tree.nodes:
        if is_leaf(tree, node_id):
            continue
        existing_projection = node_pca_projections.get(node_id)
        if existing_projection is not None and not _canonical_one_feature_basis(existing_projection):
            continue

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
            continue

        column_variances = np.var(descendant_feature_matrix, axis=0)
        active_indices = np.flatnonzero(column_variances > 0)
        if active_indices.size != 1:
            continue

        feature_index = int(active_indices[0])
        parent_node = next(iter(tree.predecessors(node_id)), None)
        parent_lambda_max = None
        if parent_node is not None:
            parent_eigs = node_pca_eigenvalues.get(parent_node)
            if parent_eigs is not None:
                parent_eigs = np.asarray(parent_eigs, dtype=np.float64)
                if parent_eigs.size > 0:
                    parent_lambda_max = float(np.max(parent_eigs))
        variance_ratio = (
            float(column_variances[feature_index] / parent_lambda_max)
            if parent_lambda_max is not None and parent_lambda_max > 0
            else None
        )
        internal_rows = max(
            int(descendant_feature_matrix.shape[0] - len(descendant_leaf_indices_by_node[node_id])),
            0,
        )
        one_hot_basis = np.zeros((1, feature_count), dtype=np.float64)
        one_hot_basis[0, feature_index] = 1.0
        one_active_candidates.append(
            {
                "node_id": node_id,
                "active_feature": feature_index,
                "active_variance": float(column_variances[feature_index]),
                "parent_node": parent_node,
                "parent_lambda_max": parent_lambda_max,
                "variance_ratio": variance_ratio,
                "n_leaves": int(len(descendant_leaf_indices_by_node[node_id])),
                "n_rows": int(descendant_feature_matrix.shape[0]),
                "internal_rows": internal_rows,
                "projection": one_hot_basis,
            }
        )

    has_low_group, low_ratio_threshold = _fit_low_ratio_split(
        [candidate["variance_ratio"] for candidate in one_active_candidates if candidate["variance_ratio"] is not None]
    )

    low_node_ids: set[str] = set()
    if has_low_group and low_ratio_threshold is not None:
        low_node_ids = {
            str(candidate["node_id"])
            for candidate in one_active_candidates
            if candidate["variance_ratio"] is not None
            and float(candidate["variance_ratio"]) <= float(low_ratio_threshold)
        }

    low_candidates = [
        candidate for candidate in one_active_candidates if str(candidate["node_id"]) in low_node_ids
    ]
    high_candidates = [
        candidate for candidate in one_active_candidates if str(candidate["node_id"]) not in low_node_ids
    ]

    low_count = int(len(low_candidates))
    high_count = int(len(high_candidates))
    low_rows = int(sum(int(candidate["n_rows"]) for candidate in low_candidates))
    high_rows = int(sum(int(candidate["n_rows"]) for candidate in high_candidates))
    low_internal_rows = int(sum(int(candidate["internal_rows"]) for candidate in low_candidates))

    dangerous_tree = bool(
        has_low_group
        and low_count > high_count
        and low_rows > high_rows
        and low_internal_rows > 0
    )

    blocked_node_ids = sorted(low_node_ids) if dangerous_tree else []
    allowed_node_ids = sorted(
        str(candidate["node_id"])
        for candidate in one_active_candidates
        if str(candidate["node_id"]) not in set(blocked_node_ids)
    )

    return {
        "mode": str(getattr(config, "ONE_ACTIVE_1D_MODE", "off")),
        "candidate_nodes": int(len(one_active_candidates)),
        "has_low_group": bool(has_low_group),
        "low_ratio_threshold": float(low_ratio_threshold) if low_ratio_threshold is not None else None,
        "low_count": low_count,
        "high_count": high_count,
        "low_rows": low_rows,
        "high_rows": high_rows,
        "low_internal_rows": low_internal_rows,
        "dangerous_tree": bool(dangerous_tree),
        "allowed_node_ids": allowed_node_ids,
        "blocked_node_ids": blocked_node_ids,
        "nodes": [
            {
                "node_id": str(candidate["node_id"]),
                "active_feature": int(candidate["active_feature"]),
                "active_variance": float(candidate["active_variance"]),
                "parent_node": candidate["parent_node"],
                "parent_lambda_max": (
                    float(candidate["parent_lambda_max"])
                    if candidate["parent_lambda_max"] is not None
                    else None
                ),
                "variance_ratio": (
                    float(candidate["variance_ratio"])
                    if candidate["variance_ratio"] is not None
                    else None
                ),
                "n_leaves": int(candidate["n_leaves"]),
                "n_rows": int(candidate["n_rows"]),
                "internal_rows": int(candidate["internal_rows"]),
                "is_low_leverage": str(candidate["node_id"]) in low_node_ids,
                "allowed_one_active_1d": str(candidate["node_id"]) not in set(blocked_node_ids),
            }
            for candidate in one_active_candidates
        ],
        "allowed_nodes": [
            candidate
            for candidate in one_active_candidates
            if str(candidate["node_id"]) in set(allowed_node_ids)
        ],
    }


def _apply_one_active_per_tree_load_guard(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    node_spectral_dimensions: dict[str, int],
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> tuple[dict[str, int], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    audit = _build_one_active_guard_audit(
        tree,
        leaf_data,
        node_spectral_dimensions,
        node_pca_projections,
        node_pca_eigenvalues,
    )

    for candidate in audit.pop("allowed_nodes"):
        node_id = str(candidate["node_id"])
        node_spectral_dimensions[node_id] = 1
        node_pca_projections[node_id] = np.asarray(candidate["projection"], dtype=np.float64)
        node_pca_eigenvalues[node_id] = np.array([1.0], dtype=np.float64)

    return node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues, audit


def compute_child_parent_spectral_context(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame | None,
    spectral_method: str | None,
) -> tuple[dict[str, int] | None, dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Prepare per-node spectral dimensions and PCA projections for Gate 2."""
    tree.graph.pop("_one_active_guard_audit", None)

    if spectral_method is None:
        return None, None, None

    if leaf_data is None:
        raise ValueError(f"spectral_method={spectral_method!r} requires leaf_data to be provided.")

    spectral_minimum_projection_dimension = getattr(
        config,
        "SPECTRAL_MINIMUM_DIMENSION",
        2,
    )

    (
        node_spectral_dimensions,
        computed_node_pca_projections,
        computed_node_pca_eigenvalues,
    ) = compute_spectral_decomposition(
        tree,
        leaf_data,
        method=spectral_method,
        minimum_projection_dimension=spectral_minimum_projection_dimension,
        compute_projections=True,
    )

    node_pca_projections = dict(computed_node_pca_projections or {})
    node_pca_eigenvalues = dict(computed_node_pca_eigenvalues or {})

    one_active_mode = str(getattr(config, "ONE_ACTIVE_1D_MODE", "off"))
    if one_active_mode == "per_tree_load_guard":
        (
            node_spectral_dimensions,
            node_pca_projections,
            node_pca_eigenvalues,
            one_active_guard_audit,
        ) = _apply_one_active_per_tree_load_guard(
            tree,
            leaf_data,
            node_spectral_dimensions,
            node_pca_projections,
            node_pca_eigenvalues,
        )
        tree.graph["_one_active_guard_audit"] = one_active_guard_audit

    node_pca_projections = node_pca_projections if node_pca_projections else None
    node_pca_eigenvalues = node_pca_eigenvalues if node_pca_eigenvalues else None

    return node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues


__all__ = ["compute_child_parent_spectral_context"]
