"""Single-feature subtree projection policy for Gate 2."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .single_feature_subtree_audit import _build_single_feature_subtree_audit_payload
from .single_feature_subtree_candidates import _prepare_single_feature_subtree_context
from .single_feature_subtree_low_information_threshold import (
    _classify_low_information_subtrees,
    _summarize_single_feature_candidate_groups,
)


def _analyze_single_feature_subtree_projection_policy(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return allowed one-active candidates plus the persisted audit payload."""
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
    allowed_node_id_set = set(group_summary["allowed_node_ids"])
    allowed_single_feature_subtrees = [
        candidate
        for candidate in single_feature_candidates
        if str(candidate["node_id"]) in allowed_node_id_set
    ]
    audit = _build_single_feature_subtree_audit_payload(
        single_feature_candidates,
        has_low_group=has_low_group,
        low_ratio_threshold=low_ratio_threshold,
        low_node_ids=low_node_ids,
        group_summary=group_summary,
    )
    return allowed_single_feature_subtrees, audit


def _build_single_feature_subtree_audit(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> dict[str, Any]:
    _, audit = _analyze_single_feature_subtree_projection_policy(
        tree,
        leaf_data,
        node_pca_projections,
        node_pca_eigenvalues,
    )
    return audit


def _apply_single_feature_subtree_projection_policy(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    node_spectral_dimensions: dict[str, int],
    node_pca_projections: dict[str, np.ndarray],
    node_pca_eigenvalues: dict[str, np.ndarray],
) -> tuple[dict[str, int], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    allowed_single_feature_subtrees, audit = _analyze_single_feature_subtree_projection_policy(
        tree,
        leaf_data,
        node_pca_projections,
        node_pca_eigenvalues,
    )

    for candidate in allowed_single_feature_subtrees:
        node_id = str(candidate["node_id"])
        node_spectral_dimensions[node_id] = 1
        node_pca_projections[node_id] = np.asarray(candidate["projection"], dtype=np.float64)
        node_pca_eigenvalues[node_id] = np.array([1.0], dtype=np.float64)

    return node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues, audit


__all__ = [
    "_apply_single_feature_subtree_projection_policy",
    "_build_single_feature_subtree_audit",
]
