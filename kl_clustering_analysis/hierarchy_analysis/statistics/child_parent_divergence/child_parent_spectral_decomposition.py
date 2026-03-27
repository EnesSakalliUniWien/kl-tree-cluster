"""Spectral setup for child-parent divergence tests."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis import config

from ._single_feature_subtree_policy import _apply_single_feature_subtree_policy
from ..projection.spectral import compute_spectral_decomposition


def _compute_child_parent_spectral_context_with_audit(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
) -> tuple[
    dict[str, int] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, object] | None,
]:
    """Prepare Marchenko-Pastur spectral context plus optional policy audit."""
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
        minimum_projection_dimension=spectral_minimum_projection_dimension,
        compute_projections=True,
    )

    node_pca_projections = dict(computed_node_pca_projections or {})
    node_pca_eigenvalues = dict(computed_node_pca_eigenvalues or {})
    single_feature_subtree_audit: dict[str, object] | None = None

    single_feature_subtree_mode = str(getattr(config, "SINGLE_FEATURE_SUBTREE_MODE", "off"))
    if single_feature_subtree_mode == "block_low_information_subtrees":
        (
            node_spectral_dimensions,
            node_pca_projections,
            node_pca_eigenvalues,
            single_feature_subtree_audit,
        ) = _apply_single_feature_subtree_policy(
            tree,
            leaf_data,
            node_spectral_dimensions,
            node_pca_projections,
            node_pca_eigenvalues,
        )

    node_pca_projections = node_pca_projections if node_pca_projections else None
    node_pca_eigenvalues = node_pca_eigenvalues if node_pca_eigenvalues else None

    return (
        node_spectral_dimensions,
        node_pca_projections,
        node_pca_eigenvalues,
        single_feature_subtree_audit,
    )


def compute_child_parent_spectral_context(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
) -> tuple[dict[str, int] | None, dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Prepare Marchenko-Pastur spectral context for Gate 2."""
    node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues, _ = (
        _compute_child_parent_spectral_context_with_audit(tree, leaf_data)
    )
    return node_spectral_dimensions, node_pca_projections, node_pca_eigenvalues


__all__ = ["compute_child_parent_spectral_context"]
