"""Spectral context preparation for Gate 2 child-parent annotation."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ...projection.spectral.tree_estimator import compute_spectral_decomposition

GATE2_SPECTRAL_MINIMUM_PROJECTION_DIMENSION = 2


def _validate_spectral_context_outputs(
    spectral_projection_dimensions_by_node: dict[str, int],
    principal_component_projections_by_node: dict[str, np.ndarray],
    principal_component_eigenvalues_by_node: dict[str, np.ndarray],
) -> None:
    """Validate the paired Gate 2 PCA outputs before they enter Gate 3."""
    projection_nodes = set(principal_component_projections_by_node)
    eigenvalue_nodes = set(principal_component_eigenvalues_by_node)
    if projection_nodes != eigenvalue_nodes:
        missing_eigenvalues = sorted(projection_nodes - eigenvalue_nodes)
        missing_projections = sorted(eigenvalue_nodes - projection_nodes)
        raise ValueError(
            "Gate 2 spectral context requires matching PCA projection/eigenvalue node keys. "
            f"Missing eigenvalues for {missing_eigenvalues}; "
            f"missing projections for {missing_projections}."
        )

    dimension_nodes = set(spectral_projection_dimensions_by_node)
    missing_dimensions = sorted(projection_nodes - dimension_nodes)
    if missing_dimensions:
        raise ValueError(
            "Gate 2 spectral context has PCA outputs without spectral dimensions for "
            f"node(s): {missing_dimensions}."
        )
    missing_projection_outputs = sorted(
        node_id
        for node_id, projection_dimension in spectral_projection_dimensions_by_node.items()
        if int(projection_dimension) > 0 and node_id not in projection_nodes
    )
    if missing_projection_outputs:
        raise ValueError(
            "Gate 2 spectral context has positive spectral dimensions without PCA outputs "
            f"for node(s): {missing_projection_outputs}."
        )

    for node_id in sorted(projection_nodes):
        projection = np.asarray(principal_component_projections_by_node[node_id])
        eigenvalues = np.asarray(principal_component_eigenvalues_by_node[node_id])
        if projection.ndim != 2:
            raise ValueError(
                f"Gate 2 PCA projection for node {node_id!r} must be a 2-D matrix; "
                f"got shape {projection.shape}."
            )
        if eigenvalues.ndim != 1:
            raise ValueError(
                f"Gate 2 PCA eigenvalues for node {node_id!r} must be a 1-D vector; "
                f"got shape {eigenvalues.shape}."
            )
        if projection.shape[0] != eigenvalues.shape[0]:
            raise ValueError(
                f"Gate 2 PCA projection/eigenvalue row count mismatch for node {node_id!r}: "
                f"{projection.shape[0]} projection row(s), {eigenvalues.shape[0]} eigenvalue(s)."
            )
        if projection.shape[0] != int(spectral_projection_dimensions_by_node[node_id]):
            raise ValueError(
                f"Gate 2 PCA projection row count for node {node_id!r} must match its "
                "spectral projection dimension."
            )


def compute_child_parent_spectral_context(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Prepare Marchenko-Pastur spectral context for Gate 2."""
    (
        node_spectral_dimensions,
        computed_node_pca_projections,
        computed_node_pca_eigenvalues,
    ) = compute_spectral_decomposition(
        tree,
        leaf_data,
        minimum_projection_dimension=GATE2_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    )

    node_pca_projections = dict(computed_node_pca_projections)
    node_pca_eigenvalues = dict(computed_node_pca_eigenvalues)

    _validate_spectral_context_outputs(
        node_spectral_dimensions,
        node_pca_projections,
        node_pca_eigenvalues,
    )

    return (
        node_spectral_dimensions,
        node_pca_projections,
        node_pca_eigenvalues,
    )


__all__ = [
    "GATE2_SPECTRAL_MINIMUM_PROJECTION_DIMENSION",
    "compute_child_parent_spectral_context",
]
