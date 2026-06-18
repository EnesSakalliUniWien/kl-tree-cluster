"""Spectral context preparation for edge-divergence annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis.tree.feature_space import FeatureSpace

from ...projection.spectral.tree_estimator import compute_spectral_decomposition
from ...projection.spectral.tree_estimator import (
    INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
)

EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION = 2


@dataclass
class SpectralContext:
    """Edge-gate spectral outputs reused by sibling-divergence tests."""

    test_projection_dimensions_by_node: dict[str, int]
    raw_mp_signal_counts_by_node: dict[str, int]
    effective_independent_rows_by_node: dict[str, int]
    mp_threshold_rows_by_node: dict[str, int]
    principal_component_projections_by_node: dict[str, np.ndarray]
    principal_component_eigenvalues_by_node: dict[str, np.ndarray]
    full_component_eigenvalues_by_node: dict[str, np.ndarray] = field(default_factory=dict)
    active_feature_counts_by_node: dict[str, int] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)


def _validate_spectral_context_outputs(
    test_projection_dimensions_by_node: dict[str, int],
    principal_component_projections_by_node: dict[str, np.ndarray],
    principal_component_eigenvalues_by_node: dict[str, np.ndarray],
) -> None:
    """Validate paired edge-gate PCA outputs before sibling testing."""
    projection_nodes = set(principal_component_projections_by_node)
    eigenvalue_nodes = set(principal_component_eigenvalues_by_node)
    if projection_nodes != eigenvalue_nodes:
        missing_eigenvalues = sorted(projection_nodes - eigenvalue_nodes)
        missing_projections = sorted(eigenvalue_nodes - projection_nodes)
        raise ValueError(
            "Edge-gate spectral context requires matching PCA projection/eigenvalue node keys. "
            f"Missing eigenvalues for {missing_eigenvalues}; "
            f"missing projections for {missing_projections}."
        )

    dimension_nodes = set(test_projection_dimensions_by_node)
    missing_dimensions = sorted(projection_nodes - dimension_nodes)
    if missing_dimensions:
        raise ValueError(
            "Edge-gate spectral context has PCA outputs without test projection dimensions for "
            f"node(s): {missing_dimensions}."
        )
    missing_projection_outputs = sorted(
        node_id
        for node_id, projection_dimension in test_projection_dimensions_by_node.items()
        if int(projection_dimension) > 0 and node_id not in projection_nodes
    )
    if missing_projection_outputs:
        raise ValueError(
            "Edge-gate spectral context has positive test projection dimensions without PCA outputs "
            f"for node(s): {missing_projection_outputs}."
        )

    for node_id in sorted(projection_nodes):
        projection = np.asarray(principal_component_projections_by_node[node_id])
        eigenvalues = np.asarray(principal_component_eigenvalues_by_node[node_id])
        if projection.ndim != 2:
            raise ValueError(
                f"Edge-gate PCA projection for node {node_id!r} must be a 2-D matrix; "
                f"got shape {projection.shape}."
            )
        if eigenvalues.ndim != 1:
            raise ValueError(
                f"Edge-gate PCA eigenvalues for node {node_id!r} must be a 1-D vector; "
                f"got shape {eigenvalues.shape}."
            )
        if projection.shape[0] != eigenvalues.shape[0]:
            raise ValueError(
                f"Edge-gate PCA projection/eigenvalue row count mismatch for node {node_id!r}: "
                f"{projection.shape[0]} projection row(s), {eigenvalues.shape[0]} eigenvalue(s)."
            )
        if projection.shape[0] != int(test_projection_dimensions_by_node[node_id]):
            raise ValueError(
                f"Edge-gate PCA projection row count for node {node_id!r} must match its "
                "test projection dimension."
            )


def compute_child_parent_spectral_context(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    *,
    minimum_projection_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    feature_space: FeatureSpace | None = None,
    include_internal_barycenters: bool = False,
    internal_distribution_mode: str = INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    mp_row_count_mode: str = MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
) -> SpectralContext:
    """Prepare Marchenko-Pastur spectral context for the edge gate."""
    start_sec = perf_counter()
    spectral_decomposition = compute_spectral_decomposition(
        tree,
        leaf_data,
        feature_space=feature_space,
        minimum_projection_dimension=int(minimum_projection_dimension),
        include_internal_barycenters=bool(include_internal_barycenters),
        internal_distribution_mode=str(internal_distribution_mode),
        mp_row_count_mode=str(mp_row_count_mode),
    )

    _validate_spectral_context_outputs(
        spectral_decomposition.test_projection_dimensions_by_node,
        spectral_decomposition.principal_component_projections_by_node,
        spectral_decomposition.principal_component_eigenvalues_by_node,
    )
    stage_timings = dict(spectral_decomposition.stage_timings)
    stage_timings["spectral_context_sec"] = float(perf_counter() - start_sec)

    return SpectralContext(
        test_projection_dimensions_by_node=(
            spectral_decomposition.test_projection_dimensions_by_node
        ),
        raw_mp_signal_counts_by_node=spectral_decomposition.raw_mp_signal_counts_by_node,
        effective_independent_rows_by_node=(
            spectral_decomposition.effective_independent_rows_by_node
        ),
        mp_threshold_rows_by_node=spectral_decomposition.mp_threshold_rows_by_node,
        principal_component_projections_by_node=(
            spectral_decomposition.principal_component_projections_by_node
        ),
        principal_component_eigenvalues_by_node=(
            spectral_decomposition.principal_component_eigenvalues_by_node
        ),
        full_component_eigenvalues_by_node=(
            spectral_decomposition.full_component_eigenvalues_by_node
        ),
        active_feature_counts_by_node=spectral_decomposition.active_feature_counts_by_node,
        stage_timings=stage_timings,
    )


__all__ = [
    "EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION",
    "MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS",
    "SpectralContext",
    "compute_child_parent_spectral_context",
]
