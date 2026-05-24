"""Parent principal-component inputs reused by Gate 3 sibling tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    SpectralContext,
)

if TYPE_CHECKING:
    import numpy as np


def collect_parent_principal_component_inputs_for_sibling_tests(
    sibling_projection_dimensions_from_child_edge_comparisons: dict[str, int] | None,
    *,
    spectral_context: SpectralContext,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Collect parent principal-component inputs for sibling tests.

    Returns the subset of Gate 2 principal-component projections and eigenvalues
    corresponding to parents that have valid sibling projection dimensions.
    """
    if sibling_projection_dimensions_from_child_edge_comparisons is None:
        raise ValueError("Gate 3 requires sibling projection dimensions from Gate 2.")

    principal_component_projections_by_node = (
        spectral_context.principal_component_projections_by_node
    )
    principal_component_eigenvalues_by_node = (
        spectral_context.principal_component_eigenvalues_by_node
    )

    parent_principal_component_projections: dict[str, np.ndarray] = {}
    parent_principal_component_eigenvalues: dict[str, np.ndarray] = {}

    for parent in sibling_projection_dimensions_from_child_edge_comparisons:
        parent_principal_component_projections[parent] = principal_component_projections_by_node[
            parent
        ]
        parent_principal_component_eigenvalues[parent] = principal_component_eigenvalues_by_node[
            parent
        ]

    return (
        parent_principal_component_projections,
        parent_principal_component_eigenvalues,
    )


__all__ = ["collect_parent_principal_component_inputs_for_sibling_tests"]
