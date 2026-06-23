"""Parent principal-component inputs reused by sibling-divergence tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    SpectralContext,
)

if TYPE_CHECKING:
    import numpy as np


def collect_parent_principal_component_inputs_for_sibling_tests(
    sibling_projection_dimensions_from_child_edge_comparisons: dict[str, int],
    *,
    spectral_context: SpectralContext,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Collect parent principal-component inputs for sibling tests.

    Returns the subset of edge-gate principal-component projections and eigenvalues
    corresponding to parents that have valid sibling projection dimensions.
    """
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
