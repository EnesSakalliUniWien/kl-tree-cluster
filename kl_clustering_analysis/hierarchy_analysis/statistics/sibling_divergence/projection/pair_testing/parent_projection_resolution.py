"""Resolution of parent projection inputs for one sibling test."""

from __future__ import annotations

import numpy as np


def resolve_parent_projection_inputs_for_sibling_test(
    parent_node_id: str,
    *,
    sibling_projection_dimensions_from_edge_comparisons: dict[str, int] | None,
    parent_principal_component_projections: dict[str, np.ndarray] | None,
    parent_principal_component_eigenvalues: dict[str, np.ndarray] | None,
) -> tuple[int | None, np.ndarray | None, np.ndarray | None]:
    """Resolve projection inputs for one sibling test."""
    projection_dimension_from_edge_comparisons = (
        sibling_projection_dimensions_from_edge_comparisons.get(parent_node_id)
        if sibling_projection_dimensions_from_edge_comparisons
        else None
    )
    parent_principal_component_projection = (
        parent_principal_component_projections.get(parent_node_id)
        if parent_principal_component_projections
        else None
    )
    parent_principal_component_eigenvalues_for_parent = (
        parent_principal_component_eigenvalues.get(parent_node_id)
        if parent_principal_component_eigenvalues
        else None
    )
    return (
        projection_dimension_from_edge_comparisons,
        parent_principal_component_projection,
        parent_principal_component_eigenvalues_for_parent,
    )


__all__ = ["resolve_parent_projection_inputs_for_sibling_test"]

