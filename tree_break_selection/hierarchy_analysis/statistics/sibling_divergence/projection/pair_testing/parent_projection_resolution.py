"""Resolution of parent projection inputs for one sibling test."""

from __future__ import annotations

import numpy as np


def resolve_parent_projection_inputs_for_sibling_test(
    parent_node_id: object,
    *,
    sibling_projection_dimensions_from_edge_comparisons: dict[object, int],
    parent_principal_component_projections: dict[object, np.ndarray],
    parent_principal_component_eigenvalues: dict[object, np.ndarray],
) -> tuple[int, np.ndarray, np.ndarray]:
    """Resolve projection inputs for one sibling test."""
    return (
        sibling_projection_dimensions_from_edge_comparisons[parent_node_id],
        parent_principal_component_projections[parent_node_id],
        parent_principal_component_eigenvalues[parent_node_id],
    )


__all__ = ["resolve_parent_projection_inputs_for_sibling_test"]
