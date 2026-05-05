"""Parent principal-component inputs reused by Gate 3 sibling tests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import SpectralContext

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def collect_parent_principal_component_inputs_for_sibling_tests(
    annotated_df_or_projection_dimensions,
    sibling_projection_dimensions_from_child_edge_comparisons: dict[str, int] | None = None,
    *,
    spectral_context: SpectralContext | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Collect parent principal-component inputs for sibling tests.

    Returns the subset of Gate 2 principal-component projections and eigenvalues
    corresponding to parents that have valid sibling projection dimensions.
    """
    if spectral_context is None and hasattr(annotated_df_or_projection_dimensions, "attrs"):
        spectral_context = SpectralContext(
            principal_component_projections_by_node=(
                annotated_df_or_projection_dimensions.attrs.get("_pca_projections")
            ),
            principal_component_eigenvalues_by_node=(
                annotated_df_or_projection_dimensions.attrs.get("_pca_eigenvalues")
            ),
        )
    else:
        sibling_projection_dimensions_from_child_edge_comparisons = (
            annotated_df_or_projection_dimensions
        )

    if sibling_projection_dimensions_from_child_edge_comparisons is None:
        return None, None

    if spectral_context is None:
        logger.debug("Gate 3: no spectral context found for parent principal components")
        return None, None

    principal_component_projections_by_node = (
        spectral_context.principal_component_projections_by_node
    )
    principal_component_eigenvalues_by_node = (
        spectral_context.principal_component_eigenvalues_by_node
    )

    if not principal_component_projections_by_node:
        logger.debug("Gate 3: no principal-component projections found in Gate 2 context")
        return None, None

    parent_principal_component_projections: dict[str, np.ndarray] = {}
    parent_principal_component_eigenvalues: dict[str, np.ndarray] = {}

    for parent in sibling_projection_dimensions_from_child_edge_comparisons:
        projection = principal_component_projections_by_node.get(parent)
        if projection is not None:
            parent_principal_component_projections[parent] = projection

        eigenvalues = (
            principal_component_eigenvalues_by_node.get(parent)
            if principal_component_eigenvalues_by_node
            else None
        )
        if eigenvalues is not None:
            parent_principal_component_eigenvalues[parent] = eigenvalues

    missing_parent_principal_component_inputs = (
        sibling_projection_dimensions_from_child_edge_comparisons.keys()
        - principal_component_projections_by_node.keys()
    )
    if missing_parent_principal_component_inputs:
        logger.debug(
            "Gate 3: %d parents have child-edge-derived sibling projection dimensions but no parent principal-component projections: %s",
            len(missing_parent_principal_component_inputs),
            sorted(missing_parent_principal_component_inputs)[:10],
        )

    return (
        parent_principal_component_projections if parent_principal_component_projections else None,
        parent_principal_component_eigenvalues if parent_principal_component_eigenvalues else None,
    )


__all__ = ["collect_parent_principal_component_inputs_for_sibling_tests"]
