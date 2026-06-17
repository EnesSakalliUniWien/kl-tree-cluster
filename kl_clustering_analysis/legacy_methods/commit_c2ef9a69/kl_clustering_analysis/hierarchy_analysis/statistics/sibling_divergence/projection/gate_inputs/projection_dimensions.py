"""Sibling projection-dimension inputs derived from child edge comparisons."""

from __future__ import annotations

import logging
import math

from kl_clustering_analysis.legacy_methods.commit_c2ef9a69.kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import (
    SpectralContext,
)

logger = logging.getLogger(__name__)


def derive_sibling_projection_dimensions_from_child_edge_comparisons(
    tree,
    *,
    spectral_context: SpectralContext,
) -> dict[str, int] | None:
    """Derive Gate 3 projection dimensions from child Gate 2 edge comparisons.

    Uses geometric mean of the two child edge dimensions for each binary
    sibling parent. When only one child has a positive edge-derived dimension,
    that dimension is reused directly. When neither child has a positive
    edge-derived dimension, the parent is omitted so Gate 3 can fall back
    downstream.
    """
    spectral_projection_dimensions_by_node = (
        spectral_context.spectral_projection_dimensions_by_node
    )
    if not spectral_projection_dimensions_by_node:
        logger.debug("Gate 3: no spectral projection dimensions found in Gate 2 context")
        return None

    sibling_projection_dimensions_from_child_edge_comparisons: dict[str, int] = {}

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        left, right = children
        left_projection_dimension = spectral_projection_dimensions_by_node.get(left, 0)
        right_projection_dimension = spectral_projection_dimensions_by_node.get(right, 0)

        if left_projection_dimension > 0 and right_projection_dimension > 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = max(
                1, round(math.sqrt(left_projection_dimension * right_projection_dimension))
            )
        elif left_projection_dimension > 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = (
                left_projection_dimension
            )
        elif right_projection_dimension > 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = (
                right_projection_dimension
            )

    return (
        sibling_projection_dimensions_from_child_edge_comparisons
        if sibling_projection_dimensions_from_child_edge_comparisons
        else None
    )


__all__ = ["derive_sibling_projection_dimensions_from_child_edge_comparisons"]
