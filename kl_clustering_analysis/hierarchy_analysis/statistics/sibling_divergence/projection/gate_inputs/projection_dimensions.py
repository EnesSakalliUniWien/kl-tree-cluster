"""Sibling projection-dimension inputs derived from child edge comparisons."""

from __future__ import annotations

import math

from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    SpectralContext,
)


def _require_child_test_projection_dimension(
    test_projection_dimensions_by_node: dict[str, int],
    child_node: str,
) -> int:
    """Return a child test projection dimension or fail on malformed edge-gate context."""
    if child_node not in test_projection_dimensions_by_node:
        raise ValueError(
            "Sibling-gate edge-derived projection dimensions require edge-gate "
            f"test projection dimensions for every child node; missing {child_node!r}."
        )
    return int(test_projection_dimensions_by_node[child_node])


def _cap_to_parent_projection_dimension(
    *,
    parent: str,
    parent_projection_dimension: int,
    candidate_projection_dimension: int,
) -> int:
    """Keep the sibling gate inside the parent PCA subspace used for testing."""
    if parent_projection_dimension < 0:
        raise ValueError(
            f"Sibling gate received a negative parent projection dimension for {parent!r}."
        )
    if candidate_projection_dimension > 0 and parent_projection_dimension == 0:
        raise ValueError(
            "Sibling gate cannot project a positive sibling dimension into a zero-dimensional "
            f"parent PCA basis for {parent!r}."
        )
    return min(int(candidate_projection_dimension), int(parent_projection_dimension))


def derive_sibling_projection_dimensions_from_child_edge_comparisons(
    tree,
    *,
    spectral_context: SpectralContext,
) -> dict[str, int]:
    """Derive sibling-gate projection dimensions from child edge comparisons.

    Uses geometric mean of the two child edge dimensions for each binary
    sibling parent. When only one child has a positive edge-derived dimension,
    that dimension is reused directly. When neither child has a positive
    edge-derived dimension, the parent's own edge-gate test projection dimension is used.
    """
    test_projection_dimensions_by_node = spectral_context.test_projection_dimensions_by_node
    if not test_projection_dimensions_by_node:
        raise ValueError("Sibling gate requires edge-gate test projection dimensions.")

    sibling_projection_dimensions_from_child_edge_comparisons: dict[str, int] = {}

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        left, right = children
        left_projection_dimension = _require_child_test_projection_dimension(
            test_projection_dimensions_by_node,
            left,
        )
        right_projection_dimension = _require_child_test_projection_dimension(
            test_projection_dimensions_by_node,
            right,
        )
        parent_projection_dimension = _require_child_test_projection_dimension(
            test_projection_dimensions_by_node,
            parent,
        )

        if left_projection_dimension > 0 and right_projection_dimension > 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = max(
                1,
                _cap_to_parent_projection_dimension(
                    parent=parent,
                    parent_projection_dimension=parent_projection_dimension,
                    candidate_projection_dimension=round(
                        math.sqrt(left_projection_dimension * right_projection_dimension)
                    ),
                ),
            )
        elif left_projection_dimension > 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = (
                _cap_to_parent_projection_dimension(
                    parent=parent,
                    parent_projection_dimension=parent_projection_dimension,
                    candidate_projection_dimension=left_projection_dimension,
                )
            )
        elif right_projection_dimension > 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = (
                _cap_to_parent_projection_dimension(
                    parent=parent,
                    parent_projection_dimension=parent_projection_dimension,
                    candidate_projection_dimension=right_projection_dimension,
                )
            )
        elif parent_projection_dimension >= 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = (
                parent_projection_dimension
            )
        else:
            raise ValueError(
                f"Sibling gate cannot resolve a non-negative projection dimension for parent {parent!r}."
            )

    return sibling_projection_dimensions_from_child_edge_comparisons


__all__ = ["derive_sibling_projection_dimensions_from_child_edge_comparisons"]
