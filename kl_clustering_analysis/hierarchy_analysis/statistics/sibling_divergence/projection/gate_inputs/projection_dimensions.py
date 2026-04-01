"""Sibling projection-dimension inputs derived from child edge comparisons."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def derive_sibling_projection_dimensions_from_child_edge_comparisons(
    tree,
    annotated_df: pd.DataFrame,
) -> dict[str, int] | None:
    """Derive Gate 3 projection dimensions from child Gate 2 edge comparisons.

    Uses geometric mean of the two child edge dimensions for each binary
    sibling parent. When only one child has a positive edge-derived dimension,
    that dimension is reused directly. When neither child has a positive
    edge-derived dimension, the parent is omitted so Gate 3 can fall back
    downstream.
    """
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        logger.debug("Gate 3: no _spectral_dims found on Gate 2 annotations")
        return None

    sibling_projection_dimensions_from_child_edge_comparisons: dict[str, int] = {}

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        left, right = children
        k_left = edge_spectral_dims.get(left, 0)
        k_right = edge_spectral_dims.get(right, 0)

        if k_left > 0 and k_right > 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = max(
                1, round(math.sqrt(k_left * k_right))
            )
        elif k_left > 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = k_left
        elif k_right > 0:
            sibling_projection_dimensions_from_child_edge_comparisons[parent] = k_right

    return (
        sibling_projection_dimensions_from_child_edge_comparisons
        if sibling_projection_dimensions_from_child_edge_comparisons
        else None
    )


__all__ = ["derive_sibling_projection_dimensions_from_child_edge_comparisons"]
