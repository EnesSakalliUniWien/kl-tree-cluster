"""Debug helper: extract per-child PCA projections from Gate 2 output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
        SpectralContext,
    )

logger = logging.getLogger(__name__)


def derive_sibling_child_pca_projections(
    tree,
    sibling_dims: dict[str, int] | None,
    *,
    spectral_context: SpectralContext,
) -> dict[str, list[np.ndarray]] | None:
    """Extract per-child PCA projections for debug and experimental analysis.

    For each binary parent P with children L, R, collects the child PCA
    projection matrices from the typed Gate 2 spectral context into
    ``[V_L, V_R]`` keyed by parent node ID.

    Returns
    -------
    dict[str, list[np.ndarray]] | None
        Mapping from parent to ``[child_L_pca, child_R_pca]``, or None.
    """
    if sibling_dims is None:
        return None

    pca_projections = spectral_context.principal_component_projections_by_node
    if not pca_projections:
        logger.debug("No principal-component projections found in Gate 2 context")
        return None

    child_pca_map: dict[str, list[np.ndarray]] = {}

    for parent in sibling_dims:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        child_projs = [pca_projections[c] for c in children if c in pca_projections]
        if child_projs:
            child_pca_map[parent] = child_projs

    return child_pca_map if child_pca_map else None
