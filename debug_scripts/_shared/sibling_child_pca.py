"""Debug helper: extract per-child PCA projections from Gate 2 output.

Moved from ``sibling_config.py`` during legacy-code cleanup (2026-03-27).
This function is not used by the production pipeline; it exists solely
for diagnostic and enhancement-lab scripts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


def derive_sibling_child_pca_projections(
    tree,
    annotated_df: pd.DataFrame,
    sibling_dims: dict[str, int] | None,
) -> dict[str, list[np.ndarray]] | None:
    """Extract per-child PCA projections for debug and experimental analysis.

    For each binary parent P with children L, R, collects the child PCA
    projection matrices (from Gate 2's ``_pca_projections``) into a list
    ``[V_L, V_R]`` keyed by parent node ID.

    Returns
    -------
    dict[str, list[np.ndarray]] | None
        Mapping from parent to ``[child_L_pca, child_R_pca]``, or None.
    """
    if sibling_dims is None:
        return None

    pca_projections = annotated_df.attrs.get("_pca_projections")
    if not pca_projections:
        logger.debug("No _pca_projections found on Gate 2 annotations")
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
