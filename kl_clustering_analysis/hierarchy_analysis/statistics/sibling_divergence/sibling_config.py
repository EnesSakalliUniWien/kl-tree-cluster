"""Gate 2 -> Gate 3 configuration helpers.

These helpers intentionally keep a permissive contract: when Gate 2 did not
materialize spectral metadata, they return ``None`` instead of raising. That
``None`` path is tolerated for non-standard configurations such as
``tree.decompose(..., leaf_data=None)`` or explicit experimental overrides.

In standard runs with leaf data, missing sibling dims are expected only for
binary parents whose two children are both leaves. Gate 2 assigns leaves
spectral dimension ``0``, so Gate 3 has no positive child-derived subspace to
combine for those parents and the sibling test falls back downstream.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


def derive_sibling_spectral_dims(
    tree,
    annotated_df: pd.DataFrame,
) -> dict[str, int] | None:
    """Derive Gate 3 projection dimensions from Gate 2 spectral output.

    Uses **geometric-mean-of-children** strategy: for each binary parent P
    with children L, R, the sibling projection dimension is
    ``round(sqrt(k_L * k_R))``.  When only one child has a positive
    spectral dimension, that value is used directly. When both children have
    non-positive spectral dims, the parent is omitted from the returned
    mapping. In the standard pipeline this means leaf-leaf parents are omitted,
    because leaves carry ``k=0``.

    Parameters
    ----------
    tree :
        Hierarchical tree structure.
    annotated_df : pd.DataFrame
        Gate 2 output with `_spectral_dims` attribute.

    Returns
    -------
    dict[str, int] | None
        Mapping from parent node to projection dimension k, or None if
        no valid dimensions found. ``None`` means Gate 3 has no parent-specific
        spectral override and should use its fallback policy downstream.
    """
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        logger.debug("Gate 3: no _spectral_dims found on Gate 2 annotations")
        return None

    sibling_dims: dict[str, int] = {}

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        left, right = children
        k_left = edge_spectral_dims.get(left, 0)
        k_right = edge_spectral_dims.get(right, 0)

        if k_left > 0 and k_right > 0:
            sibling_dims[parent] = max(1, round(math.sqrt(k_left * k_right)))
        elif k_left > 0:
            sibling_dims[parent] = k_left
        elif k_right > 0:
            sibling_dims[parent] = k_right

    return sibling_dims if sibling_dims else None


def derive_sibling_pca_projections(
    annotated_df: pd.DataFrame,
    sibling_dims: dict[str, int] | None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Extract parent PCA projections and eigenvalues for Gate 3 sibling tests.

    Returns the subset of Gate 2 PCA projections/eigenvalues that correspond
    to parents with valid sibling spectral dims. If no projections or dims
    are available, returns ``(None, None)``.

    Parameters
    ----------
    annotated_df : pd.DataFrame
        Gate 2 output with `_pca_projections` and `_pca_eigenvalues` attributes.
    sibling_dims : dict[str, int] | None
        Sibling spectral dimensions from :func:`derive_sibling_spectral_dims`.

    Returns
    -------
    tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]
        ``(pca_projections, pca_eigenvalues)`` mappings for valid parents.
    """
    if sibling_dims is None:
        return None, None

    pca_projections = annotated_df.attrs.get("_pca_projections")
    if not pca_projections:
        logger.debug("Gate 3: no _pca_projections found on Gate 2 annotations")
        return None, None

    pca_eigenvalues = annotated_df.attrs.get("_pca_eigenvalues")

    sibling_projections: dict[str, np.ndarray] = {}
    sibling_eigenvalues: dict[str, np.ndarray] = {}

    for parent in sibling_dims:
        proj = pca_projections.get(parent)
        if proj is not None:
            sibling_projections[parent] = proj
        eig = pca_eigenvalues.get(parent) if pca_eigenvalues else None
        if eig is not None:
            sibling_eigenvalues[parent] = eig

    # Log mismatch between sibling_dims and available PCA projections
    missing_pca = sibling_dims.keys() - pca_projections.keys()
    if missing_pca:
        logger.debug(
            "Gate 3: %d parents have sibling_dims but no PCA projections: %s",
            len(missing_pca),
            sorted(missing_pca)[:10],  # Show first 10
        )

    return (
        sibling_projections if sibling_projections else None,
        sibling_eigenvalues if sibling_eigenvalues else None,
    )


__all__ = [
    "derive_sibling_spectral_dims",
    "derive_sibling_pca_projections",
]
