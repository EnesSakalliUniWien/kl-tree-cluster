"""Sibling test configuration helpers (Gate 3).

Provides utilities for deriving projection dimensions and PCA directions
from Gate 2 (edge test) output for use in Gate 3 (sibling divergence) tests.

These functions bridge the Gate 2 → Gate 3 data flow:
- Extract spectral dimensions (k) from child-parent edge tests
- Extract PCA projections/eigenvalues from parent node decompositions
- Package configuration for sibling Wald tests
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def derive_sibling_spectral_dims(
    tree,
    annotated_df: pd.DataFrame,
) -> dict[str, int] | None:
    """Derive Gate 3 projection dimensions from Gate 2 spectral output.

    Uses the **min-child** strategy: for each binary parent P with children
    L, R, the sibling projection dimension is ``min(k_L, k_R)`` (from the
    per-child Marchenko-Pastur eigendecomposition in the edge test).

    The *projection directions and eigenvalues* come from the parent's PCA
    (automatically derived by :func:`derive_sibling_pca_projections`), which
    captures between-group variance.  The dimension k is kept conservative
    via min-child to avoid diluting the test with within-group-only PCs.

    Children with spectral k = 0 are excluded from the minimum.
    Parents where no child has a positive spectral k are omitted —
    the sibling test is skipped (merge) for those.

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
        no valid dimensions found.
    """
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        return None

    sibling_dims: dict[str, int] = {}

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        child_ks = [
            k
            for k in (
                edge_spectral_dims.get(left, 0),
                edge_spectral_dims.get(right, 0),
            )
            if k > 0
        ]
        if not child_ks:
            continue
        sibling_dims[parent] = min(child_ks)

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
    missing_pca = set(sibling_dims.keys()) - set(pca_projections.keys())
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


def derive_sibling_child_pca_projections(
    tree,
    annotated_df: pd.DataFrame,
    sibling_dims: dict[str, int] | None,
) -> dict[str, list[np.ndarray]] | None:
    """Extract per-child PCA projections for orthogonal-complement padding.

    For each binary parent P with children L, R, collects the child PCA
    projection matrices (from Gate 2's ``_pca_projections``) into a list
    ``[V_L, V_R]`` keyed by parent node ID.

    These are used by :func:`build_projection_basis_with_padding` to construct
    orthogonal-complement padding rows when the parent PCA has fewer than
    ``spectral_k`` components.

    Returns
    -------
    dict[str, list[np.ndarray]] | None
        Mapping from parent to ``[child_L_pca, child_R_pca]``, or None.
    """
    if sibling_dims is None:
        return None

    pca_projections = annotated_df.attrs.get("_pca_projections")
    if not pca_projections:
        return None

    child_pca_map: dict[str, list[np.ndarray]] = {}

    for parent in sibling_dims:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        child_projs = [
            pca_projections[c] for c in children if c in pca_projections
        ]
        if child_projs:
            child_pca_map[parent] = child_projs

    return child_pca_map if child_pca_map else None


__all__ = [
    "derive_sibling_spectral_dims",
    "derive_sibling_pca_projections",
    "derive_sibling_child_pca_projections",
]
