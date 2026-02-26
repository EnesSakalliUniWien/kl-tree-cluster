"""Sibling-specific spectral dimension estimation (pooled within-cluster).

Computes the pooled within-cluster effective rank for each binary parent
node. This provides the correct degrees of freedom for the sibling χ² test,
where the z-vector measures the difference between sibling distributions
and the relevant covariance is the *within-cluster* correlation — not the
overall correlation, which is inflated by between-cluster structure.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .eigen_decomposition import eigendecompose_correlation, estimate_spectral_k
from .spectral_dimension import _get_n_jobs, compute_spectral_decomposition
from .tree_helpers import is_leaf, precompute_descendants

logger = logging.getLogger(__name__)


# =====================================================================
# Worker (module-level for joblib pickling / clarity)
# =====================================================================


def _process_sibling_node(
    node_id: str,
    left_idx: List[int],
    right_idx: List[int],
    X: np.ndarray,
    method: str,
    min_k: int,
) -> Tuple[str, int | None]:
    """Compute pooled within-cluster spectral dim for one sibling pair."""
    left_rows = X[left_idx, :]
    right_rows = X[right_idx, :]
    resid_left = left_rows - left_rows.mean(axis=0)
    resid_right = right_rows - right_rows.mean(axis=0)
    pooled_resid = np.vstack([resid_left, resid_right])

    eig = eigendecompose_correlation(pooled_resid, need_eigh=False)
    if eig is None:
        return (node_id, None)

    if method == "active_features":
        k = eig.d_active
    else:
        k = estimate_spectral_k(
            eig.eigenvalues,
            method,
            pooled_resid.shape[0],
            eig.d_active,
            min_k,
        )
    return (node_id, k)


# =====================================================================
# Public API
# =====================================================================


def compute_sibling_spectral_dimensions(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    *,
    method: str = "effective_rank",
    min_k: int = 1,
) -> Dict[str, int]:
    """Compute pooled within-cluster effective rank for each binary parent.

    For the sibling test at parent P with children L and R, the z-vector
    measures ``(θ_L - θ_R) / sqrt(Var)``.  Under H₀ (siblings same), the
    covariance of z is determined by the *within-cluster* correlation — not
    the overall correlation, which is inflated by between-cluster structure.

    This function computes:
      1. Get leaf data for L-descendants and R-descendants
      2. Center each child's data around its own mean
      3. Pool the residuals
      4. Compute effective rank of the pooled correlation matrix

    The result is typically 1.5–2× the parent's overall effective rank,
    providing the correct degrees of freedom for the sibling χ² test.

    For nodes where pooled within-cluster estimation is impossible (e.g.,
    non-binary nodes, child has < 2 leaves), falls back to the parent's
    overall spectral dimension from ``compute_spectral_decomposition()``.

    Parameters
    ----------
    tree
        Directed hierarchy.
    leaf_data
        DataFrame with leaf labels as index and features as columns.
    method
        Dimension estimator (``"effective_rank"`` recommended).
    min_k
        Floor on the returned dimension.

    Returns
    -------
    dict[str, int]
        node_id → k for each internal node.  Uses pooled within-cluster
        estimate where available, falls back to parent's overall spectral dim.
    """
    t0 = time.perf_counter()

    d = leaf_data.shape[1]
    label_to_idx = {label: i for i, label in enumerate(leaf_data.index)}
    X = leaf_data.values.astype(np.float64)

    # Precompute descendant leaf indices (reuse shared helper)
    desc_indices, _ = precompute_descendants(tree, label_to_idx)

    # Get parent's overall spectral dims as fallback
    parent_dims, _, _ = compute_spectral_decomposition(
        tree,
        leaf_data,
        method=method,
        min_k=min_k,
        compute_projections=False,
    )

    sibling_dims: Dict[str, int] = {}

    # Categorize nodes: leaves, non-binary / too-few-leaves (fallback), eligible
    eligible_nodes = []
    for node_id in tree.nodes:
        if is_leaf(tree, node_id):
            sibling_dims[node_id] = 0
            continue

        children = list(tree.successors(node_id))
        if len(children) != 2:
            sibling_dims[node_id] = parent_dims.get(node_id, max(min_k, 1))
            continue

        left, right = children[0], children[1]
        left_idx = desc_indices.get(left, [])
        right_idx = desc_indices.get(right, [])

        if len(left_idx) < 2 or len(right_idx) < 2:
            sibling_dims[node_id] = parent_dims.get(node_id, max(min_k, 1))
            continue

        eligible_nodes.append((node_id, left_idx, right_idx))

    n_jobs = _get_n_jobs(len(eligible_nodes))
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_sibling_node)(nid, li, ri, X, method, min_k)
        for nid, li, ri in eligible_nodes
    )

    for node_id, k in results:
        if k is None:
            sibling_dims[node_id] = parent_dims.get(node_id, max(min_k, 1))
        else:
            sibling_dims[node_id] = k

    elapsed = time.perf_counter() - t0
    internal_ks = [v for nid, v in sibling_dims.items() if not is_leaf(tree, nid)]
    if internal_ks:
        logger.info(
            "Sibling spectral dimensions (pooled within-cluster, %s): "
            "median=%d, mean=%.1f, min=%d, max=%d "
            "(across %d internal nodes, d=%d) [%.2fs]",
            method,
            int(np.median(internal_ks)),
            float(np.mean(internal_ks)),
            min(internal_ks),
            max(internal_ks),
            len(internal_ks),
            d,
            elapsed,
        )

    return sibling_dims


__all__ = [
    "compute_sibling_spectral_dimensions",
]
