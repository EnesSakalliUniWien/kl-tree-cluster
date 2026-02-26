"""Per-node spectral dimension estimation.

Computes the projection dimension for each internal node by eigendecomposing
the local **correlation** matrix of descendant leaf data. This replaces the
Johnson-Lindenstrauss-based dimension selection which is misapplied to the
single-vector projected Wald test.

Using the **correlation matrix** (not the covariance) is essential because
the projected Wald z-vector is standardised per-feature:
    z_i = (θ̂_child_i − θ̂_parent_i) / √Var_i
so Cov(z) under H₀ equals the Pearson correlation matrix C of the data.
Eigendecomposing C and whitening by its eigenvalues gives an exact χ²(k)
null: T = Σ (vᵢᵀz)² / λᵢ ~ χ²(k).

Two estimators are provided:

1. **Effective rank** (Roy & Vetterli, 2007): continuous dimensionality
   from the Shannon entropy of the normalised eigenvalue spectrum.
   ``erank(C) = exp(−Σ pᵢ log pᵢ)``  where ``pᵢ = λᵢ / Σλⱼ``.

2. **Marchenko-Pastur signal count**: number of eigenvalues exceeding the
   MP upper bound ``σ² (1 + √(d/n))²``; σ² estimated from the bulk median.

The chosen dimension ``k_v`` is used:
  - to set the degrees of freedom of the χ²(k) null for the projected Wald test,
  - to build the PCA projection that concentrates signal while discarding
    noise-only directions,
  - together with the top-k eigenvalues for whitening (exact χ² calibration).

References
----------
Roy, O. & Vetterli, M. (2007). "The effective rank: A measure of effective
    dimensionality". EUSIPCO.
Marchenko, V. A. & Pastur, L. A. (1967). "Distribution of eigenvalues for
    some sets of random matrices". Mathematics of the USSR-Sbornik.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .eigen_decomposition import (
    build_pca_projection,
    eigendecompose_correlation,
    estimate_spectral_k,
)

# Re-export estimators so existing ``from spectral_dimension import …``
# continues to work without changes.
from .estimators import (  # noqa: F401
    count_active_features,
    effective_rank,
    marchenko_pastur_signal_count,
)
from .tree_helpers import build_subtree_data, is_leaf, precompute_descendants

logger = logging.getLogger(__name__)


# =====================================================================
# Parallelism control
# =====================================================================

# Default thread count for joblib.Parallel eigendecomposition.
# Set KL_TE_N_JOBS env var to override (e.g. "1" to disable parallelism).
_DEFAULT_MIN_NODES_FOR_PARALLEL = 8


def _get_n_jobs(n_tasks: int) -> int:
    """Resolve the number of parallel workers.

    Returns 1 (sequential) when the number of tasks is small or the user
    explicitly sets ``KL_TE_N_JOBS=1``.
    """
    env = os.environ.get("KL_TE_N_JOBS")
    if env is not None:
        try:
            return max(int(env), 1)
        except ValueError:
            pass
    if n_tasks < _DEFAULT_MIN_NODES_FOR_PARALLEL:
        return 1
    return -1  # joblib: use all available cores


# =====================================================================
# Per-node workers (module-level for joblib pickling / clarity)
# =====================================================================


def _process_node(node_id, data_sub, method, min_k, d, need_eigh):
    """Eigendecompose one node and return (node_id, k, proj, ev)."""
    if data_sub is None:
        return (node_id, max(min_k, 1), None, None)

    if method == "active_features":
        k = count_active_features(data_sub)
        k = max(k, min_k)
        k = min(k, d)
        return (node_id, k, None, None)

    eig = eigendecompose_correlation(data_sub, need_eigh)
    if eig is None:
        return (node_id, max(min_k, 1), None, None)

    k = estimate_spectral_k(
        eig.eigenvalues,
        method,
        data_sub.shape[0],
        eig.d_active,
        min_k,
    )

    proj, ev = None, None
    if need_eigh:
        proj, ev = build_pca_projection(eig, k, d)
        if proj is None or ev is None:
            proj, ev = None, None

    return (node_id, k, proj, ev)


# =====================================================================
# Per-node computation
# =====================================================================


def compute_node_spectral_dimensions(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    *,
    method: str = "effective_rank",
    min_k: int = 1,
) -> Dict[str, int]:
    """Compute per-node projection dimension via eigendecomposition.

    Thin wrapper around ``compute_spectral_decomposition`` that returns
    only the dimension dict. Use ``compute_spectral_decomposition`` directly
    if you also need PCA projections (avoids a redundant eigendecomposition).
    """
    dims, _, _ = compute_spectral_decomposition(
        tree,
        leaf_data,
        method=method,
        min_k=min_k,
        compute_projections=False,
    )
    return dims


def compute_node_pca_projections(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    spectral_dims: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Compute PCA projection matrices for each internal node.

    Thin wrapper — prefer ``compute_spectral_decomposition`` with
    ``compute_projections=True`` to avoid a redundant eigendecomposition.
    """
    _, projs, _ = compute_spectral_decomposition(
        tree,
        leaf_data,
        method="effective_rank",
        min_k=1,
        compute_projections=True,
    )
    return projs


def compute_spectral_decomposition(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    *,
    method: str = "effective_rank",
    min_k: int = 1,
    compute_projections: bool = True,
    include_internal: bool | None = None,
) -> Tuple[Dict[str, int], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute spectral dimensions, PCA projections, and eigenvalues.

    Performs exactly one eigendecomposition per internal node of the local
    **correlation** matrix, extracting the dimension estimate, the top-k
    eigenvector projection matrix, and the corresponding eigenvalues for
    whitening.

    The correlation matrix is used (not the covariance) because the Wald
    z-vector is per-feature standardised, so its covariance under H₀ equals
    the Pearson correlation C = D^{-1/2} Σ D^{-1/2}.  Only features with
    non-zero variance ("active" features) are included; constant features
    receive zero weight in the projection.

    Parameters
    ----------
    tree
        Directed hierarchy with leaf labels accessible via
        ``tree.nodes[n].get("label", n)``.
    leaf_data
        DataFrame with leaf labels as index and features as columns.
    method
        Dimension estimator: ``"effective_rank"`` (default),
        ``"marchenko_pastur"``, or ``"active_features"``.
    min_k
        Floor on the returned dimension.
    compute_projections
        If True, also returns PCA projection matrices (k_v × d) and
        eigenvalue arrays for eigenvalue-based methods. For
        ``"active_features"`` these are always empty.
    include_internal
        If True, include internal node distribution vectors in the data
        matrix used for eigendecomposition.  If None, reads from
        ``config.INCLUDE_INTERNAL_IN_SPECTRAL`` (default False).
        Internal distributions are convex combinations of leaf data — they
        do NOT increase rank and typically reduce effective rank by ~30%.

    Returns
    -------
    (spectral_dims, pca_projections, pca_eigenvalues)
        spectral_dims : dict[str, int] — node_id → k_v
        pca_projections : dict[str, np.ndarray] — node_id → (k_v × d) matrix
        pca_eigenvalues : dict[str, np.ndarray] — node_id → (k_v,) eigenvalues
    """
    from kl_clustering_analysis import config as _config

    t0 = time.perf_counter()

    if include_internal is None:
        include_internal = _config.INCLUDE_INTERNAL_IN_SPECTRAL

    if method not in ("effective_rank", "marchenko_pastur", "active_features"):
        raise ValueError(
            f"Unknown spectral dimension method {method!r}. "
            f"Choose from 'effective_rank', 'marchenko_pastur', 'active_features'."
        )

    d = leaf_data.shape[1]
    label_to_idx = {label: i for i, label in enumerate(leaf_data.index)}
    X = leaf_data.values.astype(np.float64)

    desc_indices, desc_internal = precompute_descendants(tree, label_to_idx)

    need_eigh = compute_projections and method in ("effective_rank", "marchenko_pastur")

    dims: Dict[str, int] = {}
    projs: Dict[str, np.ndarray] = {}
    eig_vals: Dict[str, np.ndarray] = {}

    # Separate leaves (trivial) from internal nodes (expensive).
    internal_nodes = []
    for node_id in tree.nodes:
        if is_leaf(tree, node_id):
            dims[node_id] = 0
        else:
            internal_nodes.append(node_id)

    # Pre-build subtree data for internal nodes (fast, read-only slicing).
    node_data = {}
    for node_id in internal_nodes:
        data_sub = build_subtree_data(
            tree,
            X,
            desc_indices,
            desc_internal,
            node_id,
            d,
            include_internal,
        )
        node_data[node_id] = data_sub

    # --- Parallel eigendecomposition via joblib ---
    # eigh / eigvalsh release the GIL (LAPACK), so threads give real
    # parallelism without process-spawn overhead.
    n_jobs = _get_n_jobs(len(internal_nodes))
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_node)(nid, node_data[nid], method, min_k, d, need_eigh)
        for nid in internal_nodes
    )

    for node_id, k, proj, ev in results:
        dims[node_id] = k
        if proj is not None and ev is not None:
            projs[node_id] = proj
            eig_vals[node_id] = ev

    elapsed = time.perf_counter() - t0

    # Log summary statistics
    internal_ks = [v for nid, v in dims.items() if not is_leaf(tree, nid)]
    if internal_ks:
        logger.info(
            "Spectral dimensions (%s): median=%d, mean=%.1f, min=%d, max=%d "
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

    if compute_projections:
        logger.info(
            "Computed PCA projections for %d internal nodes [%.2fs total]",
            len(projs),
            elapsed,
        )

    return dims, projs, eig_vals


__all__ = [
    "effective_rank",
    "marchenko_pastur_signal_count",
    "count_active_features",
    "compute_node_spectral_dimensions",
    "compute_node_pca_projections",
    "compute_spectral_decomposition",
]
