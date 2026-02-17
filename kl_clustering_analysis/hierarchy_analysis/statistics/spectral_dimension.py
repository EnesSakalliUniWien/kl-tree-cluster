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
import time
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Core estimators
# =====================================================================


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Continuous effective rank via Shannon entropy of the eigenvalue spectrum.

    Parameters
    ----------
    eigenvalues
        Non-negative eigenvalues, any order.

    Returns
    -------
    float
        exp(−Σ pᵢ log pᵢ). Returns 1.0 when the spectrum is degenerate.
    """
    eigs = np.maximum(eigenvalues, 0.0)
    total = eigs.sum()
    if total <= 0:
        return 1.0
    p = eigs / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def marchenko_pastur_signal_count(
    eigenvalues: np.ndarray,
    n: int,
    d: int,
) -> int:
    """Count eigenvalues above the Marchenko-Pastur upper bound.

    Parameters
    ----------
    eigenvalues
        Sample covariance eigenvalues (descending order preferred but not required).
    n
        Number of observations (descendant leaves).
    d
        Number of features.

    Returns
    -------
    int
        Number of signal eigenvalues exceeding σ² (1 + √(d/n))².
    """
    if n < 2 or d < 1:
        return 0
    gamma = d / n
    sigma2 = float(np.median(eigenvalues[eigenvalues > 0])) if np.any(eigenvalues > 0) else 0.0
    if sigma2 <= 0:
        return 0
    upper = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
    return int(np.sum(eigenvalues > upper))


def count_active_features(data_sub: np.ndarray) -> int:
    """Count features with non-zero variance (not all-0 or all-1).

    Parameters
    ----------
    data_sub
        Binary data matrix (n × d) for a subtree's descendants.

    Returns
    -------
    int
        Number of columns with var > 0.
    """
    if data_sub.shape[0] <= 1:
        return 0
    return int(np.sum(np.var(data_sub, axis=0) > 0))


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

    Returns
    -------
    (spectral_dims, pca_projections, pca_eigenvalues)
        spectral_dims : dict[str, int] — node_id → k_v
        pca_projections : dict[str, np.ndarray] — node_id → (k_v × d) matrix
        pca_eigenvalues : dict[str, np.ndarray] — node_id → (k_v,) eigenvalues
    """
    t0 = time.perf_counter()

    if method not in ("effective_rank", "marchenko_pastur", "active_features"):
        raise ValueError(
            f"Unknown spectral dimension method {method!r}. "
            f"Choose from 'effective_rank', 'marchenko_pastur', 'active_features'."
        )

    d = leaf_data.shape[1]
    label_to_idx = {label: i for i, label in enumerate(leaf_data.index)}
    X = leaf_data.values.astype(np.float64)

    # --- Precompute descendant leaf indices bottom-up (O(N) total) ---
    # Avoids O(N²) nx.descendants() calls.
    def _is_leaf(node_id: str) -> bool:
        is_leaf_attr = tree.nodes[node_id].get("is_leaf")
        if is_leaf_attr is not None:
            return bool(is_leaf_attr)
        return tree.out_degree(node_id) == 0

    desc_indices: Dict[str, list[int]] = {}
    # Process in reverse topological order (leaves first, root last)
    for node_id in reversed(list(nx.topological_sort(tree))):
        if _is_leaf(node_id):
            lbl = tree.nodes[node_id].get("label", node_id)
            desc_indices[node_id] = [label_to_idx[lbl]] if lbl in label_to_idx else []
        else:
            indices = []
            for child in tree.successors(node_id):
                indices.extend(desc_indices.get(child, []))
            desc_indices[node_id] = indices

    need_eigh = compute_projections and method in ("effective_rank", "marchenko_pastur")

    dims: Dict[str, int] = {}
    projs: Dict[str, np.ndarray] = {}
    eig_vals: Dict[str, np.ndarray] = {}

    for node_id in tree.nodes:
        if _is_leaf(node_id):
            dims[node_id] = 0
            continue

        row_indices = desc_indices[node_id]
        n_desc = len(row_indices)

        if n_desc < 2:
            dims[node_id] = max(min_k, 1)
            continue

        data_sub = X[row_indices, :]

        if method == "active_features":
            k = count_active_features(data_sub)
            k = max(k, min_k)
            k = min(k, d)
            dims[node_id] = k
            continue

        # --- Identify active features (non-constant columns) ---
        col_var = np.var(data_sub, axis=0)
        active_mask = col_var > 0
        d_active = int(np.sum(active_mask))

        if d_active < 2:
            dims[node_id] = max(min_k, 1)
            continue

        # --- Correlation matrix of active features ---
        # Using the correlation matrix (not covariance) is critical:
        # the Wald z-vector is per-feature standardised, so Cov(z) = Corr(X)
        # under H₀.  Eigendecomposing C and whitening by its eigenvalues
        # gives exact χ²(k) rather than an inflated weighted chi-square.
        data_active = data_sub[:, active_mask]
        corr = np.corrcoef(data_active.T)
        # Guard against numerical artefacts (NaN from near-constant cols)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        if need_eigh:
            eigenvalues, eigenvectors_active = np.linalg.eigh(corr)
            # eigh returns ascending order — reverse to descending
            eigenvalues = eigenvalues[::-1]
            eigenvectors_active = eigenvectors_active[:, ::-1]
        else:
            eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]
            eigenvectors_active = None

        # Clamp negative eigenvalues (numerical noise in correlation matrices)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        if method == "effective_rank":
            k = int(np.round(effective_rank(eigenvalues)))
        else:  # marchenko_pastur
            k = marchenko_pastur_signal_count(eigenvalues, n_desc, d_active)

        k = max(k, min_k)
        k = min(k, d_active)
        dims[node_id] = k

        # Build projection matrix and eigenvalue array from top-k eigenvectors.
        # Eigenvectors are in the active-feature subspace; embed into full
        # d-space by placing zeros at constant-feature positions.
        if need_eigh and eigenvectors_active is not None and k > 0:
            full_eigvecs = np.zeros((d, k), dtype=np.float64)
            full_eigvecs[active_mask, :] = eigenvectors_active[:, :k]
            projs[node_id] = full_eigvecs.T  # (k × d)
            # Store eigenvalues for whitening; clamp to small positive floor
            # to avoid division by zero in degenerate cases.
            eig_vals[node_id] = np.maximum(eigenvalues[:k], 1e-12).astype(np.float64)

    elapsed = time.perf_counter() - t0

    # Log summary statistics
    internal_ks = [v for nid, v in dims.items() if not _is_leaf(nid)]
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
