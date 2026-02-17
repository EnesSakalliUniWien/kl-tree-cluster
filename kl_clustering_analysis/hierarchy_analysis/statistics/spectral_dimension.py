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

    # --- Precompute descendant leaf indices bottom-up (O(N) total) ---
    # Avoids O(N²) nx.descendants() calls.
    def _is_leaf(node_id: str) -> bool:
        is_leaf_attr = tree.nodes[node_id].get("is_leaf")
        if is_leaf_attr is not None:
            return bool(is_leaf_attr)
        return tree.out_degree(node_id) == 0

    desc_indices: Dict[str, list[int]] = {}
    desc_internal: Dict[str, list[str]] = {}  # internal descendant node IDs
    # Process in reverse topological order (leaves first, root last)
    for node_id in reversed(list(nx.topological_sort(tree))):
        if _is_leaf(node_id):
            lbl = tree.nodes[node_id].get("label", node_id)
            desc_indices[node_id] = [label_to_idx[lbl]] if lbl in label_to_idx else []
            desc_internal[node_id] = []
        else:
            indices = []
            internals = []
            for child in tree.successors(node_id):
                indices.extend(desc_indices.get(child, []))
                if not _is_leaf(child):
                    internals.append(child)
                internals.extend(desc_internal.get(child, []))
            desc_indices[node_id] = indices
            desc_internal[node_id] = internals

    need_eigh = compute_projections and method in ("effective_rank", "marchenko_pastur")

    dims: Dict[str, int] = {}
    projs: Dict[str, np.ndarray] = {}
    eig_vals: Dict[str, np.ndarray] = {}

    for node_id in tree.nodes:
        if _is_leaf(node_id):
            dims[node_id] = 0
            continue

        row_indices = desc_indices[node_id]
        n_leaves = len(row_indices)

        if n_leaves < 2:
            dims[node_id] = max(min_k, 1)
            continue

        # Build data matrix from descendant leaves.
        # By default, internal node distribution vectors are EXCLUDED because
        # they are convex combinations of leaf data (do not increase rank)
        # and shift the mean toward the global average, concentrating
        # variance in the top PCs and reducing effective rank (~30%).
        leaf_rows = X[row_indices, :]
        if include_internal:
            internal_rows = []
            for inode in desc_internal[node_id]:
                dist = tree.nodes[inode].get("distribution")
                if dist is not None:
                    dist_arr = np.asarray(dist, dtype=np.float64)
                    if dist_arr.shape == (d,):
                        internal_rows.append(dist_arr)
            if internal_rows:
                data_sub = np.vstack([leaf_rows, np.array(internal_rows)])
            else:
                data_sub = leaf_rows
        else:
            data_sub = leaf_rows
        n_desc = data_sub.shape[0]

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

        # --- Dual-form optimisation (n ≪ d) ---
        # When n_desc < d_active, the d×d correlation matrix is rank ≤ n.
        # Instead of O(d³) eigh on the d×d matrix, compute the n×n Gram
        # matrix G = X_std @ X_std.T / d_active and recover the same non-
        # zero eigenvalues in O(n²d + n³) time.  For n=10, d=2000 this is
        # 10×10 eigh instead of 2000×2000.
        use_dual = n_desc < d_active

        if use_dual:
            # Standardise columns (zero-mean, unit-variance) for correlation
            col_means = data_active.mean(axis=0)
            col_stds = data_active.std(axis=0, ddof=0)
            col_stds[col_stds == 0] = 1.0  # already filtered but guard
            X_std = (data_active - col_means) / col_stds
            # Gram matrix: G = (1/d_active) X_std @ X_std.T  (n × n)
            gram = X_std @ X_std.T / d_active

            if need_eigh:
                eigenvalues, gram_vecs = np.linalg.eigh(gram)
                eigenvalues = eigenvalues[::-1]
                gram_vecs = gram_vecs[:, ::-1]
                eigenvectors_active = None  # recovered below after k is determined
            else:
                eigenvalues = np.sort(np.linalg.eigvalsh(gram))[::-1]
                gram_vecs = None
                eigenvectors_active = None
            _X_std = X_std  # save for eigenvector recovery
        else:
            corr = np.corrcoef(data_active.T)
            # Guard against numerical artefacts (NaN from near-constant cols)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            gram_vecs = None

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
        # dims stores the UNCAPPED spectral dimension (effective rank).
        # This is the authoritative projection dimension used by both
        # edge and sibling tests.  When the PCA projection has fewer
        # eigenvectors (dual form), the test functions pad with random
        # projection vectors.
        dims[node_id] = k

        # For eigenvector recovery, cap at the number of recoverable
        # eigenvectors (n_desc in dual form, d_active in primal form).
        k_avail = min(k, n_desc) if use_dual else k

        # Build projection matrix and eigenvalue array from top-k_avail eigenvectors.
        # Eigenvectors are in the active-feature subspace; embed into full
        # d-space by placing zeros at constant-feature positions.
        if need_eigh and k_avail > 0:
            # --- Recover d-space eigenvectors for dual form ---
            if use_dual and gram_vecs is not None:
                # V_active = X_std.T @ U_k / (sqrt(λ_k) * sqrt(d_active))
                # where U_k are the top-k gram eigenvectors, λ_k their eigenvalues.
                top_eigs = np.maximum(eigenvalues[:k_avail], 1e-12)
                eigenvectors_active = (
                    _X_std.T @ gram_vecs[:, :k_avail] / (np.sqrt(top_eigs) * np.sqrt(d_active))
                )
                # Normalise columns to unit length
                norms = np.linalg.norm(eigenvectors_active, axis=0)
                norms[norms == 0] = 1.0
                eigenvectors_active = eigenvectors_active / norms

            if eigenvectors_active is not None:
                full_eigvecs = np.zeros((d, k_avail), dtype=np.float64)
                full_eigvecs[active_mask, :] = eigenvectors_active[:, :k_avail]
                projs[node_id] = full_eigvecs.T  # (k_avail × d)
                # Store eigenvalues for whitening; clamp to small positive floor
                # to avoid division by zero in degenerate cases.
                eig_vals[node_id] = np.maximum(eigenvalues[:k_avail], 1e-12).astype(np.float64)

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


# =====================================================================
# Sibling-specific spectral dimension (pooled within-cluster)
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

    def _is_leaf(node_id: str) -> bool:
        is_leaf_attr = tree.nodes[node_id].get("is_leaf")
        if is_leaf_attr is not None:
            return bool(is_leaf_attr)
        return tree.out_degree(node_id) == 0

    # Precompute descendant leaf indices bottom-up
    desc_indices: Dict[str, list[int]] = {}
    for node_id in reversed(list(nx.topological_sort(tree))):
        if _is_leaf(node_id):
            lbl = tree.nodes[node_id].get("label", node_id)
            desc_indices[node_id] = [label_to_idx[lbl]] if lbl in label_to_idx else []
        else:
            indices = []
            for child in tree.successors(node_id):
                indices.extend(desc_indices.get(child, []))
            desc_indices[node_id] = indices

    # Get parent's overall spectral dims as fallback
    parent_dims, _, _ = compute_spectral_decomposition(
        tree,
        leaf_data,
        method=method,
        min_k=min_k,
        compute_projections=False,
    )

    sibling_dims: Dict[str, int] = {}

    for node_id in tree.nodes:
        if _is_leaf(node_id):
            sibling_dims[node_id] = 0
            continue

        children = list(tree.successors(node_id))
        if len(children) != 2:
            # Non-binary: use parent's spectral dim
            sibling_dims[node_id] = parent_dims.get(node_id, max(min_k, 1))
            continue

        left, right = children[0], children[1]
        left_idx = desc_indices.get(left, [])
        right_idx = desc_indices.get(right, [])
        n_left = len(left_idx)
        n_right = len(right_idx)

        # Need ≥2 leaves in each child for within-cluster centering
        if n_left < 2 or n_right < 2:
            sibling_dims[node_id] = parent_dims.get(node_id, max(min_k, 1))
            continue

        left_rows = X[left_idx, :]
        right_rows = X[right_idx, :]

        # Center each child's data around its own mean (remove between-group)
        resid_left = left_rows - left_rows.mean(axis=0)
        resid_right = right_rows - right_rows.mean(axis=0)
        pooled_resid = np.vstack([resid_left, resid_right])
        n_pooled = pooled_resid.shape[0]

        # Identify active features in pooled residuals
        col_var = np.var(pooled_resid, axis=0)
        active_mask = col_var > 0
        d_active = int(np.sum(active_mask))

        if d_active < 2:
            sibling_dims[node_id] = parent_dims.get(node_id, max(min_k, 1))
            continue

        # Compute correlation matrix of pooled residuals
        data_active = pooled_resid[:, active_mask]
        use_dual = n_pooled < d_active

        if use_dual:
            col_means = data_active.mean(axis=0)
            col_stds = data_active.std(axis=0, ddof=0)
            col_stds[col_stds == 0] = 1.0
            X_std = (data_active - col_means) / col_stds
            gram = X_std @ X_std.T / d_active
            eigenvalues = np.sort(np.linalg.eigvalsh(gram))[::-1]
        else:
            corr = np.corrcoef(data_active.T)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]

        eigenvalues = np.maximum(eigenvalues, 0.0)

        if method == "effective_rank":
            k = int(np.round(effective_rank(eigenvalues)))
        elif method == "marchenko_pastur":
            k = marchenko_pastur_signal_count(eigenvalues, n_pooled, d_active)
        elif method == "active_features":
            k = d_active
        else:
            k = int(np.round(effective_rank(eigenvalues)))

        k = max(k, min_k)
        k = min(k, d_active)
        sibling_dims[node_id] = k

    elapsed = time.perf_counter() - t0
    internal_ks = [v for nid, v in sibling_dims.items() if not _is_leaf(nid)]
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
    "effective_rank",
    "marchenko_pastur_signal_count",
    "count_active_features",
    "compute_node_spectral_dimensions",
    "compute_node_pca_projections",
    "compute_spectral_decomposition",
    "compute_sibling_spectral_dimensions",
]
