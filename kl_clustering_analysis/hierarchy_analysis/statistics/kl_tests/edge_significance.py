"""Edge significance testing for hierarchical clustering.

Tests whether each child's distribution significantly diverges from its parent
using a projected Wald test. Supports both binary (Bernoulli) and categorical
(multinomial) distributions.

Test statistic: T = ||R·z||² ~ χ²(k) where:
- z_i = (θ_child - θ_parent) / √(Var(θ̂)) is the standardized difference
- R is a k × d random projection matrix (k = O(log n))
- k is the projection dimension from the JL lemma

References
----------
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
    mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.
Bogomolov et al. (2021). "Hypotheses on a tree: new error rates and
    testing strategies". Biometrika, 108(3), 575-590.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    assign_divergence_results,
    extract_leaf_counts,
)
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths

from ..branch_length_utils import compute_mean_branch_length as _compute_mean_branch_length
from ..branch_length_utils import (
    sanitize_positive_branch_length as _sanitize_positive_branch_length,
)
from ..multiple_testing import apply_multiple_testing_correction
from ..projection.random_projection import (
    compute_projection_dimension,
    derive_projection_seed,
    generate_projection_matrix,
)
from ..projection.spectral_dimension import compute_spectral_decomposition

logger = logging.getLogger(__name__)


# =============================================================================
# Core Statistical Functions
# =============================================================================


def _compute_standardized_z(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    branch_length: float | None = None,
    mean_branch_length: float | None = None,
) -> np.ndarray:
    """Compute standardized z-scores for child vs parent.

    Supports both binary (1D) and categorical (2D) distributions.
    Under H₀ (child ~ parent), each z_i ~ N(0, 1).

    Uses nested variance formula that accounts for child being within parent:
        Var(θ̂_child - θ̂_parent) = θ(1-θ) × (1/n_child - 1/n_parent)

    Applies normalized branch-length scaling following Felsenstein's (1985)
    Phylogenetic Independent Contrasts:
        Var_adjusted = Var × (1 + BL/mean_BL)

    The normalization ensures BL_norm ≥ 1, preventing variance shrinkage
    that would cause over-splitting.

    Parameters
    ----------
    child_dist
        Child node's distribution. Shape (d,) for binary or (d, K) for categorical.
    parent_dist
        Parent node's distribution. Same shape as child_dist.
    n_child
        Sample size (leaf count) for the child node.
    n_parent
        Sample size (leaf count) for the parent node.
    branch_length
        Distance from parent to child in the tree.
    mean_branch_length
        Mean branch length across tree for normalization.

    Returns
    -------
    np.ndarray
        Standardized z-scores, flattened to 1D.
    """
    # Nested variance formula: Var = θ(1-θ) × (1/n_child - 1/n_parent)
    # This accounts for the covariance between child and parent estimates
    # since child data is included in parent's calculation.
    nested_factor = 1.0 / n_child - 1.0 / n_parent

    # Child must be a proper subset of parent: n_child < n_parent
    # If nested_factor <= 0, the tree structure is invalid (child >= parent)
    if nested_factor <= 0:
        raise ValueError(
            f"Invalid tree structure: child sample size ({n_child}) must be strictly "
            f"less than parent sample size ({n_parent}). Got nested_factor={nested_factor:.6f}. "
            f"This indicates a degenerate or incorrectly constructed tree."
        )

    var = parent_dist * (1 - parent_dist) * nested_factor

    # Felsenstein (1985) branch-length adjustment with normalization:
    # BL_norm = 1 + BL/mean_BL ensures multiplier ≥ 1
    # - Longer branches → larger variance → smaller z → harder to split
    # - Shorter branches → less extra variance → easier to split
    if (
        branch_length is not None
        and np.isfinite(branch_length)
        and branch_length > 0
        and mean_branch_length is not None
        and np.isfinite(mean_branch_length)
        and mean_branch_length > 0
    ):
        bl_normalized = 1.0 + branch_length / mean_branch_length
        var = var * bl_normalized

    var = np.maximum(var, 1e-10)
    z = (child_dist - parent_dist) / np.sqrt(var)

    # [0.25, 0.75] -  [0.1, 0.9] / 2

    # Flatten if categorical (2D -> 1D)
    return z.ravel()


def _compute_projected_test(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    seed: int,
    branch_length: float | None = None,
    mean_branch_length: float | None = None,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
) -> tuple[float, float, float, bool]:
    """Compute projected Wald test for one edge.

    Projects z-scores to k dimensions and computes T = ||R·z||² ~ χ²(k).

    When *pca_projection* and *pca_eigenvalues* are provided (from correlation-
    matrix eigendecomposition), the test uses eigenvalue whitening for an exact
    χ²(k) null:
        T = Σ (vᵢᵀz)² / λᵢ ~ χ²(k)
    where vᵢ are eigenvectors of Corr(X) and λᵢ the corresponding eigenvalues.

    When *pca_projection* is provided without eigenvalues, or when a random
    projection is used, falls back to the unwhitened statistic T = ||Rz||².

    Parameters
    ----------
    child_dist
        Child node's distribution.
    parent_dist
        Parent node's distribution.
    n_child
        Sample size for child node.
    n_parent
        Sample size for parent node.
    seed
        Random seed for projection matrix (used only for random projection fallback).
    branch_length
        Distance from parent to child. Used for Felsenstein adjustment.
    mean_branch_length
        Mean branch length across tree for normalization.
    spectral_k
        Per-node projection dimension from spectral decomposition.  If ``None``,
        the JL formula is used.
    pca_projection
        Per-node PCA projection matrix of shape ``(spectral_k, d)``.  If ``None``
        but *spectral_k* is set, a random projection of dimension *spectral_k*
        is generated instead.
    pca_eigenvalues
        Top-k eigenvalues of the correlation matrix, shape ``(k,)``.
        Used for whitening when *pca_projection* is also provided.

    Returns
    -------
    tuple[float, float, float, bool]
        (test_statistic, degrees_of_freedom, p_value, invalid_test)
    """
    z = _compute_standardized_z(
        child_dist, parent_dist, n_child, n_parent, branch_length, mean_branch_length
    )

    # Explicit invalid-test path: never coerce non-finite z-scores.
    # Keep raw statistics as NaN and route p=1.0 only in correction step.
    non_finite = ~np.isfinite(z)
    if np.any(non_finite):
        logger.warning(
            "Found %d non-finite z-scores in edge test; marking test invalid "
            "(raw outputs NaN, conservative p=1.0 for correction).",
            int(np.sum(non_finite)),
        )
        return np.nan, np.nan, np.nan, True
    z = z.astype(np.float64, copy=False)

    d = len(z)

    # For categorical data, account for simplex constraint (probs sum to 1)
    # Drop the last category column - only K-1 categories are independent
    # This properly handles the correlation between categories
    if child_dist.ndim == 2 and child_dist.shape[1] > 1:
        n_features = child_dist.shape[0]
        n_categories = child_dist.shape[1]
        # Reshape to (n_features, n_categories), drop last column, flatten
        z = z.reshape(n_features, n_categories)[:, :-1].ravel()

    d = len(z)

    # --- Determine projection dimension and matrix ---
    # spectral_k is the AUTHORITATIVE dimension when available.
    # PCA projections may have fewer rows (dual-form cap at n_desc);
    # in that case we pad with random projection vectors.
    if spectral_k is not None and spectral_k > 0:
        k = min(spectral_k, d)
    else:
        # Fallback: JL-based dimension
        k = compute_projection_dimension(n_child, d)

    if pca_projection is not None:
        k_pca = pca_projection.shape[0]
        if k_pca >= k:
            # Truncate PCA projection to the authoritative k rows.
            R = pca_projection[:k]
            eig_for_whitening = (
                np.asarray(pca_eigenvalues[:k], dtype=np.float64)
                if pca_eigenvalues is not None
                else None
            )
        else:
            # Pad with random projection vectors to reach k rows.
            R_pad = generate_projection_matrix(d, k - k_pca, seed, use_cache=False)
            R = np.vstack([pca_projection, R_pad])
            # Eigenvalues only cover the first k_pca (PCA) rows.
            eig_for_whitening = (
                np.asarray(pca_eigenvalues, dtype=np.float64)
                if pca_eigenvalues is not None
                else None
            )
    else:
        # Random projection (JL or spectral-k driven).
        # Per-test projections are one-off; bypass cache.
        R = generate_projection_matrix(d, k, seed, use_cache=False)
        eig_for_whitening = None

    try:
        if hasattr(R, "dot"):
            projected = R.dot(z)
        else:
            projected = R @ z
    except Exception as e:
        logger.error(
            f"Projection failed (Edge): z.shape={z.shape}, R.shape={R.shape}, z_stats={np.min(z)}/{np.max(z)}"
        )
        raise e

    # Test statistic: whitened or Satterthwaite depending on config.
    # See sibling_divergence_test._compute_chi_square_pvalue for full docs.
    if eig_for_whitening is not None and len(eig_for_whitening) > 0:
        n_eig = len(eig_for_whitening)
        k_rand = k - n_eig if n_eig < k else 0
        stat_rand = float(np.sum(projected[n_eig:] ** 2)) if k_rand > 0 else 0.0

        if config.EIGENVALUE_WHITENING:
            # Whitened: T_pca = Σ w²/λ ~ χ²(k_pca)
            stat_pca = float(np.sum(projected[:n_eig] ** 2 / eig_for_whitening))
            stat = stat_pca + stat_rand
            pval = float(chi2.sf(stat, df=k))
        else:
            # Satterthwaite: T_pca = Σ w² ~ Σ λᵢ·χ²(1)
            stat_pca = float(np.sum(projected[:n_eig] ** 2))
            stat = stat_pca + stat_rand
            eigs = eig_for_whitening
            sum_eig = float(np.sum(eigs))
            sum_eig2 = float(np.sum(eigs**2))
            if sum_eig2 > 0 and sum_eig > 0:
                c = sum_eig2 / sum_eig
                nu_pca = sum_eig**2 / sum_eig2
                nu_total = nu_pca + k_rand
                stat_scaled = stat_pca / c + stat_rand
                pval = float(chi2.sf(stat_scaled, df=nu_total))
            else:
                pval = float(chi2.sf(stat, df=k))
    else:
        stat = float(np.sum(projected**2))
        pval = float(chi2.sf(stat, df=k))

    return stat, float(k), pval, False


def _compute_p_values_via_projection(
    tree: nx.DiGraph,
    child_ids: list[str],
    parent_ids: list[str],
    child_leaf_counts: np.ndarray,
    parent_leaf_counts: np.ndarray,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
    pca_eigenvalues: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute p-values for all edges via random projection.

    Extracts branch lengths from tree edges and applies Felsenstein's (1985)
    branch-length adjustment to variance. Uses corrected variance formula
    that accounts for child being nested within parent.

    Parameters
    ----------
    tree
        Hierarchy with 'distribution' attribute on nodes and 'branch_length'
        attribute on edges.
    child_ids
        List of child node IDs.
    parent_ids
        List of parent node IDs (aligned with child_ids).
    child_leaf_counts
        Sample sizes for child nodes.
    parent_leaf_counts
        Sample sizes for parent nodes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (test_statistics, degrees_of_freedom, p_values, invalid_mask)
    """
    n_edges = len(child_ids)
    stats = np.full(n_edges, np.nan)
    dfs = np.full(n_edges, np.nan)
    pvals = np.full(n_edges, np.nan)
    invalid_mask = np.zeros(n_edges, dtype=bool)

    # Compute mean branch length for normalization (gated by config)
    mean_branch_length = _compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    for i in range(n_edges):
        child_dist = tree.nodes[child_ids[i]].get("distribution")
        parent_dist = tree.nodes[parent_ids[i]].get("distribution")

        if child_dist is None or parent_dist is None or child_leaf_counts[i] < 1:
            stats[i], dfs[i], pvals[i] = 0.0, 0.0, 1.0
            continue

        # Extract branch length from tree edge (parent → child)
        branch_length: float | None = None
        if tree.has_edge(parent_ids[i], child_ids[i]):
            branch_length = _sanitize_positive_branch_length(
                tree.edges[parent_ids[i], child_ids[i]].get("branch_length")
            )

        test_seed = derive_projection_seed(
            config.PROJECTION_RANDOM_SEED,
            f"edge:{parent_ids[i]}->{child_ids[i]}",
        )

        # Per-node spectral dimension and PCA projection (if available).
        # The parent node determines the local subspace: the child edge
        # test asks "does this child diverge from its parent?", so the
        # relevant covariance structure is the parent's descendant data.
        _spectral_k: int | None = None
        _pca_proj: np.ndarray | None = None
        _pca_eig: np.ndarray | None = None
        if spectral_dims is not None:
            _spectral_k = spectral_dims.get(parent_ids[i])
        if pca_projections is not None:
            _pca_proj = pca_projections.get(parent_ids[i])
        if pca_eigenvalues is not None:
            _pca_eig = pca_eigenvalues.get(parent_ids[i])

        stat_i, df_i, pval_i, invalid_i = _compute_projected_test(
            np.asarray(child_dist, dtype=np.float64),
            np.asarray(parent_dist, dtype=np.float64),
            int(child_leaf_counts[i]),
            int(parent_leaf_counts[i]),
            test_seed,
            branch_length,
            mean_branch_length,
            spectral_k=_spectral_k,
            pca_projection=_pca_proj,
            pca_eigenvalues=_pca_eig,
        )
        stats[i], dfs[i], pvals[i] = stat_i, df_i, pval_i
        invalid_mask[i] = bool(invalid_i)

    return stats, dfs, pvals, invalid_mask


# =============================================================================
# Public API
# =============================================================================


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    significance_level_alpha: float = 0.05,
    fdr_method: str = "tree_bh",
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    min_k: int | None = None,
) -> pd.DataFrame:
    """Test child-parent divergence using projected Wald test.

    Supports both binary (Bernoulli) and categorical (multinomial) distributions.

    When *leaf_data* and *spectral_method* are provided, per-node
    eigendecomposition replaces the JL-based projection dimension.
    The spectral method determines both the projection dimension k_v
    and (for "effective_rank" / "marchenko_pastur") the PCA-based
    informed projection at each internal node.

    Parameters
    ----------
    tree
        Directed acyclic graph representing the hierarchy. Nodes must have
        'distribution' attribute containing feature parameters.
    nodes_statistics_dataframe
        DataFrame indexed by node_id with 'leaf_count' column.
    significance_level_alpha
        FDR threshold for correction.
    fdr_method
        FDR correction method: "tree_bh", "flat", or "level_wise".
    leaf_data
        Raw binary data matrix (samples × features).  Required for
        per-node spectral dimension estimation.
    spectral_method
        Dimension estimator: ``"effective_rank"``, ``"marchenko_pastur"``,
        or ``"active_features"``.  When ``None`` (default), the legacy
        JL-based dimension is used.
    min_k
        Minimum projection dimension (floor).  When ``None``, uses
        ``config.PROJECTION_MIN_K`` (resolved to int at pipeline entry).

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with divergence test results.
    """
    nodes_dataframe = nodes_statistics_dataframe.copy()
    alpha = float(significance_level_alpha)

    edge_list = list(tree.edges())
    parent_ids = [parent_id for parent_id, _ in edge_list]
    child_ids = [child_id for _, child_id in edge_list]

    if not child_ids:
        raise ValueError("Tree has no edges. Cannot compute child-parent divergence.")

    child_leaf_counts = extract_leaf_counts(nodes_dataframe, child_ids)
    parent_leaf_counts = extract_leaf_counts(nodes_dataframe, parent_ids)

    # --- Per-node spectral dimension (replaces JL when configured) ---
    spectral_dims: dict[str, int] | None = None
    pca_projections: dict[str, np.ndarray] | None = None
    pca_eigenvalues: dict[str, np.ndarray] | None = None

    if spectral_method is not None:
        if leaf_data is None:
            raise ValueError(
                f"spectral_method={spectral_method!r} requires leaf_data to be provided."
            )
        # Only compute DIMENSIONS from the correlation eigendecomposition.
        # PCA projections are NOT used for the test statistic — random
        # orthonormal projection with spectral k gives an approximately
        # valid χ²(k) null via concentration of measure, while preserving
        # power (no eigenvalue whitening that down-weights signal directions).
        # Eigenvalues and projections are still available in df.attrs for
        # downstream consumers that want whitened statistics.
        spectral_dims, pca_proj_dict, pca_eig_dict = compute_spectral_decomposition(
            tree,
            leaf_data,
            method=spectral_method,
            min_k=min_k if isinstance(min_k, int) else 1,
            compute_projections=True,
        )
        pca_projections = pca_proj_dict if pca_proj_dict else None
        pca_eigenvalues = pca_eig_dict if pca_eig_dict else None

    # Stash computed spectral info in df.attrs so sibling tests can reuse them
    # without recomputing the eigendecomposition.
    nodes_dataframe.attrs["_spectral_dims"] = spectral_dims
    nodes_dataframe.attrs["_pca_projections"] = pca_projections
    nodes_dataframe.attrs["_pca_eigenvalues"] = pca_eigenvalues

    test_stats, degrees_of_freedom, p_values, invalid_mask = _compute_p_values_via_projection(
        tree,
        child_ids,
        parent_ids,
        child_leaf_counts,
        parent_leaf_counts,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    )

    # Stash raw test data in attrs so the post-hoc edge calibration
    # (calibrate_edges_from_sibling_neighborhood) can use them after Gate 3.
    nodes_dataframe.attrs["_edge_raw_test_data"] = {
        "child_ids": child_ids,
        "parent_ids": parent_ids,
        "test_stats": test_stats.copy(),
        "degrees_of_freedom": degrees_of_freedom.copy(),
        "p_values": p_values.copy(),
        "child_leaf_counts": child_leaf_counts.copy(),
        "parent_leaf_counts": parent_leaf_counts.copy(),
    }

    node_depths = compute_node_depths(tree)
    child_depths = np.array([node_depths.get(cid, 0) for cid in child_ids])

    # Preserve raw NaN outputs for invalid tests; use conservative surrogate
    # p-values only for multiple-testing correction.
    p_values_for_correction = np.where(np.isfinite(p_values), p_values, 1.0)

    nonfinite_p_mask = ~np.isfinite(p_values)
    n_invalid = int(np.sum(invalid_mask))
    n_nonfinite_p = int(np.sum(nonfinite_p_mask))
    if n_invalid or n_nonfinite_p:
        bad_indices = [i for i, v in enumerate(p_values) if not np.isfinite(v)]
        bad_ids = [child_ids[i] for i in bad_indices]
        preview = ", ".join(map(repr, bad_ids[:5]))
        logger.warning(
            "Child-parent divergence audit: total_tests=%d, invalid_tests=%d, "
            "nonfinite_p_values=%d. Conservative correction path applied "
            "(p=1.0, reject=False) for nodes: %s",
            len(child_ids),
            n_invalid,
            n_nonfinite_p,
            preview,
        )

    # node_depths and child_depths already computed above (before calibration)

    reject_null, p_values_corrected = apply_multiple_testing_correction(
        p_values=p_values_for_correction,
        child_ids=child_ids,
        child_depths=child_depths,
        alpha=alpha,
        method=fdr_method,
        tree=tree,
    )
    reject_null = np.where(nonfinite_p_mask, False, reject_null)

    # Attach run-level audit counters for downstream diagnostics.
    nodes_dataframe.attrs["child_parent_divergence_audit"] = {
        "total_tests": int(len(child_ids)),
        "invalid_tests": n_invalid,
        "nonfinite_p_values": n_nonfinite_p,
        "conservative_path_tests": n_nonfinite_p,
    }

    return assign_divergence_results(
        nodes_dataframe=nodes_dataframe,
        child_ids=child_ids,
        p_values=p_values,
        p_values_corrected=p_values_corrected,
        reject_null=reject_null,
        degrees_of_freedom=degrees_of_freedom,
        invalid_mask=invalid_mask,
    )


__all__ = ["annotate_child_parent_divergence"]

__all__ = ["annotate_child_parent_divergence"]
