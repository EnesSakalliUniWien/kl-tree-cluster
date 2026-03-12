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

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    assign_divergence_results,
    extract_leaf_counts,
)
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths

from ...decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_projection_dimension,
)
from ...decomposition.backends.random_projection_backend import (
    derive_projection_seed_backend as derive_projection_seed,
)
from ...decomposition.methods.projected_wald import run_projected_wald_kernel
from ..branch_length_utils import compute_mean_branch_length as _compute_mean_branch_length
from ..branch_length_utils import (
    sanitize_positive_branch_length as _sanitize_positive_branch_length,
)
from ..multiple_testing import apply_multiple_testing_correction
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

    variance = parent_dist * (1 - parent_dist) * nested_factor

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
        normalized_branch_length_multiplier = 1.0 + branch_length / mean_branch_length
        variance = variance * normalized_branch_length_multiplier

    variance = np.maximum(variance, 1e-10)
    z_scores = (child_dist - parent_dist) / np.sqrt(variance)

    # [0.25, 0.75] -  [0.1, 0.9] / 2

    # Flatten if categorical (2D -> 1D)
    return z_scores.ravel()


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
    minimum_projection_dimension: int | None = None,
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
    minimum_projection_dimension
        Minimum projection dimension floor used by the JL fallback path when
        *spectral_k* is ``None``.

    Returns
    -------
    tuple[float, float, float, bool]
        (test_statistic, degrees_of_freedom, p_value, invalid_test)
    """
    standardized_z_scores = _compute_standardized_z(
        child_dist, parent_dist, n_child, n_parent, branch_length, mean_branch_length
    )

    # Explicit invalid-test path: never coerce non-finite z-scores.
    # Keep raw statistics as NaN and route p=1.0 only in correction step.
    nonfinite_z_score_mask = ~np.isfinite(standardized_z_scores)
    if np.any(nonfinite_z_score_mask):
        logger.warning(
            "Found %d non-finite z-scores in edge test; marking test invalid "
            "(raw outputs NaN, conservative p=1.0 for correction).",
            int(np.sum(nonfinite_z_score_mask)),
        )
        return np.nan, np.nan, np.nan, True

    standardized_z_scores = standardized_z_scores.astype(np.float64, copy=False)

    # For categorical data, account for simplex constraint (probs sum to 1)
    # Drop the last category column - only K-1 categories are independent
    # This properly handles the correlation between categories
    if child_dist.ndim == 2 and child_dist.shape[1] > 1:
        n_features = child_dist.shape[0]
        n_categories = child_dist.shape[1]
        # Reshape to (n_features, n_categories), drop last column, flatten
        standardized_z_scores = standardized_z_scores.reshape(n_features, n_categories)[
            :, :-1
        ].ravel()

    # Shared projected-test kernel.  Keep edge return semantics unchanged:
    # return nominal df=k (not the Satterthwaite effective df).
    try:
        test_statistic, projection_dim, _effective_degrees_of_freedom, p_value = (
            run_projected_wald_kernel(
                standardized_z_scores,
                seed=seed,
                spectral_k=spectral_k,
                pca_projection=pca_projection,
                pca_eigenvalues=pca_eigenvalues,
                k_fallback=lambda dim: compute_projection_dimension(
                    n_child,
                    dim,
                    minimum_projection_dimension=minimum_projection_dimension,
                ),
            )
        )
    except Exception as e:
        logger.error(
            "Projection failed (Edge): z.shape=%s, z_stats=%s/%s",
            standardized_z_scores.shape,
            np.min(standardized_z_scores),
            np.max(standardized_z_scores),
        )
        raise e

    return test_statistic, float(projection_dim), p_value, False


def _compute_p_values_via_projection(
    tree: nx.DiGraph,
    child_ids: list[str],
    parent_ids: list[str],
    child_leaf_counts: np.ndarray,
    parent_leaf_counts: np.ndarray,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
    pca_eigenvalues: dict[str, np.ndarray] | None = None,
    minimum_projection_dimension: int | None = None,
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
    n_edge_tests = len(child_ids)
    test_statistics = np.full(n_edge_tests, np.nan)
    degrees_of_freedom = np.full(n_edge_tests, np.nan)
    p_values = np.full(n_edge_tests, np.nan)
    invalid_test_mask = np.zeros(n_edge_tests, dtype=bool)

    # Compute mean branch length for normalization (gated by config)
    mean_branch_length = _compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    for edge_index in range(n_edge_tests):
        child_dist = tree.nodes[child_ids[edge_index]].get("distribution")
        parent_dist = tree.nodes[parent_ids[edge_index]].get("distribution")

        if child_dist is None or parent_dist is None or child_leaf_counts[edge_index] < 1:
            test_statistics[edge_index], degrees_of_freedom[edge_index], p_values[edge_index] = (
                0.0,
                0.0,
                1.0,
            )
            continue

        # Extract branch length from tree edge (parent → child)
        branch_length: float | None = None
        if tree.has_edge(parent_ids[edge_index], child_ids[edge_index]):
            branch_length = _sanitize_positive_branch_length(
                tree.edges[parent_ids[edge_index], child_ids[edge_index]].get("branch_length")
            )

        test_seed = derive_projection_seed(
            config.PROJECTION_RANDOM_SEED,
            f"edge:{parent_ids[edge_index]}->{child_ids[edge_index]}",
        )

        # Per-node spectral dimension and PCA projection (if available).
        # The parent node determines the local subspace: the child edge
        # test asks "does this child diverge from its parent?", so the
        # relevant covariance structure is the parent's descendant data.
        node_spectral_dimension: int | None = None
        node_pca_projection: np.ndarray | None = None
        node_pca_eigenvalues: np.ndarray | None = None
        if spectral_dims is not None:
            node_spectral_dimension = spectral_dims.get(parent_ids[edge_index])
        if pca_projections is not None:
            node_pca_projection = pca_projections.get(parent_ids[edge_index])
        if pca_eigenvalues is not None:
            node_pca_eigenvalues = pca_eigenvalues.get(parent_ids[edge_index])

        projected_test_kwargs: dict[str, object] = {
            "spectral_k": node_spectral_dimension,
            "pca_projection": node_pca_projection,
            "pca_eigenvalues": node_pca_eigenvalues,
        }
        if minimum_projection_dimension is not None:
            projected_test_kwargs["minimum_projection_dimension"] = int(
                minimum_projection_dimension
            )

        (
            edge_test_statistic,
            edge_degrees_of_freedom,
            edge_p_value,
            edge_test_invalid,
        ) = _compute_projected_test(
            np.asarray(child_dist, dtype=np.float64),
            np.asarray(parent_dist, dtype=np.float64),
            int(child_leaf_counts[edge_index]),
            int(parent_leaf_counts[edge_index]),
            test_seed,
            branch_length,
            mean_branch_length,
            **projected_test_kwargs,
        )
        test_statistics[edge_index], degrees_of_freedom[edge_index], p_values[edge_index] = (
            edge_test_statistic,
            edge_degrees_of_freedom,
            edge_p_value,
        )
        invalid_test_mask[edge_index] = bool(edge_test_invalid)

    return test_statistics, degrees_of_freedom, p_values, invalid_test_mask


# =============================================================================
# Public API
# =============================================================================


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    significance_level_alpha: float = 0.05,
    fdr_method: str = "tree_bh",
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    minimum_projection_dimension: int | None = None,
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
    annotations_df
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
    minimum_projection_dimension
        Minimum projection dimension (floor) used by the JL fallback path.
        When ``None``, uses ``config.PROJECTION_MINIMUM_DIMENSION`` via backend
        defaulting. Spectral decomposition still uses ``SPECTRAL_MINIMUM_DIMENSION``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with divergence test results.
    """
    annotations_df = annotations_df.copy()
    edge_alpha = float(significance_level_alpha)

    tree_edges = list(tree.edges())
    parent_ids = [parent_id for parent_id, _ in tree_edges]
    child_ids = [child_id for _, child_id in tree_edges]

    if not child_ids:
        raise ValueError("Tree has no edges. Cannot compute child-parent divergence.")

    child_leaf_counts = extract_leaf_counts(annotations_df, child_ids)
    parent_leaf_counts = extract_leaf_counts(annotations_df, parent_ids)

    # --- Per-node spectral dimension (replaces JL when configured) ---
    node_spectral_dimensions: dict[str, int] | None = None
    node_pca_projections: dict[str, np.ndarray] | None = None
    node_pca_eigenvalues: dict[str, np.ndarray] | None = None

    if spectral_method is not None:
        if leaf_data is None:
            raise ValueError(
                f"spectral_method={spectral_method!r} requires leaf_data to be provided."
            )
        # Compute spectral dimensions and (when available) PCA projections/
        # eigenvalues for the current Gate 2 run. These are consumed
        # immediately below by _compute_p_values_via_projection().
        #
        # The spectral path uses its own small floor (SPECTRAL_MINIMUM_DIMENSION=2)
        # instead of the global JL-derived minimum_projection_dimension.  The per-node effective
        # rank IS the signal dimensionality; flooring it with the global
        # erank inflates df with noise-only χ² components and kills power.
        from kl_clustering_analysis import config as _config

        spectral_minimum_projection_dimension = getattr(
            _config,
            "SPECTRAL_MINIMUM_DIMENSION",
            2,
        )

        (
            node_spectral_dimensions,
            computed_node_pca_projections,
            computed_node_pca_eigenvalues,
        ) = compute_spectral_decomposition(
            tree,
            leaf_data,
            method=spectral_method,
            minimum_projection_dimension=spectral_minimum_projection_dimension,
            compute_projections=True,
        )
        node_pca_projections = (
            computed_node_pca_projections if computed_node_pca_projections else None
        )
        node_pca_eigenvalues = (
            computed_node_pca_eigenvalues if computed_node_pca_eigenvalues else None
        )

    # Keep only lightweight spectral dimensions in attrs for diagnostics.
    annotations_df.attrs["_spectral_dims"] = node_spectral_dimensions

    projection_test_kwargs: dict[str, object] = {
        "tree": tree,
        "child_ids": child_ids,
        "parent_ids": parent_ids,
        "child_leaf_counts": child_leaf_counts,
        "parent_leaf_counts": parent_leaf_counts,
        "spectral_dims": node_spectral_dimensions,
        "pca_projections": node_pca_projections,
        "pca_eigenvalues": node_pca_eigenvalues,
    }
    if minimum_projection_dimension is not None:
        projection_test_kwargs["minimum_projection_dimension"] = int(minimum_projection_dimension)

    (
        edge_test_statistics,
        edge_degrees_of_freedom,
        edge_p_values,
        invalid_test_mask,
    ) = _compute_p_values_via_projection(**projection_test_kwargs)

    # Stash raw test data in attrs so the post-hoc edge calibration
    # (calibrate_edges_from_sibling_neighborhood) can use them after Gate 3.
    annotations_df.attrs["_edge_raw_test_data"] = {
        "child_ids": child_ids,
        "parent_ids": parent_ids,
        "test_stats": edge_test_statistics.copy(),
        "degrees_of_freedom": edge_degrees_of_freedom.copy(),
        "p_values": edge_p_values.copy(),
        "child_leaf_counts": child_leaf_counts.copy(),
        "parent_leaf_counts": parent_leaf_counts.copy(),
    }

    node_depths = compute_node_depths(tree)
    child_depths_for_correction = np.array([node_depths.get(cid, 0) for cid in child_ids])

    # Preserve raw NaN outputs for invalid tests; use conservative surrogate
    # p-values only for multiple-testing correction.
    p_values_for_correction = np.where(np.isfinite(edge_p_values), edge_p_values, 1.0)

    nonfinite_p_value_mask = ~np.isfinite(edge_p_values)
    invalid_test_count = int(np.sum(invalid_test_mask))
    nonfinite_p_value_count = int(np.sum(nonfinite_p_value_mask))
    if invalid_test_count or nonfinite_p_value_count:
        nonfinite_p_value_indices = [
            edge_index
            for edge_index, p_value in enumerate(edge_p_values)
            if not np.isfinite(p_value)
        ]
        nonfinite_p_value_node_ids = [
            child_ids[edge_index] for edge_index in nonfinite_p_value_indices
        ]
        preview_node_ids = ", ".join(map(repr, nonfinite_p_value_node_ids[:5]))
        logger.warning(
            "Child-parent divergence audit: total_tests=%d, invalid_tests=%d, "
            "nonfinite_p_values=%d. Conservative correction path applied "
            "(p=1.0, reject=False) for nodes: %s",
            len(child_ids),
            invalid_test_count,
            nonfinite_p_value_count,
            preview_node_ids,
        )

    # node_depths and child_depths_for_correction already computed above (before calibration)

    reject_null_hypothesis, corrected_p_values = apply_multiple_testing_correction(
        p_values=p_values_for_correction,
        child_ids=child_ids,
        child_depths=child_depths_for_correction,
        alpha=edge_alpha,
        method=fdr_method,
        tree=tree,
    )

    reject_null_hypothesis = np.where(nonfinite_p_value_mask, False, reject_null_hypothesis)

    # Attach run-level audit counters for downstream diagnostics.
    annotations_df.attrs["child_parent_divergence_audit"] = {
        "total_tests": int(len(child_ids)),
        "invalid_tests": invalid_test_count,
        "nonfinite_p_values": nonfinite_p_value_count,
        "conservative_path_tests": nonfinite_p_value_count,
    }

    return assign_divergence_results(
        annotations_df=annotations_df,
        child_ids=child_ids,
        p_values=edge_p_values,
        p_values_corrected=corrected_p_values,
        reject_null=reject_null_hypothesis,
        degrees_of_freedom=edge_degrees_of_freedom,
        invalid_mask=invalid_test_mask,
    )


__all__ = ["annotate_child_parent_divergence"]
