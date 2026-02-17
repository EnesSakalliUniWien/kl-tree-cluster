"""Sibling divergence test for hierarchical clustering.

Tests whether sibling nodes have significantly different distributions using
a Wald chi-square statistic with random projection for high dimensions.

Test statistic: T = ||R·z||² ~ χ²(k) where z is the standardized difference
and R is a random projection matrix reducing d features to k = O(log n).
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2, hmean

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    extract_node_distribution,
    extract_node_sample_size,
    initialize_sibling_divergence_columns,
)

from ..branch_length_utils import compute_mean_branch_length, sanitize_positive_branch_length
from ..categorical_mahalanobis import categorical_whitened_vector
from ..multiple_testing import benjamini_hochberg_correction
from ..pooled_variance import _is_categorical, standardize_proportion_difference
from ..random_projection import (
    compute_projection_dimension,
    derive_projection_seed,
    generate_projection_matrix,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Core Statistical Test
# =============================================================================


def _compute_chi_square_pvalue(
    projected: np.ndarray,
    df: int,
    eigenvalues: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    """Compute test statistic and p-value from projected z-scores.

    Two modes controlled by ``config.EIGENVALUE_WHITENING``:

    **Whitened** (``True``): T = Σ (vᵢᵀz)²/λᵢ ~ χ²(k).
    Exact under H₀ but divides signal by large eigenvalues → lower power.
    Supports split whitening when ``len(eigenvalues) < len(projected)``:
    the first k_pca components are whitened, remaining (random-padding)
    use plain sum of squares.

    **Satterthwaite** (``False``): T = Σ (vᵢᵀz)² ~ Σ λᵢ·χ²(1).
    Approximate as c·χ²(ν) where c = Σλᵢ²/Σλᵢ, ν = (Σλᵢ)²/Σλᵢ².
    Preserves power because signal in high-eigenvalue directions is not dampened.
    Any random-padding components contribute plain χ²(1) each.

    When *eigenvalues* is ``None``, the plain sum of squares with χ²(k) is used.
    """
    if eigenvalues is not None and len(eigenvalues) > 0:
        k_pca = len(eigenvalues)

        if config.EIGENVALUE_WHITENING:
            # --- Whitened mode: T = Σ wᵢ²/λᵢ ~ χ²(k) ---
            stat_pca = float(np.sum(projected[:k_pca] ** 2 / eigenvalues))
            if k_pca < len(projected):
                stat_rand = float(np.sum(projected[k_pca:] ** 2))
            else:
                stat_rand = 0.0
            stat = stat_pca + stat_rand
            return stat, float(df), float(chi2.sf(stat, df=df))
        else:
            # --- Satterthwaite mode: T = Σ wᵢ² ~ Σ λᵢ·χ²(1) ---
            # PCA part: unwhitened sum of squares
            sq_pca = projected[:k_pca] ** 2
            stat_pca = float(np.sum(sq_pca))
            # Random-padding part (if any): plain χ²(1) per component
            k_rand = len(projected) - k_pca
            stat_rand = float(np.sum(projected[k_pca:] ** 2)) if k_rand > 0 else 0.0
            stat = stat_pca + stat_rand

            # Satterthwaite: T_pca ~ Σ λᵢ·χ²(1), approximate as c·χ²(ν)
            # c = Σλᵢ²/Σλᵢ,  ν = (Σλᵢ)²/Σλᵢ²
            eigs = np.asarray(eigenvalues, dtype=np.float64)
            sum_eig = float(np.sum(eigs))
            sum_eig2 = float(np.sum(eigs**2))

            if sum_eig2 > 0 and sum_eig > 0:
                c = sum_eig2 / sum_eig
                nu_pca = sum_eig**2 / sum_eig2
                # Combined df: Satterthwaite ν for PCA + k_rand for random padding
                nu_total = nu_pca + k_rand
                # Scale only the PCA component; random part is already χ²
                stat_scaled = stat_pca / c + stat_rand
                return stat, float(nu_total), float(chi2.sf(stat_scaled, df=nu_total))
            else:
                # Degenerate eigenvalues — fall back to plain χ²(k)
                return stat, float(df), float(chi2.sf(stat, df=df))
    else:
        stat = float(np.sum(projected**2))
    return stat, float(df), float(chi2.sf(stat, df=df))


def sibling_divergence_test(
    left_dist: np.ndarray,
    right_dist: np.ndarray,
    n_left: float,
    n_right: float,
    branch_length_left: float | None = None,
    branch_length_right: float | None = None,
    mean_branch_length: float | None = None,
    *,
    test_id: str | None = None,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    """Two-sample Wald test for sibling divergence.

    Uses random projection on standardized vectors for both binary and
    categorical inputs. For categorical data, the vector is first covariance-
    whitened via a multinomial Mahalanobis construction (drop-last basis).

    Optionally applies Felsenstein's (1985) Phylogenetic Independent Contrasts
    adjustment by scaling variance by the sum of branch lengths.

    Parameters
    ----------
    left_dist : np.ndarray
        Distribution of left sibling.
    right_dist : np.ndarray
        Distribution of right sibling.
    n_left : float
        Sample size of left sibling.
    n_right : float
        Sample size of right sibling.
    branch_length_left : float, optional
        Branch length (distance to parent) for left sibling.
    branch_length_right : float, optional
        Branch length (distance to parent) for right sibling.
    mean_branch_length : float, optional
        Mean branch length across the tree for Felsenstein normalization.

    Returns
    -------
    Tuple[float, float, float]
        (test_statistic, degrees_of_freedom, p_value).
    """
    # Compute branch length sum for Felsenstein adjustment.
    # When mean_branch_length is None (Felsenstein disabled via config),
    # skip branch-length computation entirely to avoid triggering the
    # ValueError in standardize_proportion_difference().
    branch_length_sum = None
    if mean_branch_length is not None:
        bl_left = sanitize_positive_branch_length(branch_length_left)
        bl_right = sanitize_positive_branch_length(branch_length_right)
        if bl_left is not None and bl_right is not None:
            branch_length_sum = bl_left + bl_right
            if branch_length_sum <= 0:
                logger.warning(
                    "Non-positive sibling branch length sum encountered "
                    "(left=%s, right=%s). Disabling branch-length variance adjustment "
                    "for this test.",
                    bl_left,
                    bl_right,
                )
                branch_length_sum = None

    n_eff = hmean([n_left, n_right])

    if _is_categorical(np.asarray(left_dist)):
        z = categorical_whitened_vector(
            np.asarray(left_dist, dtype=np.float64),
            np.asarray(right_dist, dtype=np.float64),
            float(n_left),
            float(n_right),
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
        )
    else:
        z, _ = standardize_proportion_difference(
            left_dist,
            right_dist,
            n_left,
            n_right,
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
        )

    # Explicit invalid-test path: never coerce non-finite z-scores.
    # Keep raw statistics as NaN and route p=1.0 only in correction step.
    if not np.isfinite(z).all():
        logger.warning(
            "Found %d non-finite z-scores in sibling test; marking test invalid "
            "(raw outputs NaN, conservative p=1.0 for correction).",
            int(np.sum(~np.isfinite(z))),
        )
        return np.nan, np.nan, np.nan
    z = z.astype(np.float64, copy=False)  # Force float64

    d = len(z)

    # --- Determine projection dimension and matrix ---
    # spectral_k is the AUTHORITATIVE dimension when available.
    # PCA projections may have fewer rows (dual-form cap at n_desc);
    # in that case we pad with random projection vectors.
    if spectral_k is not None and spectral_k > 0:
        k = min(spectral_k, d)
    else:
        # Fallback: JL-based dimension
        # Use n_left + n_right (total observations) for the information cap,
        # NOT hmean.  The data matrix spanning the difference has rank ≤ n_L + n_R,
        # so that many z-components can carry signal.  hmean is the correct
        # effective sample size for *variance* estimation (pooled_variance.py),
        # but the rank constraint governs how many projection dimensions are
        # informative.
        n_total = int(n_left + n_right)
        k = compute_projection_dimension(n_total, d)

    # Project and compute test statistic
    if test_id is None:
        test_id = (
            f"sibling:shapeL={tuple(np.shape(left_dist))}:shapeR={tuple(np.shape(right_dist))}:"
            f"nL={float(n_left):.6g}:nR={float(n_right):.6g}"
        )
    test_seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, test_id)

    if pca_projection is not None:
        k_pca = pca_projection.shape[0]
        if k_pca >= k:
            # Truncate PCA projection to the authoritative k rows.
            R = pca_projection[:k]
            pca_eigenvalues = pca_eigenvalues[:k] if pca_eigenvalues is not None else None
        else:
            # Pad with random projection vectors to reach k rows.
            R_pad = generate_projection_matrix(d, k - k_pca, test_seed, use_cache=False)
            R = np.vstack([pca_projection, R_pad])
            # pca_eigenvalues stays as-is (only covers first k_pca rows)
    else:
        # Random projection (JL or spectral-k driven).
        # Per-test projections are one-off; bypass cache to avoid unbounded cache growth.
        R = generate_projection_matrix(d, k, test_seed, use_cache=False)
        pca_eigenvalues = None  # no whitening for pure random projection

    try:
        # Optimize matmul by ensuring arrays are ostensibly aligned (though numpy handles this)
        if hasattr(R, "dot"):
            projected = R.dot(z)
        else:
            projected = R @ z
    except Exception as e:
        logging.error(
            f"Projection failed (Sibling): z.shape={z.shape}, R.shape={R.shape}, z_stats={np.min(z)}/{np.max(z)}"
        )
        raise e

    return _compute_chi_square_pvalue(projected, k, eigenvalues=pca_eigenvalues)


# =============================================================================
# Tree Traversal Helpers
# =============================================================================


def _get_binary_children(tree: nx.DiGraph, parent: str) -> Optional[Tuple[str, str]]:
    """Return (left, right) children if parent has exactly 2, else None."""
    children = list(tree.successors(parent))
    if len(children) != 2:
        return None
    return children[0], children[1]


def _either_child_significant(
    left: str,
    right: str,
    sig_map: Dict[str, bool],
) -> bool:
    """Check if at least one child has significant child-parent divergence."""
    return sig_map.get(left, False) or sig_map.get(right, False)


def _get_sibling_data(
    tree: nx.DiGraph,
    parent: str,
    left: str,
    right: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    float | None,
    float | None,
]:
    """Extract distributions, sample sizes, and branch lengths for sibling pair.

    Branch lengths are extracted from the tree edges (parent → child).
    If not available, returns None for the branch lengths.
    """
    # Extract branch lengths from tree edges
    left_branch = (
        tree.edges[parent, left].get("branch_length") if tree.has_edge(parent, left) else None
    )
    right_branch = (
        tree.edges[parent, right].get("branch_length") if tree.has_edge(parent, right) else None
    )

    return (
        extract_node_distribution(tree, left),
        extract_node_distribution(tree, right),
        extract_node_sample_size(tree, left),
        extract_node_sample_size(tree, right),
        left_branch,
        right_branch,
    )


# =============================================================================
# Test Collection and Execution
# =============================================================================


def _collect_test_arguments(
    tree: nx.DiGraph,
    nodes_df: pd.DataFrame,
) -> Tuple[
    List[str],
    List[Tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]],
    List[str],
]:
    """Collect sibling pairs eligible for testing.

    Returns (parent_nodes, test_args, skipped_nodes).
    Each test_args tuple contains: (left_dist, right_dist, n_left, n_right, branch_left, branch_right)
    """
    if "Child_Parent_Divergence_Significant" not in nodes_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. " "Run child-parent test first."
        )

    sig_map = nodes_df["Child_Parent_Divergence_Significant"].to_dict()

    parents: List[str] = []
    args: List[Tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]] = []
    skipped: List[str] = []

    for parent in tree.nodes:
        children = _get_binary_children(tree, parent)
        if children is None:
            continue

        left, right = children

        # Skip if neither child diverged from parent
        if not _either_child_significant(left, right, sig_map):
            skipped.append(parent)
            continue

        left_dist, right_dist, n_left, n_right, bl_left, bl_right = _get_sibling_data(
            tree, parent, left, right
        )

        parents.append(parent)
        args.append((left_dist, right_dist, n_left, n_right, bl_left, bl_right))

    return parents, args, skipped


def _run_tests(
    parents: List[str],
    args: List[Tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]],
    mean_branch_length: float | None = None,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
) -> List[Tuple[float, float, float]]:
    """Execute sibling divergence tests for all collected pairs."""
    if len(parents) != len(args):
        raise ValueError(f"parents and args length mismatch: {len(parents)} != {len(args)}")
    results = []
    for parent, (left, right, n_l, n_r, bl_l, bl_r) in zip(parents, args, strict=False):
        _spectral_k: int | None = None
        _pca_proj: np.ndarray | None = None
        if spectral_dims is not None:
            _spectral_k = spectral_dims.get(parent)
        if pca_projections is not None:
            _pca_proj = pca_projections.get(parent)
        results.append(
            sibling_divergence_test(
                left,
                right,
                n_l,
                n_r,
                bl_l,
                bl_r,
                mean_branch_length,
                test_id=f"sibling:{parent}",
                spectral_k=_spectral_k,
                pca_projection=_pca_proj,
            )
        )
    return results


# =============================================================================
# DataFrame Updates
# =============================================================================


def _apply_results(
    df: pd.DataFrame,
    parents: List[str],
    results: List[Tuple[float, float, float]],
    alpha: float,
) -> pd.DataFrame:
    """Apply test results with BH correction to dataframe."""
    if not results:
        return df

    stats = np.array([r[0] for r in results])
    dfs = np.array([r[1] for r in results])
    pvals = np.array([r[2] for r in results])

    invalid_mask = (~np.isfinite(stats)) | (~np.isfinite(dfs)) | (~np.isfinite(pvals))
    pvals_for_correction = np.where(np.isfinite(pvals), pvals, 1.0)

    reject, pvals_adj, _ = benjamini_hochberg_correction(pvals_for_correction, alpha=alpha)
    reject = np.where(invalid_mask, False, reject)
    n_invalid = int(np.sum(invalid_mask))
    if n_invalid:
        logger.warning(
            "Sibling divergence audit: total_tests=%d, invalid_tests=%d. "
            "Conservative correction path applied (p=1.0, reject=False).",
            len(results),
            n_invalid,
        )

    df.loc[parents, "Sibling_Test_Statistic"] = stats
    df.loc[parents, "Sibling_Degrees_of_Freedom"] = dfs
    df.loc[parents, "Sibling_Divergence_P_Value"] = pvals
    df.loc[parents, "Sibling_Divergence_P_Value_Corrected"] = pvals_adj
    df.loc[parents, "Sibling_Divergence_Invalid"] = invalid_mask
    df.loc[parents, "Sibling_BH_Different"] = reject
    df.loc[parents, "Sibling_BH_Same"] = ~reject

    return df


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Test sibling divergence and annotate results in dataframe.

    Parameters
    ----------
    tree : nx.DiGraph
        Hierarchical tree with 'distribution' attribute on nodes.
    nodes_statistics_dataframe : pd.DataFrame
        Must contain 'Child_Parent_Divergence_Significant' column.
    significance_level_alpha : float
        FDR level for BH correction.

    Returns
    -------
    pd.DataFrame
        Updated with Sibling_Test_Statistic, Sibling_Degrees_of_Freedom,
        Sibling_Divergence_P_Value, Sibling_Divergence_P_Value_Corrected,
        Sibling_BH_Different, Sibling_BH_Same columns.
    """
    if len(nodes_statistics_dataframe) == 0:
        raise ValueError("Empty dataframe")

    df = nodes_statistics_dataframe.copy()
    df = initialize_sibling_divergence_columns(df)

    parents, args, skipped = _collect_test_arguments(tree, df)

    if not parents:
        warnings.warn("No eligible parent nodes for sibling tests", UserWarning)
        return df

    if skipped:
        df.loc[skipped, "Sibling_Divergence_Skipped"] = True
        logger.debug(f"Skipped {len(skipped)} nodes")

    # Compute mean branch length from tree for Felsenstein normalization
    # using the shared sanitization policy.  Gated by config.
    _mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    results = _run_tests(
        parents,
        args,
        mean_branch_length=_mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )
    df = _apply_results(df, parents, results, significance_level_alpha)
    sibling_invalid = int(df.loc[parents, "Sibling_Divergence_Invalid"].sum())
    df.attrs["sibling_divergence_audit"] = {
        "total_tests": int(len(parents)),
        "invalid_tests": sibling_invalid,
        "conservative_path_tests": sibling_invalid,
    }
    return df


__all__ = ["annotate_sibling_divergence", "sibling_divergence_test"]
