"""Cousin-calibrated sibling divergence test (F-test).

Replaces the standard Wald χ² sibling test with a calibrated F-test that
cancels post-selection bias from the linkage tree.

Architecture
------------
At a parent node P with children L, R and uncle U (sibling of P under
grandparent G), we compare the sibling test statistic T_LR to U's own
sibling statistic T_{UL,UR}:

            G
           / \\
          P   U
         / \\ / \\
        L  R UL UR

Both T_LR and T_{UL,UR} are projected Wald χ² statistics at the *same*
tree depth, so they experience the same post-selection inflation.
Their ratio follows an F distribution:

    F = (T_LR / k_LR) / (T_{UL,UR} / k_{UL,UR})  ~  F(k_LR, k_{UL,UR})

Under the null (L = R = homogeneous), the inflation factor c cancels:

    T_LR ~ c · χ²(k_LR),  T_{UL,UR} ~ c · χ²(k_{UL,UR})
    ⟹  F ~ F(k_LR, k_{UL,UR})   regardless of c

Empirical calibration: 5.0% rejection under null, KS p = 0.98 (n=200, p=50).

Fallback
--------
When the cousin reference is unavailable (uncle is a leaf, uncle has non-binary
children, or node is root's child), falls back to the standard Wald χ² test
with a flag indicating the fallback was used.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import initialize_sibling_divergence_columns

from ..branch_length_utils import compute_mean_branch_length
from ..multiple_testing import benjamini_hochberg_correction
from .sibling_divergence_test import (
    _either_child_significant,
    _get_binary_children,
    _get_sibling_data,
    sibling_divergence_test,
)

logger = logging.getLogger(__name__)

# Core: Compute projected Wald χ² for a sibling pair
# =============================================================================


def _compute_sibling_stat(
    left_dist: np.ndarray,
    right_dist: np.ndarray,
    n_left: int,
    n_right: int,
    branch_length_left: float | None,
    branch_length_right: float | None,
    mean_branch_length: float | None,
    test_id: str,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
) -> Tuple[float, int, float]:
    """Compute projected Wald χ² statistic for a sibling pair.

    Returns (statistic, degrees_of_freedom, p_value).
    """
    result = sibling_divergence_test(
        left_dist,
        right_dist,
        float(n_left),
        float(n_right),
        branch_length_left=branch_length_left,
        branch_length_right=branch_length_right,
        mean_branch_length=mean_branch_length,
        test_id=test_id,
        spectral_k=spectral_k,
        pca_projection=pca_projection,
    )
    return result  # (stat, df, pval)


# =============================================================================
# Uncle / Cousin lookup
# =============================================================================


def _get_uncle(tree: nx.DiGraph, parent: str) -> Tuple[Optional[str], Optional[str]]:
    """Find the uncle node (parent's sibling) and grandparent.

    Returns (grandparent, uncle) or (None, None) if unavailable.
    """
    predecessors = list(tree.predecessors(parent))
    if not predecessors:
        return None, None

    grandparent = predecessors[0]
    gp_children = list(tree.successors(grandparent))
    if len(gp_children) != 2:
        return None, None

    uncle = gp_children[0] if gp_children[1] == parent else gp_children[1]
    return grandparent, uncle


def _get_cousin_reference(
    tree: nx.DiGraph,
    uncle: str,
    mean_branch_length: float | None,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
) -> Tuple[float, int, bool]:
    """Compute the cousin-level reference statistic T_{UL,UR}.

    Returns (stat, df, valid). If the uncle doesn't have binary children
    or another issue prevents testing, returns (nan, nan, False).
    """
    uncle_children = _get_binary_children(tree, uncle)
    if uncle_children is None:
        return np.nan, 0, False

    ul, ur = uncle_children
    ul_dist, ur_dist, n_ul, n_ur, bl_ul, bl_ur = _get_sibling_data(tree, uncle, ul, ur)

    if n_ul < 2 or n_ur < 2:
        return np.nan, 0, False

    stat, df, pval = _compute_sibling_stat(
        ul_dist,
        ur_dist,
        n_ul,
        n_ur,
        bl_ul,
        bl_ur,
        mean_branch_length,
        test_id=f"cousin_ref:{uncle}",
        spectral_k=spectral_k,
        pca_projection=pca_projection,
    )

    if not np.isfinite(stat) or stat <= 0:
        return np.nan, 0, False

    return stat, int(df), True


# =============================================================================
# Cousin F-test
# =============================================================================


def cousin_ftest(
    tree: nx.DiGraph,
    parent: str,
    left: str,
    right: str,
    mean_branch_length: float | None,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
) -> Tuple[float, float, float, bool]:
    """Cousin-calibrated F-test for sibling divergence.

    Parameters
    ----------
    tree
        Hierarchical tree with distributions and branch lengths.
    parent
        The parent node whose children are being tested.
    left, right
        The two children of parent.
    mean_branch_length
        Mean branch length for Felsenstein normalization.

    Returns
    -------
    stat : float
        F-statistic (or Wald χ² if fallback).
    df : float
        Degrees of freedom (or tuple encoded as float for F-test).
    pval : float
        p-value.
    used_ftest : bool
        True if cousin F-test was used, False if fell back to Wald.
    """
    # Compute T_LR (sibling stat at parent)
    left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(tree, parent, left, right)

    stat_lr, k_lr, pval_lr = _compute_sibling_stat(
        left_dist,
        right_dist,
        n_l,
        n_r,
        bl_l,
        bl_r,
        mean_branch_length,
        test_id=f"sibling:{parent}",
        spectral_k=spectral_k,
        pca_projection=pca_projection,
    )

    if not np.isfinite(stat_lr):
        return np.nan, np.nan, np.nan, False

    # Find uncle
    grandparent, uncle = _get_uncle(tree, parent)
    if uncle is None:
        # Root's children — no uncle available, fallback to Wald
        logger.debug(
            "No uncle available for %s (root child); falling back to Wald χ²",
            parent,
        )
        return stat_lr, k_lr, pval_lr, False

    # Get cousin reference T_{UL,UR}
    # Use uncle's spectral info if available
    uncle_spectral_k = spectral_dims.get(uncle) if spectral_dims else None
    uncle_pca_proj = pca_projections.get(uncle) if pca_projections else None
    stat_uu, k_uu, valid = _get_cousin_reference(
        tree,
        uncle,
        mean_branch_length,
        spectral_k=uncle_spectral_k,
        pca_projection=uncle_pca_proj,
    )

    if not valid:
        # Uncle is a leaf or has non-binary children — fallback to Wald
        logger.debug(
            "Uncle %s has no binary children for %s; falling back to Wald χ²",
            uncle,
            parent,
        )
        return stat_lr, k_lr, pval_lr, False

    # F-test: ratio of mean chi-square values
    f_stat = (stat_lr / k_lr) / (stat_uu / k_uu)
    f_pval = float(f_dist.sf(f_stat, dfn=k_lr, dfd=k_uu))

    return f_stat, float(k_lr), f_pval, True


# =============================================================================
# Annotation pipeline (mirrors annotate_sibling_divergence)
# =============================================================================


def _collect_test_arguments_cousin(
    tree: nx.DiGraph,
    nodes_df: pd.DataFrame,
) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """Collect sibling pairs eligible for testing.

    Returns (parent_nodes, child_pairs, skipped_nodes).
    """
    if "Child_Parent_Divergence_Significant" not in nodes_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. " "Run child-parent test first."
        )

    sig_map = nodes_df["Child_Parent_Divergence_Significant"].to_dict()

    parents: List[str] = []
    child_pairs: List[Tuple[str, str]] = []
    skipped: List[str] = []

    for parent in tree.nodes:
        children = _get_binary_children(tree, parent)
        if children is None:
            continue

        left, right = children

        # Skip if neither child diverged from parent (same gating as current)
        if not _either_child_significant(left, right, sig_map):
            skipped.append(parent)
            continue

        parents.append(parent)
        child_pairs.append((left, right))

    return parents, child_pairs, skipped


def _run_cousin_tests(
    tree: nx.DiGraph,
    parents: List[str],
    child_pairs: List[Tuple[str, str]],
    mean_branch_length: float | None,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
) -> Tuple[List[Tuple[float, float, float]], List[bool]]:
    """Execute cousin F-tests for all collected pairs.

    Returns (results, used_ftest_flags).
    """
    results = []
    ftest_flags = []

    for parent, (left, right) in zip(parents, child_pairs, strict=False):
        # Look up spectral info for this parent
        _spectral_k = spectral_dims.get(parent) if spectral_dims else None
        _pca_proj = pca_projections.get(parent) if pca_projections else None
        stat, df, pval, used_ftest = cousin_ftest(
            tree,
            parent,
            left,
            right,
            mean_branch_length,
            spectral_k=_spectral_k,
            pca_projection=_pca_proj,
            spectral_dims=spectral_dims,
            pca_projections=pca_projections,
        )
        results.append((stat, df, pval))
        ftest_flags.append(used_ftest)

    return results, ftest_flags


def _apply_results_cousin(
    df: pd.DataFrame,
    parents: List[str],
    results: List[Tuple[float, float, float]],
    ftest_flags: List[bool],
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
            "Cousin F-test audit: total_tests=%d, invalid_tests=%d. "
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
    df.loc[parents, "Sibling_Test_Method"] = [
        "cousin_ftest" if f else "wald_fallback" for f in ftest_flags
    ]

    return df


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence_cousin(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Test sibling divergence using cousin-calibrated F-test.

    This is a drop-in replacement for ``annotate_sibling_divergence`` that uses
    the cousin F-test for calibrated p-values. Falls back to standard Wald χ²
    when the cousin reference is unavailable (uncle is leaf or root's children).

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
        Updated with sibling divergence columns (same schema as standard test)
        plus ``Sibling_Test_Method`` column indicating "cousin_ftest" or
        "wald_fallback" per node.
    """
    if len(nodes_statistics_dataframe) == 0:
        raise ValueError("Empty dataframe")

    df = nodes_statistics_dataframe.copy()
    df = initialize_sibling_divergence_columns(df)

    parents, child_pairs, skipped = _collect_test_arguments_cousin(tree, df)

    if not parents:
        warnings.warn("No eligible parent nodes for sibling tests", UserWarning)
        return df

    if skipped:
        df.loc[skipped, "Sibling_Divergence_Skipped"] = True
        logger.debug("Skipped %d nodes (no child-parent signal)", len(skipped))

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    results, ftest_flags = _run_cousin_tests(
        tree,
        parents,
        child_pairs,
        mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )

    n_ftest = sum(ftest_flags)
    n_fallback = len(ftest_flags) - n_ftest
    logger.info(
        "Cousin F-test: %d/%d nodes used F-test, %d fell back to Wald",
        n_ftest,
        len(parents),
        n_fallback,
    )

    df = _apply_results_cousin(df, parents, results, ftest_flags, significance_level_alpha)

    ftest_count = sum(ftest_flags)
    fallback_count = len(ftest_flags) - ftest_count
    invalid_count = int(df.loc[parents, "Sibling_Divergence_Invalid"].sum())

    df.attrs["sibling_divergence_audit"] = {
        "total_tests": len(parents),
        "invalid_tests": invalid_count,
        "conservative_path_tests": invalid_count,
        "cousin_ftest_count": ftest_count,
        "wald_fallback_count": fallback_count,
        "test_method": "cousin_ftest",
    }

    return df


__all__ = ["annotate_sibling_divergence_cousin", "cousin_ftest"]
