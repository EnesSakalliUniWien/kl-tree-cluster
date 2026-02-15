"""Tree-guided cousin calibration for the sibling divergence test.

Corrects the post-selection inflation of the Wald χ² statistic by
finding *topologically local* null-like calibration pairs for each
focal pair.  Instead of fitting a global regression model, this method
**walks the tree bidirectionally** from each focal node to find the nearest
null-like relatives and uses their observed inflation ratios.

Architecture
------------
1. **Compute all raw Wald stats** T_i, k_i for every eligible sibling pair
   (same as the standard pipeline).
2. **Index null-like pairs by tree node**: pairs where neither child is
   edge-significant.  For these, r_i = T_i / k_i estimates the pure
   post-selection inflation at that location.
3. **For each focal pair**, search bidirectionally:
   a. **Walk UP** from the focal parent through ancestors, inspecting uncle
      subtrees for null-like pairs.  Use the nearest ones found.
   b. **Walk DOWN** into the focal pair's own children's subtrees if the
      upward walk found nothing.  Descendants experienced similar selection
      pressure from the same tree construction process.
   c. **Across the root**: the upward walk naturally crosses to the other side
      of the tree when it reaches the root and searches its opposite subtree.
4. **Deflate focal pairs**: T_adj = T / ĉ, p = χ²_sf(T_adj, k).

Why bidirectional?
------------------
The post-selection inflation c is *local* — it depends on how the linkage
algorithm chose the split at that particular point in the tree.  Walking
only upward can fail when the focal pair is near the root or when the
opposite side of the tree has no null-like pairs.  The downward fallback
uses descendants that share similar selection pressure, providing a
calibration reference even when no uncle-side null pairs exist.

Fallback tiers
--------------
- Walk UP finds null-like pairs in uncle subtree → local median ĉ
- Walk DOWN finds null-like pairs in descendants → local median ĉ
- Neither direction finds any → global median ĉ
- <3 null-like pairs in entire tree → no calibration (raw Wald; flag warning)

Configuration
-------------
Toggle via ``config.SIBLING_TEST_METHOD = "cousin_tree_guided"``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    extract_node_sample_size,
    initialize_sibling_divergence_columns,
)

from ..branch_length_utils import compute_mean_branch_length
from ..multiple_testing import benjamini_hochberg_correction
from .sibling_divergence_test import (
    _either_child_significant,
    _get_binary_children,
    _get_sibling_data,
    sibling_divergence_test,
)

logger = logging.getLogger(__name__)

# Minimum null-like pairs for global fallback
_MIN_GLOBAL_MEDIAN = 3


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class _SiblingRecord:
    """Per–sibling-pair record for calibration pipeline."""

    parent: str
    left: str
    right: str
    stat: float  # raw Wald T
    df: int  # projection dimension k
    pval: float  # raw Wald p
    bl_sum: float  # branch-length sum (left + right)
    n_parent: int  # number of leaves under parent
    is_null_like: bool  # neither child edge-significant


# =============================================================================
# Tree-guided cousin search
# =============================================================================


def _build_null_index(
    records: List[_SiblingRecord],
) -> Dict[str, _SiblingRecord]:
    """Index null-like pairs by their parent node for O(1) lookup.

    Returns a dict mapping parent node id → _SiblingRecord for that pair.
    Only includes records with valid (finite, positive) T/k ratios.
    """
    null_index: Dict[str, _SiblingRecord] = {}
    for rec in records:
        if rec.is_null_like and np.isfinite(rec.stat) and rec.df > 0 and rec.stat > 0:
            null_index[rec.parent] = rec
    return null_index


def _collect_null_ratios_in_subtree(
    tree: nx.DiGraph,
    subtree_root: str,
    null_index: Dict[str, _SiblingRecord],
) -> List[float]:
    """Collect T/k ratios from null-like pairs within a subtree.

    Searches descendants of ``subtree_root`` (including subtree_root itself)
    for nodes that appear in the null index.
    """
    ratios: List[float] = []

    # Check the subtree root itself
    if subtree_root in null_index:
        rec = null_index[subtree_root]
        ratios.append(rec.stat / rec.df)

    # Check all descendants
    for desc in nx.descendants(tree, subtree_root):
        if desc in null_index:
            rec = null_index[desc]
            ratios.append(rec.stat / rec.df)

    return ratios


def _find_nearest_null_cousin(
    tree: nx.DiGraph,
    focal_parent: str,
    null_index: Dict[str, _SiblingRecord],
) -> Tuple[List[float], int, str]:
    """Walk the tree bidirectionally from the focal parent to find nearest null-like relatives.

    Strategy — walk UP first, then DOWN:

    Phase 1 (upward):
    1. Start at the focal parent node.
    2. Move to its parent (grandparent of the focal pair's children).
    3. Look at the uncle (grandparent's other child) and search the uncle's
       entire subtree for null-like pairs.
    4. If found, return those ratios.
    5. If not found, continue walking up to great-grandparent, etc.
    6. Stop at the root.

    Phase 2 (downward — if upward found nothing):
    7. Search the focal pair's own children's subtrees for null-like pairs.
       These descendants experienced similar selection pressure from the same
       tree construction process.

    Parameters
    ----------
    tree
        The hierarchy tree.
    focal_parent
        The parent node of the focal sibling pair being calibrated.
    null_index
        Mapping from parent node → _SiblingRecord for null-like pairs.

    Returns
    -------
    Tuple[List[float], int, str]
        (ratios, levels_walked, direction) where ratios is T/k values from
        nearby null-like pairs, levels_walked is distance traversed, and
        direction is ``"up"`` or ``"down"`` indicating search direction.
    """
    # --- Phase 1: walk UP ---
    current = focal_parent
    levels = 0

    while True:
        predecessors = list(tree.predecessors(current))
        if not predecessors:
            # Reached the root without finding null-like cousins
            break

        ancestor = predecessors[0]
        levels += 1

        # Find the uncle(s) — ancestor's other children that are not on our path
        ancestor_children = list(tree.successors(ancestor))
        uncles = [c for c in ancestor_children if c != current]

        # Search each uncle's subtree for null-like pairs
        all_ratios: List[float] = []
        for uncle in uncles:
            ratios = _collect_null_ratios_in_subtree(tree, uncle, null_index)
            all_ratios.extend(ratios)

        if all_ratios:
            return all_ratios, levels, "up"

        # Also check if the ancestor itself is a null-like pair
        if ancestor in null_index:
            rec = null_index[ancestor]
            return [rec.stat / rec.df], levels, "up"

        # Continue walking up
        current = ancestor

    # --- Phase 2: walk DOWN (search own descendants) ---
    focal_children = list(tree.successors(focal_parent))
    down_ratios: List[float] = []
    for child in focal_children:
        # Search each child's subtree (but not the focal parent itself)
        ratios = _collect_null_ratios_in_subtree(tree, child, null_index)
        down_ratios.extend(ratios)

    if down_ratios:
        return down_ratios, 1, "down"

    return [], 0, "none"


def _compute_local_c_hat(
    tree: nx.DiGraph,
    focal_parent: str,
    null_index: Dict[str, _SiblingRecord],
    global_c_hat: float,
) -> Tuple[float, str]:
    """Compute the inflation factor ĉ for a focal pair using tree-guided search.

    Parameters
    ----------
    tree
        The hierarchy tree.
    focal_parent
        Parent node of the focal sibling pair.
    null_index
        Index of null-like pairs by parent node.
    global_c_hat
        Fallback global median from all null-like pairs.

    Returns
    -------
    Tuple[float, str]
        (c_hat, method) where method describes what calibration was used:
        "local_up_L{n}" for upward tree-guided (n = levels walked),
        "local_down" for descendant-based, "global_median"
        for fallback, "none" when no calibration is available.
    """
    ratios, levels, direction = _find_nearest_null_cousin(tree, focal_parent, null_index)

    if ratios:
        c_hat = float(np.median(ratios))
        # ĉ must be ≥ 1 (inflation, never deflation)
        c_hat = max(c_hat, 1.0)
        if direction == "up":
            return c_hat, f"local_up_L{levels}"
        else:
            return c_hat, "local_down"

    # Fallback: global median
    if global_c_hat > 0:
        return max(global_c_hat, 1.0), "global_median"

    return 1.0, "none"


# =============================================================================
# Pipeline: collect → index → deflate per-pair
# =============================================================================


def _collect_all_pairs(
    tree: nx.DiGraph,
    nodes_df: pd.DataFrame,
    mean_bl: float | None,
) -> List[_SiblingRecord]:
    """Collect ALL binary-child parent nodes and compute raw Wald stats."""
    if "Child_Parent_Divergence_Significant" not in nodes_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. " "Run child-parent test first."
        )

    sig_map = nodes_df["Child_Parent_Divergence_Significant"].to_dict()
    records: List[_SiblingRecord] = []

    for parent in tree.nodes:
        children = _get_binary_children(tree, parent)
        if children is None:
            continue

        left, right = children
        left_dist, right_dist, n_l, n_r, bl_l, bl_r = _get_sibling_data(tree, parent, left, right)

        # Compute raw Wald stat
        stat, df, pval = sibling_divergence_test(
            left_dist,
            right_dist,
            float(n_l),
            float(n_r),
            branch_length_left=bl_l,
            branch_length_right=bl_r,
            mean_branch_length=mean_bl,
            test_id=f"sibling:{parent}",
        )

        # Branch-length sum
        bl_sum = 0.0
        if bl_l is not None and bl_r is not None:
            bl_sum = bl_l + bl_r
        elif bl_l is not None:
            bl_sum = bl_l
        elif bl_r is not None:
            bl_sum = bl_r

        n_parent = extract_node_sample_size(tree, parent)

        # Is this a null-like pair?  Neither child edge-significant.
        is_null = not _either_child_significant(left, right, sig_map)

        records.append(
            _SiblingRecord(
                parent=parent,
                left=left,
                right=right,
                stat=stat,
                df=int(df) if np.isfinite(df) else 0,
                pval=pval,
                bl_sum=bl_sum,
                n_parent=n_parent,
                is_null_like=is_null,
            )
        )

    return records


def _deflate_and_test(
    records: List[_SiblingRecord],
    tree: nx.DiGraph,
    null_index: Dict[str, _SiblingRecord],
    global_c_hat: float,
) -> Tuple[List[str], List[Tuple[float, float, float]], List[str]]:
    """Deflate focal pairs using tree-guided cousin search.

    For each focal pair, walks the tree to find the nearest null-like
    relatives and uses their median T/k as the deflation factor.

    Returns
    -------
    Tuple[List[str], List[Tuple[float, float, float]], List[str]]
        (focal_parents, focal_results, calibration_methods)
    """
    focal_parents: List[str] = []
    focal_results: List[Tuple[float, float, float]] = []
    methods: List[str] = []

    for rec in records:
        if rec.is_null_like:
            continue  # null-like pairs are skipped (treated as no-split)

        if not np.isfinite(rec.stat) or rec.df <= 0:
            focal_parents.append(rec.parent)
            focal_results.append((np.nan, np.nan, np.nan))
            methods.append("invalid")
            continue

        c_hat, method = _compute_local_c_hat(
            tree,
            rec.parent,
            null_index,
            global_c_hat,
        )
        t_adj = rec.stat / c_hat
        p_adj = float(chi2.sf(t_adj, df=rec.df))

        focal_parents.append(rec.parent)
        focal_results.append((t_adj, float(rec.df), p_adj))
        methods.append(f"tree_guided_{method}")

    return focal_parents, focal_results, methods


def _apply_results(
    df: pd.DataFrame,
    focal_parents: List[str],
    focal_results: List[Tuple[float, float, float]],
    calibration_methods: List[str],
    skipped_parents: List[str],
    alpha: float,
) -> pd.DataFrame:
    """Apply deflated results with BH correction to DataFrame."""
    if skipped_parents:
        df.loc[skipped_parents, "Sibling_Divergence_Skipped"] = True

    if not focal_results:
        return df

    stats = np.array([r[0] for r in focal_results])
    dfs = np.array([r[1] for r in focal_results])
    pvals = np.array([r[2] for r in focal_results])

    invalid_mask = (~np.isfinite(stats)) | (~np.isfinite(dfs)) | (~np.isfinite(pvals))
    pvals_for_correction = np.where(np.isfinite(pvals), pvals, 1.0)

    reject, pvals_adj, _ = benjamini_hochberg_correction(pvals_for_correction, alpha=alpha)
    reject = np.where(invalid_mask, False, reject)

    n_invalid = int(np.sum(invalid_mask))
    if n_invalid:
        logger.warning(
            "Tree-guided cousin audit: total_tests=%d, invalid_tests=%d. "
            "Conservative correction path applied (p=1.0, reject=False).",
            len(focal_results),
            n_invalid,
        )

    df.loc[focal_parents, "Sibling_Test_Statistic"] = stats
    df.loc[focal_parents, "Sibling_Degrees_of_Freedom"] = dfs
    df.loc[focal_parents, "Sibling_Divergence_P_Value"] = pvals
    df.loc[focal_parents, "Sibling_Divergence_P_Value_Corrected"] = pvals_adj
    df.loc[focal_parents, "Sibling_Divergence_Invalid"] = invalid_mask
    df.loc[focal_parents, "Sibling_BH_Different"] = reject
    df.loc[focal_parents, "Sibling_BH_Same"] = ~reject
    df.loc[focal_parents, "Sibling_Test_Method"] = calibration_methods

    return df


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence_tree_guided(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
) -> pd.DataFrame:
    """Test sibling divergence using tree-guided cousin calibration.

    For each focal sibling pair (at least one child edge-significant),
    walks up the tree to find the nearest null-like relatives (neither
    child edge-significant) and uses their observed T/k ratios to
    estimate the local post-selection inflation factor ĉ.

    This produces per-node calibration that adapts to local tree
    structure, rather than fitting a single global regression.

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
        plus ``Sibling_Test_Method`` column indicating calibration source.
    """
    if len(nodes_statistics_dataframe) == 0:
        raise ValueError("Empty dataframe")

    df = nodes_statistics_dataframe.copy()
    df = initialize_sibling_divergence_columns(df)

    mean_bl = compute_mean_branch_length(tree)

    # Pass 1: compute ALL raw Wald stats
    records = _collect_all_pairs(tree, df, mean_bl)

    if not records:
        warnings.warn("No eligible parent nodes for sibling tests", UserWarning)
        return df

    n_null = sum(1 for r in records if r.is_null_like)
    n_focal = sum(1 for r in records if not r.is_null_like)
    logger.info(
        "Tree-guided cousin: %d total pairs (%d null-like, %d focal).",
        len(records),
        n_null,
        n_focal,
    )

    # Pass 2: build null-like index and compute global fallback
    null_index = _build_null_index(records)
    null_ratios = [rec.stat / rec.df for rec in null_index.values()]

    if len(null_ratios) < _MIN_GLOBAL_MEDIAN:
        logger.warning(
            "Tree-guided cousin: only %d null-like pairs (need ≥%d) — "
            "raw Wald stats will be used (ĉ = 1.0).",
            len(null_ratios),
            _MIN_GLOBAL_MEDIAN,
        )
        global_c_hat = 1.0
    else:
        global_c_hat = float(np.median(null_ratios))
        logger.info(
            "Tree-guided cousin: global median ĉ = %.3f from %d null-like pairs "
            "(used as fallback only).",
            global_c_hat,
            len(null_ratios),
        )

    # Pass 3: deflate focals using tree-guided search
    focal_parents, focal_results, cal_methods = _deflate_and_test(
        records,
        tree,
        null_index,
        global_c_hat,
    )

    # Null-like parents are skipped (they are noise splits)
    skipped_parents = [r.parent for r in records if r.is_null_like]

    df = _apply_results(
        df,
        focal_parents,
        focal_results,
        cal_methods,
        skipped_parents,
        significance_level_alpha,
    )

    # Audit metadata
    # Count how many focal pairs used local-up, local-down, global, or none
    n_local_up = sum(1 for m in cal_methods if m.startswith("tree_guided_local_up"))
    n_local_down = sum(1 for m in cal_methods if m == "tree_guided_local_down")
    n_global_fallback = sum(1 for m in cal_methods if m == "tree_guided_global_median")
    n_none = sum(1 for m in cal_methods if m == "tree_guided_none")
    n_invalid = sum(1 for m in cal_methods if m == "invalid")

    df.attrs["sibling_divergence_audit"] = {
        "total_pairs": len(records),
        "null_like_pairs": n_null,
        "focal_pairs": n_focal,
        "calibration_local_up": n_local_up,
        "calibration_local_down": n_local_down,
        "calibration_global_fallback": n_global_fallback,
        "calibration_none": n_none,
        "calibration_invalid": n_invalid,
        "global_c_hat": global_c_hat,
        "null_index_size": len(null_index),
        "test_method": "cousin_tree_guided",
    }

    return df


__all__ = ["annotate_sibling_divergence_tree_guided"]
