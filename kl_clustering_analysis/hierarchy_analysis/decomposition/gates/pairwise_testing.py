"""Pairwise divergence testing between tree nodes and cluster roots.

This module provides stateless functions that compare two nodes (or two
cluster roots) in a :class:`~tree.poset_tree.PosetTree` using the projected
Wald chi-square sibling divergence test, with optional Felsenstein branch-
length adjustment and calibration-model deflation.

These functions are used as callbacks by:

* **Signal localization** — ``test_node_pair_divergence`` compares arbitrary
  subtree pairs during recursive cross-boundary testing.
* **Post-hoc merge** — ``test_cluster_pair_divergence`` compares cluster roots
  across sibling boundaries, deflating by the calibration model used during
  annotation so that merge and split decisions are on the same scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from kl_clustering_analysis.tree.poset_tree import PosetTree

from ...statistics.branch_length_utils import sanitize_positive_branch_length
from ...statistics.sibling_divergence import (
    CalibrationModel,
    WeightedCalibrationModel,
    predict_inflation_factor,
    predict_weighted_inflation_factor,
)
from ...statistics.sibling_divergence.sibling_divergence_test import sibling_divergence_test


@dataclass(frozen=True)
class BranchDistanceCache:
    """Precomputed ancestry and root-distance data for O(1)-ish queries."""

    root: str
    root_distance: Dict[str, float | None]
    entry_time: Dict[str, int]
    exit_time: Dict[str, int]


def build_branch_distance_cache(
    tree: "PosetTree",
    *,
    root: str | None = None,
    distance_mode: str = "phylogeny",
) -> BranchDistanceCache | None:
    """Build branch-distance cache for a rooted tree.

    Distances are cumulative from root:
    - ``distance_mode="phylogeny"`` uses finite positive ``branch_length`` edge
      attributes. If any edge on the root->node path is missing/invalid, that
      node receives ``None`` distance.
    - ``distance_mode="topology"`` uses unit edge lengths (hop distance), so
      every root-reachable node gets a finite distance.
    """
    if distance_mode not in {"phylogeny", "topology"}:
        raise ValueError(
            "distance_mode must be 'phylogeny' or 'topology'. "
            f"Got {distance_mode!r}."
        )

    if root is None:
        root = tree.graph.get("root")
    if root is None:
        roots = [n for n, d in tree.in_degree() if d == 0]
        if len(roots) != 1:
            return None
        root = roots[0]
    if root not in tree:
        return None

    root_distance: Dict[str, float | None] = {root: 0.0}
    entry_time: Dict[str, int] = {}
    exit_time: Dict[str, int] = {}
    time_counter = 0

    # Iterative DFS to compute Euler tour timestamps and root distances.
    stack: list[tuple[str, bool]] = [(root, False)]
    while stack:
        node, exiting = stack.pop()
        if not exiting:
            if node in entry_time:
                continue
            entry_time[node] = time_counter
            time_counter += 1
            stack.append((node, True))
            children = list(tree.successors(node))
            for child in reversed(children):
                if child in root_distance:
                    continue
                parent_dist = root_distance.get(node)
                if parent_dist is None:
                    root_distance[child] = None
                else:
                    if distance_mode == "topology":
                        edge_bl = 1.0
                    else:
                        edge_bl = sanitize_positive_branch_length(
                            tree.edges[node, child].get("branch_length")
                        )
                    root_distance[child] = None if edge_bl is None else parent_dist + edge_bl
                stack.append((child, False))
        else:
            exit_time[node] = time_counter
            time_counter += 1

    # Disconnected/non-root-reachable nodes are marked unavailable.
    for node in tree:
        if node not in root_distance:
            root_distance[node] = None
        entry_time.setdefault(node, -1)
        exit_time.setdefault(node, -1)

    return BranchDistanceCache(
        root=root,
        root_distance=root_distance,
        entry_time=entry_time,
        exit_time=exit_time,
    )


def _is_ancestor_cached(cache: BranchDistanceCache, ancestor: str, node: str) -> bool:
    """Check ancestor relation in O(1) using Euler tour timestamps."""
    in_ancestor = cache.entry_time.get(ancestor, -1)
    in_node = cache.entry_time.get(node, -1)
    out_ancestor = cache.exit_time.get(ancestor, -1)
    if in_ancestor < 0 or in_node < 0 or out_ancestor < 0:
        return False
    return in_ancestor <= in_node <= out_ancestor


def _cached_branch_distance_from_ancestor(
    cache: BranchDistanceCache,
    ancestor: str,
    node: str,
) -> float | None:
    """Lookup ancestor->node distance via cumulative root distances."""
    if ancestor == node:
        return None
    if not _is_ancestor_cached(cache, ancestor, node):
        return None

    ancestor_dist = cache.root_distance.get(ancestor)
    node_dist = cache.root_distance.get(node)
    if ancestor_dist is None or node_dist is None:
        return None

    return sanitize_positive_branch_length(node_dist - ancestor_dist)


def _strict_branch_distance_from_ancestor(
    tree: "PosetTree",
    ancestor: str,
    node: str,
) -> float | None:
    """Sum validated ``branch_length`` values along ancestor -> node path.

    This function intentionally avoids weighted shortest-path calls because
    NetworkX treats missing edge weights as 1.0 when ``weight="branch_length"``
    is provided. Here, every traversed edge must carry a finite positive
    ``branch_length``; otherwise ``None`` is returned so branch-length scaling
    is disabled for that comparison.
    """
    if ancestor not in tree or node not in tree:
        return None
    if ancestor == node:
        return None

    total = 0.0
    current = node

    while current != ancestor:
        parents = list(tree.predecessors(current))
        if len(parents) != 1:
            return None

        parent = parents[0]
        branch_length = sanitize_positive_branch_length(
            tree.edges[parent, current].get("branch_length")
        )
        if branch_length is None:
            return None

        total += branch_length
        current = parent

    return total if total > 0 else None


def _branch_distance_from_ancestor(
    tree: "PosetTree",
    ancestor: str,
    node: str,
    branch_distance_cache: BranchDistanceCache | None,
) -> float | None:
    """Get ancestor->node branch distance using cache when available."""
    if branch_distance_cache is not None:
        return _cached_branch_distance_from_ancestor(branch_distance_cache, ancestor, node)
    return _strict_branch_distance_from_ancestor(tree, ancestor, node)


def _deflate_by_calibration(
    test_stat: float,
    degrees_of_freedom: float,
    p_value: float,
    *,
    bl_a: float | None,
    bl_b: float | None,
    tree: "PosetTree",
    node_a: str,
    node_b: str,
    calibration_model: Union[CalibrationModel, WeightedCalibrationModel],
    ancestor_override: str | None = None,
) -> Tuple[float, float, float]:
    """Deflate a raw Wald statistic by the calibration inflation factor.

    Parameters
    ----------
    test_stat, degrees_of_freedom, p_value
        Raw test results.
    bl_a, bl_b
        Branch lengths from the two nodes to their common ancestor.
    tree
        Tree providing ``leaf_count`` node attributes.
    node_a, node_b
        Nodes being compared (used to find the ancestor sample size).
    calibration_model
        Fitted calibration model.
    ancestor_override
        If given, use this node's leaf count as ``n_ancestor`` instead
        of computing the LCA.  Used by ``test_cluster_pair_divergence``
        where the ancestor is already known.

    Returns
    -------
    Tuple[float, float, float]
        ``(deflated_test_stat, degrees_of_freedom, deflated_p_value)``
    """
    bl_sum = 0.0
    if bl_a is not None:
        bl_sum += bl_a
    if bl_b is not None:
        bl_sum += bl_b

    if ancestor_override is not None:
        n_ancestor = int(tree.nodes[ancestor_override]["leaf_count"])
    else:
        lca = tree.find_lca(node_a, node_b)
        n_ancestor = int(tree.nodes[lca]["leaf_count"])

    if isinstance(calibration_model, WeightedCalibrationModel):
        c_hat = predict_weighted_inflation_factor(calibration_model, bl_sum, n_ancestor)
    else:
        c_hat = predict_inflation_factor(calibration_model, bl_sum, n_ancestor)

    if c_hat > 0 and np.isfinite(test_stat) and degrees_of_freedom > 0:
        from scipy.stats import chi2 as chi2_dist

        test_stat = test_stat / c_hat
        p_value = float(chi2_dist.sf(test_stat, df=degrees_of_freedom))

    return test_stat, degrees_of_freedom, p_value


def _compute_branch_lengths_to_lca(
    tree: "PosetTree",
    node_a: str,
    node_b: str,
    mean_branch_length: float | None,
    branch_distance_cache: BranchDistanceCache | None = None,
) -> Tuple[float | None, float | None]:
    """Compute patristic distances from two nodes to their LCA.

    Only computed when the tree actually has branch-length annotations
    (``mean_branch_length is not None``).  NetworkX defaults missing
    ``weight`` attributes to 1 (hop count), which would be inconsistent
    with a ``None`` mean branch length.

    Parameters
    ----------
    tree
        The hierarchy tree.
    node_a, node_b
        Nodes whose distances to the LCA are needed.
    mean_branch_length
        Pre-computed mean branch length for the tree, or ``None`` if the
        tree has no branch-length annotations.

    Returns
    -------
    Tuple[float | None, float | None]
        ``(dist_a, dist_b)`` — patristic distances, or ``(None, None)``
        when branch lengths are unavailable.
    """
    if mean_branch_length is None:
        return None, None

    lca = tree.find_lca(node_a, node_b)
    dist_a = _branch_distance_from_ancestor(tree, lca, node_a, branch_distance_cache)
    dist_b = _branch_distance_from_ancestor(tree, lca, node_b, branch_distance_cache)
    return dist_a, dist_b


def _compute_branch_lengths_from_ancestor(
    tree: "PosetTree",
    node_a: str,
    node_b: str,
    common_ancestor: str,
    mean_branch_length: float | None,
    branch_distance_cache: BranchDistanceCache | None = None,
) -> Tuple[float | None, float | None]:
    """Compute patristic distances from two nodes to a known common ancestor.

    Same logic as :func:`_compute_branch_lengths_to_lca` but the ancestor
    is provided explicitly (e.g. the sibling-boundary node found by the
    post-hoc merge traversal).

    Parameters
    ----------
    tree
        The hierarchy tree.
    node_a, node_b
        Nodes whose distances to *common_ancestor* are needed.
    common_ancestor
        A known common ancestor of *node_a* and *node_b*.
    mean_branch_length
        Pre-computed mean branch length for the tree, or ``None``.

    Returns
    -------
    Tuple[float | None, float | None]
        ``(dist_a, dist_b)`` — patristic distances, or ``(None, None)``
        when branch lengths are unavailable.
    """
    if mean_branch_length is None:
        return None, None

    dist_a = _branch_distance_from_ancestor(
        tree, common_ancestor, node_a, branch_distance_cache
    )
    dist_b = _branch_distance_from_ancestor(
        tree, common_ancestor, node_b, branch_distance_cache
    )
    return dist_a, dist_b


def test_node_pair_divergence(
    tree: "PosetTree",
    node_a: str,
    node_b: str,
    mean_branch_length: float | None,
    calibration_model: Union[CalibrationModel, WeightedCalibrationModel, None] = None,
    branch_distance_cache: BranchDistanceCache | None = None,
) -> Tuple[float, float, float]:
    """Test divergence between two arbitrary tree nodes.

    Extracts distributions and sample sizes from the tree, computes
    patristic distances to the LCA for Felsenstein adjustment, and
    delegates to :func:`sibling_divergence_test`.

    When a *calibration_model* is provided the raw Wald statistic is
    deflated by the predicted inflation factor ĉ (same model that was
    used during the annotation phase).  This ensures that the signal-
    localization sub-tests in v2 operate on the **same statistical
    scale** as the Gate 3 decision that triggered localization.

    Parameters
    ----------
    tree
        The hierarchy tree with pre-populated ``distribution`` and
        ``leaf_count`` node attributes.
    node_a, node_b
        Tree node identifiers to compare.
    mean_branch_length
        Pre-computed mean branch length for the tree, or ``None`` to
        disable Felsenstein adjustment.
    calibration_model
        Optional calibration model for deflating the raw Wald statistic.
        When provided, the test statistic is divided by the predicted
        inflation factor ``ĉ`` and the p-value is recomputed.  This
        makes localization sub-tests commensurate with the calibrated
        Gate 3 sibling test.

    Returns
    -------
    Tuple[float, float, float]
        ``(test_statistic, degrees_of_freedom, p_value)``
    """
    dist_a = np.asarray(tree.nodes[node_a]["distribution"], dtype=float)
    dist_b = np.asarray(tree.nodes[node_b]["distribution"], dtype=float)
    n_a = float(tree.nodes[node_a]["leaf_count"])
    n_b = float(tree.nodes[node_b]["leaf_count"])

    bl_a, bl_b = _compute_branch_lengths_to_lca(
        tree,
        node_a,
        node_b,
        mean_branch_length,
        branch_distance_cache=branch_distance_cache,
    )

    test_stat, degrees_of_freedom, p_value = sibling_divergence_test(
        left_dist=dist_a,
        right_dist=dist_b,
        n_left=n_a,
        n_right=n_b,
        branch_length_left=bl_a,
        branch_length_right=bl_b,
        mean_branch_length=mean_branch_length,
        test_id=f"nodepair:{node_a}|{node_b}",
    )

    # Deflate by the calibration model so that localization sub-tests
    # operate on the same scale as the Gate 3 sibling test.
    if calibration_model is not None:
        test_stat, degrees_of_freedom, p_value = _deflate_by_calibration(
            test_stat,
            degrees_of_freedom,
            p_value,
            bl_a=bl_a,
            bl_b=bl_b,
            tree=tree,
            node_a=node_a,
            node_b=node_b,
            calibration_model=calibration_model,
        )

    return test_stat, degrees_of_freedom, p_value


def test_cluster_pair_divergence(
    tree: "PosetTree",
    cluster_a: str,
    cluster_b: str,
    common_ancestor: str,
    mean_branch_length: float | None,
    calibration_model: Union[CalibrationModel, WeightedCalibrationModel, None] = None,
    branch_distance_cache: BranchDistanceCache | None = None,
) -> Tuple[float, float, float]:
    """Test whether two cluster roots are significantly different.

    Computes a Wald chi-square statistic (optionally after JL projection)
    from the standardized difference between the two cluster mean vectors,
    then optionally deflates by a calibration model so that the post-hoc
    merge test operates on the same scale as the decomposition.

    Parameters
    ----------
    tree
        The hierarchy tree with pre-populated ``distribution`` and
        ``leaf_count`` node attributes.
    cluster_a, cluster_b
        Cluster root node identifiers to compare.
    common_ancestor
        The lowest common ancestor in the tree (used for branch-length
        computation).
    mean_branch_length
        Pre-computed mean branch length for the tree, or ``None``.
    calibration_model
        Optional calibration model for deflating the raw Wald statistic.
        When provided, the test statistic is divided by the predicted
        inflation factor ``ĉ`` and the p-value is recomputed from the
        deflated statistic.

    Returns
    -------
    Tuple[float, float, float]
        ``(test_statistic, degrees_of_freedom, p_value)``
    """
    dist_a = np.asarray(tree.nodes[cluster_a]["distribution"], dtype=float)
    dist_b = np.asarray(tree.nodes[cluster_b]["distribution"], dtype=float)
    size_a = int(tree.nodes[cluster_a]["leaf_count"])
    size_b = int(tree.nodes[cluster_b]["leaf_count"])

    bl_a, bl_b = _compute_branch_lengths_from_ancestor(
        tree,
        cluster_a,
        cluster_b,
        common_ancestor,
        mean_branch_length,
        branch_distance_cache=branch_distance_cache,
    )

    test_stat, degrees_of_freedom, p_value = sibling_divergence_test(
        left_dist=dist_a,
        right_dist=dist_b,
        n_left=float(size_a),
        n_right=float(size_b),
        branch_length_left=bl_a,
        branch_length_right=bl_b,
        mean_branch_length=mean_branch_length,
        test_id=f"clusterpair:{cluster_a}|{cluster_b}|ancestor:{common_ancestor}",
    )

    # Deflate by the same calibration model used during annotation so that
    # the post-hoc merge operates on the same scale as the decomposition.
    if calibration_model is not None:
        test_stat, degrees_of_freedom, p_value = _deflate_by_calibration(
            test_stat,
            degrees_of_freedom,
            p_value,
            bl_a=bl_a,
            bl_b=bl_b,
            tree=tree,
            node_a=cluster_a,
            node_b=cluster_b,
            calibration_model=calibration_model,
            ancestor_override=common_ancestor,
        )

    return test_stat, degrees_of_freedom, p_value
