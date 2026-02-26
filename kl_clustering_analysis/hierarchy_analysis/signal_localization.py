"""Signal localization for finding where divergence originates in a tree.

This module implements **depth-1 cross-boundary testing** to identify
which immediate subtrees are truly different vs. which could be merged.

The key insight is that when two sibling subtrees are "significantly different"
at the aggregate level, the difference may be localized to specific
sub-branches. By expanding one level into the immediate children and testing
all cross-boundary pairs, we can identify:
- Which child-level sub-clusters are truly different (hard boundaries)
- Which child-level sub-clusters are similar despite being on opposite sides

Every cross-product test is **terminal**: its result (similar or different)
is recorded directly with no further drilling.  Finer structure within each
child is handled by the main DFS traversal when it visits that child node.

This depth-1 design avoids two failure modes of recursive drill-down:
1. **Power collapse** — recursive expansion reaches individual leaves
   (n=1 tests) that have zero discriminative power, producing false
   similarity edges across true cluster boundaries.
2. **Multi-level overlap** — nodes from different tree depths enter the
   same Union-Find graph, creating overlapping leaf sets that fragment
   the partition.

References
----------
- Goeman & Mansmann (2008). "Multiple testing on the directed acyclic graph of GO"
- Yekutieli (2008). "Hierarchical False Discovery Rate-Controlling Methodology"
- Meinshausen (2008). "Hierarchical testing of variable importance"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from .statistics.multiple_testing import benjamini_hochberg_correction

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SimilarityEdge:
    """An edge indicating two nodes are NOT significantly different."""

    node_a: str
    node_b: str
    p_value: float
    test_statistic: float = 0.0
    degrees_of_freedom: float = 0.0

    def __hash__(self) -> int:
        return hash(frozenset([self.node_a, self.node_b]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimilarityEdge):
            return False
        return frozenset([self.node_a, self.node_b]) == frozenset([other.node_a, other.node_b])


@dataclass
class LocalizationResult:
    """Result of signal localization between two subtrees.

    Attributes
    ----------
    left_root : str
        Root node of the left subtree.
    right_root : str
        Root node of the right subtree.
    aggregate_p_value : float
        P-value from testing left_root vs right_root directly.
    aggregate_significant : bool
        Whether the aggregate test was significant.
    similarity_edges : List[SimilarityEdge]
        Edges connecting nodes that are NOT significantly different.
        These represent potential cross-boundary merges.
    difference_pairs : List[Tuple[str, str]]
        Pairs of nodes that ARE significantly different.
    nodes_tested : int
        Total number of pairwise tests performed.
    depth_reached : int
        Maximum recursion depth reached during localization.
    """

    left_root: str
    right_root: str
    aggregate_p_value: float
    aggregate_significant: bool
    similarity_edges: List[SimilarityEdge] = field(default_factory=list)
    difference_pairs: List[Tuple[str, str]] = field(default_factory=list)
    nodes_tested: int = 0
    depth_reached: int = 0

    @property
    def has_soft_boundaries(self) -> bool:
        """True if there are cross-boundary similarities (potential merges)."""
        return len(self.similarity_edges) > 0

    @property
    def all_different(self) -> bool:
        """True if all tested pairs are significantly different."""
        return len(self.similarity_edges) == 0 and self.aggregate_significant

    def get_similarity_graph(self) -> nx.Graph:
        """Build a graph where edges indicate non-significant differences."""
        G = nx.Graph()
        for edge in self.similarity_edges:
            G.add_edge(
                edge.node_a,
                edge.node_b,
                p_value=edge.p_value,
                test_statistic=edge.test_statistic,
            )
        return G


# =============================================================================
# Helper Functions
# =============================================================================


def _get_children(tree: nx.DiGraph, node: str) -> List[str]:
    """Get immediate children of a node."""
    return list(tree.successors(node))


def _is_leaf(tree: nx.DiGraph, node: str) -> bool:
    """Check if a node is a leaf (no children)."""
    return tree.out_degree(node) == 0


# =============================================================================
# Core Localization Algorithm
# =============================================================================


def localize_divergence_signal(
    tree: nx.DiGraph,
    left_root: str,
    right_root: str,
    test_divergence: Callable[[str, str], Tuple[float, float, float]],
    alpha: float = 0.05,
    max_depth: int | None = None,
    apply_fdr: bool = True,
    is_edge_significant: Callable[[str], bool] | None = None,
    max_pairs: int | None = None,
) -> LocalizationResult:
    """Localize divergence between two subtrees via depth-1 cross-product tests.

    When Gate 3 declares that ``left_root`` and ``right_root`` are
    significantly different at the aggregate level, this function asks
    *where* the difference lives by expanding **one level** into the
    immediate children of each root and testing all cross-boundary pairs.

    Every cross-product test is **terminal** — its result (similar or
    different) is recorded directly.  No further drilling occurs.  This
    avoids the power collapse that happens when recursive drill-down
    reaches individual leaves (``n = 1`` tests), and eliminates the
    multi-level overlap problem in the Union-Find graph.

    The internal structure of each child node is handled separately by
    the main DFS traversal when it visits that node — localization only
    determines *which sub-parts of the left subtree relate to which
    sub-parts of the right subtree*.

    Parameters
    ----------
    tree
        Directed hierarchy.
    left_root, right_root
        Roots of the two subtrees to compare.
    test_divergence
        ``(node_a, node_b) -> (stat, df, p_value)`` pairwise test callable.
    alpha
        Significance level for raw comparisons and optional BH correction.
    max_depth
        Accepted for API compatibility but not used (depth is always 1).
    apply_fdr
        Apply Benjamini-Hochberg correction to the collected terminal tests.
    is_edge_significant
        ``node_id -> bool`` callback (Gate 2).  Children whose edge to their
        parent is NOT significant are treated as atomic (not expanded).
        ``None`` treats all children as valid.
    max_pairs
        Accepted for API compatibility but not used.
    """
    # First, test at aggregate level
    agg_stat, agg_df, agg_pval = test_divergence(left_root, right_root)
    agg_significant = agg_pval < alpha

    result = LocalizationResult(
        left_root=left_root,
        right_root=right_root,
        aggregate_p_value=agg_pval,
        aggregate_significant=agg_significant,
        nodes_tested=1,
        depth_reached=0,
    )

    # If not significant at aggregate level, no need to localize
    if not agg_significant:
        result.similarity_edges.append(
            SimilarityEdge(
                node_a=left_root,
                node_b=right_root,
                p_value=agg_pval,
                test_statistic=agg_stat,
                degrees_of_freedom=agg_df,
            )
        )
        return result

    # ---- helpers ----

    def _get_valid_children(node: str) -> List[str]:
        """Get children whose edge to parent passes Gate 2."""
        children = _get_children(tree, node)
        if is_edge_significant is None:
            return children
        return [c for c in children if is_edge_significant(c)]

    # ---- depth-1 cross-product ----
    #
    # Expand each root into its immediate children (filtered by Gate 2).
    # If a root has no valid children it is kept as-is (atomic node).
    # Then test every (left_child, right_child) pair.  Each result is
    # terminal — no further drilling.

    left_children = _get_valid_children(left_root)
    right_children = _get_valid_children(right_root)
    left_nodes = left_children if left_children else [left_root]
    right_nodes = right_children if right_children else [right_root]

    all_test_results: List[Tuple[str, str, float, float, float]] = []

    for li in left_nodes:
        for ri in right_nodes:
            stat, df, pval = test_divergence(li, ri)
            result.nodes_tested += 1
            all_test_results.append((li, ri, stat, df, pval))

    result.depth_reached = 1 if all_test_results else 0

    if not all_test_results:
        return result

    # Apply FDR correction if requested
    p_values = np.array([r[4] for r in all_test_results])

    if apply_fdr and len(p_values) > 1:
        reject, p_adjusted, _ = benjamini_hochberg_correction(p_values, alpha=alpha)
    else:
        reject = p_values < alpha
        p_adjusted = p_values

    # Categorize results — every test is terminal
    for i, (l_node, r_node, stat, df, pval) in enumerate(all_test_results):
        if reject[i]:
            result.difference_pairs.append((l_node, r_node))
        else:
            result.similarity_edges.append(
                SimilarityEdge(
                    node_a=l_node,
                    node_b=r_node,
                    p_value=float(p_adjusted[i]),
                    test_statistic=stat,
                    degrees_of_freedom=df,
                )
            )

    return result


# =============================================================================
# Cluster Extraction from Similarity Graph
# =============================================================================


def merge_difference_graphs(
    localization_results: Dict[str, LocalizationResult],
) -> nx.Graph:
    """Merge all difference pairs into a single graph.

    Parameters
    ----------
    localization_results : Dict[str, LocalizationResult]
        Localization results from all split points.

    Returns
    -------
    nx.Graph
        Graph where edges indicate significant differences (Cannot-Link).
    """
    G = nx.Graph()

    for _, result in localization_results.items():
        for node_a, node_b in result.difference_pairs:
            G.add_edge(node_a, node_b)

    return G


def extract_constrained_clusters(
    similarity_graph: nx.Graph,
    difference_graph: nx.Graph,
    tree: nx.DiGraph,
    merge_points: List[str],
) -> List[Set[str]]:
    """Extract clusters using similarity edges constrained by difference pairs.

    This replaces simple connected components (Single Linkage) with a
    greedy agglomerative approach that respects "Cannot-Link" constraints.
    It prevents "Over-Merging" by ensuring that we do not merge clusters
    that contain significantly different members.

    Algorithm
    ---------
    1. Start with initial clusters (leaves of merge points + nodes in similarity/diff graphs).
    2. Sort similarity edges by p-value descending (most similar first).
    3. Iterate edges:
       - If endpoints are in different clusters:
         - Check for ANY difference edge between the two clusters.
         - If NO difference edge: Merge.
         - If difference edge exists: Skip (Conflict).

    Parameters
    ----------
    similarity_graph : nx.Graph
        Graph of similar node pairs (potential merges).
    difference_graph : nx.Graph
        Graph of different node pairs (forbidden merges).
    tree : nx.DiGraph
        The original hierarchy tree.
    merge_points : List[str]
        Nodes that were explicitly merged/not split in traversal.

    Returns
    -------
    List[Set[str]]
        List of leaf sets, each representing a cluster.
    """
    # 1. Initialize clusters
    # We must account for all nodes involved in ANY graph + merge points.
    # Prune merge_points that are ancestors of nodes already represented
    # in the similarity/difference graphs.  Those sub-tree leaves are
    # already tracked at finer granularity by the graph nodes; keeping
    # the ancestor would produce duplicate leaf assignments across
    # clusters ("lowest-node-wins" policy).
    graph_nodes = set(similarity_graph.nodes) | set(difference_graph.nodes)
    pruned_merge_points: List[str] = []
    for mp in merge_points:
        if mp in graph_nodes:
            # Node itself is in the graph — keep it there, skip as merge_point
            continue
        descendants_of_mp = nx.descendants(tree, mp) if not _is_leaf(tree, mp) else set()
        if descendants_of_mp & graph_nodes:
            # mp is an ancestor of at least one graph node → skip to avoid
            # leaf overlap.  The graph-node's leaves are already covered.
            continue
        pruned_merge_points.append(mp)

    all_nodes = graph_nodes | set(pruned_merge_points)

    # Map node_id -> set of leaf labels
    node_to_leaves: Dict[str, Set[str]] = {}
    for node in all_nodes:
        node_to_leaves[node] = _get_all_leaves(tree, node)

    # Initialize Disjoint Set (Union-Find) structure
    # We map each node_id to a cluster_id (using the representative node)
    parent = {n: n for n in all_nodes}

    def find(n: str) -> str:
        if parent[n] != n:
            parent[n] = find(parent[n])
        return parent[n]

    def union(n1: str, n2: str) -> None:
        root1 = find(n1)
        root2 = find(n2)
        if root1 != root2:
            parent[root2] = root1

    # Pre-compute cluster members for fast lookups during merge checks
    # Map root -> set of member nodes
    cluster_members: Dict[str, Set[str]] = {n: {n} for n in all_nodes}

    # 2. Sort similarity edges
    # We want to process the "most similar" pairs first.
    # High p-value = Fail to reject H0 = Similar.
    edges_with_p = []
    for u, v, data in similarity_graph.edges(data=True):
        p_val = data.get("p_value", 0.0)
        edges_with_p.append((u, v, p_val))

    # Sort descending by p-value
    edges_with_p.sort(key=lambda x: x[2], reverse=True)

    # 3. Greedy Merge with Constraints
    for u, v, p_val in edges_with_p:
        if u not in parent or v not in parent:
            # Should not happen given initialization
            continue

        root_u = find(u)
        root_v = find(v)

        if root_u == root_v:
            continue

        # Check for conflicts
        members_u = cluster_members[root_u]
        members_v = cluster_members[root_v]

        conflict_found = False
        # If ANY member of U has a difference edge to ANY member of V, abort.
        # Optimization: Check smaller set against larger set's adjacencies?
        # Difference graph is likely sparse.

        for m_u in members_u:
            if conflict_found:
                break
            # Neighbors of m_u in difference graph
            if m_u in difference_graph:
                for diff_neighbor in difference_graph.neighbors(m_u):
                    if diff_neighbor in members_v:
                        conflict_found = True
                        break

        if not conflict_found:
            # Merge
            # Update union-find
            union(root_u, root_v)

            # Merge membership sets
            # (If union logic swapped roots, we need to be careful, but here parent[root2]=root1)
            cluster_members[root_u].update(cluster_members[root_v])
            del cluster_members[root_v]

    # 4. Extract final leaf sets
    final_clusters: List[Set[str]] = []

    # Only iterate over active roots
    for root in cluster_members:
        # Collect all leaves from all members
        cluster_leaf_set = set()
        for member in cluster_members[root]:
            cluster_leaf_set.update(node_to_leaves[member])

        if cluster_leaf_set:
            final_clusters.append(cluster_leaf_set)

    # 5. Deduplicate overlapping leaves
    # Nodes at different tree depths (from different localization calls or
    # merge_points) can share descendant leaves.  When a leaf appears in
    # multiple clusters, assign it to the *smallest* cluster (most specific
    # / deepest in the hierarchy).  This is the "lowest-node-wins" policy.
    leaf_to_cluster: Dict[str, int] = {}  # leaf_label -> best cluster index
    for idx, cluster_set in enumerate(final_clusters):
        for leaf in cluster_set:
            if leaf not in leaf_to_cluster:
                leaf_to_cluster[leaf] = idx
            else:
                # Keep the cluster with fewer leaves (more specific)
                prev_idx = leaf_to_cluster[leaf]
                if len(final_clusters[idx]) < len(final_clusters[prev_idx]):
                    leaf_to_cluster[leaf] = idx

    # Rebuild clusters from the deduplicated mapping
    deduped: Dict[int, Set[str]] = {}
    for leaf, idx in leaf_to_cluster.items():
        deduped.setdefault(idx, set()).add(leaf)

    return [s for s in deduped.values() if s]


def _get_all_leaves(tree: nx.DiGraph, node: str) -> Set[str]:
    """Get all leaf labels under a given node.

    Returns the ``label`` attribute of each leaf (e.g. ``"sample_0"``),
    falling back to the node id only when no label is stored.  This
    ensures consistency with ``TreeDecomposition._get_all_leaves``
    which returns labels, not raw node ids.
    """
    if _is_leaf(tree, node):
        return {tree.nodes[node].get("label", node)}

    descendants = nx.descendants(tree, node)
    return {tree.nodes[d].get("label", d) for d in descendants if _is_leaf(tree, d)}


# =============================================================================
# Integration with TreeDecomposition
# =============================================================================


def build_cross_boundary_similarity(
    tree: nx.DiGraph,
    split_points: List[Tuple[str, str, str]],  # (parent, left_child, right_child)
    test_divergence: Callable[[str, str], Tuple[float, float, float]],
    alpha: float = 0.05,
    max_depth: int = 3,
    max_pairs: int | None = None,
) -> Dict[str, LocalizationResult]:
    """Build localization results for all split points in the tree.

    Parameters
    ----------
    tree : nx.DiGraph
        The hierarchy tree.
    split_points : List[Tuple[str, str, str]]
        List of (parent, left_child, right_child) tuples representing
        nodes where a split was made.
    test_divergence : Callable
        Function (node_a, node_b) -> (test_stat, df, p_value).
    alpha : float
        Significance level.
    max_depth : int
        Maximum recursion depth for localization.

    Returns
    -------
    Dict[str, LocalizationResult]
        Mapping from parent node to localization result.
    """
    results: Dict[str, LocalizationResult] = {}

    for parent, left_child, right_child in split_points:
        result = localize_divergence_signal(
            tree=tree,
            left_root=left_child,
            right_root=right_child,
            test_divergence=test_divergence,
            alpha=alpha,
            max_depth=max_depth,
            max_pairs=max_pairs,
        )
        results[parent] = result

        if result.has_soft_boundaries:
            logger.info(
                f"Found {len(result.similarity_edges)} soft boundary edges "
                f"at split point {parent}"
            )

    return results


def merge_similarity_graphs(
    localization_results: Dict[str, LocalizationResult],
) -> nx.Graph:
    """Merge all similarity edges into a single graph.

    Parameters
    ----------
    localization_results : Dict[str, LocalizationResult]
        Localization results from all split points.

    Returns
    -------
    nx.Graph
        Combined similarity graph with all cross-boundary edges.
    """
    G = nx.Graph()

    for parent, result in localization_results.items():
        for edge in result.similarity_edges:
            # Add edge with metadata
            if G.has_edge(edge.node_a, edge.node_b):
                # Keep the edge with higher p-value (strongest similarity evidence:
                # high p-value means we FAIL to reject H₀ of "same distribution")
                existing_p = G.edges[edge.node_a, edge.node_b].get("p_value", 1.0)
                if edge.p_value > existing_p:
                    G.edges[edge.node_a, edge.node_b]["p_value"] = edge.p_value
                    G.edges[edge.node_a, edge.node_b]["split_parent"] = parent
            else:
                G.add_edge(
                    edge.node_a,
                    edge.node_b,
                    p_value=edge.p_value,
                    test_statistic=edge.test_statistic,
                    split_parent=parent,
                )

    return G


__all__ = [
    "SimilarityEdge",
    "LocalizationResult",
    "localize_divergence_signal",
    "extract_constrained_clusters",
    "build_cross_boundary_similarity",
    "merge_similarity_graphs",
    "merge_difference_graphs",
]
