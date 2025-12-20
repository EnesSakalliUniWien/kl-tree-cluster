"""Tree decomposition logic for KL-based clustering.

This module contains :class:`~kl_clustering_analysis.hierarchy_analysis.tree_decomposer.ClusterDecomposer`,
which traverses a hierarchy and decides where to split or merge to form clusters.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Set

import numpy as np
import pandas as pd
import networkx as nx

from ..core_utils.data_utils import extract_bool_column_dict
from .. import config


class ClusterDecomposer:
    """Annotate a hierarchy with significance tests and carve it into clusters.

    The decomposer walks a :class:`~tree.poset_tree.PosetTree` top-down and decides
    whether to split or merge at each internal node based on two statistical gates:

    #. **Binary structure gate** – parent must have exactly two children to split.
    #. **Sibling divergence gate** – siblings must have significantly different
       distributions according to a Jensen-Shannon divergence permutation test
       with Benjamini-Hochberg multiple-testing correction. If siblings are
       significantly different, the split proceeds; otherwise the children are
       merged into a single cluster.

    Nodes that pass all gates become cluster boundaries. Leaves under the same
    boundary node are assigned the same cluster identifier. The resulting report
    captures the cluster root node, member leaves, and cluster size.
    """

    def __init__(
        self,
        tree: nx.DiGraph,
        results_df: pd.DataFrame | None = None,
        *,
        alpha_local: float = config.ALPHA_LOCAL,
        sibling_alpha: float = config.SIBLING_ALPHA,
        significance_column: str | None = None,
        near_independence_alpha_buffer: float = 0.0,
        near_independence_kl_gap: float = 0.0,
        sibling_shortlist_size: int | None = None,
    ):
        """Configure decomposition thresholds and pre-compute reusable metadata.

        Parameters
        ----------
        tree
            Directed hierarchy (typically a :class:`~tree.poset_tree.PosetTree`).
        results_df
            DataFrame of statistical annotations (e.g., columns produced by
            ``hierarchy_analysis.statistics`` helpers). May be ``None`` if the caller
            plans to rely on on-the-fly calculations.
        alpha_local
            Significance level used when the local Kullback-Leibler divergence gate
            falls back to raw chi-square tests.
        sibling_alpha
            Significance level used by sibling-independence annotations and gating.
        """
        self.tree = tree
        self.results_df = results_df if results_df is not None else pd.DataFrame()
        self.alpha_local = float(alpha_local)
        self.sibling_alpha = float(sibling_alpha)
        self.significance_column = significance_column
        self.near_independence_alpha_buffer = float(near_independence_alpha_buffer)
        self.near_independence_kl_gap = float(near_independence_kl_gap)
        self.sibling_shortlist_size = (
            int(sibling_shortlist_size) if sibling_shortlist_size is not None else 0
        )

        # ----- root -----
        self._root = next(
            node_id for node_id, degree in self.tree.in_degree() if degree == 0
        )

        # ----- pre-cache node metadata -----
        self._cache_node_metadata()

        # ----- leaf partitions & counts (poset view) -----
        self._leaf_partition_by_node: Dict[str, frozenset] = (
            self._compute_leaf_partitions_for_all_nodes()
        )

        self._leaf_count_cache: Dict[str, int] = {
            node_id: self.tree.nodes[node_id].get(
                "leaf_count", len(self._leaf_partition_by_node.get(node_id, ()))
            )
            for node_id in self.tree.nodes
        }

        # ----- results_df → fast dictionary lookups (no .loc in hot paths) -----
        self._local_significant = extract_bool_column_dict(
            self.results_df, "Local_BH_Significant"
        )
        # Sibling divergence test: Sibling_BH_Different = True means siblings differ -> SPLIT
        self._sibling_different = extract_bool_column_dict(
            self.results_df, "Sibling_BH_Different"
        )
        self._sibling_skipped = extract_bool_column_dict(
            self.results_df, "Sibling_Divergence_Skipped"
        )

        # Precompute children list (avoids rebuilding generator repeatedly)
        self._children: Dict[str, List[str]] = {
            n: list(self.tree.successors(n)) for n in self.tree.nodes
        }

    # ---------- initialization helpers ----------

    def _cache_node_metadata(self) -> None:
        """Cache node attributes for fast repeated access during decomposition.

        Extracts and stores distributions, leaf flags, and labels.
        This one-time preprocessing avoids expensive NetworkX lookups in tight loops.
        """
        self._distribution_by_node: Dict[str, np.ndarray] = {}
        self._is_leaf: Dict[str, bool] = {}
        self._label: Dict[str, str] = {}

        for node_id in self.tree.nodes:
            node_data = self.tree.nodes[node_id]
            distribution_array = np.asarray(
                node_data["distribution"], dtype=float
            ).ravel()
            self._distribution_by_node[node_id] = distribution_array
            self._is_leaf[node_id] = node_data["is_leaf"]
            self._label[node_id] = node_data.get("label", node_id)
        self._n_features = (
            len(next(iter(self._distribution_by_node.values())))
            if self._distribution_by_node
            else 0
        )

    # ---------- poset helpers ----------

    def _compute_leaf_partitions_for_all_nodes(self) -> dict[str, frozenset]:
        """Compute the leaf partition (descendant leaves) for every node via bottom-up traversal.

        Each node in a hierarchical tree defines a partition of the leaf set - the collection
        of terminal samples that descend from that node. This method precomputes these partitions
        for all nodes using dynamic programming, enabling O(1) cluster membership queries during
        the decomposition phase.

        The algorithm uses topological sorting on the reversed tree to process nodes in bottom-up
        order (leaves first, then parents). For leaf nodes, the partition is a singleton set
        containing only that leaf's label. For internal nodes, the partition is the union of
        all child partitions, representing the transitive closure of descendant leaves.

        This precomputation trades O(N) space and O(N) initialization time for O(1) access
        during the iterative tree decomposition, where cluster boundaries are identified and
        all leaves under a boundary node must be collected into a cluster.

        Returns
        -------
        dict[str, frozenset]
            Dictionary mapping each node_id to a frozenset of leaf labels. The frozenset
            contains all terminal samples (leaves) that are descendants of that node in
            the hierarchy. For leaf nodes, this is a singleton set; for internal nodes,
            it's the union of all descendant leaf partitions.

        Notes
        -----
        - Uses NetworkX's topological_sort on tree.reverse() to ensure proper bottom-up ordering
        - Returns frozenset (immutable) to prevent accidental modification of cached partitions
        - Empty frozenset returned for malformed nodes without children (defensive fallback)

        Examples
        --------
        For a tree with structure::

                 root
                /    \\
               A      B
              / \\     \\
             D   E     F

        The computed partitions would be::

            D → {D}
            E → {E}
            F → {F}
            A → {D, E}
            B → {F}
            root → {D, E, F}
        """
        leaf_partition_by_node: Dict[str, frozenset] = {}

        # Process children before parents (bottom-up)
        for node_id in nx.topological_sort(self.tree.reverse()):
            if self._is_leaf[node_id]:
                # Base case: leaf partition contains only itself
                leaf_partition_by_node[node_id] = frozenset([self._label[node_id]])
            else:
                # Recursive case: union child partitions to form parent partition
                child_partitions = [
                    leaf_partition_by_node[child_id]
                    for child_id in self.tree.successors(node_id)
                ]
                leaf_partition_by_node[node_id] = (
                    frozenset().union(*child_partitions)
                    if child_partitions
                    else frozenset()
                )

        return leaf_partition_by_node

    # ---------- utilities ----------

    def _get_all_leaves(self, node_id: str) -> set[str]:
        """Return the leaf partition beneath a node using precomputed cache.

        Parameters
        ----------
        node_id
            The node whose leaf partition to retrieve

        Returns
        -------
        set
            Set of leaf labels in the node's partition
        """
        return set(self._leaf_partition_by_node[node_id])

    # ---------- LCA ----------

    def _find_cluster_root(self, leaf_labels: Set[str]) -> str:
        """Identify the lowest common ancestor for a collection of leaf labels."""
        leaf_nodes: List[str] = []
        for n in self.tree.nodes:
            if self._is_leaf[n] and self._label[n] in leaf_labels:
                leaf_nodes.append(n)
        if not leaf_nodes:
            return self._root
        if len(leaf_nodes) == 1:
            return leaf_nodes[0]

        common = set(nx.ancestors(self.tree, leaf_nodes[0]))
        common.add(leaf_nodes[0])
        for lf in leaf_nodes[1:]:
            anc = set(nx.ancestors(self.tree, lf))
            anc.add(lf)
            common &= anc
            if not common:
                return self._root
        for anc in common:
            if not (set(nx.descendants(self.tree, anc)) & common):
                return anc
        return self._root

    # ---------- local KL (child vs parent) ----------

    def _leaf_count(self, node_id: str) -> int:
        return self._leaf_count_cache[node_id]

    def _child_diverges_from_parent(self, child: str, parent: str) -> bool:
        """Determine whether the local divergence test flags ``child`` as divergent.

        Relies on precomputed ``Local_BH_Significant`` annotations from
        :func:`~kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance.annotate_child_parent_divergence`.
        No fallback computation is performed; missing annotations raise to signal
        a misconfigured pipeline.
        """
        annotated = self._local_significant.get(child)
        if annotated is None:
            raise ValueError(
                "Local divergence annotations missing for node "
                f"{child!r}; run annotate_child_parent_divergence first."
            )
        return bool(annotated)

    # ---------- core decomposition (iterative, no recursion) ----------

    def _process_node_for_decomposition(
        self,
        node_id: str,
        nodes_to_visit: List[str],
        final_leaf_sets: List[set[str]],
    ) -> None:
        """Apply split-or-merge decision for one node during traversal.

        Parameters
        ----------
        node_id
            Node to process
        nodes_to_visit
            Pending nodes list used as a last in, first out worklist.
        final_leaf_sets
            Accumulator for cluster leaf sets
        """
        if self._should_split(node_id):
            children = self._children[node_id]
            left_child, right_child = children
            # Because this is a last in, first out worklist, pushing right then left
            # means the left child is processed first (left-first depth-first traversal).
            nodes_to_visit.append(right_child)
            nodes_to_visit.append(left_child)
        else:
            final_leaf_sets.append(self._get_all_leaves(node_id))

    def _iterate_nodes_to_visit(
        self, nodes_to_visit: List[str], processed: Set[str]
    ) -> Iterator[str]:
        """Yield nodes from a mutable last in, first out list exactly once.

        This is a small readability helper that encapsulates the common:
        pop → skip processed → mark processed → yield pattern.

        Notes
        -----
        The underlying algorithm still relies on mutating ``nodes_to_visit`` during
        traversal (e.g., pushing children when a node should split).
        """
        while nodes_to_visit:
            node_id = nodes_to_visit.pop()
            if node_id in processed:
                continue
            processed.add(node_id)
            yield node_id

    def _build_cluster_assignments(
        self, final_leaf_sets: List[set[str]]
    ) -> dict[int, dict[str, object]]:
        """Build cluster assignment dictionary from collected leaf sets.

        Parameters
        ----------
        final_leaf_sets
            List of leaf sets, one per cluster

        Returns
        -------
        dict
            Cluster assignments mapping cluster_index to cluster metadata
        """
        cluster_assignments: dict[int, dict[str, object]] = {}
        for cluster_index, leaf_set in enumerate(final_leaf_sets):
            if not leaf_set:
                continue
            cluster_assignments[cluster_index] = {
                "root_node": self._find_cluster_root(leaf_set),
                "leaves": sorted(leaf_set),
                "size": len(leaf_set),
            }
        return cluster_assignments

    def _should_split(self, parent: str) -> bool:
        """Evaluate statistical gates and return ``True`` when parent should split.

        Gates evaluated in order:

        1. **Binary structure gate** (always active)
           - OPEN: parent has exactly 2 children → proceed to statistical tests
           - CLOSED: parent has <2 or >2 children → merge at parent, form cluster boundary

        2. **Sibling divergence gate** (always active when annotated)
           - OPEN: siblings are significantly different (JSD permutation test with BH correction) → SPLIT
           - CLOSED: siblings are not significantly different → MERGE
           - Missing annotation treated as error

        Outcome Logic:
        - ALL gates OPEN → return True → children added to traversal stack, split continues recursively
        - ANY gate CLOSED → return False → collect all leaves under parent as single cluster, stop splitting here
        """
        # Gate 1: Binary structure requirement
        children = self._children[parent]
        if len(children) != 2:
            return False

        # Gate 2: Sibling divergence requirement
        # Sibling_BH_Different = True means siblings have significantly different distributions -> SPLIT
        is_different = self._sibling_different.get(parent)

        if is_different is None:
            if self._sibling_skipped.get(parent, False):
                # Skipped nodes default to no split (conservative)
                return False
            raise ValueError(
                "Sibling divergence annotations missing for node "
                f"{parent!r}; run annotate_sibling_divergence first."
            )

        return bool(is_different)

    def _pop_candidate(
        self, heap: List[str], queued: set[str], processed: set[str]
    ) -> str | None:
        """Pop the next candidate node for shortlist traversal."""

        while heap:
            node = heap.pop()
            if node in processed:
                continue
            return node
        return None

    def _decompose_with_shortlist(self) -> dict[str, object]:
        """Decomposition that exercises shortlist semantics for testing."""

        heap: List[str] = [self._root]
        queued: set[str] = {self._root}
        processed: Set[str] = set()
        final_leaf_sets: List[set[str]] = []

        while heap:
            node = self._pop_candidate(heap, queued, processed)
            if node is None:
                break
            if node in processed:
                continue
            processed.add(node)

            if self._should_split(node):
                children = self._children[node]
                for child in children:
                    if child not in queued and child not in processed:
                        heap.append(child)
                        queued.add(child)
            else:
                final_leaf_sets.append(self._get_all_leaves(node))

        cluster_assignments = self._build_cluster_assignments(final_leaf_sets)

        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "independence_analysis": {
                "alpha_local": self.alpha_local,
                "decision_mode": "cmi_only",
            },
        }

    def decompose_tree(self) -> dict[str, object]:
        """Return cluster assignments by iteratively traversing the hierarchy.

        Traversal order
        ---------------
        The traversal uses a last in, first out list (similar to an explicit stack),
        which produces a depth-first traversal order. When a node is split, its
        two children are appended in right-then-left order so that the left child
        is processed first on the next iteration.
        """
        if self.sibling_shortlist_size:
            return self._decompose_with_shortlist()

        nodes_to_visit: List[str] = [self._root]
        final_leaf_sets: List[set[str]] = []
        processed: Set[str] = set()

        for node in self._iterate_nodes_to_visit(nodes_to_visit, processed):
            self._process_node_for_decomposition(node, nodes_to_visit, final_leaf_sets)

        cluster_assignments = self._build_cluster_assignments(final_leaf_sets)

        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "independence_analysis": {
                "alpha_local": self.alpha_local,
                "decision_mode": "cmi_only",
            },
        }

    @staticmethod
    def build_sample_cluster_assignments(
        decomposition_results: Dict[str, object],
    ) -> pd.DataFrame:
        """Build per-sample cluster assignments from decomposition output.

        Parameters
        ----------
        decomposition_results
            A decomposition result dictionary produced by :meth:`ClusterDecomposer.decompose_tree`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame indexed by ``sample_id`` with columns:
            - ``cluster_id``: integer cluster identifier
            - ``cluster_root``: node identifier that forms the cluster boundary
            - ``cluster_size``: number of samples in the cluster
        """

        cluster_assignments = decomposition_results.get("cluster_assignments", {})
        if not cluster_assignments:
            return pd.DataFrame(columns=["cluster_id", "cluster_root", "cluster_size"])

        rows: Dict[str, Dict[str, object]] = {}
        for cluster_identifier, info in cluster_assignments.items():
            root = info.get("root_node")
            size = info.get("size", 0)
            for sample_identifier in info.get("leaves", []):
                rows[sample_identifier] = {
                    "cluster_id": cluster_identifier,
                    "cluster_root": root,
                    "cluster_size": size,
                }

        assignments_table = pd.DataFrame.from_dict(rows, orient="index")
        assignments_table.index.name = "sample_id"
        return assignments_table.sort_values("cluster_id")
