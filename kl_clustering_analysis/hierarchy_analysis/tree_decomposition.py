"""Tree decomposition logic for KL-based clustering.

This module contains :class:`~kl_clustering_analysis.hierarchy_analysis.tree_decomposition.TreeDecomposition`,
which traverses a hierarchy and decides where to split or merge to form clusters.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from ..core_utils.data_utils import extract_bool_column_dict
from .. import config
from .posthoc_merge import apply_posthoc_merge

# Statistical sibling test used for pairwise cluster comparisons
from .statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
)


class TreeDecomposition:
    """Annotate a hierarchy with significance tests and carve it into clusters.

    The decomposer walks a :class:`~tree.poset_tree.PosetTree` top-down and decides
    whether to split or merge at each internal node based on two statistical gates:

    #. **Binary structure gate** - parent must have exactly two children to split.
    #. **Sibling divergence gate** - siblings must have significantly different
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
        sibling_shortlist_size: int | None = None,
        posthoc_merge: bool = config.POSTHOC_MERGE,
        posthoc_merge_alpha: float | None = config.POSTHOC_MERGE_ALPHA,
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
        posthoc_merge
            If True, apply tree-respecting post-hoc merging after initial decomposition.
            This iteratively merges clusters whose underlying leaf clusters are NOT
            significantly different, working bottom-up through the tree.
        posthoc_merge_alpha
            Significance level for post-hoc merge tests. Defaults to sibling_alpha if None.
        """
        self.tree = tree
        self.results_df = results_df if results_df is not None else pd.DataFrame()
        self.alpha_local = float(alpha_local)
        self.sibling_alpha = float(sibling_alpha)

        self.sibling_shortlist_size = (
            int(sibling_shortlist_size) if sibling_shortlist_size is not None else 0
        )
        self.posthoc_merge = bool(posthoc_merge)
        self.posthoc_merge_alpha = (
            float(posthoc_merge_alpha)
            if posthoc_merge_alpha is not None
            else self.sibling_alpha
        )

        # ----- root -----
        self._root = next(
            node_id for node_id, degree in self.tree.in_degree() if degree == 0
        )

        # ----- pre-cache node metadata -----
        self._cache_node_metadata()

        # ----- leaf partitions & counts (poset view) -----
        if not hasattr(self.tree, "compute_descendant_sets"):
            raise TypeError(
                "TreeDecomposition requires a tree implementation that provides "
                "`compute_descendant_sets(use_labels=...)` (e.g., PosetTree)."
            )
        self._leaf_partition_by_node = self.tree.compute_descendant_sets(  # type: ignore[attr-defined]
            use_labels=True
        )

        self._leaf_count_cache: Dict[str, int] = {
            node_id: self.tree.nodes[node_id].get(
                "leaf_count", len(self._leaf_partition_by_node.get(node_id, ()))
            )
            for node_id in self.tree.nodes
        }

        # ----- results_df → fast dictionary lookups (no .loc in hot paths) -----
        self._local_significant = extract_bool_column_dict(
            self.results_df, "Child_Parent_Divergence_Significant"
        )
        # Sibling divergence test: Sibling_BH_Different = True means siblings differ -> SPLIT
        self._sibling_different = extract_bool_column_dict(
            self.results_df, "Sibling_BH_Different"
        )
        self._sibling_skipped = extract_bool_column_dict(
            self.results_df, "Sibling_Divergence_Skipped"
        )

        for column_name, mapping in (
            ("Child_Parent_Divergence_Significant", self._local_significant),
            ("Sibling_BH_Different", self._sibling_different),
            ("Sibling_Divergence_Skipped", self._sibling_skipped),
        ):
            missing = set(self.tree.nodes) - set(mapping.keys())
            if missing:
                preview = ", ".join(map(repr, list(missing)[:5]))
                raise ValueError(
                    f"Missing {column_name!r} values for nodes: {preview}."
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
            # Preserve shape: 1D for binary (Bernoulli), 2D for categorical (multinomial)
            distribution_array = np.asarray(node_data["distribution"], dtype=float)
            self._distribution_by_node[node_id] = distribution_array
            self._is_leaf[node_id] = node_data["is_leaf"]
            self._label[node_id] = node_data.get("label", node_id)
        
        # n_features is the first dimension (number of features/variables)
        if self._distribution_by_node:
            first_dist = next(iter(self._distribution_by_node.values()))
            self._n_features = first_dist.shape[0] if first_dist.ndim >= 1 else 0
        else:
            self._n_features = 0

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
        """Identify the lowest common ancestor for a collection of leaf labels.

        This implementation delegates the LCA calculation to the tree object, which is
        expected to have an efficient implementation, and caches a label-to-node mapping
        to accelerate leaf lookups.
        """
        if not hasattr(self, "_label_to_node_map"):
            # This is a one-time cache population for all leaves.
            self._label_to_node_map = {
                self._label[n]: n for n in self.tree.nodes if self._is_leaf.get(n)
            }

        leaf_nodes = [
            self._label_to_node_map[label]
            for label in leaf_labels
            if label in self._label_to_node_map
        ]

        if not leaf_nodes:
            return self._root

        # The `self.tree` object is a PosetTree, which has `find_lca_for_set`.
        return self.tree.find_lca_for_set(leaf_nodes)

    # ---------- local KL (child vs parent) ----------

    def _leaf_count(self, node_id: str) -> int:
        return self._leaf_count_cache[node_id]

    def _child_diverges_from_parent(self, child: str, parent: str) -> bool:
        """Determine whether the local divergence test flags ``child`` as divergent.

        Relies on precomputed ``Child_Parent_Divergence_Significant`` annotations from
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

        2. **Child-parent divergence gate** (child vs parent) - SIGNAL DETECTION
           - OPEN: at least one child significantly diverges from parent → proceed
           - CLOSED: neither child diverges from parent → MERGE
           - Checked FIRST to confirm there's actual signal (not just noise)
           - This matches the annotation order: child-parent test runs before sibling test

        3. **Sibling divergence gate** (sibling vs sibling) - CLUSTER SEPARATION
           - OPEN: siblings are significantly different → SPLIT
           - CLOSED: siblings are not significantly different → MERGE
           - Only checked if child-parent gate passes (consistent with annotation)

        Outcome Logic:
        - ALL gates OPEN → return True → children added to traversal stack, split continues recursively
        - ANY gate CLOSED → return False → collect all leaves under parent as single cluster, stop splitting here

        Rationale for gate order:
        - Child-parent test detects signal: "Did the children diverge from the parent?"
        - Sibling test confirms separation: "Did they diverge in different directions?"
        - This order matches annotation: sibling test is only computed when child-parent passes
        """
        # Gate 1: Binary structure requirement
        children = self._children[parent]
        if len(children) != 2:
            return False

        left_child, right_child = children

        # Gate 2: Child-parent divergence requirement (check FIRST - detects signal)
        # At least one child must significantly diverge from parent to confirm there's real signal
        left_diverges = self._local_significant.get(left_child)
        right_diverges = self._local_significant.get(right_child)

        if left_diverges is None or right_diverges is None:
            raise ValueError(
                "Missing child-parent divergence annotations for "
                f"{left_child!r} or {right_child!r}; annotate before decomposing."
            )

        # If neither child diverges from parent, it's just noise - merge
        if not (left_diverges or right_diverges):
            return False

        # Gate 3: Sibling divergence requirement (confirms cluster separation)
        # Sibling_BH_Different = True means siblings have significantly different distributions
        is_different = self._sibling_different.get(parent)

        if is_different is None:
            raise ValueError(
                "Sibling divergence annotations missing for node "
                f"{parent!r}; run annotate_sibling_divergence first."
            )

        if self._sibling_skipped.get(parent, False):
            raise ValueError(
                "Sibling divergence test skipped for node "
                f"{parent!r}; annotations incomplete."
            )

        # Siblings must be significantly different to justify splitting
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

        cluster_assignments = self._maybe_apply_posthoc_merge(cluster_assignments)

        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "independence_analysis": {
                "alpha_local": self.alpha_local,
                "decision_mode": "sibling_divergence",
                "posthoc_merge": self.posthoc_merge,
            },
        }

    # ---------- post-hoc merge helpers ----------

    def _compute_cluster_distribution(
        self, cluster_root: str
    ) -> Tuple[np.ndarray, int]:
        """Compute the distribution for a cluster.

        Uses the precomputed distribution from the tree node.

        Parameters
        ----------
        cluster_root
            The root node of the cluster.

        Returns
        -------
        Tuple[np.ndarray, int]
            (distribution, sample_size)
        """
        distribution = self._distribution_by_node[cluster_root]
        sample_size = self._leaf_count(cluster_root)
        return distribution, sample_size

    def _test_cluster_pair_divergence(
        self, cluster_a: str, cluster_b: str, common_ancestor: str
    ) -> Tuple[float, float, float]:
        """Test if two clusters are significantly different.

        Wrapper around :func:`sibling_divergence_test`.

        Notes
        -----
        - Computes a Wald χ² statistic (optionally after JL projection) from the
          standardized difference between the two cluster mean vectors.
        - The sibling test returns ``(test_statistic, degrees_of_freedom, p_value)``.
        - The ``common_ancestor`` parameter is accepted for API symmetry but is
          not used by the current implementation.

        Parameters
        ----------
        cluster_a, cluster_b
            The two cluster root nodes to compare.
        common_ancestor
            The lowest common ancestor in the tree (present for API symmetry,
            not used by this wrapper).

        Returns
        -------
        Tuple[float, float, float]
            (test_statistic, degrees_of_freedom, p_value)
        """
        dist_a, size_a = self._compute_cluster_distribution(cluster_a)
        dist_b, size_b = self._compute_cluster_distribution(cluster_b)

        test_stat, df, p_value = sibling_divergence_test(
            left_dist=dist_a,
            right_dist=dist_b,
            n_left=float(size_a),
            n_right=float(size_b),
        )

        return test_stat, df, p_value

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

        cluster_assignments = self._maybe_apply_posthoc_merge(cluster_assignments)

        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "independence_analysis": {
                "alpha_local": self.alpha_local,
                "decision_mode": "sibling_divergence",
                "posthoc_merge": self.posthoc_merge,
            },
        }

    def _maybe_apply_posthoc_merge(
        self, cluster_assignments: dict[int, dict[str, object]]
    ) -> dict[int, dict[str, object]]:
        """Optionally apply tree-respecting post-hoc merge to an existing decomposition.

        Parameters
        ----------
        cluster_assignments
            Mapping from cluster id to cluster metadata, including ``root_node``.

        Returns
        -------
        dict[int, dict[str, object]]
            Updated cluster assignments after optional post-hoc merging.
        """
        if not (self.posthoc_merge and self._distribution_by_node):
            return cluster_assignments

        cluster_roots: Set[str] = set()
        for info in cluster_assignments.values():
            root_node = info.get("root_node")
            if isinstance(root_node, str) and root_node:
                cluster_roots.add(root_node)

        if not cluster_roots:
            return cluster_assignments

        merged_roots = apply_posthoc_merge(
            cluster_roots=cluster_roots,
            alpha=self.posthoc_merge_alpha,
            tree=self.tree,
            children=self._children,
            root=self._root,
            test_divergence=self._test_cluster_pair_divergence,
        )

        # Deterministic cluster ids: avoid iterating an unordered set.
        merged_leaf_sets: List[set[str]] = []
        for root_node in sorted(merged_roots):
            leaf_labels = self._get_all_leaves(root_node)
            if leaf_labels:
                merged_leaf_sets.append(set(leaf_labels))

        return self._build_cluster_assignments(merged_leaf_sets)

    @staticmethod
    def build_sample_cluster_assignments(
        decomposition_results: Dict[str, object],
    ) -> pd.DataFrame:
        """Build per-sample cluster assignments from decomposition output.

        Parameters
        ----------
        decomposition_results
            A decomposition result dictionary produced by :meth:`TreeDecomposition.decompose_tree`.

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
