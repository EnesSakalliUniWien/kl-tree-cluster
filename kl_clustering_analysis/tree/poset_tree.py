from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments as _build_sample_cluster_assignments,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.information_metrics.kl_divergence.divergence_metrics import (
    _extract_hierarchy_statistics,
    _populate_global_kl,
    _populate_local_kl,
)
from kl_clustering_analysis.tree.distributions import populate_distributions

if TYPE_CHECKING:
    import pandas as pd


# ============================================================
# 1) PosetTree (NetworkX.DiGraph subclass)
# ============================================================


class PosetTree(nx.DiGraph):
    """Directed tree wrapper that exposes hierarchy operations.

    The class augments ``networkx.DiGraph`` with helpers that make hierarchical
    clustering workflows easier to manage:

    * the root (in-degree 0) is tracked and can be retrieved via :meth:`root`.
    * leaves carry a ``label`` attribute so downstream consumers can recover the
      original sample identifiers.
    * multiple constructors (:meth:`from_agglomerative`, :meth:`from_linkage`,
      :meth:`from_undirected_edges`) turn clustering output or undirected edge
      lists into a consistent directed representation where edges always point
      from parent to child.
    * utility accessors (:meth:`get_leaves`, :meth:`compute_descendant_sets`)
      provide common tree queries needed by statistical routines.

    Leaves have ``out_degree == 0`` and are expected to carry ``is_leaf=True``.
    """

    # ---------------- Constructors ----------------

    def __init__(self, *args, **kwargs):
        """Initialize PosetTree with stats_df property."""
        super().__init__(*args, **kwargs)
        self.stats_df: Optional["pd.DataFrame"] = None
        self._depths: Optional[Dict[str, int]] = None

    @classmethod
    def from_agglomerative(
        cls,
        X: np.ndarray,
        leaf_names: Optional[List[str]] = None,
        linkage: str = "average",
        metric: str = "euclidean",
        compute_distances: bool = True,
    ) -> "PosetTree":
        """Construct a tree from an :class:`sklearn.cluster.AgglomerativeClustering` fit.

        Delegates to :func:`~kl_clustering_analysis.tree.io.tree_from_agglomerative`.
        """
        from kl_clustering_analysis.tree.io import tree_from_agglomerative

        return tree_from_agglomerative(
            X,
            leaf_names=leaf_names,
            linkage=linkage,
            metric=metric,
            compute_distances=compute_distances,
        )

    @classmethod
    def from_undirected_edges(cls, edges: Iterable[Tuple]) -> "PosetTree":
        """Orient an undirected tree and promote it to :class:`PosetTree`.

        Delegates to :func:`~kl_clustering_analysis.tree.io.tree_from_undirected_edges`.
        """
        from kl_clustering_analysis.tree.io import tree_from_undirected_edges

        return tree_from_undirected_edges(edges)

    @classmethod
    def from_linkage(
        cls,
        linkage_matrix: np.ndarray,
        leaf_names: Optional[List[str]] = None,
    ) -> "PosetTree":
        """Build a tree from a SciPy linkage matrix.

        Delegates to :func:`~kl_clustering_analysis.tree.io.tree_from_linkage`.
        """
        from kl_clustering_analysis.tree.io import tree_from_linkage

        return tree_from_linkage(linkage_matrix, leaf_names=leaf_names)

    # ---------------- Poset helpers ----------------

    def root(self) -> str:
        """Return the cached root node, discovering it if necessary."""
        r = self.graph.get("root")
        if r is None:
            roots = [u for u, d in self.in_degree() if d == 0]
            if len(roots) != 1:
                raise ValueError(f"Expected one root, got {roots}")
            r = roots[0]
            self.graph["root"] = r
        return r

    def get_leaves(
        self,
        node: Optional[str] = None,
        return_labels: bool = True,
        sort: bool = True,
    ) -> List[str]:
        """Collect leaf nodes globally or within a subtree.

        Parameters
        ----------
        node
            When ``None`` (default), returns all leaves. Otherwise restricts the search
            to the descendants of ``node``.
        return_labels
            If ``True`` (default) return the ``label`` attribute; otherwise return raw
            node ids.
        sort
            Whether to sort the returned values in ascending order.

        Returns
        -------
        list[str]
            Leaf labels or ids, depending on ``return_labels``.
        """
        if node is None:
            leaf_nodes = [n for n in self.nodes if self._is_leaf(n)]
        else:
            if self._is_leaf(node):
                leaf_nodes = [node]
            else:
                leaf_nodes = [d for d in nx.descendants(self, node) if self._is_leaf(d)]

        out = [self.nodes[n].get("label", n) for n in leaf_nodes] if return_labels else leaf_nodes
        return sorted(out) if sort else out

    def _is_leaf(self, node_id: str) -> bool:
        """Check if a node is a leaf."""
        is_leaf_attr = self.nodes[node_id].get("is_leaf")
        if is_leaf_attr is not None:
            return bool(is_leaf_attr)
        return self.out_degree(node_id) == 0

    @property
    def distribution_map(self) -> Dict[str, np.ndarray]:
        """Map each node to its distribution vector.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary whose keys are node ids and values are the node
            distribution arrays (populated by :meth:`populate_node_divergences`).
        """
        return {
            n: np.asarray(self.nodes[n]["distribution"], dtype=float)
            for n in self.nodes
            if "distribution" in self.nodes[n]
        }

    @property
    def leaf_count_map(self) -> Dict[str, int]:
        """Map each node to its descendant leaf count.

        Returns
        -------
        dict[str, int]
            Dictionary whose keys are node ids and values are the ``leaf_count``
            attribute (populated by :meth:`populate_node_divergences`).
        """
        return {
            n: int(self.nodes[n]["leaf_count"]) for n in self.nodes if "leaf_count" in self.nodes[n]
        }

    def compute_descendant_sets(self, use_labels: bool = True) -> Dict[str, frozenset]:
        """Map each node to the set of leaf labels under it.

        Parameters
        ----------
        use_labels
            When ``True`` (default), map to stored ``label`` values; otherwise use
            internal node identifiers.

        Returns
        -------
        dict[str, frozenset]
            Dictionary whose keys are node ids and whose values are the descendant leaf
            labels/ids as a frozenset.
        """
        desc_sets: Dict[str, frozenset] = {}
        # process leaves first (reverse topological order)
        for node in nx.topological_sort(self.reverse()):
            if self.nodes[node].get("is_leaf", False) or self.out_degree(node) == 0:
                val = self.nodes[node].get("label", node) if use_labels else node
                desc_sets[node] = frozenset([val])
            else:
                child_sets = [desc_sets[c] for c in self.successors(node)]
                desc_sets[node] = frozenset.union(*child_sets)
        return desc_sets

    def _get_depths(self) -> Dict[str, int]:
        """Computes and caches node depths from the root."""
        if self._depths is None:
            self._depths = compute_node_depths(self)
        return self._depths

    def find_lca(self, node_a: str, node_b: str) -> str:
        """Find the lowest common ancestor (LCA) of two nodes.

        This implementation assumes the graph is a tree (each node has one parent)
        and uses node depths for an efficient O(depth) search.

        Parameters
        ----------
        node_a, node_b
            Node identifiers whose LCA is sought.

        Returns
        -------
        str
            The node id of the lowest common ancestor.
        """
        if node_a == node_b:
            return node_a

        depths = self._get_depths()
        depth_a = depths.get(node_a)
        depth_b = depths.get(node_b)

        if depth_a is None or depth_b is None:
            # Fallback for nodes not in the main tree structure. This can happen
            # if the graph is not a single connected tree. The original
            # implementation is a safe fallback for general DAGs.
            ancestors_a = set(nx.ancestors(self, node_a))
            ancestors_a.add(node_a)
            current = node_b
            while current not in ancestors_a:
                parents = list(self.predecessors(current))
                if not parents:
                    return self.root()
                current = parents[0]
            return current

        # Use depths to find LCA
        current_a, current_b = node_a, node_b
        # 1. Bring nodes to the same depth
        if depth_a > depth_b:
            for _ in range(depth_a - depth_b):
                current_a = next(self.predecessors(current_a))
        elif depth_b > depth_a:
            for _ in range(depth_b - depth_a):
                current_b = next(self.predecessors(current_b))

        # 2. Walk up until they meet
        while current_a != current_b:
            current_a = next(self.predecessors(current_a))
            current_b = next(self.predecessors(current_b))

        return current_a

    def find_lca_for_set(self, nodes: Iterable[str]) -> str:
        """Find the lowest common ancestor for a collection of nodes.

        Iteratively applies the two-node LCA function to find the LCA for the set.

        Parameters
        ----------
        nodes
            An iterable of node identifiers.

        Returns
        -------
        str
            The node id of the lowest common ancestor for the set.
        """
        node_iterator = iter(nodes)
        try:
            # Start with the first node as the initial LCA
            lca = next(node_iterator)
        except StopIteration:
            # If the input iterable is empty, return the tree's root.
            return self.root()

        root = self.root()
        for node in node_iterator:
            lca = self.find_lca(lca, node)
            # Optimization: if LCA is the root, it cannot get any higher.
            if lca == root:
                return root
        return lca

    def populate_node_divergences(self, leaf_data: "pd.DataFrame") -> None:
        """Populate tree nodes with distributions and KL divergences.

        Populates each node with:
        - distribution: weighted mean of leaf/child distributions
        - leaf_count: number of descendant leaves
        - kl_divergence_global: KL(node||root)
        - kl_divergence_local: KL(child||parent)
        - per-column versions of both KL metrics

        Assumes each feature is a Bernoulli probability in [0,1].

        Note: Root's global KL is set to NaN (self-comparison is meaningless).

        Parameters
        ----------
        leaf_data
            DataFrame where rows are leaf labels and columns are feature probabilities.

        Notes
        -----
        Results are stored in tree.stats_df for later access.
        """
        root = self.root()
        populate_distributions(self, leaf_data)
        _populate_global_kl(self, root)
        _populate_local_kl(self)
        self.stats_df = _extract_hierarchy_statistics(self)

    # ---------------- Decomposition helper ----------------

    def decompose(
        self,
        results_df: Optional["pd.DataFrame"] = None,
        leaf_data: Optional["pd.DataFrame"] = None,
        **decomposer_kwargs,
    ) -> Dict[str, object]:
        """Run ``TreeDecomposition`` directly from the tree.

        Parameters
        ----------
        results_df
            Optional statistics/annotations DataFrame. When omitted, falls back to
            ``self.stats_df`` (populated by :meth:`populate_node_divergences`).
        leaf_data
            Optional leaf-level probability DataFrame. When provided and ``results_df``
            is not supplied, ``populate_node_divergences`` will be invoked to
            initialize node distributions/KL metrics prior to decomposition.
        **decomposer_kwargs
            Extra keyword arguments forwarded to ``TreeDecomposition`` (e.g.,
            ``alpha_local``, ``sibling_alpha``).

        Returns
        -------
        dict
            Decomposition output from ``TreeDecomposition.decompose_tree``.
        """
        # Extract alpha values from kwargs (with defaults from config)
        alpha_local = decomposer_kwargs.pop("alpha_local", config.ALPHA_LOCAL)
        sibling_alpha = decomposer_kwargs.pop("sibling_alpha", config.SIBLING_ALPHA)

        if results_df is None:
            if self.stats_df is None:
                if leaf_data is None:
                    raise ValueError(
                        "Tree has no stats_df; provide results_df or leaf_data to populate."
                    )
                self.populate_node_divergences(leaf_data)
            results_df = self.stats_df.copy()

        decomposer = TreeDecomposition(
            tree=self,
            results_df=results_df,
            alpha_local=alpha_local,
            sibling_alpha=sibling_alpha,
            leaf_data=leaf_data,
            **decomposer_kwargs,
        )

        # Cache annotated results back so stats_df reflects the full pipeline
        self.stats_df = decomposer.results_df

        if decomposer.use_signal_localization:
            return decomposer.decompose_tree_v2()
        return decomposer.decompose_tree()

    def build_sample_cluster_assignments(
        self, decomposition_results: Dict[str, object]
    ) -> "pd.DataFrame":
        """Build a per-sample cluster assignment table from decomposition output.

        This method is a convenience wrapper around
        :meth:`kl_clustering_analysis.hierarchy_analysis.tree_decomposition.TreeDecomposition.build_sample_cluster_assignments`.

        Parameters
        ----------
        decomposition_results
            A decomposition result dictionary produced by :meth:`decompose`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame indexed by ``sample_id`` with cluster assignment columns.
        """

        return _build_sample_cluster_assignments(decomposition_results)
