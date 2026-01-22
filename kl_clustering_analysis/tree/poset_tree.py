from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Iterable, TYPE_CHECKING
import numpy as np
import networkx as nx

from sklearn.cluster import AgglomerativeClustering
from kl_clustering_analysis.information_metrics.kl_divergence.divergence_metrics import (
    _populate_distributions,
    _populate_global_kl,
    _populate_local_kl,
    _extract_hierarchy_statistics,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import (
    TreeDecomposition,
)
from kl_clustering_analysis.hierarchy_analysis.statistics import (
    annotate_child_parent_divergence,
    annotate_sibling_divergence,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths

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

        Parameters
        ----------
        X
            Feature matrix of shape ``(n_samples, n_features)`` used for hierarchical
            clustering.
        leaf_names
            Optional list of labels; defaults to ``leaf_{i}`` when omitted.
        linkage, metric, compute_distances
            Passed straight through to :class:`AgglomerativeClustering` to control the
            linkage strategy.

        Returns
        -------
        PosetTree
            Directed tree whose leaves correspond to the fitted samples and whose
            internal nodes track merge heights provided by scikit-learn.
        """
        n = int(X.shape[0])
        if leaf_names is None:
            leaf_names = [f"leaf_{i}" for i in range(n)]

        model = AgglomerativeClustering(
            n_clusters=1,
            linkage=linkage,
            metric=metric,
            compute_distances=compute_distances,
        )
        model.fit(X)
        children = model.children_  # (n-1, 2) merges; indices in [0, 2n-2]
        distances = getattr(
            model, "distances_", np.arange(children.shape[0], dtype=float)
        )

        def _id(idx: int) -> str:
            return f"L{idx}" if idx < n else f"N{idx}"

        G = cls()
        # add leaves
        for i, name in enumerate(leaf_names):
            G.add_node(_id(i), is_leaf=True, label=str(name))

        # add internal merges
        for k, (a, b) in enumerate(children):
            nid = _id(n + k)
            G.add_node(nid, is_leaf=False, height=float(distances[k]))
            G.add_edge(nid, _id(a))
            G.add_edge(nid, _id(b))

        # annotate root
        roots = [u for u, d in G.in_degree() if d == 0]
        if len(roots) != 1:
            raise ValueError(f"Expected one root, got {roots}")
        G.graph["root"] = roots[0]
        return G

    @classmethod
    def from_undirected_edges(cls, edges: Iterable[Tuple]) -> "PosetTree":
        """Orient an undirected tree and promote it to :class:`PosetTree`.

        Parameters
        ----------
        edges
            Iterable of ``(u, v, weight)`` tuples describing an undirected weighted
            tree.

        Returns
        -------
        PosetTree
            Directed version of the input tree with a deterministic root and leaf
            annotations.
        """
        U = nx.Graph()
        U.add_weighted_edges_from(edges)
        # pick a leaf as root
        leaves = [n for n, d in U.degree() if d == 1]
        root = leaves[0] if leaves else next(iter(U.nodes))

        G = cls()
        for n in U.nodes():
            G.add_node(n)

        visited = {root}
        q = [root]
        while q:
            u = q.pop(0)
            for v, attr in U[u].items():
                if v not in visited:
                    visited.add(v)
                    G.add_edge(u, v, weight=float(attr.get("weight", 1.0)))
                    q.append(v)
        # annotate leaves
        for n in G.nodes:
            G.nodes[n]["is_leaf"] = G.out_degree(n) == 0
        G.graph["root"] = root
        return G

    @classmethod
    def from_linkage(
        cls,
        linkage_matrix: np.ndarray,
        leaf_names: Optional[List[str]] = None,
    ) -> "PosetTree":
        """
        Builds a tree from a SciPy linkage matrix.

        Assumes the input is a valid `(n-1, 4)` linkage matrix.

        Args:
            linkage_matrix: A NumPy array from `scipy.cluster.hierarchy.linkage`.
            leaf_names: An optional list of names for the leaf nodes.

        Returns:
            A `PosetTree` instance.
        """
        n_leaves = linkage_matrix.shape[0] + 1
        if leaf_names is None:
            leaf_names = [f"leaf_{i}" for i in range(n_leaves)]

        G = cls()
        # Store linkage matrix for later use (e.g., inconsistency coefficient)
        G.graph["linkage_matrix"] = linkage_matrix
        G.graph["n_leaves"] = n_leaves
        
        for i, name in enumerate(leaf_names):
            G.add_node(f"L{i}", label=name, is_leaf=True)

        for merge_idx, (left_idx, right_idx, dist, _) in enumerate(linkage_matrix):
            node_idx = n_leaves + merge_idx
            node_id = f"N{int(node_idx)}"
            left_id = (
                f"L{int(left_idx)}" if left_idx < n_leaves else f"N{int(left_idx)}"
            )
            right_id = (
                f"L{int(right_idx)}" if right_idx < n_leaves else f"N{int(right_idx)}"
            )

            G.add_node(node_id, is_leaf=False, height=float(dist), merge_idx=merge_idx)
            G.add_edge(node_id, left_id, weight=float(dist))
            G.add_edge(node_id, right_id, weight=float(dist))
        return G

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

        out = (
            [self.nodes[n].get("label", n) for n in leaf_nodes]
            if return_labels
            else leaf_nodes
        )
        return sorted(out) if sort else out

    def _is_leaf(self, node_id: str) -> bool:
        """Check if a node is a leaf."""
        is_leaf_attr = self.nodes[node_id].get("is_leaf")
        if is_leaf_attr is not None:
            return bool(is_leaf_attr)
        return self.out_degree(node_id) == 0

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
        _populate_distributions(self, root, leaf_data)
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
            ``alpha_local``, ``near_independence_alpha_buffer``).

        Returns
        -------
        dict
            Decomposition output from ``TreeDecomposition.decompose_tree``.
        """
        if results_df is None:
            if self.stats_df is None:
                if leaf_data is None:
                    raise ValueError(
                        "Tree has no stats_df; provide results_df or leaf_data to populate."
                    )
                self.populate_node_divergences(leaf_data)
            results_df = self.stats_df.copy()

            # Run statistical annotations to align with pipeline_helpers
            results_df = annotate_child_parent_divergence(
                self,
                results_df,
                significance_level_alpha=config.SIGNIFICANCE_ALPHA,
            )
            results_df = annotate_sibling_divergence(
                self,
                results_df,
                significance_level_alpha=config.SIGNIFICANCE_ALPHA,
            )
            # Update the tree's stats_df with the annotated results so they are available later
            self.stats_df = results_df

        decomposer = TreeDecomposition(
            tree=self,
            results_df=results_df,
            **decomposer_kwargs,
        )

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

        return TreeDecomposition.build_sample_cluster_assignments(decomposition_results)
