from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
import networkx as nx

from sklearn.cluster import AgglomerativeClustering

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

            G.add_node(node_id, is_leaf=False, height=float(dist))
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

        def _is_leaf(n: str) -> bool:
            v = self.nodes[n].get("is_leaf")
            return bool(v) if v is not None else (self.out_degree(n) == 0)

        if node is None:
            leaf_nodes = [n for n in self.nodes if _is_leaf(n)]
        else:
            if _is_leaf(node):
                leaf_nodes = [node]
            else:
                leaf_nodes = [d for d in nx.descendants(self, node) if _is_leaf(d)]

        out = (
            [self.nodes[n].get("label", n) for n in leaf_nodes]
            if return_labels
            else leaf_nodes
        )
        return sorted(out) if sort else out

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
