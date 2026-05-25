from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths
from kl_clustering_analysis.tree.distributions import populate_distributions
from kl_clustering_analysis.tree.topology import (
    compute_descendant_leaf_sets,
    get_leaf_values,
    is_leaf,
    lowest_common_ancestor,
    lowest_common_ancestor_for_set,
)

if TYPE_CHECKING:
    import pandas as pd

    from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.annotation_bundle import (
        GateAnnotationBundle,
    )
    from kl_clustering_analysis.tree.feature_space import FeatureSpace


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
        """Initialize PosetTree with annotations_df property."""
        super().__init__(*args, **kwargs)
        self.annotations_df: pd.DataFrame | None = None
        self._depths: dict[object, int] | None = None

    @classmethod
    def from_agglomerative(
        cls,
        X: np.ndarray,
        leaf_names: list[str] | None = None,
        linkage: str = "average",
        metric: str = "euclidean",
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
        )

    @classmethod
    def from_undirected_edges(cls, edges: Iterable[tuple]) -> "PosetTree":
        """Orient an undirected tree and promote it to :class:`PosetTree`.

        Delegates to :func:`~kl_clustering_analysis.tree.io.tree_from_undirected_edges`.
        """
        from kl_clustering_analysis.tree.io import tree_from_undirected_edges

        return tree_from_undirected_edges(edges)

    @classmethod
    def from_linkage(
        cls,
        linkage_matrix: np.ndarray,
        leaf_names: list[str] | None = None,
    ) -> "PosetTree":
        """Build a tree from a SciPy linkage matrix.

        Delegates to :func:`~kl_clustering_analysis.tree.io.tree_from_linkage`.
        """
        from kl_clustering_analysis.tree.io import tree_from_linkage

        return tree_from_linkage(linkage_matrix, leaf_names=leaf_names)

    # ---------------- Poset helpers ----------------

    def root(self) -> object:
        """Return the cached root node, discovering it if necessary."""
        if "root" in self.graph:
            return self.graph["root"]

        roots = [u for u, d in self.in_degree() if d == 0]
        if len(roots) != 1:
            raise ValueError(f"Expected one root, got {roots}")
        r = roots[0]
        self.graph["root"] = r
        return r

    def get_leaves(
        self,
        node: object | None = None,
        return_labels: bool = True,
        sort: bool = True,
    ) -> list[str]:
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
        return get_leaf_values(self, node=node, use_labels=return_labels, sort=sort)

    def _is_leaf(self, node_id: object) -> bool:
        """Check if a node is a leaf."""
        return is_leaf(self, node_id)

    def compute_descendant_sets(self, use_labels: bool = True) -> dict[object, frozenset]:
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
        return compute_descendant_leaf_sets(self, use_labels=use_labels)

    def _get_depths(self) -> dict[object, int]:
        """Computes and caches node depths from the root."""
        if self._depths is None:
            self._depths = compute_node_depths(self)
        return self._depths

    def find_lca(self, node_a: object, node_b: object) -> object:
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
        return lowest_common_ancestor(self, node_a, node_b, depths=self._get_depths())

    def find_lca_for_set(self, nodes: Iterable[object]) -> object:
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
        return lowest_common_ancestor_for_set(self, nodes)

    def populate_node_divergences(
        self,
        leaf_data: "pd.DataFrame",
        *,
        feature_space: "FeatureSpace | None" = None,
    ) -> None:
        """Populate tree nodes with distributions and build stats DataFrame.

        Populates each node with:
        - distribution: weighted mean of leaf/child distributions
        - leaf_count: number of descendant leaves

        Node distributions are stored as flat raw-coordinate vectors. The
        optional feature-space contract supplies Bernoulli/categorical/continuous block
        structure for downstream covariance code.

        Parameters
        ----------
        leaf_data
            DataFrame where rows are leaf labels and columns are raw feature coordinates.

        Notes
        -----
        Results are stored in ``tree.annotations_df`` for later access.
        """
        import pandas as pd

        populate_distributions(
            self,
            leaf_data,
            feature_space=feature_space,
        )
        node_records = []
        for node_id in self.nodes():
            node_attrs = self.nodes[node_id]
            node_records.append(
                {
                    "node_id": node_id,
                    "distribution": node_attrs["distribution"],
                    "leaf_count": node_attrs["leaf_count"],
                    "is_leaf": node_attrs["is_leaf"],
                }
            )
        self.annotations_df = pd.DataFrame.from_records(node_records).set_index(
            "node_id", drop=True
        )

    # ---------------- Decomposition helper ----------------

    def decompose(
        self,
        annotations_df: pd.DataFrame | None = None,
        gate_annotation_bundle: GateAnnotationBundle | None = None,
        leaf_data: pd.DataFrame | None = None,
        feature_space: FeatureSpace | None = None,
        **decomposer_kwargs,
    ) -> dict[str, object]:
        """Run ``TreeDecomposition`` directly from the tree.

        Parameters
        ----------
        annotations_df
            Required statistics/annotations DataFrame when no gate annotation
            bundle is provided.
        gate_annotation_bundle
            Explicit reusable output from ``run_gate_annotation_pipeline``.
        leaf_data
            Optional leaf-level probability DataFrame used by statistical gate
            annotation.
        **decomposer_kwargs
            Extra keyword arguments forwarded to ``TreeDecomposition`` (e.g.,
            ``alpha_local``, ``sibling_alpha``).

        Returns
        -------
        dict
            Decomposition output from ``TreeDecomposition.decompose_tree``.
        """
        # Extract alpha values from kwargs (with defaults from config)
        alpha_local = decomposer_kwargs.pop("alpha_local", config.EDGE_ALPHA)
        sibling_alpha = decomposer_kwargs.pop("sibling_alpha", config.SIBLING_ALPHA)

        if annotations_df is not None and gate_annotation_bundle is not None:
            raise ValueError("Pass either annotations_df or gate_annotation_bundle, not both.")

        if annotations_df is None and gate_annotation_bundle is None:
            annotations_df = self.annotations_df

        if annotations_df is None and gate_annotation_bundle is None:
            raise ValueError(
                "annotations_df or gate_annotation_bundle is required. Call "
                "populate_node_divergences(leaf_data) first and pass "
                "tree.annotations_df explicitly, or pass the bundle returned by "
                "run_gate_annotation_pipeline."
            )

        from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition

        decomposer = TreeDecomposition(
            tree=self,
            annotations_df=annotations_df,
            gate_annotation_bundle=gate_annotation_bundle,
            alpha_local=alpha_local,
            sibling_alpha=sibling_alpha,
            leaf_data=leaf_data,
            feature_space=feature_space,
            **decomposer_kwargs,
        )

        # Cache annotated results back so annotations_df reflects the full pipeline
        self.annotations_df = decomposer.annotations_df

        return decomposer.decompose_tree()

    def build_sample_cluster_assignments(
        self, decomposition_results: dict[str, object]
    ) -> pd.DataFrame:
        """Build a per-sample cluster assignment table from decomposition output.

        This method is a convenience wrapper around
        :func:`kl_clustering_analysis.hierarchy_analysis.cluster_assignments.build_sample_cluster_assignments`.

        Parameters
        ----------
        decomposition_results
            A decomposition result dictionary produced by :meth:`decompose`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame indexed by ``sample_id`` with cluster assignment columns.
        """

        from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
            build_sample_cluster_assignments,
        )

        return build_sample_cluster_assignments(decomposition_results)
