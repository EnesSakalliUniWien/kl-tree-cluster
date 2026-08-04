from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import networkx as nx

from tree_break_selection.tree.distributions import populate_distributions
from tree_break_selection.tree.topology import (
    compute_descendant_leaf_sets,
    get_leaf_values,
    lowest_common_ancestor_for_set,
)

if TYPE_CHECKING:
    import pandas as pd

    from tree_break_selection.tree.feature_space import FeatureSpace


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
    * construction modules create a consistent directed representation where
      edges always point from parent to child.
    * utility accessors (:meth:`get_leaves`, :meth:`compute_descendant_sets`)
      provide common tree queries needed by statistical routines.

    Leaves have ``out_degree == 0`` and are expected to carry ``is_leaf=True``.
    """

    def __init__(self, *args, **kwargs):
        """Initialize PosetTree with annotations_df property."""
        super().__init__(*args, **kwargs)
        self.annotations_df: pd.DataFrame | None = None

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
        - distribution: empirical subtree barycenter, computed as the
          leaf-count-weighted mean of leaf/child distributions
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

    def build_sample_cluster_assignments(
        self, decomposition_results: dict[str, object]
    ) -> pd.DataFrame:
        """Build a per-sample cluster assignment table from decomposition output.

        This method is a convenience wrapper around
        :func:`tree_break_selection.hierarchy_analysis.cluster_assignments.build_sample_cluster_assignments`.

        Parameters
        ----------
        decomposition_results
            A decomposition result dictionary produced by :meth:`decompose`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame indexed by ``sample_id`` with cluster assignment columns.
        """

        from tree_break_selection.hierarchy_analysis.cluster_assignments import (
            build_sample_cluster_assignments,
        )

        return build_sample_cluster_assignments(decomposition_results)
