"""Tree decomposition logic for KL-based clustering.

This module contains :class:`~kl_clustering_analysis.hierarchy_analysis.tree_decomposition.TreeDecomposition`,
which traverses a hierarchy and decides where to split or merge to form clusters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tree.poset_tree import PosetTree

import numpy as np
import pandas as pd

from .. import config
from ..core_utils.data_utils import extract_row_column_maps
from .cluster_assignments import build_cluster_assignments
from .decomposition.backends.random_projection_backend import (
    resolve_minimum_projection_dimension_backend,
)
from .decomposition.gates.gate_evaluator import GateEvaluator
from .decomposition.gates.orchestrator import run_gate_annotation_pipeline


class TreeDecomposition:
    """Annotate a hierarchy with significance tests and carve it into clusters.

    The decomposer walks a :class:`~tree.poset_tree.PosetTree` top-down and decides
    whether to split or merge at each internal node based on three statistical gates:

    #. **Binary structure gate** - parent must have exactly two children to split.
    #. **Child-parent divergence gate** - at least one child must significantly
       diverge from the parent (projected Wald chi-square test), confirming
       there is real signal to split on.
    #. **Sibling divergence gate** - siblings must have significantly different
       distributions according to a projected Wald chi-square test with
       Benjamini-Hochberg FDR correction.  If siblings are significantly
       different, the split proceeds; otherwise the children are merged
       into a single cluster.

    Nodes that pass all gates become cluster boundaries. Leaves under the same
    boundary node are assigned the same cluster identifier. The resulting report
    captures the cluster root node, member leaves, and cluster size.
    """

    def __init__(
        self,
        tree: PosetTree,
        annotations_df: pd.DataFrame | None = None,
        *,
        alpha_local: float = config.EDGE_ALPHA,
        sibling_alpha: float = config.SIBLING_ALPHA,
        leaf_data: pd.DataFrame | None = None,
        passthrough: bool = config.PASSTHROUGH,
    ):
        """Configure decomposition thresholds and pre-compute reusable metadata.

        Parameters
        ----------
        tree
            Directed hierarchy (typically a :class:`~tree.poset_tree.PosetTree`).
        annotations_df
            DataFrame of statistical annotations (e.g., columns produced by
            ``hierarchy_analysis.statistics`` helpers). May be ``None`` if the caller
            plans to rely on on-the-fly calculations.
        alpha_local
            Significance level used when the local Kullback-Leibler divergence gate
            falls back to raw chi-square tests.
        sibling_alpha
            Significance level used by sibling-independence annotations and gating.
        leaf_data
            Raw binary data matrix (samples × features).  Required for per-node
            spectral dimension estimation.  When ``None``, spectral projection
            is disabled and tests are skipped (treated as merge).
        """
        self.tree = tree
        self.annotations_df = annotations_df if annotations_df is not None else pd.DataFrame()
        self.alpha_local = float(alpha_local)
        self.sibling_alpha = float(sibling_alpha)
        self._leaf_data = leaf_data
        resolve_minimum_projection_dimension_backend(
            config.PROJECTION_MINIMUM_DIMENSION,
            leaf_data=leaf_data,
        )

        # ----- root -----
        self._root = next(node_id for node_id, degree in self.tree.in_degree() if degree == 0)

        # ----- pre-cache node metadata -----
        self._cache_node_metadata()

        # ----- leaf partitions & counts (poset view) -----
        self._descendant_leaf_sets = self.tree.compute_descendant_sets(use_labels=True)

        # ----- ensure statistical annotations are present -----
        self.annotations_df = self._prepare_annotations(self.annotations_df)

        # ----- annotations_df → fast row/column dictionary lookups (no .loc in hot paths) -----
        self._annotations_by_row, self._annotations_by_column = extract_row_column_maps(
            self.annotations_df
        )
        self._local_significant = self._extract_required_bool_annotation_column(
            "Child_Parent_Divergence_Significant"
        )
        # Sibling divergence test: Sibling_BH_Different = True means siblings differ -> SPLIT
        self._sibling_different = self._extract_required_bool_annotation_column(
            "Sibling_BH_Different"
        )

        self._sibling_skipped = self._extract_required_bool_annotation_column(
            "Sibling_Divergence_Skipped"
        )

        for column_name, mapping in (
            ("Child_Parent_Divergence_Significant", self._local_significant),
            ("Sibling_BH_Different", self._sibling_different),
            ("Sibling_Divergence_Skipped", self._sibling_skipped),
        ):
            missing = set(self._node_ids) - set(mapping.keys())
            if missing:
                preview = ", ".join(map(repr, list(missing)[:5]))
                raise ValueError(f"Missing {column_name!r} values for nodes: {preview}.")

        # Precompute children list (avoids rebuilding generator repeatedly)
        self._children: dict[str, list[str]] = {
            n: list(self.tree.successors(n)) for n in self._node_ids
        }

        # ----- precompute has_descendant_split (bottom-up O(n)) -----
        self._passthrough = bool(passthrough)
        self._has_descendant_split: dict[str, bool] = {}
        if self._passthrough:
            self._has_descendant_split = self._compute_has_descendant_split()

        # ----- construct the GateEvaluator -----
        self._gate = GateEvaluator(
            tree=self.tree,
            local_significant=self._local_significant,
            sibling_different=self._sibling_different,
            sibling_skipped=self._sibling_skipped,
            children_map=self._children,
            descendant_leaf_sets=self._descendant_leaf_sets,
            has_descendant_split=self._has_descendant_split,
            passthrough=self._passthrough,
        )

    # ---------- initialization helpers ----------

    def _compute_has_descendant_split(self) -> dict[str, bool]:
        """Precompute bottom-up flag: does any descendant have a significant sibling split?

        For each internal node, ``has_descendant_split[node]`` is ``True``
        when at least one descendant (not the node itself) has
        ``Sibling_BH_Different == True`` and was not skipped.

        Computed in a single reverse-topological pass (O(n)):

        - Leaves → ``False``
        - Internal node → ``True`` if any child is itself a split node or
          has ``has_descendant_split == True``.

        Returns
        -------
        dict[str, bool]
            Mapping from node ID to boolean flag.
        """
        from ..core_utils.tree_utils import bottom_up_nodes

        has_split: dict[str, bool] = {}

        for node in bottom_up_nodes(self.tree):
            children = self._children.get(node, [])
            if not children:
                # Leaf node
                has_split[node] = False
                continue

            # Check if any child is itself a split point or has a descendant split
            found = False
            for child in children:
                # Child itself is a significant split point
                child_is_split = self._sibling_different.get(
                    child, False
                ) and not self._sibling_skipped.get(child, True)
                if child_is_split or has_split.get(child, False):
                    found = True
                    break

            has_split[node] = found

        return has_split

    def _prepare_annotations(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure statistical annotation columns are present on *annotations_df*.

        Delegates directly to canonical gate orchestrator.
        """
        return run_gate_annotation_pipeline(
            self.tree,
            annotations_df,
            alpha_local=self.alpha_local,
            sibling_alpha=self.sibling_alpha,
            leaf_data=self._leaf_data,
            sibling_method=config.SIBLING_TEST_METHOD,
            sibling_whitening=config.SIBLING_WHITENING,
        ).annotated_df

    def _extract_required_bool_annotation_column(self, column_name: str) -> dict[str, bool]:
        """Extract a required boolean annotation column from cached column mappings."""
        if column_name not in self._annotations_by_column:
            raise KeyError(f"Missing required column {column_name!r} in dataframe.")

        raw_mapping = self._annotations_by_column[column_name]
        missing_nodes = [node_id for node_id, value in raw_mapping.items() if pd.isna(value)]
        if missing_nodes:
            preview = ", ".join(map(repr, missing_nodes[:5]))
            raise ValueError(
                f"Column {column_name!r} contains missing values for nodes: {preview}. "
                "Ensure all nodes are annotated before extraction."
            )

        return {node_id: bool(value) for node_id, value in raw_mapping.items()}

    def _cache_node_metadata(self) -> None:
        """Cache node attributes for fast repeated access during decomposition.

        Extracts and stores distributions, leaf flags, and labels.
        This one-time preprocessing avoids expensive NetworkX lookups in tight loops.
        """
        self._node_ids: tuple[str, ...] = tuple()
        self._node_attrs_by_id: dict[str, dict] = {}
        self._distribution_by_node: dict[str, np.ndarray] = {}
        self._is_leaf: dict[str, bool] = {}
        self._label: dict[str, str] = {}

        node_ids: list[str] = []
        for node_id, node_data in self.tree.nodes(data=True):
            node_ids.append(node_id)
            self._node_attrs_by_id[node_id] = node_data
            # Preserve shape: 1D for binary (Bernoulli), 2D for categorical (multinomial)
            distribution_array = np.asarray(node_data["distribution"], dtype=float)
            self._distribution_by_node[node_id] = distribution_array
            self._is_leaf[node_id] = node_data["is_leaf"]
            self._label[node_id] = node_data.get("label", node_id)
        self._node_ids = tuple(node_ids)

    # ---------- utilities ----------

    # ---------- LCA ----------

    def _find_cluster_root(self, leaf_labels: set[str]) -> str:
        """Identify the lowest common ancestor for a collection of leaf labels.

        This implementation delegates the LCA calculation to the tree object, which is
        expected to have an efficient implementation, and caches a label-to-node mapping
        to accelerate leaf lookups.
        """
        if not hasattr(self, "_label_to_node_map"):
            # This is a one-time cache population for all leaves.
            self._label_to_node_map = {
                self._label[n]: n for n in self._node_ids if self._is_leaf.get(n)
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

    # ---------- core decomposition (iterative, no recursion) ----------

    def decompose_tree(self) -> dict[str, object]:
        """Return cluster assignments by iteratively traversing the hierarchy.

        Traversal order
        ---------------
        The traversal uses a last in, first out list (similar to an explicit stack),
        which produces a depth-first traversal order. When a node is split, its
        two children are appended in right-then-left order so that the left child
        is processed first on the next iteration.
        """
        nodes_to_visit: list[str] = [self._root]
        final_leaf_sets: list[set[str]] = []
        processed: set[str] = set()

        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if node in processed:
                continue
            processed.add(node)

            if self._gate.should_split(node) or self._gate.should_pass_through(node):
                left_child, right_child = self._children[node]
                nodes_to_visit.append(right_child)
                nodes_to_visit.append(left_child)
                continue

            final_leaf_sets.append(set(self._descendant_leaf_sets[node]))

        cluster_assignments = build_cluster_assignments(
            final_leaf_sets,
            self._find_cluster_root,
        )

        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "independence_analysis": {
                "alpha_local": self.alpha_local,
                "decision_mode": "sibling_divergence",
            },
        }
