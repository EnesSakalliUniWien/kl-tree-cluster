"""Tree decomposition logic for KL-based clustering.

This module contains :class:`~kl_clustering_analysis.hierarchy_analysis.tree_decomposition.TreeDecomposition`,
which traverses a hierarchy and decides where to split or merge to form clusters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tree.poset_tree import PosetTree

import pandas as pd

from .. import config
from ..core_utils.data_utils import extract_bool_column_dict
from .cluster_assignments import ClusterBoundary, build_cluster_assignments
from .decomposition.gates.annotation_bundle import GateAnnotationBundle
from .decomposition.gates.column_contracts import (
    validate_edge_gate_columns,
    validate_sibling_gate_columns,
)
from .decomposition.gates.gate_evaluator import GateEvaluator, TraversalDecision
from .decomposition.gates.orchestrator import (
    build_gate_annotation_config_metadata,
    build_gate_annotation_leaf_data_metadata,
    run_gate_annotation_pipeline,
)


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
        gate_annotation_bundle: GateAnnotationBundle | None = None,
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
            ``hierarchy_analysis.statistics`` helpers). Used as input to the
            gate annotation pipeline.
        gate_annotation_bundle
            Explicit reusable output from ``run_gate_annotation_pipeline``.
            This is the only cache-valid gate annotation contract.
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
        if annotations_df is not None and gate_annotation_bundle is not None:
            raise ValueError("Pass either annotations_df or gate_annotation_bundle, not both.")

        self.tree = tree
        self._gate_annotation_bundle = gate_annotation_bundle
        if gate_annotation_bundle is not None:
            self.annotations_df = gate_annotation_bundle.annotated_df
        elif annotations_df is not None:
            self.annotations_df = annotations_df
        else:
            self.annotations_df = pd.DataFrame()
        self.alpha_local = float(alpha_local)
        self.sibling_alpha = float(sibling_alpha)
        self._leaf_data = leaf_data

        # ----- root -----
        self._root = self.tree.root()

        self._node_ids = tuple(self.tree.nodes)

        # ----- leaf partitions & counts (poset view) -----
        self._descendant_leaf_sets = self.tree.compute_descendant_sets(use_labels=True)

        # ----- ensure statistical annotations are present -----
        self.annotations_df = self._prepare_annotations(self.annotations_df)

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

        # Precompute children list (avoids rebuilding generator repeatedly)
        self._children: dict[object, list[object]] = {
            n: list(self.tree.successors(n)) for n in self._node_ids
        }

        # ----- construct the GateEvaluator -----
        self._gate = GateEvaluator(
            tree=self.tree,
            local_significant=self._local_significant,
            sibling_different=self._sibling_different,
            sibling_skipped=self._sibling_skipped,
            children_map=self._children,
            passthrough=bool(passthrough),
        )

    # ---------- initialization helpers ----------

    def _prepare_annotations(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure statistical annotation columns are present on *annotations_df*.

        Reuses precomputed gate annotations only when they are supplied as an
        explicit ``GateAnnotationBundle`` whose metadata matches this
        decomposition request.
        """
        if self._gate_annotation_bundle is not None and self._can_reuse_gate_annotation_bundle(
            self._gate_annotation_bundle
        ):
            return self._gate_annotation_bundle.annotated_df

        annotation_bundle = run_gate_annotation_pipeline(
            self.tree,
            annotations_df,
            alpha_local=self.alpha_local,
            sibling_alpha=self.sibling_alpha,
            leaf_data=self._leaf_data,
        )
        self._gate_annotation_bundle = annotation_bundle
        return annotation_bundle.annotated_df

    def _can_reuse_gate_annotation_bundle(
        self,
        gate_annotation_bundle: GateAnnotationBundle,
    ) -> bool:
        """Return whether existing gate annotations can be trusted as current."""
        annotations_df = gate_annotation_bundle.annotated_df
        if annotations_df.empty:
            return False

        validate_edge_gate_columns(annotations_df)
        validate_sibling_gate_columns(annotations_df)

        required_gate_decision_columns = (
            "Child_Parent_Divergence_Significant",
            "Sibling_BH_Different",
            "Sibling_Divergence_Skipped",
        )
        if any(column not in annotations_df.columns for column in required_gate_decision_columns):
            return False
        if any(annotations_df[column].isna().any() for column in required_gate_decision_columns):
            return False
        if set(self._node_ids) - set(annotations_df.index):
            return False

        metadata = gate_annotation_bundle.metadata

        return (
            metadata.pipeline == "gate_annotation"
            and metadata.edge.alpha == self.alpha_local
            and metadata.sibling.alpha == self.sibling_alpha
            and metadata.config == build_gate_annotation_config_metadata()
            and metadata.leaf_data == build_gate_annotation_leaf_data_metadata(self._leaf_data)
        )

    def _extract_required_bool_annotation_column(self, column_name: str) -> dict[object, bool]:
        """Extract a required boolean annotation column keyed by tree node id."""
        return extract_bool_column_dict(
            self.annotations_df,
            column_name,
            coerce_index_to_str=False,
        )

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
        nodes_to_visit: list[object] = [self._root]
        final_boundaries: list[ClusterBoundary] = []
        processed: set[object] = set()

        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if node in processed:
                continue
            processed.add(node)

            decision = self._gate.decision(node)
            if decision in (TraversalDecision.SPLIT, TraversalDecision.PASS_THROUGH):
                left_child, right_child = self._children[node]
                nodes_to_visit.append(right_child)
                nodes_to_visit.append(left_child)
                continue

            final_boundaries.append(
                ClusterBoundary(root_node=node, leaves=self._descendant_leaf_sets[node])
            )

        cluster_assignments = build_cluster_assignments(final_boundaries)

        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "independence_analysis": {
                "alpha_local": self.alpha_local,
                "decision_mode": "sibling_divergence",
            },
        }
