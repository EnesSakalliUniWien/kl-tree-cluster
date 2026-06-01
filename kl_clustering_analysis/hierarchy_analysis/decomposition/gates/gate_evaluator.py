"""Split-decision evaluator for tree decomposition.

:class:`GateEvaluator` encapsulates the structure prerequisite and statistical
gates that decide whether to split or stop at each internal node during
top-down traversal:

#. **Binary structure prerequisite** — parent must have exactly two children.
#. **Edge divergence gate** — at least one child must significantly
   diverge from the parent (projected Wald chi-square test).
#. **Sibling divergence gate** — siblings must have significantly different
   distributions.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kl_clustering_analysis.tree.poset_tree import PosetTree

from kl_clustering_analysis.core_utils.tree_utils import bottom_up_nodes


class TraversalDecision(Enum):
    """Top-down decomposition action for a tree node."""

    BOUNDARY = "boundary"
    SPLIT = "split"
    PASS_THROUGH = "pass_through"


class GateEvaluator:
    """Evaluate split-or-merge gates at each internal node.

    This is a lightweight, stateless-after-init object that holds the pre-
    computed annotation dictionaries and the tree reference.  It does NOT
    own the tree, the annotations, or any configuration — those are injected
    by the caller (typically :class:`TreeDecomposition`).

    Parameters
    ----------
    tree
        The hierarchy tree (a :class:`PosetTree`).
    edge_divergent
        ``{node_id: bool}`` — child-parent divergence significance.
    sibling_different
        ``{node_id: bool}`` — sibling BH-corrected divergence.
    sibling_skipped
        ``{node_id: bool}`` — whether the sibling test was skipped.
    children_map
        ``{node_id: [child_1, child_2, ...]}`` — pre-computed children list.
    """

    def __init__(
        self,
        tree: "PosetTree",
        edge_divergent: dict[object, bool],
        sibling_different: dict[object, bool],
        sibling_skipped: dict[object, bool],
        children_map: dict[object, list[object]],
        *,
        passthrough: bool = False,
    ) -> None:
        self.tree = tree
        self._edge_divergent = edge_divergent
        self._sibling_different = sibling_different
        self._sibling_skipped = sibling_skipped
        self._children_map = children_map
        self._passthrough = passthrough
        self._node_ids = tuple(self.tree.nodes)
        self._validate_contract()
        if self._passthrough:
            self._split_prerequisites_by_node = self._compute_split_prerequisites_by_node()
            self._can_split_by_node = self._compute_can_split_by_node(
                self._split_prerequisites_by_node
            )
            self._has_descendant_split = self._compute_has_descendant_split(
                self._can_split_by_node
            )
        else:
            self._split_prerequisites_by_node = {}
            self._can_split_by_node = {}
            self._has_descendant_split = {}

    # ------------------------------------------------------------------
    # Shared gate logic
    # ------------------------------------------------------------------

    def _validate_contract(self) -> None:
        """Validate injected maps once at the component boundary."""
        node_ids = self._node_ids
        for name, mapping in (
            ("children_map", self._children_map),
            ("edge_divergent", self._edge_divergent),
            ("sibling_different", self._sibling_different),
            ("sibling_skipped", self._sibling_skipped),
        ):
            missing = [node for node in node_ids if node not in mapping]
            if missing:
                preview = ", ".join(map(repr, missing[:5]))
                raise ValueError(f"Missing {name} values for nodes: {preview}.")

    def _passes_split_prerequisites(self, parent: object) -> bool:
        """Run the binary-structure prerequisite and edge-divergence gate.

        Returns
        -------
        bool
            ``True`` only when the node is binary and at least one child
            significantly diverges from the parent.
        """
        children = self._children_map[parent]
        if len(children) != 2:
            return False

        left_child, right_child = children

        return bool(
            self._edge_divergent[left_child] or self._edge_divergent[right_child]
        )

    def _sibling_gate_is_open(self, parent: object) -> bool:
        """Run the sibling-divergence gate.

        Returns ``True`` when siblings are significantly different and the
        test was not skipped.
        """
        if self._sibling_skipped[parent]:
            return False

        return bool(self._sibling_different[parent])

    def _can_split(self, parent: object) -> bool:
        """Return whether all split gates are open for *parent*."""
        return self._passes_split_prerequisites(parent) and self._sibling_gate_is_open(parent)

    def _compute_split_prerequisites_by_node(self) -> dict[object, bool]:
        """Return cached binary-structure plus edge-gate status for every node."""
        return {node: self._passes_split_prerequisites(node) for node in self._node_ids}

    def _compute_can_split_by_node(
        self,
        split_prerequisites_by_node: dict[object, bool],
    ) -> dict[object, bool]:
        """Return cached full split-gate status for every node."""
        return {
            node: (
                split_prerequisites_by_node[node] and self._sibling_gate_is_open(node)
            )
            for node in self._node_ids
        }

    def _compute_has_descendant_split(
        self,
        can_split_by_node: dict[object, bool],
    ) -> dict[object, bool]:
        """Return whether each node has a descendant that can split under all gates."""
        has_split: dict[object, bool] = {}
        for node in bottom_up_nodes(self.tree):
            has_split[node] = any(
                can_split_by_node[child] or has_split[child]
                for child in self._children_map[node]
            )
        return has_split

    def decision(self, parent: object) -> TraversalDecision:
        """Evaluate the top-down clustering action for *parent*.

        ``SPLIT`` and ``PASS_THROUGH`` both continue traversal into the two
        children. ``BOUNDARY`` means the node's descendant leaves form a final
        cluster.
        """
        if self._passthrough:
            if not self._split_prerequisites_by_node[parent]:
                return TraversalDecision.BOUNDARY
            if self._can_split_by_node[parent]:
                return TraversalDecision.SPLIT
            if self._has_descendant_split[parent]:
                return TraversalDecision.PASS_THROUGH
            return TraversalDecision.BOUNDARY

        if not self._passes_split_prerequisites(parent):
            return TraversalDecision.BOUNDARY

        if self._sibling_gate_is_open(parent):
            return TraversalDecision.SPLIT

        return TraversalDecision.BOUNDARY
