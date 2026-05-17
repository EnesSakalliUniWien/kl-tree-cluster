"""Statistical gate evaluator for tree decomposition.

:class:`GateEvaluator` encapsulates the three statistical gates that decide
whether to split or merge at each internal node during top-down traversal:

#. **Binary structure gate** — parent must have exactly two children.
#. **Child-parent divergence gate** — at least one child must significantly
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
    local_significant
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
        local_significant: dict[object, bool],
        sibling_different: dict[object, bool],
        sibling_skipped: dict[object, bool],
        children_map: dict[object, list[object]],
        *,
        passthrough: bool = False,
    ) -> None:
        self.tree = tree
        self._local_significant = local_significant
        self._sibling_different = sibling_different
        self._sibling_skipped = sibling_skipped
        self._children_map = children_map
        self._passthrough = passthrough
        self._has_descendant_split = (
            self._compute_has_descendant_split() if self._passthrough else {}
        )

    # ------------------------------------------------------------------
    # Shared gate logic
    # ------------------------------------------------------------------

    def _passes_split_prerequisites(self, parent: object) -> bool:
        """Run Gates 1 (binary structure) and 2 (child-parent divergence).

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

        left_diverges = self._local_significant.get(left_child)
        right_diverges = self._local_significant.get(right_child)

        if left_diverges is None or right_diverges is None:
            raise ValueError(
                "Missing child-parent divergence annotations for "
                f"{left_child!r} or {right_child!r}; annotate before decomposing."
            )

        return bool(left_diverges or right_diverges)

    def _sibling_gate_is_open(self, parent: object) -> bool:
        """Run Gate 3 (sibling divergence).

        Returns ``True`` when siblings are significantly different and the
        test was not skipped.
        """
        is_different = self._sibling_different.get(parent)

        if is_different is None:
            raise ValueError(
                "Sibling divergence annotations missing for node "
                f"{parent!r}; run annotate_sibling_divergence first."
            )

        if self._sibling_skipped.get(parent, False):
            return False

        return bool(is_different)

    def _can_split(self, parent: object) -> bool:
        """Return whether all split gates are open for *parent*."""
        return self._passes_split_prerequisites(parent) and self._sibling_gate_is_open(parent)

    def _compute_has_descendant_split(self) -> dict[object, bool]:
        """Return whether each node has a descendant that can split under all gates."""
        has_split: dict[object, bool] = {}
        for node in bottom_up_nodes(self.tree):
            has_split[node] = any(
                self._can_split(child) or has_split.get(child, False)
                for child in self._children_map.get(node, [])
            )
        return has_split

    def decision(self, parent: object) -> TraversalDecision:
        """Evaluate the top-down clustering action for *parent*.

        ``SPLIT`` and ``PASS_THROUGH`` both continue traversal into the two
        children. ``BOUNDARY`` means the node's descendant leaves form a final
        cluster.
        """
        if not self._passes_split_prerequisites(parent):
            return TraversalDecision.BOUNDARY

        if self._sibling_gate_is_open(parent):
            return TraversalDecision.SPLIT

        if not self._passthrough:
            return TraversalDecision.BOUNDARY

        if self._has_descendant_split.get(parent, False):
            return TraversalDecision.PASS_THROUGH

        return TraversalDecision.BOUNDARY
