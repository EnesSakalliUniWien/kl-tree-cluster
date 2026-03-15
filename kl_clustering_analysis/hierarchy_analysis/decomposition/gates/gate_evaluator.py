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

from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from kl_clustering_analysis.tree.poset_tree import PosetTree


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
    descendant_leaf_sets
        ``{node_id: set_of_leaf_labels}`` — pre-computed leaf partitions.
    """

    def __init__(
        self,
        tree: "PosetTree",
        local_significant: Dict[str, bool],
        sibling_different: Dict[str, bool],
        sibling_skipped: Dict[str, bool],
        children_map: Dict[str, List[str]],
        descendant_leaf_sets: Dict[str, frozenset],
        *,
        has_descendant_split: Dict[str, bool] | None = None,
        passthrough: bool = False,
    ) -> None:
        self.tree = tree
        self._local_significant = local_significant
        self._sibling_different = sibling_different
        self._sibling_skipped = sibling_skipped
        self._children_map = children_map
        self._descendant_leaf_sets = descendant_leaf_sets
        self._has_descendant_split = has_descendant_split or {}
        self._passthrough = passthrough

    # ------------------------------------------------------------------
    # Shared gate logic
    # ------------------------------------------------------------------

    def _evaluate_gates_1_and_2(self, parent: str) -> Tuple[bool, str | None, str | None]:
        """Run Gates 1 (binary structure) and 2 (child-parent divergence).

        Returns
        -------
        (passed, left_child, right_child)
            ``passed`` is ``True`` only when both gates are open.
            When ``False``, the children are ``None``.
        """
        children = self._children_map[parent]
        if len(children) != 2:
            return False, None, None

        left_child, right_child = children

        left_diverges = self._local_significant.get(left_child)
        right_diverges = self._local_significant.get(right_child)

        if left_diverges is None or right_diverges is None:
            raise ValueError(
                "Missing child-parent divergence annotations for "
                f"{left_child!r} or {right_child!r}; annotate before decomposing."
            )

        if not (left_diverges or right_diverges):
            return False, None, None

        return True, left_child, right_child

    def _evaluate_gate_3(self, parent: str) -> bool:
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

    # ------------------------------------------------------------------
    # Gate evaluation — v1 (hard split)
    # ------------------------------------------------------------------

    def should_split(self, parent: str) -> bool:
        """Evaluate statistical gates and return ``True`` when *parent* should split.

        Gates evaluated in order:

        1. **Binary structure gate** — parent must have exactly 2 children.
        2. **Child-parent divergence gate** — at least one child must
           significantly diverge from the parent.
        3. **Sibling divergence gate** — siblings must be significantly
           different.

        All gates must be OPEN to return ``True``.
        """
        passed, _, _ = self._evaluate_gates_1_and_2(parent)
        if not passed:
            return False
        return self._evaluate_gate_3(parent)

    def should_pass_through(self, parent: str) -> bool:
        """Return ``True`` when the DFS should continue past a Gate 3 failure.

        This is checked only when :meth:`should_split` returns ``False`` and
        ``passthrough`` mode is enabled.  The conditions are:

        1. Gates 1 and 2 **pass** (binary structure + edge signal).
        2. Gate 3 **fails** (siblings declared same or test skipped).
        3. At least one descendant has ``Sibling_BH_Different == True``
           (precomputed in ``has_descendant_split``).

        When all conditions hold, the traversal continues into both children
        instead of merging — eventually reaching the deeper split.
        """
        if not self._passthrough:
            return False
        passed, _, _ = self._evaluate_gates_1_and_2(parent)
        if not passed:
            return False
        # Gate 3 must be failing (if it passed, should_split would be True)
        if self._evaluate_gate_3(parent):
            return False
        return self._has_descendant_split.get(parent, False)
