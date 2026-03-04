"""Statistical gate evaluation for tree decomposition.

This module provides :class:`GateEvaluator`, which encapsulates the three
statistical gates that decide whether to split or merge at each internal
node during top-down traversal:

#. **Binary structure gate** — parent must have exactly two children.
#. **Child-parent divergence gate** — at least one child must significantly
   diverge from the parent (projected Wald chi-square test).
#. **Sibling divergence gate** — siblings must have significantly different
   distributions.

It also provides traversal helpers (``process_node``, ``iterate_worklist``)
that are used by :class:`TreeDecomposition` during the DFS walk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Hashable, Iterator, List, Set, Tuple

if TYPE_CHECKING:
    from ..tree.poset_tree import PosetTree


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
    root
        The root node identifier.
    """

    def __init__(
        self,
        tree: "PosetTree",
        local_significant: Dict[Hashable, bool],
        sibling_different: Dict[Hashable, bool],
        sibling_skipped: Dict[Hashable, bool],
        children_map: Dict[Hashable, List[str]],
        descendant_leaf_sets: Dict[Hashable, set],
        root: str,
    ) -> None:
        self.tree = tree
        self._local_significant = local_significant
        self._sibling_different = sibling_different
        self._sibling_skipped = sibling_skipped
        self._children_map = children_map
        self._descendant_leaf_sets = descendant_leaf_sets
        self._root = root

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


# ======================================================================
# Traversal helpers (stateless functions)
# ======================================================================


def process_node(
    node_id: str,
    gate: GateEvaluator,
    nodes_to_visit: List[str],
    final_leaf_sets: List[set[str]],
) -> None:
    """Apply split-or-merge decision for one node during DFS traversal.

    If the gate says SPLIT, the two children are pushed onto *nodes_to_visit*
    (right-then-left so left is processed first).  Otherwise, all descendant
    leaves are collected as a single cluster.

    Parameters
    ----------
    node_id
        Node to process.
    gate
        Gate evaluator with pre-loaded annotations.
    nodes_to_visit
        Mutable LIFO worklist.
    final_leaf_sets
        Accumulator for cluster leaf sets.
    """
    if gate.should_split(node_id):
        children = gate._children_map[node_id]
        left_child, right_child = children
        nodes_to_visit.append(right_child)
        nodes_to_visit.append(left_child)
    else:
        final_leaf_sets.append(set(gate._descendant_leaf_sets[node_id]))


def iterate_worklist(
    nodes_to_visit: List[str],
    processed: Set[str],
) -> Iterator[str]:
    """Yield nodes from a mutable LIFO list exactly once.

    Encapsulates the pop → skip-if-processed → mark-processed → yield
    pattern.  The caller may mutate *nodes_to_visit* between yields
    (e.g. by pushing children when a split occurs).
    """
    while nodes_to_visit:
        node_id = nodes_to_visit.pop()
        if node_id in processed:
            continue
        processed.add(node_id)
        yield node_id
