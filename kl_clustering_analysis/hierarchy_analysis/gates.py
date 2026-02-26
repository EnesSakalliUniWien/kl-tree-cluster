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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Set, Tuple

if TYPE_CHECKING:
    from ..tree.poset_tree import PosetTree

from .signal_localization import LocalizationResult, localize_divergence_signal


@dataclass
class V2TraversalState:
    """Mutable accumulators populated during v2 DFS traversal."""

    split_points: List[Tuple[str, str, str]]  # (parent, left, right)
    merge_points: List[str]
    localization_results: Dict[str, LocalizationResult]


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
        local_significant: Dict[str, bool],
        sibling_different: Dict[str, bool],
        sibling_skipped: Dict[str, bool],
        children_map: Dict[str, List[str]],
        descendant_leaf_sets: Dict[str, set],
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

    # ------------------------------------------------------------------
    # Gate evaluation — v2 (with signal localization)
    # ------------------------------------------------------------------

    def should_split_v2(
        self,
        parent: str,
        *,
        test_divergence: Callable[[str, str], Tuple[float, float, float]],
        sibling_alpha: float,
        localization_max_depth: int | None = None,
        localization_max_pairs: int | None = None,
    ) -> Tuple[bool, LocalizationResult | None]:
        """Enhanced split decision with depth-1 signal localization.

        Like :meth:`should_split` but when siblings are "different", expands
        one level into their immediate children and tests all cross-boundary
        pairs to identify which sub-parts are similar vs. different.  Every
        cross-product test is terminal — no further drilling occurs.

        Parameters
        ----------
        parent
            The parent node to evaluate.
        test_divergence
            Callable ``(node_a, node_b) -> (stat, df, p_value)`` for pairwise
            divergence testing (used by signal localization).
        sibling_alpha
            Significance level for the localization tests.
        localization_max_depth
            Maximum recursion depth for signal localization.
        localization_max_pairs
            Maximum terminal cross-boundary pairs per localization call.

        Returns
        -------
        Tuple[bool, LocalizationResult | None]
            ``(should_split, localization_result)``
        """
        passed, left_child, right_child = self._evaluate_gates_1_and_2(parent)
        if not passed:
            return False, None

        if not self._evaluate_gate_3(parent):
            return False, None

        # Siblings ARE different — LOCALIZE the signal
        localization_result = localize_divergence_signal(
            tree=self.tree,
            left_root=left_child,
            right_root=right_child,
            test_divergence=test_divergence,
            alpha=sibling_alpha,
            max_depth=localization_max_depth,
            is_edge_significant=self._check_edge_significance,
            max_pairs=localization_max_pairs,
        )

        # Localization power guard: the aggregate Gate 3 said "different",
        # but after drilling down and applying BH correction on the finer-
        # grained sub-pair tests, NONE of the sub-pairs remained significant.
        # This does NOT mean the siblings are the same — the aggregate test
        # has higher power (one test, pooled signal) while localization
        # fragments the signal into many sub-tests with a BH penalty.
        # We trust the aggregate: SPLIT, but discard the localization result
        # to prevent misleading similarity edges from causing cross-boundary
        # merges.  Returning (True, None) triggers a hard v1-style split.
        if len(localization_result.difference_pairs) == 0:
            return True, None

        return True, localization_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_edge_significance(self, node_id: str) -> bool:
        """Check if a node is significantly different from its parent.

        Used as a callback for signal localization.
        """
        if node_id == self._root:
            return True
        return bool(self._local_significant.get(node_id, False))


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


def process_node_v2(
    node_id: str,
    gate: GateEvaluator,
    nodes_to_visit: List[str],
    state: V2TraversalState,
    *,
    test_divergence: Callable[[str, str], Tuple[float, float, float]],
    sibling_alpha: float,
    localization_max_depth: int | None = None,
    localization_max_pairs: int | None = None,
) -> None:
    """Apply v2 split-or-merge decision for one node during DFS traversal.

    Like :func:`process_node` but collects split/merge metadata and
    localization results needed by the v2 decomposition pipeline.

    Parameters
    ----------
    node_id
        Node to process.
    gate
        Gate evaluator with pre-loaded annotations.
    nodes_to_visit
        Mutable LIFO worklist.
    state
        Mutable accumulators for split points, merge points, and
        localization results.
    test_divergence
        Callable ``(node_a, node_b) -> (stat, df, p_value)``.
    sibling_alpha
        Significance level for localization tests.
    localization_max_depth
        Maximum recursion depth for signal localization.
    localization_max_pairs
        Maximum terminal cross-boundary pairs per localization call.
    """
    should_split, loc_result = gate.should_split_v2(
        node_id,
        test_divergence=test_divergence,
        sibling_alpha=sibling_alpha,
        localization_max_depth=localization_max_depth,
        localization_max_pairs=localization_max_pairs,
    )

    if should_split:
        children = gate._children_map[node_id]
        left_child, right_child = children
        state.split_points.append((node_id, left_child, right_child))

        if loc_result is not None:
            state.localization_results[node_id] = loc_result

        # Push right-then-left so left is processed first
        nodes_to_visit.append(right_child)
        nodes_to_visit.append(left_child)
    else:
        state.merge_points.append(node_id)
