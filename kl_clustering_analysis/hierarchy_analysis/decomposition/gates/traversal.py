"""Traversal helpers for tree decomposition.

Provides ``process_node`` and ``iterate_worklist`` — stateless functions
used by :class:`TreeDecomposition` during the top-down DFS walk.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gate_evaluator import GateEvaluator

# ======================================================================
# Traversal helpers (stateless functions)
# ======================================================================


def process_node(
    node_id: str,
    gate: GateEvaluator,
    nodes_to_visit: list[str],
    final_leaf_sets: list[set[str]],
) -> None:
    """Apply split, pass-through, or merge decision for one node during DFS.

    Decision order:

    1. **SPLIT** — all three gates pass → push both children.
    2. **PASS-THROUGH** — Gates 1+2 pass, Gate 3 fails, but a descendant
       has a significant sibling split → push both children so the DFS
       can reach the deeper structure.
    3. **MERGE** — otherwise, collect all descendant leaves as one cluster.

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
    elif gate.should_pass_through(node_id):
        children = gate._children_map[node_id]
        left_child, right_child = children
        nodes_to_visit.append(right_child)
        nodes_to_visit.append(left_child)
    else:
        final_leaf_sets.append(set(gate._descendant_leaf_sets[node_id]))


def iterate_worklist(
    nodes_to_visit: list[str],
    processed: set[str],
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
