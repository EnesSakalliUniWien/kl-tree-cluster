"""Data models for the Tree-BH hierarchical multiple-testing procedure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TreeBHSiblingGroupOutcome:
    """Outcome of applying BH within one sibling group."""

    depth: int
    sibling_group_alpha: float
    tested_child_ids: list[str]
    raw_p_values: list[float]
    child_hypotheses_rejected_by_bh: list[bool]

    @property
    def rejection_count(self) -> int:
        return int(sum(self.child_hypotheses_rejected_by_bh))

    @property
    def test_count(self) -> int:
        return len(self.raw_p_values)


@dataclass(frozen=True)
class ChildParentEdgeTreeBHResult:
    """Tree-BH correction output for Gate 2 child-parent edge hypotheses."""

    child_parent_edge_null_rejected_by_tree_bh: np.ndarray
    child_parent_edge_corrected_p_values_by_tree_bh: np.ndarray
    child_parent_edge_tested_by_tree_bh: np.ndarray
    tree_bh_base_alpha_by_depth: dict[int, float]
    sibling_group_outcomes: dict[str, TreeBHSiblingGroupOutcome]


__all__ = ["TreeBHSiblingGroupOutcome", "ChildParentEdgeTreeBHResult"]
