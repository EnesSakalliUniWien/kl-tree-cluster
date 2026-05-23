"""Audit helpers for Gate 2 child-parent divergence annotation."""

from __future__ import annotations

def build_child_parent_divergence_audit(
    *,
    total_tests: int,
    child_parent_edge_tested_by_tree_bh,
    ancestor_blocked_edge_flags,
) -> dict[str, int]:
    """Build the persisted child-parent divergence audit payload."""
    return {
        "total_tests": int(total_tests),
        "tested_edges": int(child_parent_edge_tested_by_tree_bh.sum()),
        "ancestor_blocked_edges": int(ancestor_blocked_edge_flags.sum()),
    }


__all__ = ["build_child_parent_divergence_audit"]
