"""Audit helpers for Gate 2 child-parent divergence annotation."""

from __future__ import annotations

import logging

import numpy as np


def build_child_parent_divergence_audit(
    *,
    total_tests: int,
    invalid_test_count: int,
    non_finite_p_value_count: int,
    child_parent_edge_tested_by_tree_bh,
    ancestor_blocked_edge_flags,
) -> dict[str, int]:
    """Build the persisted child-parent divergence audit payload."""
    return {
        "total_tests": int(total_tests),
        "invalid_tests": int(invalid_test_count),
        "non_finite_p_values": int(non_finite_p_value_count),
        "conservative_path_tests": int(non_finite_p_value_count),
        "tested_edges": int(child_parent_edge_tested_by_tree_bh.sum()),
        "ancestor_blocked_edges": int(ancestor_blocked_edge_flags.sum()),
    }


def log_non_finite_child_parent_divergence_audit(
    logger: logging.Logger,
    *,
    child_ids: list[str],
    edge_p_values,
    invalid_test_count: int,
    non_finite_p_value_count: int,
) -> None:
    """Log the conservative-path audit summary for invalid/non-finite edge tests."""
    if not (invalid_test_count or non_finite_p_value_count):
        return

    non_finite_p_value_indices = [
        edge_index for edge_index, p_value in enumerate(edge_p_values) if not np.isfinite(p_value)
    ]
    non_finite_p_value_node_ids = [
        child_ids[edge_index] for edge_index in non_finite_p_value_indices
    ]
    preview_node_ids = ", ".join(map(repr, non_finite_p_value_node_ids[:5]))

    logger.warning(
        "Child-parent divergence audit: total_tests=%d, invalid_tests=%d, "
        "non_finite_p_values=%d. Conservative correction path applied "
        "(p=1.0, reject=False) for nodes: %s",
        len(child_ids),
        invalid_test_count,
        non_finite_p_value_count,
        preview_node_ids,
    )


__all__ = [
    "build_child_parent_divergence_audit",
    "log_non_finite_child_parent_divergence_audit",
]
