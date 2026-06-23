"""Audit serialization for the Gate 2 single-feature subtree policy."""

from __future__ import annotations

from typing import Any

from tree_break_selection.legacy_methods.commit_c2ef9a69.tree_break_selection import config


def _serialize_single_feature_subtree_candidate(
    candidate: dict[str, Any],
    *,
    low_node_ids: set[str],
    blocked_node_ids: set[str],
) -> dict[str, Any]:
    """Convert internal candidate state into the persisted audit schema."""
    node_id = str(candidate["node_id"])
    return {
        "node_id": node_id,
        "active_feature": int(candidate["active_feature"]),
        "active_variance": float(candidate["active_variance"]),
        "parent_node": candidate["parent_node"],
        "parent_lambda_max": (
            float(candidate["parent_lambda_max"])
            if candidate["parent_lambda_max"] is not None
            else None
        ),
        "variance_ratio": (
            float(candidate["variance_ratio"]) if candidate["variance_ratio"] is not None else None
        ),
        "n_leaves": int(candidate["n_leaves"]),
        "n_rows": int(candidate["n_rows"]),
        "internal_rows": int(candidate["internal_rows"]),
        "is_low_leverage": node_id in low_node_ids,
        "allowed_one_active_1d": node_id not in blocked_node_ids,
    }


def _build_single_feature_subtree_audit_payload(
    single_feature_candidates: list[dict[str, Any]],
    *,
    has_low_group: bool,
    low_ratio_threshold: float | None,
    low_node_ids: set[str],
    group_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the persisted audit payload for the policy."""
    blocked_node_id_set = group_summary["blocked_node_id_set"]
    return {
        "mode": str(getattr(config, "SINGLE_FEATURE_SUBTREE_MODE", "off")),
        "candidate_nodes": int(len(single_feature_candidates)),
        "has_low_group": bool(has_low_group),
        "low_ratio_threshold": (
            float(low_ratio_threshold) if low_ratio_threshold is not None else None
        ),
        "low_count": int(group_summary["low_count"]),
        "high_count": int(group_summary["high_count"]),
        "low_rows": int(group_summary["low_rows"]),
        "high_rows": int(group_summary["high_rows"]),
        "low_internal_rows": int(group_summary["low_internal_rows"]),
        "dangerous_tree": bool(group_summary["dangerous_tree"]),
        "allowed_node_ids": group_summary["allowed_node_ids"],
        "blocked_node_ids": group_summary["blocked_node_ids"],
        "nodes": [
            _serialize_single_feature_subtree_candidate(
                candidate,
                low_node_ids=low_node_ids,
                blocked_node_ids=blocked_node_id_set,
            )
            for candidate in single_feature_candidates
        ],
    }
