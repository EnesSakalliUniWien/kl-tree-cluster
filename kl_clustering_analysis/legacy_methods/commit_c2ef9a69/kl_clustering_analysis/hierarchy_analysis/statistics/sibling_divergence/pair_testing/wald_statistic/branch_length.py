"""Branch-length preprocessing for sibling Wald testing."""

from __future__ import annotations

from ....branch_length_utils import sanitize_positive_branch_length


def _resolve_sibling_branch_length_sum(
    branch_length_left: float | None,
    branch_length_right: float | None,
    mean_branch_length: float | None,
) -> float | None:
    """Return the sibling branch-length sum when variance adjustment is enabled."""
    if mean_branch_length is None:
        return None

    sanitized_left = sanitize_positive_branch_length(branch_length_left)
    sanitized_right = sanitize_positive_branch_length(branch_length_right)
    if sanitized_left is None or sanitized_right is None:
        return None

    return sanitized_left + sanitized_right


__all__ = ["_resolve_sibling_branch_length_sum"]
