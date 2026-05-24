"""Branch-length preprocessing for sibling Wald testing."""

from __future__ import annotations

from ....branch_length_utils import compute_sibling_branch_length_sum


def _resolve_sibling_branch_length_sum(
    branch_length_left: float | None,
    branch_length_right: float | None,
    mean_branch_length: float | None,
) -> float | None:
    """Return the sibling branch-length sum when variance adjustment is enabled."""
    if mean_branch_length is None:
        return None

    branch_length_sum = compute_sibling_branch_length_sum(
        branch_length_left,
        branch_length_right,
    )
    if branch_length_sum == 0.0:
        return None

    return branch_length_sum


__all__ = ["_resolve_sibling_branch_length_sum"]
