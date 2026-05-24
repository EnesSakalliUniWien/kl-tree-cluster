"""Construction helpers for sibling pair records."""

from __future__ import annotations

import numpy as np

from ..types.sibling_pair_record import SiblingPairRecord


def build_sibling_pair_record(
    *,
    parent_node_id: object,
    left_child_id: object,
    right_child_id: object,
    test_statistic: float,
    reference_scale: float,
    degrees_of_freedom: float,
    p_value: float,
    branch_length_sum: float,
    parent_sample_size: int,
    is_null_like: bool,
    is_edge_blocked: bool,
    sibling_null_weight: float,
    sibling_projection_dimension: float,
) -> SiblingPairRecord:
    """Construct a sibling-pair record from resolved statistical inputs."""
    if not np.isfinite(test_statistic):
        raise ValueError(
            "Sibling pair record requires a finite test statistic; "
            f"parent={parent_node_id!r}."
        )
    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError(
            "Sibling pair record requires a finite positive reference_scale; "
            f"parent={parent_node_id!r}."
        )
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom < 0:
        raise ValueError(
            "Sibling pair record requires finite non-negative degrees of freedom; "
            f"parent={parent_node_id!r}."
        )
    if not np.isfinite(p_value) or p_value < 0.0 or p_value > 1.0:
        raise ValueError(
            "Sibling pair record requires a finite p-value in [0, 1]; "
            f"parent={parent_node_id!r}."
        )
    if not np.isfinite(sibling_null_weight) or not 0.0 <= sibling_null_weight <= 1.0:
        raise ValueError(
            "Sibling pair record requires a finite sibling_null_weight in [0, 1]; "
            f"parent={parent_node_id!r}."
        )
    if not np.isfinite(sibling_projection_dimension) or sibling_projection_dimension < 0.0:
        raise ValueError(
            "Sibling pair record requires a finite non-negative sibling_projection_dimension; "
            f"parent={parent_node_id!r}."
        )
    return SiblingPairRecord(
        parent=parent_node_id,
        left=left_child_id,
        right=right_child_id,
        stat=test_statistic,
        reference_scale=float(reference_scale),
        degrees_of_freedom=float(degrees_of_freedom),
        p_value=p_value,
        branch_length_sum=branch_length_sum,
        n_parent=parent_sample_size,
        is_null_like=is_null_like,
        is_edge_blocked=is_edge_blocked,
        sibling_null_weight=sibling_null_weight,
        sibling_projection_dimension=sibling_projection_dimension,
    )


__all__ = ["build_sibling_pair_record"]
