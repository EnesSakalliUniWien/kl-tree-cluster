"""Construction helpers for sibling pair records."""

from __future__ import annotations

import numpy as np

from ..types import SiblingPairRecord


def build_sibling_pair_record(
    *,
    parent_node_id: str,
    left_child_id: str,
    right_child_id: str,
    test_statistic: float,
    degrees_of_freedom: float,
    p_value: float,
    branch_length_sum: float,
    parent_sample_size: int,
    is_null_like: bool,
    is_gate2_blocked: bool,
    sibling_null_prior_from_edge_pvalue: float,
    sibling_test_calibration_scale: float,
    projection_dimension_source: str,
    resolved_projection_dimension: float,
    used_parent_principal_component_basis: bool,
) -> SiblingPairRecord:
    """Construct a sibling-pair record from resolved statistical inputs."""
    return SiblingPairRecord(
        parent=parent_node_id,
        left=left_child_id,
        right=right_child_id,
        stat=test_statistic,
        degrees_of_freedom=float(degrees_of_freedom) if np.isfinite(degrees_of_freedom) else 0.0,
        p_value=p_value,
        branch_length_sum=branch_length_sum,
        n_parent=parent_sample_size,
        is_null_like=is_null_like,
        is_gate2_blocked=is_gate2_blocked,
        sibling_null_prior_from_edge_pvalue=sibling_null_prior_from_edge_pvalue,
        sibling_test_calibration_scale=sibling_test_calibration_scale,
        projection_dimension_source=projection_dimension_source,
        resolved_projection_dimension=resolved_projection_dimension,
        used_parent_principal_component_basis=used_parent_principal_component_basis,
    )


__all__ = ["build_sibling_pair_record"]
