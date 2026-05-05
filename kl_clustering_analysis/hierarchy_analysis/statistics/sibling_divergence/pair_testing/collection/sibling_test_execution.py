"""Execution wrapper for sibling divergence Wald tests."""

from __future__ import annotations

import numpy as np

from ..wald_statistic.sibling_divergence_test import sibling_divergence_test


def run_sibling_divergence_wald_test_with_diagnostics(
    *,
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    left_sample_size: float,
    right_sample_size: float,
    branch_length_left: float | None,
    branch_length_right: float | None,
    mean_branch_length: float | None,
    parent_node_id: str,
    projection_dimension_from_edge_comparisons: int | None,
    parent_principal_component_projection: np.ndarray | None,
    parent_principal_component_eigenvalues: np.ndarray | None,
) -> tuple[float, float, float, dict[str, object]]:
    """Run the sibling Wald test and return projection diagnostics."""
    projection_diagnostics: dict[str, object] = {}
    test_statistic, degrees_of_freedom, p_value = sibling_divergence_test(
        left_distribution,
        right_distribution,
        float(left_sample_size),
        float(right_sample_size),
        branch_length_left=branch_length_left,
        branch_length_right=branch_length_right,
        mean_branch_length=mean_branch_length,
        test_id=f"sibling:{parent_node_id}",
        projection_dimension_from_edge_comparisons=projection_dimension_from_edge_comparisons,
        parent_principal_component_projection=parent_principal_component_projection,
        parent_principal_component_eigenvalues=parent_principal_component_eigenvalues,
        projection_diagnostics=projection_diagnostics,
    )
    return test_statistic, degrees_of_freedom, p_value, projection_diagnostics


__all__ = ["run_sibling_divergence_wald_test_with_diagnostics"]
