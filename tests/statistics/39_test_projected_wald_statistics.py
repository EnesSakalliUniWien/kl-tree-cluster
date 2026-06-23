"""Regression tests for projected Wald statistic construction."""

from __future__ import annotations

import numpy as np
import pytest
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_projected_wald.child_parent_projected_wald_test import (
    run_child_parent_projected_wald_test,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.projected_wald.projected_wald_kernel import (
    run_projected_wald_kernel,
)


def test_child_parent_projected_wald_uses_projection_dimension_as_df() -> None:
    """Child-parent metadata must expose the df used by the Wald reference law."""
    child_distribution = np.array([0.25, 0.55, 0.20])
    parent_distribution = np.array([0.20, 0.50, 0.30])
    pca_projection = np.eye(3, dtype=np.float64)[:2]
    pca_eigenvalues = np.array([9.0, 1.0])

    statistic, degrees_of_freedom, p_value, is_invalid = run_child_parent_projected_wald_test(
        child_distribution,
        parent_distribution,
        n_child=20,
        n_parent=100,
        spectral_k=2,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
    )

    assert not is_invalid
    assert np.isfinite(statistic)
    assert np.isfinite(p_value)
    assert np.isclose(degrees_of_freedom, 2.0)


def test_projected_wald_kernel_accepts_exact_zero_dimensional_context() -> None:
    result = run_projected_wald_kernel(
        np.zeros(4, dtype=np.float64),
        spectral_k=0,
        pca_projection=np.zeros((0, 4), dtype=np.float64),
        pca_eigenvalues=np.zeros(0, dtype=np.float64),
    )

    assert result.statistic == 0.0
    assert result.projection_dimension == 0
    assert result.reference_scale == 1.0
    assert result.degrees_of_freedom == 0.0
    assert result.p_value == 1.0


def test_projected_wald_kernel_rejects_projection_width_mismatch() -> None:
    with pytest.raises(ValueError, match="PCA projection width"):
        run_projected_wald_kernel(
            np.zeros(4, dtype=np.float64),
            spectral_k=2,
            pca_projection=np.zeros((2, 6), dtype=np.float64),
            pca_eigenvalues=np.ones(2, dtype=np.float64),
        )
