"""Regression tests for projected Wald statistic construction."""

from __future__ import annotations

import numpy as np
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_projected_wald.child_parent_projected_wald_test import (
    run_child_parent_projected_wald_test,
)


def test_child_parent_projected_wald_returns_satterthwaite_effective_df() -> None:
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

    expected_effective_df = float(np.sum(pca_eigenvalues) ** 2 / np.sum(pca_eigenvalues**2))

    assert not is_invalid
    assert np.isfinite(statistic)
    assert np.isfinite(p_value)
    assert not np.isclose(expected_effective_df, 2.0)
    assert np.isclose(degrees_of_freedom, expected_effective_df)
