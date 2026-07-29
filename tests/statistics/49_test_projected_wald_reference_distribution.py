"""Contracts for the projected-Wald chi-square reference distribution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2
from tree_break_selection.hierarchy_analysis.statistics.projection.projected_wald.projected_wald_reference_distribution import (
    compute_projected_pvalue,
)


def test_projected_pvalue_rejects_missing_eigenvalues() -> None:
    rng = np.random.default_rng(42)
    projected = rng.standard_normal(10)

    with pytest.raises(ValueError, match="requires PCA eigenvalues"):
        compute_projected_pvalue(projected, eigenvalues=None)


def test_projected_pvalue_rejects_empty_eigenvalues() -> None:
    rng = np.random.default_rng(42)
    projected = rng.standard_normal(5)

    with pytest.raises(ValueError, match="same length"):
        compute_projected_pvalue(projected, eigenvalues=np.array([]))


def test_projected_chi_square_uses_orthonormal_dimension() -> None:
    projected = np.array([1.0, 2.0, 3.0])
    eigenvalues = np.array([2.0, 1.0, 0.5])

    reference = compute_projected_pvalue(projected, eigenvalues=eigenvalues)
    expected_statistic = float(np.sum(projected**2))
    expected_degrees_of_freedom = float(projected.shape[0])

    assert abs(reference.statistic - expected_statistic) < 1e-10
    assert reference.reference_scale == 1.0
    assert abs(reference.degrees_of_freedom - expected_degrees_of_freedom) < 1e-10
    assert (
        abs(
            reference.p_value
            - float(chi2.sf(expected_statistic, df=expected_degrees_of_freedom))
        )
        < 1e-10
    )
