from __future__ import annotations

import numpy as np
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    estimate_marchenko_pastur_dimension,
    marchenko_pastur_signal_count,
)


def test_marchenko_pastur_k_fixed_fixture() -> None:
    """Marchenko-Pastur estimation returns the expected fixed-fixture dimension."""
    eigenvalues = np.array([6.2, 3.4, 1.3, 0.9, 0.6, 0.2], dtype=np.float64)
    minimum_projection_dimension = 1
    d_active = 6
    n_desc = 24

    dimension_estimate = estimate_marchenko_pastur_dimension(
        eigenvalues,
        n_samples=n_desc,
        n_features=d_active,
        minimum_projection_dimension=minimum_projection_dimension,
    )

    assert dimension_estimate.raw_mp_signal_count == 2
    assert dimension_estimate.test_projection_dimension == 2


def test_marchenko_pastur_uses_correlation_unit_noise_scale() -> None:
    """Correlation spectra use MP upper edge with sigma^2 fixed at one."""
    eigenvalues = np.array([2.4, 2.1, 1.9, 1.7], dtype=np.float64)

    dimension_estimate = estimate_marchenko_pastur_dimension(
        eigenvalues,
        n_samples=100,
        n_features=4,
        minimum_projection_dimension=1,
    )

    assert dimension_estimate.raw_mp_signal_count == 4
    assert dimension_estimate.test_projection_dimension == 4


def test_marchenko_pastur_raw_signal_count_can_be_zero() -> None:
    """The raw MP count is not the floored projected-Wald test dimension."""
    eigenvalues = np.array([1.1, 0.9, 0.7], dtype=np.float64)

    raw_signal_count = marchenko_pastur_signal_count(
        eigenvalues,
        mp_threshold_rows=30,
        n_active_features=3,
    )
    dimension_estimate = estimate_marchenko_pastur_dimension(
        eigenvalues,
        n_samples=30,
        n_features=3,
        minimum_projection_dimension=2,
    )

    assert raw_signal_count == 0
    assert dimension_estimate.raw_mp_signal_count == 0
    assert dimension_estimate.test_projection_dimension == 2
    assert dimension_estimate.effective_independent_rows == 30
    assert dimension_estimate.mp_threshold_rows == 30


def test_marchenko_pastur_dimension_uses_explicit_effective_independent_rows() -> None:
    """Internal spectral rows do not have to define the MP threshold row count."""
    eigenvalues = np.array([2.0, 1.5, 1.0], dtype=np.float64)

    augmented_row_estimate = estimate_marchenko_pastur_dimension(
        eigenvalues,
        n_samples=100,
        n_features=3,
        minimum_projection_dimension=0,
    )
    leaf_row_estimate = estimate_marchenko_pastur_dimension(
        eigenvalues,
        n_samples=100,
        n_features=3,
        effective_independent_rows=10,
        minimum_projection_dimension=0,
    )

    assert augmented_row_estimate.effective_independent_rows == 100
    assert leaf_row_estimate.effective_independent_rows == 10
    assert augmented_row_estimate.mp_threshold_rows == 100
    assert leaf_row_estimate.mp_threshold_rows == 10
    assert (
        augmented_row_estimate.raw_mp_signal_count
        > leaf_row_estimate.raw_mp_signal_count
    )
