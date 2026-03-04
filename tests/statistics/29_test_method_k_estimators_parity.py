from __future__ import annotations

import numpy as np

from kl_clustering_analysis.hierarchy_analysis.decomposition.methods.k_estimators import (
    estimate_k_effective_rank,
    estimate_k_marchenko_pastur,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.eigen_decomposition import (
    estimate_spectral_k,
)


def test_effective_rank_k_parity_fixed_fixture() -> None:
    """Legacy compatibility estimator and new method estimator should agree."""
    eigenvalues = np.array([4.0, 2.0, 1.0, 0.5], dtype=np.float64)
    min_k = 2
    d_active = 4
    n_desc = 40

    k_legacy = estimate_spectral_k(
        eigenvalues,
        method="effective_rank",
        n_desc=n_desc,
        d_active=d_active,
        min_k=min_k,
    )
    k_method = estimate_k_effective_rank(
        eigenvalues,
        min_k=min_k,
        d_active=d_active,
    )
    assert k_legacy == k_method


def test_marchenko_pastur_k_parity_fixed_fixture() -> None:
    """Legacy compatibility estimator and new method estimator should agree."""
    eigenvalues = np.array([6.2, 3.4, 1.3, 0.9, 0.6, 0.2], dtype=np.float64)
    min_k = 1
    d_active = 6
    n_desc = 24

    k_legacy = estimate_spectral_k(
        eigenvalues,
        method="marchenko_pastur",
        n_desc=n_desc,
        d_active=d_active,
        min_k=min_k,
    )
    k_method = estimate_k_marchenko_pastur(
        eigenvalues,
        n_desc=n_desc,
        d_active=d_active,
        min_k=min_k,
    )
    assert k_legacy == k_method

