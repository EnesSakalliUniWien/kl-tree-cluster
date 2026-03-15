from __future__ import annotations

import numpy as np

from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    estimate_k_marchenko_pastur,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends import (
    estimate_spectral_k_backend as estimate_spectral_k,
)


def test_marchenko_pastur_k_parity_fixed_fixture() -> None:
    """Legacy compatibility estimator and new method estimator should agree."""
    eigenvalues = np.array([6.2, 3.4, 1.3, 0.9, 0.6, 0.2], dtype=np.float64)
    minimum_projection_dimension = 1
    d_active = 6
    n_desc = 24

    k_legacy = estimate_spectral_k(
        eigenvalues,
        method="marchenko_pastur",
        n_samples=n_desc,
        n_features=d_active,
        minimum_projection_dimension=minimum_projection_dimension,
    )
    k_method = estimate_k_marchenko_pastur(
        eigenvalues,
        n_samples=n_desc,
        n_features=d_active,
        minimum_projection_dimension=minimum_projection_dimension,
    )
    assert k_legacy == k_method
