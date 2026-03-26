from __future__ import annotations

import numpy as np

from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    estimate_k_marchenko_pastur,
)


def test_marchenko_pastur_k_fixed_fixture() -> None:
    """Marchenko-Pastur estimation returns the expected fixed-fixture dimension."""
    eigenvalues = np.array([6.2, 3.4, 1.3, 0.9, 0.6, 0.2], dtype=np.float64)
    minimum_projection_dimension = 1
    d_active = 6
    n_desc = 24

    k_estimated = estimate_k_marchenko_pastur(
        eigenvalues,
        n_samples=n_desc,
        n_features=d_active,
        minimum_projection_dimension=minimum_projection_dimension,
    )

    assert k_estimated == 2
