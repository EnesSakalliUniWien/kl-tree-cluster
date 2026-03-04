from __future__ import annotations

import numpy as np

from kl_clustering_analysis.hierarchy_analysis.decomposition.methods import k_estimators
from kl_clustering_analysis.hierarchy_analysis.statistics.projection import estimators


def test_projection_estimator_shims_match_canonical_methods() -> None:
    eigenvalues = np.array([6.2, 3.4, 1.3, 0.9, 0.6, 0.2], dtype=np.float64)
    data = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    assert estimators.effective_rank(eigenvalues) == k_estimators.effective_rank(eigenvalues)
    assert estimators.marchenko_pastur_signal_count(eigenvalues, n=24, d=6) == (
        k_estimators.marchenko_pastur_signal_count(eigenvalues, n_desc=24, d_active=6)
    )
    assert estimators.count_active_features(data) == k_estimators.count_active_features(data)
