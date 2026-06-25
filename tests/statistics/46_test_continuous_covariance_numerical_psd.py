from __future__ import annotations

import numpy as np
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    compute_whitened_wald_contrast,
)
from tree_break_selection.tree.feature_space import continuous_feature_space_from_columns


def test_continuous_covariance_roundoff_psd_survives_branch_time_scaling() -> None:
    feature_space = continuous_feature_space_from_columns(["x", "y"])
    covariance = np.array(
        [
            [1.0, 0.0],
            [0.0, -1e-12],
        ],
        dtype=float,
    )

    z_scores = compute_whitened_wald_contrast(
        np.array([0.5, 0.1], dtype=float),
        np.array([0.0, 0.0], dtype=float),
        5.0,
        10.0,
        comparison="child_parent",
        feature_space=feature_space,
        continuous_covariance_by_block={"continuous": covariance},
        tree_time=10_000.0,
        tree_time_normalizer=1.0,
    )

    assert np.isfinite(z_scores).all()
    assert z_scores.shape == (2,)
