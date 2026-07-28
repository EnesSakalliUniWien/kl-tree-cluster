from __future__ import annotations

import numpy as np
import pytest
from benchmarks.validation.statistics.calibration_support import (
    continuous_covariance,
    summarize_p_value_calibration,
)


def test_continuous_covariance_profiles_have_stable_shapes() -> None:
    identity = continuous_covariance(dimension=3, profile="identity")
    ar1 = continuous_covariance(dimension=3, profile="ar1_0.6")
    ill_conditioned = continuous_covariance(dimension=3, profile="ill_conditioned")

    np.testing.assert_array_equal(identity, np.eye(3))
    np.testing.assert_allclose(
        ar1,
        np.array(
            [
                [1.0, 0.6, 0.36],
                [0.6, 1.0, 0.6],
                [0.36, 0.6, 1.0],
            ]
        ),
    )
    np.testing.assert_allclose(np.diag(ill_conditioned), np.geomspace(1.0, 1e-3, num=3))


def test_p_value_summary_preserves_report_schema() -> None:
    summary = summarize_p_value_calibration(
        p_values=np.array([0.01, 0.20, 0.50, 0.90]),
        n_replicates=4,
        alpha=0.05,
    )

    assert list(summary) == [
        "n_replicates",
        "alpha",
        "rejection_count",
        "rejection_rate",
        "confidence_interval",
        "effect_estimate",
        "p_value_uniformity_summary",
    ]
    assert summary["rejection_count"] == 1
    assert summary["rejection_rate"] == 0.25
    assert summary["confidence_interval"]["method"] == "wilson_95"
    assert summary["effect_estimate"] == {
        "name": "rejection_rate_minus_alpha",
        "value": 0.20,
    }


def test_p_value_summary_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite values in"):
        summarize_p_value_calibration(
            p_values=np.array([0.2, np.nan]),
            n_replicates=2,
            alpha=0.05,
        )
