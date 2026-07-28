from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.spectral.stability.covariance_axis_stability import (
    procrustes_subspace_metrics,
    run_weighting_stability,
    sign_aligned_axis_metrics,
)


def test_sign_aligned_axis_metrics_are_sign_invariant() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    candidate = -reference

    metrics = sign_aligned_axis_metrics(reference, candidate)

    assert metrics["axis1_alignment_sign"] == -1.0
    assert metrics["axis1_abs_cosine"] == pytest.approx(1.0)
    assert metrics["axis1_cosine_after_alignment"] == pytest.approx(1.0)


def test_procrustes_subspace_metrics_are_rotation_invariant() -> None:
    reference = np.eye(4, 2)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    candidate = reference @ rotation

    metrics = procrustes_subspace_metrics(reference, candidate)

    assert metrics["subspace_components"] == 2
    assert metrics["subspace_mean_canonical_corr"] == pytest.approx(1.0)
    assert metrics["procrustes_residual"] < 1e-12


def test_low_rank_covariance_axis_is_stable_under_feature_subsampling() -> None:
    rng = np.random.default_rng(1729)
    n_samples = 40
    n_features = 120
    latent = np.linspace(-2.0, 2.0, n_samples)
    loadings = rng.normal(size=n_features)
    noise = 0.05 * rng.normal(size=(n_samples, n_features))
    values = np.outer(latent, loadings) + noise
    data = pd.DataFrame(
        values,
        index=[f"g{i}" for i in range(n_samples)],
        columns=[f"f{j}" for j in range(n_features)],
    )

    replicates, summary = run_weighting_stability(
        data,
        weighting="binary",
        n_components=3,
        replicates=8,
        feature_fraction=0.7,
        resample_mode="subsample",
        random_state=11,
        stability_threshold=0.9,
    )

    assert replicates.shape[0] == 8
    assert summary["axis1_abs_cosine_q50"] > 0.99
    assert summary["axis1_stable_fraction"] == pytest.approx(1.0)
    assert summary["subspace_mean_canonical_corr_q50"] > 0.8
