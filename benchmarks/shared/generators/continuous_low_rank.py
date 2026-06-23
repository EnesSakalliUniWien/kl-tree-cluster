"""Continuous low-rank p>>n generators for covariance diagnostics."""

from __future__ import annotations

import numpy as np

from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
    continuous_dataframe_and_metadata,
    require_case_value,
)


def generate_continuous_low_rank_factor(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate clustered continuous data from low-rank factors plus diagonal noise."""
    n_samples = int(require_case_value(test_case, "n_samples", "Low-rank factor"))
    n_features = int(require_case_value(test_case, "n_features", "Low-rank factor"))
    n_clusters = int(require_case_value(test_case, "n_clusters", "Low-rank factor"))
    rank = int(require_case_value(test_case, "rank", "Low-rank factor"))
    noise_variance = float(test_case.get("noise_variance", 1.0))
    rng = np.random.default_rng(seed)

    labels = np.arange(n_samples, dtype=int) % n_clusters
    rng.shuffle(labels)
    loadings = rng.normal(size=(n_features, rank))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=0, keepdims=True), 1e-12)
    factors = rng.normal(size=(n_samples, rank))
    if str(test_case.get("mean_signal", "cluster_low_rank")) == "cluster_low_rank":
        for cluster in range(n_clusters):
            factors[labels == cluster, cluster % rank] += 2.0
    matrix = factors @ loadings.T
    matrix += rng.normal(scale=np.sqrt(noise_variance), size=(n_samples, n_features))

    data_df, feature_space, distance_condensed = continuous_dataframe_and_metadata(
        matrix,
        [f"S{j}" for j in range(n_samples)],
        [f"F{j}" for j in range(n_features)],
    )
    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        noise=noise_variance,
        generator="continuous_low_rank_factor",
        source_family="continuous_low_rank_factor",
        feature_representation="continuous",
        requires_precomputed_tbs_distance=True,
        precomputed_distance_condensed=distance_condensed,
        distance_metric="mahalanobis_time",
        extra={
            "feature_space": feature_space,
            "rank": rank,
            "noise_variance": noise_variance,
            "mean_signal": str(test_case.get("mean_signal", "cluster_low_rank")),
            "covariance_modes": [
                "empirical_dense",
                "linear_shrinkage_identity",
                "low_rank_plus_diagonal",
                "POET_like_factor_sparse",
            ],
        },
    )
    return data_df, labels, matrix.astype(float, copy=False), metadata
