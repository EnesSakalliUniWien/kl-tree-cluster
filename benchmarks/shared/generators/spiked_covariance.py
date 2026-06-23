"""Spiked continuous covariance generators for MP/k-min diagnostics."""

from __future__ import annotations

import numpy as np

from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
    continuous_dataframe_and_metadata,
    require_case_value,
)


def generate_continuous_spiked_covariance(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate identity-plus-rank-one continuous data near a BBP threshold."""
    n_samples = int(require_case_value(test_case, "n_samples", "Spiked covariance"))
    n_features = int(require_case_value(test_case, "n_features", "Spiked covariance"))
    n_clusters = int(require_case_value(test_case, "n_clusters", "Spiked covariance"))
    rng = np.random.default_rng(seed)

    aspect = n_features / max(n_samples, 1)
    bbp_scale = np.sqrt(aspect)
    spike_name = str(test_case.get("spike_strength", "below_bbp"))
    spike = 0.55 * bbp_scale if spike_name == "below_bbp" else 1.65 * bbp_scale

    direction = rng.normal(size=n_features)
    direction /= np.linalg.norm(direction)
    scores = rng.normal(size=(n_samples, 1))
    noise = rng.normal(size=(n_samples, n_features))
    matrix = noise + np.sqrt(max(spike, 0.0)) * scores @ direction[None, :]

    if n_clusters > 1:
        labels = np.arange(n_samples, dtype=int) % n_clusters
        rng.shuffle(labels)
        centered = labels - float(np.mean(labels))
        denom = max(float(np.max(np.abs(centered))), 1.0)
        matrix += (centered / denom)[:, None] * (0.85 * direction[None, :])
    else:
        labels = np.zeros(n_samples, dtype=int)

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
        noise=1.0,
        generator="continuous_spiked_covariance",
        source_family="continuous_spiked_covariance",
        feature_representation="continuous",
        requires_precomputed_tbs_distance=True,
        precomputed_distance_condensed=distance_condensed,
        distance_metric="mahalanobis_time",
        extra={
            "feature_space": feature_space,
            "spike_strength": spike_name,
            "spike_value": float(spike),
            "covariance_profile": str(test_case.get("covariance_profile", "identity_plus_rank1")),
        },
    )
    return data_df, labels, matrix.astype(float, copy=False), metadata
