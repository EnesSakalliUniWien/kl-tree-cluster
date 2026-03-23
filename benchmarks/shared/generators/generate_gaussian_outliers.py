"""Generate Gaussian clustering benchmarks with explicit outlier contamination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.datasets import make_blobs

from benchmarks.shared.generators.common import calculate_cluster_sizes

OutlierLabelMode = Literal["singleton", "grouped"]


@dataclass(frozen=True)
class GaussianOutlierConfig:
    n_samples: int
    n_features: int
    n_inlier_clusters: int
    cluster_std: float = 0.7
    outlier_count: int = 1
    outlier_distance: float = 8.0
    outlier_std: float = 0.2
    spatial_mode: OutlierSpatialMode = "clustered"
    label_mode: OutlierLabelMode = "singleton"
    balanced_clusters: bool = True
    random_seed: int | None = None


def _sample_uniform_shell_outliers(
    rng: np.random.Generator,
    *,
    inlier_min: np.ndarray,
    inlier_max: np.ndarray,
    outlier_count: int,
    outlier_distance: float,
) -> np.ndarray:
    low = inlier_min - outlier_distance
    high = inlier_max + outlier_distance
    center = 0.5 * (inlier_min + inlier_max)
    min_radius = 0.5 * np.linalg.norm(inlier_max - inlier_min) + 0.5 * outlier_distance

    collected: list[np.ndarray] = []
    max_attempts = outlier_count * 200
    attempts = 0
    while len(collected) < outlier_count and attempts < max_attempts:
        candidate = rng.uniform(low=low, high=high)
        if np.linalg.norm(candidate - center) >= min_radius:
            collected.append(candidate)
        attempts += 1

    if len(collected) != outlier_count:
        raise ValueError(
            "Failed to sample sufficiently separated shell outliers; consider increasing "
            "outlier_distance or reducing outlier_count."
        )
    return np.vstack(collected)


def generate_gaussian_outliers(
    config: GaussianOutlierConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if config.n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {config.n_samples}")
    if config.n_features <= 0:
        raise ValueError(f"n_features must be positive, got {config.n_features}")
    if config.n_inlier_clusters <= 0:
        raise ValueError(f"n_inlier_clusters must be positive, got {config.n_inlier_clusters}")
    if config.outlier_count <= 0:
        raise ValueError(f"outlier_count must be positive, got {config.outlier_count}")
    if config.outlier_count >= config.n_samples:
        raise ValueError(
            f"outlier_count must be smaller than n_samples; got {config.outlier_count} >= {config.n_samples}"
        )
    if config.cluster_std <= 0:
        raise ValueError(f"cluster_std must be positive, got {config.cluster_std}")
    if config.outlier_std < 0:
        raise ValueError(f"outlier_std must be non-negative, got {config.outlier_std}")
    if config.outlier_distance <= 0:
        raise ValueError(f"outlier_distance must be positive, got {config.outlier_distance}")

    rng = np.random.default_rng(config.random_seed)
    n_inliers = config.n_samples - config.outlier_count
    cluster_sizes = calculate_cluster_sizes(
        n_inliers,
        config.n_inlier_clusters,
        config.balanced_clusters,
        rng=rng,
    )
    inlier_centers = rng.uniform(
        low=-10.0,
        high=10.0,
        size=(config.n_inlier_clusters, config.n_features),
    )

    x_inliers, y_inliers = make_blobs(
        n_samples=cluster_sizes,
        n_features=config.n_features,
        centers=inlier_centers,
        cluster_std=config.cluster_std,
        random_state=config.random_seed,
    )

    inlier_min = x_inliers.min(axis=0)
    inlier_max = x_inliers.max(axis=0)
    box_center = 0.5 * (inlier_min + inlier_max)
    direction = rng.normal(size=config.n_features)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        direction = np.ones(config.n_features, dtype=float)
        direction_norm = np.linalg.norm(direction)
    direction = direction / direction_norm
    outlier_center = box_center + direction * config.outlier_distance

    if config.spatial_mode == "clustered":
        if config.outlier_std == 0:
            x_outliers = np.repeat(outlier_center[None, :], config.outlier_count, axis=0)
        else:
            x_outliers = rng.normal(
                loc=outlier_center,
                scale=config.outlier_std,
                size=(config.outlier_count, config.n_features),
            )
    elif config.spatial_mode == "uniform_shell":
        x_outliers = _sample_uniform_shell_outliers(
            rng,
            inlier_min=inlier_min,
            inlier_max=inlier_max,
            outlier_count=config.outlier_count,
            outlier_distance=config.outlier_distance,
        )
    else:
        raise ValueError(f"Unsupported spatial_mode: {config.spatial_mode}")

    if config.label_mode == "singleton":
        y_outliers = np.arange(
            config.n_inlier_clusters,
            config.n_inlier_clusters + config.outlier_count,
            dtype=int,
        )
    elif config.label_mode == "grouped":
        y_outliers = np.full(config.outlier_count, config.n_inlier_clusters, dtype=int)
    else:
        raise ValueError(f"Unsupported label_mode: {config.label_mode}")

    x_full = np.vstack([x_inliers, x_outliers])
    y_full = np.concatenate([y_inliers.astype(int), y_outliers])

    permutation = rng.permutation(config.n_samples)
    x_full = x_full[permutation]
    y_full = y_full[permutation]
    outlier_mask = permutation >= n_inliers
    outlier_indices = np.flatnonzero(outlier_mask)

    metadata: dict[str, Any] = {
        "n_samples": config.n_samples,
        "n_features": config.n_features,
        "n_inlier_clusters": config.n_inlier_clusters,
        "n_clusters": int(np.unique(y_full).size),
        "cluster_std": config.cluster_std,
        "outlier_count": config.outlier_count,
        "outlier_distance": config.outlier_distance,
        "outlier_std": config.outlier_std,
        "outlier_spatial_mode": config.spatial_mode,
        "outlier_label_mode": config.label_mode,
        "outlier_indices": outlier_indices.tolist(),
        "outlier_center": outlier_center,
        "balanced_clusters": config.balanced_clusters,
        "cluster_sizes": cluster_sizes,
    }
    return x_full, y_full, metadata


__all__ = ["GaussianOutlierConfig", "generate_gaussian_outliers"]
