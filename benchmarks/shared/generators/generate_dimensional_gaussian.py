"""Generate Gaussian benchmark data with fixed signal subspace and varying total dimension.

The generator is designed for dimensionality stress tests where only a subset of
features is informative and the remaining features are irrelevant noise. This
matches common benchmark practice in the clustering literature more closely than
simply increasing the number of informative dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from benchmarks.shared.generators.common import calculate_cluster_sizes


@dataclass(frozen=True)
class DimensionalGaussianConfig:
    n_samples: int
    n_clusters: int
    informative_dims: int
    noise_dims: int
    separation: float = 2.5
    informative_std: float = 1.0
    noise_std: float = 1.0
    informative_corr: float = 0.0
    noise_corr: float = 0.0
    signal_mode: SignalMode = "consolidated"
    balanced_clusters: bool = True
    random_seed: int | None = None

    @property
    def n_features(self) -> int:
        return self.informative_dims + self.noise_dims


def _validate_exchangeable_corr(n_dims: int, corr: float, name: str) -> None:
    if n_dims <= 1:
        return
    lower_bound = -1.0 / (n_dims - 1)
    if corr < lower_bound or corr >= 1.0:
        raise ValueError(
            f"{name} must be in [{lower_bound:.4f}, 1.0) for n_dims={n_dims}; got {corr}."
        )


def _make_exchangeable_covariance(n_dims: int, std: float, corr: float) -> np.ndarray:
    if n_dims <= 0:
        raise ValueError(f"n_dims must be positive, got {n_dims}")
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    _validate_exchangeable_corr(n_dims, corr, "corr")
    if n_dims == 1:
        return np.array([[std**2]], dtype=float)
    base = np.full((n_dims, n_dims), corr, dtype=float)
    np.fill_diagonal(base, 1.0)
    return (std**2) * base


def _build_consolidated_means(
    n_clusters: int,
    informative_dims: int,
    separation: float,
) -> np.ndarray:
    means = np.empty((n_clusters, informative_dims), dtype=float)
    feature_blocks = np.array_split(np.arange(informative_dims), n_clusters)
    background_value = -separation / max(n_clusters - 1, 1)

    for cluster_id, feature_idx in enumerate(feature_blocks):
        mean = np.full(informative_dims, background_value, dtype=float)
        mean[feature_idx] = separation
        means[cluster_id] = mean

    return means


def _build_diffuse_means(
    n_clusters: int,
    informative_dims: int,
    separation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = rng.normal(size=(n_clusters, informative_dims))
    raw -= raw.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    zero_norm_mask = norms.squeeze(axis=1) == 0
    if np.any(zero_norm_mask):
        raw[zero_norm_mask] = rng.normal(size=(int(zero_norm_mask.sum()), informative_dims))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return separation * raw / norms


def _build_cluster_means(
    config: DimensionalGaussianConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.signal_mode == "consolidated":
        return _build_consolidated_means(
            n_clusters=config.n_clusters,
            informative_dims=config.informative_dims,
            separation=config.separation,
        )
    if config.signal_mode == "diffuse":
        return _build_diffuse_means(
            n_clusters=config.n_clusters,
            informative_dims=config.informative_dims,
            separation=config.separation,
            rng=rng,
        )
    raise ValueError(f"Unsupported signal_mode: {config.signal_mode}")


def generate_dimensional_gaussian(
    config: DimensionalGaussianConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if config.n_samples < config.n_clusters:
        raise ValueError(
            f"n_samples must be >= n_clusters, got {config.n_samples} and {config.n_clusters}."
        )
    if config.informative_dims <= 0:
        raise ValueError(f"informative_dims must be positive, got {config.informative_dims}")
    if config.noise_dims < 0:
        raise ValueError(f"noise_dims must be non-negative, got {config.noise_dims}")
    if config.separation <= 0:
        raise ValueError(f"separation must be positive, got {config.separation}")

    rng = np.random.default_rng(config.random_seed)
    cluster_sizes = calculate_cluster_sizes(
        config.n_samples,
        config.n_clusters,
        config.balanced_clusters,
        rng=rng,
    )

    informative_means = _build_cluster_means(config, rng)
    informative_cov = _make_exchangeable_covariance(
        config.informative_dims,
        config.informative_std,
        config.informative_corr,
    )
    noise_cov = None
    if config.noise_dims > 0:
        noise_cov = _make_exchangeable_covariance(
            config.noise_dims,
            config.noise_std,
            config.noise_corr,
        )

    matrices: list[np.ndarray] = []
    labels: list[int] = []
    for cluster_id, cluster_size in enumerate(cluster_sizes):
        informative_block = rng.multivariate_normal(
            mean=informative_means[cluster_id],
            cov=informative_cov,
            size=cluster_size,
        )
        if noise_cov is not None:
            noise_block = rng.multivariate_normal(
                mean=np.zeros(config.noise_dims, dtype=float),
                cov=noise_cov,
                size=cluster_size,
            )
            cluster_matrix = np.hstack([informative_block, noise_block])
        else:
            cluster_matrix = informative_block
        matrices.append(cluster_matrix)
        labels.extend([cluster_id] * cluster_size)

    data = np.vstack(matrices)
    label_array = np.asarray(labels, dtype=int)
    metadata: dict[str, Any] = {
        "n_samples": config.n_samples,
        "n_clusters": config.n_clusters,
        "n_features": config.n_features,
        "informative_dims": config.informative_dims,
        "noise_dims": config.noise_dims,
        "separation": config.separation,
        "informative_std": config.informative_std,
        "noise_std": config.noise_std,
        "informative_corr": config.informative_corr,
        "noise_corr": config.noise_corr,
        "signal_mode": config.signal_mode,
        "balanced_clusters": config.balanced_clusters,
        "cluster_sizes": cluster_sizes,
        "informative_feature_indices": list(range(config.informative_dims)),
        "noise_feature_indices": list(range(config.informative_dims, config.n_features)),
        "cluster_means": informative_means,
    }
    return data, label_array, metadata


__all__ = ["DimensionalGaussianConfig", "generate_dimensional_gaussian"]
