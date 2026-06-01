"""Benchmark case-data generation for Gaussian-source families."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
    continuous_dataframe_and_metadata,
    metadata_extras,
    one_hot_encode_categorical,
)
from benchmarks.shared.generators.generate_dimensional_gaussian import (
    DimensionalGaussianConfig,
    generate_dimensional_gaussian,
)
from benchmarks.shared.generators.generate_gaussian_outliers import (
    GaussianOutlierConfig,
    generate_gaussian_outliers,
)


def generate_blobs_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate Gaussian blobs and median-binarize each coordinate."""
    n_samples = int(test_case["n_samples"])
    n_features = int(test_case["n_features"])
    blobs_result = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=test_case["n_clusters"],
        cluster_std=test_case["cluster_std"],
        random_state=seed,
    )
    x_continuous: np.ndarray = blobs_result[0]
    y: np.ndarray = blobs_result[1]
    x_binary = (x_continuous > np.median(x_continuous, axis=0)).astype(int)
    data_df = pd.DataFrame(
        x_binary,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=int(test_case["n_clusters"]),
        noise=float(test_case["cluster_std"]),
        generator="blobs",
        source_family="gaussian_blobs",
        feature_representation="median_binary",
        requires_precomputed_kl_distance=False,
    )
    return data_df, y, x_binary, metadata


def generate_blobs_continuous_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate Gaussian blobs in continuous coordinates."""
    n_samples = int(test_case["n_samples"])
    n_features = int(test_case["n_features"])
    blobs_result = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=test_case["n_clusters"],
        cluster_std=test_case["cluster_std"],
        random_state=seed,
    )
    x_continuous: np.ndarray = blobs_result[0]
    y: np.ndarray = blobs_result[1]
    data_df, feature_space, distance_condensed = continuous_dataframe_and_metadata(
        x_continuous,
        [f"S{j}" for j in range(n_samples)],
        [f"F{j}" for j in range(n_features)],
    )
    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=int(test_case["n_clusters"]),
        noise=float(test_case["cluster_std"]),
        generator="blobs_continuous",
        source_family="gaussian_blobs",
        feature_representation="continuous",
        requires_precomputed_kl_distance=True,
        precomputed_distance_condensed=distance_condensed,
        distance_metric="euclidean",
        extra={"feature_space": feature_space},
    )
    return data_df, y, x_continuous.astype(float, copy=False), metadata


def generate_blobs_quantile_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate Gaussian blobs, quantile-discretize, and one-hot encode."""
    n_samples = int(test_case["n_samples"])
    n_features = int(test_case["n_features"])
    n_categories = int(test_case["n_categories"])
    blobs_result = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=test_case["n_clusters"],
        cluster_std=test_case["cluster_std"],
        random_state=seed,
    )
    x_continuous: np.ndarray = blobs_result[0]
    y: np.ndarray = blobs_result[1]

    quantiles = np.linspace(0, 1, n_categories + 1)[1:-1]
    thresholds = np.quantile(x_continuous, quantiles, axis=0)
    x_categorical = np.zeros_like(x_continuous, dtype=int)
    for q_idx in range(len(quantiles)):
        x_categorical += (x_continuous > thresholds[q_idx]).astype(int)

    sample_names = [f"S{j}" for j in range(n_samples)]
    data_df, n_binary, feature_space = one_hot_encode_categorical(
        x_categorical,
        n_categories,
        sample_names,
    )

    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_binary,
        n_clusters=int(test_case["n_clusters"]),
        noise=float(test_case["cluster_std"]),
        generator="blobs_quantile",
        source_family="gaussian_blobs",
        feature_representation="quantile_one_hot",
        requires_precomputed_kl_distance=False,
        extra={
            "n_features_original": n_features,
            "n_categories": n_categories,
            "feature_space": feature_space,
        },
    )
    return data_df, y, data_df.values.astype(float), metadata


def _resolve_dimensional_feature_counts(test_case: dict) -> tuple[int, int]:
    informative_dims = test_case["informative_dims"]
    if informative_dims is None:
        raise ValueError("Dimensional Gaussian generator requires 'informative_dims'.")

    n_features_present = "n_features" in test_case
    noise_dims_present = "noise_dims" in test_case
    if not n_features_present and not noise_dims_present:
        raise ValueError("Dimensional Gaussian generator requires 'n_features' or 'noise_dims'.")

    informative_dims = int(informative_dims)
    if n_features_present:
        n_features = int(test_case["n_features"])
        noise_dims = n_features - informative_dims
    else:
        noise_dims = int(test_case["noise_dims"])

    if informative_dims <= 0:
        raise ValueError(f"informative_dims must be positive, got {informative_dims}")
    if noise_dims < 0:
        raise ValueError(
            f"noise_dims must be non-negative after resolving total features, got {noise_dims}"
        )
    return informative_dims, noise_dims


def _dimensional_config(test_case: dict, seed: int | None) -> DimensionalGaussianConfig:
    informative_dims, noise_dims = _resolve_dimensional_feature_counts(test_case)
    return DimensionalGaussianConfig(
        n_samples=int(test_case["n_samples"]),
        n_clusters=int(test_case["n_clusters"]),
        informative_dims=informative_dims,
        noise_dims=noise_dims,
        separation=float(test_case["separation"]),
        informative_std=float(test_case["informative_std"]),
        noise_std=float(test_case["noise_std"]),
        informative_corr=float(test_case["informative_corr"]),
        noise_corr=float(test_case["noise_corr"]),
        signal_mode=str(test_case["signal_mode"]),
        balanced_clusters=bool(test_case["balanced_clusters"]),
        random_seed=seed,
    )


def generate_dimensional_gaussian_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate dimensional Gaussian data and median-binarize coordinates."""
    config = _dimensional_config(test_case, seed)
    x_continuous, y, sim_meta = generate_dimensional_gaussian(config)
    x_binary = (x_continuous > np.median(x_continuous, axis=0)).astype(int)
    feature_names = [f"F{j}" for j in range(x_binary.shape[1])]
    sample_names = [f"S{j}" for j in range(config.n_samples)]
    data_df = pd.DataFrame(x_binary, index=sample_names, columns=feature_names)

    metadata = case_metadata(
        test_case=test_case,
        n_samples=config.n_samples,
        n_features=int(x_binary.shape[1]),
        n_clusters=int(test_case["n_clusters"]),
        noise=float(test_case["noise_std"]),
        generator="dimensional_gaussian",
        source_family="dimensional_gaussian",
        feature_representation="median_binary",
        requires_precomputed_kl_distance=False,
        extra={
            "informative_dims": config.informative_dims,
            "noise_dims": config.noise_dims,
            "separation": float(test_case["separation"]),
            "informative_corr": float(test_case["informative_corr"]),
            "noise_corr": float(test_case["noise_corr"]),
            "signal_mode": str(test_case["signal_mode"]),
            "binarization": "median",
            **metadata_extras(sim_meta),
        },
    )
    return data_df, y, x_binary.astype(float), metadata


def generate_dimensional_gaussian_continuous_case(
    test_case: dict,
    seed: int | None,
) -> CaseDataResult:
    """Generate dimensional Gaussian data in continuous coordinates."""
    config = _dimensional_config(test_case, seed)
    x_continuous, y, sim_meta = generate_dimensional_gaussian(config)
    n_features = int(x_continuous.shape[1])
    data_df, feature_space, distance_condensed = continuous_dataframe_and_metadata(
        x_continuous,
        [f"S{j}" for j in range(config.n_samples)],
        [f"F{j}" for j in range(n_features)],
    )

    metadata = case_metadata(
        test_case=test_case,
        n_samples=config.n_samples,
        n_features=n_features,
        n_clusters=int(test_case["n_clusters"]),
        noise=float(test_case["noise_std"]),
        generator="dimensional_gaussian_continuous",
        source_family="dimensional_gaussian",
        feature_representation="continuous",
        requires_precomputed_kl_distance=True,
        precomputed_distance_condensed=distance_condensed,
        distance_metric="euclidean",
        extra={
            "feature_space": feature_space,
            "informative_dims": config.informative_dims,
            "noise_dims": config.noise_dims,
            "separation": float(test_case["separation"]),
            "informative_corr": float(test_case["informative_corr"]),
            "noise_corr": float(test_case["noise_corr"]),
            "signal_mode": str(test_case["signal_mode"]),
            **metadata_extras(sim_meta),
        },
    )
    return data_df, y, x_continuous.astype(float, copy=False), metadata


def _outlier_config(test_case: dict, seed: int | None) -> GaussianOutlierConfig:
    return GaussianOutlierConfig(
        n_samples=int(test_case["n_samples"]),
        n_features=int(test_case["n_features"]),
        n_inlier_clusters=int(test_case["n_inlier_clusters"]),
        cluster_std=float(test_case["cluster_std"]),
        outlier_count=int(test_case["outlier_count"]),
        outlier_distance=float(test_case["outlier_distance"]),
        outlier_std=float(test_case["outlier_std"]),
        spatial_mode=str(test_case["outlier_spatial_mode"]),
        label_mode=str(test_case["outlier_label_mode"]),
        balanced_clusters=bool(test_case["balanced_clusters"]),
        random_seed=seed,
    )


def generate_gaussian_outlier_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate Gaussian outlier data and median-binarize coordinates."""
    config = _outlier_config(test_case, seed)
    x_continuous, y, sim_meta = generate_gaussian_outliers(config)
    x_binary = (x_continuous > np.median(x_continuous, axis=0)).astype(int)
    data_df = pd.DataFrame(
        x_binary,
        index=[f"S{j}" for j in range(config.n_samples)],
        columns=[f"F{j}" for j in range(config.n_features)],
    )
    metadata = case_metadata(
        test_case=test_case,
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_clusters=int(np.unique(y).size),
        noise=float(test_case["outlier_count"]) / float(config.n_samples),
        generator="gaussian_outliers",
        source_family="gaussian_outliers",
        feature_representation="median_binary",
        requires_precomputed_kl_distance=False,
        extra={
            "n_inlier_clusters": config.n_inlier_clusters,
            "binarization": "median",
            **metadata_extras(sim_meta),
        },
    )
    return data_df, y, x_binary.astype(float), metadata


def generate_gaussian_outlier_continuous_case(
    test_case: dict,
    seed: int | None,
) -> CaseDataResult:
    """Generate Gaussian outlier data in continuous coordinates."""
    config = _outlier_config(test_case, seed)
    x_continuous, y, sim_meta = generate_gaussian_outliers(config)
    data_df, feature_space, distance_condensed = continuous_dataframe_and_metadata(
        x_continuous,
        [f"S{j}" for j in range(config.n_samples)],
        [f"F{j}" for j in range(config.n_features)],
    )
    metadata = case_metadata(
        test_case=test_case,
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_clusters=int(np.unique(y).size),
        noise=float(test_case["outlier_count"]) / float(config.n_samples),
        generator="gaussian_outliers_continuous",
        source_family="gaussian_outliers",
        feature_representation="continuous",
        requires_precomputed_kl_distance=True,
        precomputed_distance_condensed=distance_condensed,
        distance_metric="euclidean",
        extra={
            "n_inlier_clusters": config.n_inlier_clusters,
            "feature_space": feature_space,
            **metadata_extras(sim_meta),
        },
    )
    return data_df, y, x_continuous.astype(float, copy=False), metadata
