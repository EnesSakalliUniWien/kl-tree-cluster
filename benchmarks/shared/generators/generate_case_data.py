"""Create test case data for benchmarking.

Exports:
- generate_case_data(test_case: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]

This is the extraction of `_generate_case_data` previously in
`benchmarks.shared.pipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from benchmarks.shared.generators.generate_categorical_matrix import (
    generate_categorical_feature_matrix,
)
from benchmarks.shared.generators.generate_dimensional_gaussian import (
    DimensionalGaussianConfig,
    generate_dimensional_gaussian,
)
from benchmarks.shared.generators.generate_gaussian_outliers import (
    GaussianOutlierConfig,
    generate_gaussian_outliers,
)
from benchmarks.shared.generators.generate_phylogenetic import generate_phylogenetic_data
from benchmarks.shared.generators.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)
from benchmarks.shared.generators.generate_sbm import generate_sbm
from benchmarks.shared.generators.generate_temporal_evolution import (
    generate_temporal_evolution_data,
)
from kl_clustering_analysis.tree.feature_space import (
    FeatureSpace,
    continuous_feature_space_from_columns,
    infer_feature_space_from_columns,
)
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_blobs


def _require_case_value(test_case: dict, key: str, generator_name: str) -> Any:
    if key not in test_case:
        raise ValueError(f"{generator_name} generator requires '{key}'.")
    return test_case[key]


def _one_hot_encode_categorical(
    matrix: np.ndarray,
    n_categories: int,
    sample_names: list[str],
) -> tuple[pd.DataFrame, int, FeatureSpace]:
    """One-hot encode a (n_rows, n_cols) category-index matrix into binary.

    Each original feature with K categories becomes K binary indicator columns.
    This converts categorical data into a form compatible with the Bernoulli KL
    pipeline (all values in {0, 1}).

    Returns:
        (data_df, n_binary_features, feature_space) where data_df has shape
        (n_rows, n_cols * K).
    """
    n_rows, n_cols = matrix.shape
    n_binary = n_cols * n_categories
    binary = np.zeros((n_rows, n_binary), dtype=int)
    for j in range(n_cols):
        for k in range(n_categories):
            binary[:, j * n_categories + k] = (matrix[:, j] == k).astype(int)
    feature_names = [f"F{j}_c{k}" for j in range(n_cols) for k in range(n_categories)]
    data_df = pd.DataFrame(binary, index=sample_names, columns=feature_names)
    feature_space = infer_feature_space_from_columns(tuple(data_df.columns))
    return data_df, n_binary, feature_space


def _continuous_dataframe_and_metadata(
    matrix: np.ndarray,
    sample_names: list[str],
    feature_names: list[str],
) -> tuple[pd.DataFrame, FeatureSpace, np.ndarray]:
    """Return continuous benchmark data with its explicit diagonal Gaussian contract."""
    continuous_matrix = np.asarray(matrix, dtype=np.float64)
    data_df = pd.DataFrame(
        continuous_matrix,
        index=sample_names,
        columns=feature_names,
    )
    feature_space = continuous_feature_space_from_columns(tuple(data_df.columns))
    distance_condensed = pdist(continuous_matrix, metric="euclidean")
    return data_df, feature_space, distance_condensed


def _validate_binary_params(test_case: dict) -> Tuple[int, int]:
    """Validate canonical benchmark-case geometry for the binary generator.

    Returns (n_samples, n_features).
    """
    n_samples = int(_require_case_value(test_case, "n_samples", "Binary"))
    n_features = int(_require_case_value(test_case, "n_features", "Binary"))
    return n_samples, n_features


def _generate_binary_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate the 'binary' style test case using the feature matrix generator."""
    n_samples, n_features = _validate_binary_params(test_case)
    entropy = test_case["entropy_param"]
    balanced = test_case["balanced_clusters"]
    feature_sparsity = test_case["feature_sparsity"]
    noise_features = int(test_case["noise_features"])

    data_dict, cluster_assignments = generate_random_feature_matrix(
        n_rows=n_samples,
        n_cols=n_features,
        entropy_param=entropy,
        n_clusters=test_case["n_clusters"],
        random_seed=seed,
        balanced_clusters=balanced,
        feature_sparsity=feature_sparsity,
        noise_features=noise_features,
    )

    original_names = list(data_dict.keys())
    matrix = np.array([data_dict[name] for name in original_names], dtype=int)
    feature_names = [f"F{j}" for j in range(matrix.shape[1])]

    data_df = pd.DataFrame(matrix, index=original_names, columns=feature_names)
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    actual_cols = matrix.shape[1]
    metadata = {
        "n_samples": n_samples,
        "n_features": actual_cols,
        "n_clusters": test_case["n_clusters"],
        "noise": entropy,
        "name": str(test_case["name"]),
        "generator": "binary",
        "noise_features": noise_features,
        "requires_precomputed_kl_distance": False,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": None,
    }

    return data_df, true_labels, matrix.astype(float), metadata


def _generate_blobs_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate the default 'blobs' style test case (Gaussian blobs -> median binarized)."""
    n_samples = int(test_case["n_samples"])
    n_features = int(test_case["n_features"])
    blobs_result = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=test_case["n_clusters"],
        cluster_std=test_case["cluster_std"],
        random_state=seed,
    )
    X: np.ndarray = blobs_result[0]
    y: np.ndarray = blobs_result[1]
    X_bin = (X > np.median(X, axis=0)).astype(int)
    data_df = pd.DataFrame(
        X_bin,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": test_case["n_clusters"],
        "noise": test_case["cluster_std"],
        "name": str(test_case["name"]),
        "generator": "blobs",
        "requires_precomputed_kl_distance": False,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": None,
    }
    # Return binarized data as x_original so UMAP and baseline methods
    # (K-Means, Spectral) operate on the same feature space as the KL pipeline.
    return data_df, y, X_bin, metadata


def _generate_blobs_continuous_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate Gaussian blobs in continuous coordinates for A/B benchmarking."""
    n_samples = int(test_case["n_samples"])
    n_features = int(test_case["n_features"])
    blobs_result = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=test_case["n_clusters"],
        cluster_std=test_case["cluster_std"],
        random_state=seed,
    )
    X: np.ndarray = blobs_result[0]
    y: np.ndarray = blobs_result[1]
    data_df, feature_space, distance_condensed = _continuous_dataframe_and_metadata(
        X,
        [f"S{j}" for j in range(n_samples)],
        [f"F{j}" for j in range(n_features)],
    )
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": test_case["n_clusters"],
        "noise": test_case["cluster_std"],
        "name": str(test_case["name"]),
        "generator": "blobs_continuous",
        "feature_representation": "continuous",
        "feature_space": feature_space,
        "distance_metric": "euclidean",
        "requires_precomputed_kl_distance": True,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": distance_condensed,
    }
    return data_df, y, X.astype(float, copy=False), metadata


def _generate_blobs_quantile_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate Gaussian blobs discretized into quantile categories, then one-hot encoded.

    Instead of harsh median binarization (2 bins), this discretizes each feature
    into ``n_categories`` equal-frequency quantile bins.  The resulting category
    indices are one-hot encoded into binary indicators compatible with the
    Bernoulli KL pipeline.

    More categories preserve more of the continuous overlap structure, reducing
    artificial sub-cluster signal that causes over-splitting.
    """
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
    X: np.ndarray = blobs_result[0]
    y: np.ndarray = blobs_result[1]

    # Quantile-discretize each feature into n_categories bins
    quantiles = np.linspace(0, 1, n_categories + 1)[1:-1]  # e.g. [0.25, 0.5, 0.75] for 4 bins
    thresholds = np.quantile(X, quantiles, axis=0)  # shape (n_categories-1, n_features)
    X_cat = np.zeros_like(X, dtype=int)
    for q_idx in range(len(quantiles)):
        X_cat += (X > thresholds[q_idx]).astype(int)
    # X_cat now contains category indices 0..n_categories-1

    sample_names = [f"S{j}" for j in range(n_samples)]
    data_df, n_binary, feature_space = _one_hot_encode_categorical(
        X_cat,
        n_categories,
        sample_names,
    )

    metadata = {
        "n_samples": n_samples,
        "n_features": n_binary,
        "n_features_original": n_features,
        "n_categories": n_categories,
        "feature_space": feature_space,
        "n_clusters": test_case["n_clusters"],
        "noise": test_case["cluster_std"],
        "name": str(test_case["name"]),
        "generator": "blobs_quantile",
        "requires_precomputed_kl_distance": False,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": None,
    }
    # Return one-hot encoded data as x_original so UMAP and baseline methods
    # (K-Means, Spectral) operate on the same feature space as the KL pipeline.
    return data_df, y, data_df.values.astype(float), metadata


def _resolve_dimensional_feature_counts(test_case: dict) -> Tuple[int, int]:
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
        n_features = informative_dims + noise_dims

    if informative_dims <= 0:
        raise ValueError(f"informative_dims must be positive, got {informative_dims}")
    if noise_dims < 0:
        raise ValueError(
            f"noise_dims must be non-negative after resolving total features, got {noise_dims}"
        )
    return informative_dims, noise_dims


def _generate_dimensional_gaussian_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    n_samples = int(test_case["n_samples"])
    informative_dims, noise_dims = _resolve_dimensional_feature_counts(test_case)
    config = DimensionalGaussianConfig(
        n_samples=n_samples,
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

    x_continuous, y, sim_meta = generate_dimensional_gaussian(config)
    x_binary = (x_continuous > np.median(x_continuous, axis=0)).astype(int)
    feature_names = [f"F{j}" for j in range(x_binary.shape[1])]
    sample_names = [f"S{j}" for j in range(n_samples)]
    data_df = pd.DataFrame(x_binary, index=sample_names, columns=feature_names)

    metadata = {
        "n_samples": n_samples,
        "n_features": int(x_binary.shape[1]),
        "n_clusters": int(test_case["n_clusters"]),
        "noise": float(test_case["noise_std"]),
        "name": str(test_case["name"]),
        "generator": "dimensional_gaussian",
        "informative_dims": informative_dims,
        "noise_dims": noise_dims,
        "separation": float(test_case["separation"]),
        "informative_corr": float(test_case["informative_corr"]),
        "noise_corr": float(test_case["noise_corr"]),
        "signal_mode": str(test_case["signal_mode"]),
        "binarization": "median",
        "requires_precomputed_kl_distance": False,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": None,
        **sim_meta,
    }
    return data_df, y, x_binary.astype(float), metadata


def _generate_dimensional_gaussian_continuous_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate dimensional Gaussian data without median binarization."""
    n_samples = int(test_case["n_samples"])
    informative_dims, noise_dims = _resolve_dimensional_feature_counts(test_case)
    config = DimensionalGaussianConfig(
        n_samples=n_samples,
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

    x_continuous, y, sim_meta = generate_dimensional_gaussian(config)
    n_features = int(x_continuous.shape[1])
    data_df, feature_space, distance_condensed = _continuous_dataframe_and_metadata(
        x_continuous,
        [f"S{j}" for j in range(n_samples)],
        [f"F{j}" for j in range(n_features)],
    )

    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": int(test_case["n_clusters"]),
        "noise": float(test_case["noise_std"]),
        "name": str(test_case["name"]),
        "generator": "dimensional_gaussian_continuous",
        "feature_representation": "continuous",
        "feature_space": feature_space,
        "distance_metric": "euclidean",
        "informative_dims": informative_dims,
        "noise_dims": noise_dims,
        "separation": float(test_case["separation"]),
        "informative_corr": float(test_case["informative_corr"]),
        "noise_corr": float(test_case["noise_corr"]),
        "signal_mode": str(test_case["signal_mode"]),
        "requires_precomputed_kl_distance": True,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": distance_condensed,
        **sim_meta,
    }
    return data_df, y, x_continuous.astype(float, copy=False), metadata


def _generate_gaussian_outlier_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    n_samples = int(test_case["n_samples"])
    n_features = int(test_case["n_features"])
    n_inlier_clusters = int(test_case["n_inlier_clusters"])

    config = GaussianOutlierConfig(
        n_samples=n_samples,
        n_features=n_features,
        n_inlier_clusters=n_inlier_clusters,
        cluster_std=float(test_case["cluster_std"]),
        outlier_count=int(test_case["outlier_count"]),
        outlier_distance=float(test_case["outlier_distance"]),
        outlier_std=float(test_case["outlier_std"]),
        spatial_mode=str(test_case["outlier_spatial_mode"]),
        label_mode=str(test_case["outlier_label_mode"]),
        balanced_clusters=bool(test_case["balanced_clusters"]),
        random_seed=seed,
    )

    x_continuous, y, sim_meta = generate_gaussian_outliers(config)
    x_binary = (x_continuous > np.median(x_continuous, axis=0)).astype(int)
    data_df = pd.DataFrame(
        x_binary,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": int(np.unique(y).size),
        "n_inlier_clusters": n_inlier_clusters,
        "noise": float(test_case["outlier_count"]) / float(n_samples),
        "name": str(test_case["name"]),
        "generator": "gaussian_outliers",
        "binarization": "median",
        "requires_precomputed_kl_distance": False,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": None,
        **sim_meta,
    }
    return data_df, y, x_binary.astype(float), metadata


def _generate_gaussian_outlier_continuous_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate Gaussian outlier data without median binarization."""
    n_samples = int(test_case["n_samples"])
    n_features = int(test_case["n_features"])
    n_inlier_clusters = int(test_case["n_inlier_clusters"])

    config = GaussianOutlierConfig(
        n_samples=n_samples,
        n_features=n_features,
        n_inlier_clusters=n_inlier_clusters,
        cluster_std=float(test_case["cluster_std"]),
        outlier_count=int(test_case["outlier_count"]),
        outlier_distance=float(test_case["outlier_distance"]),
        outlier_std=float(test_case["outlier_std"]),
        spatial_mode=str(test_case["outlier_spatial_mode"]),
        label_mode=str(test_case["outlier_label_mode"]),
        balanced_clusters=bool(test_case["balanced_clusters"]),
        random_seed=seed,
    )

    x_continuous, y, sim_meta = generate_gaussian_outliers(config)
    data_df, feature_space, distance_condensed = _continuous_dataframe_and_metadata(
        x_continuous,
        [f"S{j}" for j in range(n_samples)],
        [f"F{j}" for j in range(n_features)],
    )
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": int(np.unique(y).size),
        "n_inlier_clusters": n_inlier_clusters,
        "noise": float(test_case["outlier_count"]) / float(n_samples),
        "name": str(test_case["name"]),
        "generator": "gaussian_outliers_continuous",
        "feature_representation": "continuous",
        "feature_space": feature_space,
        "distance_metric": "euclidean",
        "requires_precomputed_kl_distance": True,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": distance_condensed,
        **sim_meta,
    }
    return data_df, y, x_continuous.astype(float, copy=False), metadata


def _generate_sbm_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a graph test case using the SBM generator.

    Returns a DataFrame where each row corresponds to a node and columns are node indices
    (adjacency rows). Also returns ground-truth labels and the raw adjacency matrix
    as X_original for downstream use.
    """
    sizes = test_case["sizes"]
    p_intra = test_case["p_intra"]
    p_inter = test_case["p_inter"]
    directed = bool(test_case["directed"])
    allow_self_loops = bool(test_case["allow_self_loops"])

    G, ground_truth, A, sbm_meta = generate_sbm(
        sizes=sizes,
        p_intra=p_intra,
        p_inter=p_inter,
        seed=seed,
        directed=directed,
        allow_self_loops=allow_self_loops,
    )

    n_nodes = int(sbm_meta["n_nodes"])
    data_df = pd.DataFrame(
        A.astype(int),
        index=[f"S{j}" for j in range(n_nodes)],
        columns=[f"F{j}" for j in range(n_nodes)],
    )

    sbm_expected = None
    sbm_modularity = None
    sbm_modularity_shifted = None
    sbm_modularity_norm = None

    adj = A.astype(float, copy=False)
    degrees = adj.sum(axis=1)
    m = adj.sum() / 2.0
    if m > 0:
        sbm_expected = np.outer(degrees, degrees) / (2.0 * m)
        sbm_modularity = adj - sbm_expected
        sbm_modularity_shifted = sbm_modularity - sbm_modularity.min()
        sbm_modularity_norm = sbm_modularity_shifted / (sbm_modularity_shifted.max() + 1e-10)
        precomputed_distance_matrix = 1.0 - sbm_modularity_norm
        distance_metric = "sbm_shifted_modularity"
    else:
        precomputed_distance_matrix = 1.0 - adj
        distance_metric = "sbm_adjacency_complement"
    np.fill_diagonal(precomputed_distance_matrix, 0.0)

    precomputed_distance_condensed = squareform(precomputed_distance_matrix)

    metadata = {
        "n_samples": n_nodes,
        "n_features": n_nodes,
        "n_clusters": int(sbm_meta["n_blocks"]),
        "noise": float(p_inter),
        "name": str(test_case["name"]),
        "generator": "sbm",
        "distance_metric": distance_metric,
        "adjacency": A,
        "requires_precomputed_kl_distance": True,
        "precomputed_distance_matrix": precomputed_distance_matrix,
        "precomputed_distance_condensed": precomputed_distance_condensed,
        "sbm_expected": sbm_expected,
        "sbm_modularity": sbm_modularity,
        "sbm_modularity_shifted": sbm_modularity_shifted,
        "sbm_modularity_norm": sbm_modularity_norm,
    }

    return data_df, ground_truth, A, metadata


def _validate_categorical_params(test_case: dict) -> Tuple[int, int, int]:
    """Validate canonical benchmark-case geometry for the categorical generator.

    Returns (n_samples, n_features, n_categories).
    """
    n_samples = int(_require_case_value(test_case, "n_samples", "Categorical"))
    n_features = int(_require_case_value(test_case, "n_features", "Categorical"))
    n_categories = int(_require_case_value(test_case, "n_categories", "Categorical"))
    return n_samples, n_features, n_categories


def _generate_categorical_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a categorical (multinomial) test case.

    Returns a DataFrame where each cell contains a category index (0 to K-1),
    and the distributions array contains the underlying probability distributions.
    """
    n_samples, n_features, n_categories = _validate_categorical_params(test_case)
    entropy = test_case["entropy_param"]
    balanced = test_case["balanced_clusters"]

    sample_dict, cluster_assignments, distributions = generate_categorical_feature_matrix(
        n_rows=n_samples,
        n_cols=n_features,
        n_categories=n_categories,
        entropy_param=entropy,
        n_clusters=test_case["n_clusters"],
        random_seed=seed,
        balanced_clusters=balanced,
    )

    original_names = list(sample_dict.keys())
    # Matrix of sampled categories (n_rows, n_cols)
    matrix = np.array([sample_dict[name] for name in original_names], dtype=int)

    # One-hot encode: category indices → binary indicators for Bernoulli KL pipeline
    data_df, n_binary, feature_space = _one_hot_encode_categorical(
        matrix,
        n_categories,
        original_names,
    )
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    metadata = {
        "n_samples": n_samples,
        "n_features": n_binary,
        "n_features_original": n_features,
        "n_categories": n_categories,
        "feature_space": feature_space,
        "n_clusters": test_case["n_clusters"],
        "noise": entropy,
        "name": str(test_case["name"]),
        "generator": "categorical",
        "distributions": distributions,  # (n_rows, n_cols, n_categories)
        "requires_precomputed_kl_distance": False,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": None,
    }

    return data_df, true_labels, matrix.astype(float), metadata


def _generate_phylogenetic_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a phylogenetic simulation test case.

    Simulates trait evolution along a random phylogenetic tree.
    Each taxon (leaf) becomes a cluster, and samples are drawn from
    the evolved distributions at each leaf.
    """
    n_taxa = int(_require_case_value(test_case, "n_taxa", "Phylogenetic"))
    n_features = int(_require_case_value(test_case, "n_features", "Phylogenetic"))
    n_categories = int(_require_case_value(test_case, "n_categories", "Phylogenetic"))
    samples_per_taxon = int(_require_case_value(test_case, "samples_per_taxon", "Phylogenetic"))
    mutation_rate = float(_require_case_value(test_case, "mutation_rate", "Phylogenetic"))
    root_concentration = float(
        _require_case_value(test_case, "root_concentration", "Phylogenetic")
    )

    sample_dict, cluster_assignments, distributions, phylo_meta = generate_phylogenetic_data(
        n_taxa=n_taxa,
        n_features=n_features,
        n_categories=n_categories,
        samples_per_taxon=samples_per_taxon,
        mutation_rate=mutation_rate,
        root_concentration=root_concentration,
        random_seed=seed,
    )

    original_names = list(sample_dict.keys())
    matrix = np.array([sample_dict[name] for name in original_names], dtype=int)

    # One-hot encode: category indices → binary indicators for Bernoulli KL pipeline
    data_df, n_binary, feature_space = _one_hot_encode_categorical(
        matrix,
        n_categories,
        original_names,
    )
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    metadata = {
        "n_samples": len(original_names),
        "n_features": n_binary,
        "n_features_original": n_features,
        "n_categories": n_categories,
        "feature_space": feature_space,
        "n_clusters": n_taxa,
        "n_taxa": n_taxa,
        "samples_per_taxon": samples_per_taxon,
        "mutation_rate": mutation_rate,
        "noise": mutation_rate,
        "name": str(test_case["name"]),
        "generator": "phylogenetic",
        "distributions": distributions,
        "tree_structure": phylo_meta["tree_structure"],
        "leaf_distributions": phylo_meta["leaf_distributions"],
        "requires_precomputed_kl_distance": False,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": None,
    }

    return data_df, true_labels, matrix.astype(float), metadata


def _generate_temporal_evolution_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a temporal evolution test case.

    Simulates sequence evolution along a growing branch over time.
    Each time point becomes a cluster, with increasing divergence from ancestor.
    """
    n_time_points = int(_require_case_value(test_case, "n_time_points", "Temporal evolution"))
    n_features = int(_require_case_value(test_case, "n_features", "Temporal evolution"))
    n_categories = int(_require_case_value(test_case, "n_categories", "Temporal evolution"))
    samples_per_time = int(_require_case_value(test_case, "samples_per_time", "Temporal evolution"))
    mutation_rate = float(_require_case_value(test_case, "mutation_rate", "Temporal evolution"))
    shift_strength = _require_case_value(test_case, "shift_strength", "Temporal evolution")
    root_concentration = float(
        _require_case_value(test_case, "root_concentration", "Temporal evolution")
    )

    sample_dict, cluster_assignments, distributions, evo_meta = generate_temporal_evolution_data(
        n_time_points=n_time_points,
        n_features=n_features,
        n_categories=n_categories,
        samples_per_time=samples_per_time,
        mutation_rate=mutation_rate,
        shift_strength=shift_strength,
        root_concentration=root_concentration,
        random_seed=seed,
    )

    original_names = list(sample_dict.keys())
    matrix = np.array([sample_dict[name] for name in original_names], dtype=int)

    # One-hot encode: category indices → binary indicators for Bernoulli KL pipeline
    data_df, n_binary, feature_space = _one_hot_encode_categorical(
        matrix,
        n_categories,
        original_names,
    )
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    metadata = {
        "n_samples": len(original_names),
        "n_features": n_binary,
        "n_features_original": n_features,
        "n_categories": n_categories,
        "feature_space": feature_space,
        "n_clusters": n_time_points,
        "n_time_points": n_time_points,
        "samples_per_time": samples_per_time,
        "mutation_rate": mutation_rate,
        "shift_strength": shift_strength,
        "noise": mutation_rate,
        "name": str(test_case["name"]),
        "generator": "temporal_evolution",
        "distributions": distributions,
        "divergence_from_ancestor": evo_meta["divergence_from_ancestor"],
        "divergence_matrix": evo_meta["divergence_matrix"],
        "requires_precomputed_kl_distance": False,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": None,
    }

    return data_df, true_labels, matrix.astype(float), metadata


def _generate_preloaded_case(
    test_case: dict,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a pre-existing data file (TSV/CSV) as a benchmark case.

    Required test_case keys:
        - file_path: path to the data file (absolute or relative to repo root)
        - sep: separator character
        - n_clusters: expected number of clusters for display
    """
    file_path = test_case["file_path"]
    sep = test_case["sep"]

    # Resolve relative paths against the repo root
    path = Path(file_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / path

    if not path.exists():
        raise FileNotFoundError(f"Preloaded data file not found: {path}")

    data_df = pd.read_csv(path, sep=sep, index_col=0)
    n_samples, n_features = data_df.shape

    # No ground truth labels for real data
    y = np.full(n_samples, np.nan)
    x_original = data_df.values.copy()

    metadata = {
        "name": str(test_case["name"]),
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": test_case["n_clusters"],
        "noise": np.nan,
        "generator": "preloaded",
        "source_file": str(path),
        "sparsity": float(1 - data_df.values.mean()),
        "requires_precomputed_kl_distance": False,
        "precomputed_distance_matrix": None,
        "precomputed_distance_condensed": None,
    }
    return data_df, y, x_original, metadata


def generate_case_data(
    test_case: dict,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create a binary dataframe, true labels, original features, and metadata for a test case.

    This function dispatches to specialized helpers based on ``test_case['generator']``.
    """
    _require_case_value(test_case, "name", "Benchmark case")
    generator = _require_case_value(test_case, "generator", "Benchmark case")
    seed = test_case["seed"] if generator != "preloaded" else None

    if generator == "binary":
        data_df, y, x_original, metadata = _generate_binary_case(test_case, seed)
    elif generator == "blobs":
        data_df, y, x_original, metadata = _generate_blobs_case(test_case, seed)
    elif generator == "blobs_continuous":
        data_df, y, x_original, metadata = _generate_blobs_continuous_case(test_case, seed)
    elif generator == "blobs_quantile":
        data_df, y, x_original, metadata = _generate_blobs_quantile_case(test_case, seed)
    elif generator == "dimensional_gaussian":
        data_df, y, x_original, metadata = _generate_dimensional_gaussian_case(test_case, seed)
    elif generator == "dimensional_gaussian_continuous":
        data_df, y, x_original, metadata = _generate_dimensional_gaussian_continuous_case(test_case, seed)
    elif generator == "gaussian_outliers":
        data_df, y, x_original, metadata = _generate_gaussian_outlier_case(test_case, seed)
    elif generator == "gaussian_outliers_continuous":
        data_df, y, x_original, metadata = _generate_gaussian_outlier_continuous_case(test_case, seed)
    elif generator == "sbm":
        data_df, y, x_original, metadata = _generate_sbm_case(test_case, seed)
    elif generator == "categorical":
        data_df, y, x_original, metadata = _generate_categorical_case(test_case, seed)
    elif generator == "phylogenetic":
        data_df, y, x_original, metadata = _generate_phylogenetic_case(test_case, seed)
    elif generator == "temporal_evolution":
        data_df, y, x_original, metadata = _generate_temporal_evolution_case(test_case, seed)
    elif generator == "preloaded":
        data_df, y, x_original, metadata = _generate_preloaded_case(test_case)
    else:
        raise ValueError(f"Unknown generator: {generator}")

    if "category" in test_case:
        metadata["category"] = test_case["category"]

    return data_df, y, x_original, metadata
