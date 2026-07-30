"""Shared contracts for benchmark case-data generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from tree_break_selection.tree.continuous_distance import (
    CONTINUOUS_STANDARDIZED_EUCLIDEAN_TREE_DISTANCE_METRIC,
    CONTINUOUS_TREE_DISTANCE_METRIC,
    continuous_time_distance_condensed,
    standardized_euclidean_distance_condensed,
)
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    continuous_feature_space_from_columns,
    infer_feature_space_from_columns,
)

CaseDataResult = tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]
CaseGenerator = Callable[[dict, int | None], CaseDataResult]

CORE_METADATA_KEYS = frozenset(
    {
        "n_samples",
        "n_features",
        "n_clusters",
        "noise",
        "name",
        "generator",
        "source_family",
        "feature_representation",
        "simulation_model",
        "observation_model",
        "benchmark_intent",
        "scientific_caution",
        "recommended_simulation_family",
        "requires_precomputed_tbs_distance",
        "precomputed_distance_matrix",
        "precomputed_distance_condensed",
    }
)

_SIMULATION_MODELS = {
    "binary_selected_nonnull_only": "selected_edge_bernoulli_stress",
    "binary_template": "bernoulli_template_latent_class",
    "categorical_dirichlet_multinomial": "dirichlet_multinomial_latent_class",
    "categorical_multinomial": "categorical_multinomial_latent_class",
    "continuous_low_rank_factor": "low_rank_gaussian_factor_model",
    "continuous_spiked_covariance": "spiked_covariance_gaussian_model",
    "dimensional_gaussian": "sparse_subspace_gaussian_mixture",
    "gaussian_blobs": "isotropic_gaussian_mixture",
    "gaussian_outliers": "gaussian_mixture_with_outlier_contamination",
    "phylogenetic_brownian": "brownian_like_phylogenetic_continuous_model",
    "phylogenetic_sequence": "branchwise_jukes_cantor_like_categorical_model",
    "planted_hierarchy_binary": "hierarchical_bernoulli_latent_class",
    "preloaded_matrix": "external_preloaded_matrix",
    "stochastic_block_model": "stochastic_block_model_graph",
    "temporal_sequence": "sequential_categorical_drift_model",
}

_OBSERVATION_MODELS = {
    "binary": "native_binary_feature_matrix",
    "categorical_one_hot": "one_hot_encoded_categorical_feature_blocks",
    "continuous": "continuous_feature_matrix",
    "graph_adjacency": "node_by_node_graph_adjacency_matrix",
    "median_binary": "per_feature_median_thresholded_continuous_matrix",
    "preloaded_matrix": "preloaded_matrix",
    "quantile_one_hot": "quantile_discretized_continuous_one_hot_blocks",
}

_RECOMMENDED_SIMULATION_FAMILIES = {
    "binary": "bernoulli_template_or_latent_class_binary_model",
    "categorical_one_hot": "categorical_or_dirichlet_multinomial_blocks",
    "continuous": "gaussian_mixture_or_sparse_subspace_gaussian",
    "graph_adjacency": "graph_sbm_or_lfr_with_graph_native_distances",
    "median_binary": "explicit_bernoulli_threshold_model_or_continuous_gaussian_variant",
    "preloaded_matrix": "documented_external_data_contract",
    "quantile_one_hot": "ordinal_or_categorical_discretization_model",
}


def require_case_value(test_case: dict, key: str, generator_name: str) -> Any:
    if key not in test_case:
        raise ValueError(f"{generator_name} generator requires '{key}'.")
    return test_case[key]


def metadata_extras(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return generator-specific metadata without duplicating core fields."""
    return {key: value for key, value in metadata.items() if key not in CORE_METADATA_KEYS}


def _benchmark_intent(
    *,
    source_family: str,
    feature_representation: str,
    requires_precomputed_tbs_distance: bool,
) -> str:
    if feature_representation == "median_binary":
        return "discretized_continuous_stress"
    if feature_representation == "graph_adjacency":
        return "graph_community_detection_stress"
    if feature_representation == "quantile_one_hot":
        return "discretized_continuous_categorical_stress"
    if feature_representation == "continuous" and requires_precomputed_tbs_distance:
        return "continuous_reference_or_diagnostic"
    if source_family.startswith("phylogenetic"):
        return "phylogenetic_sequence_or_trait_diagnostic"
    if source_family.startswith("categorical"):
        return "categorical_distributional_recovery"
    if "binary" in source_family:
        return "binary_distributional_recovery"
    return "benchmark_case"


def _scientific_caution(
    *,
    source_family: str,
    feature_representation: str,
    distance_metric: str | None,
) -> str:
    if feature_representation == "median_binary":
        return (
            "The simulated source is continuous but the observed benchmark matrix is "
            "median-thresholded binary; interpret it as a discretized stress case, not "
            "as raw Gaussian clustering."
        )
    if feature_representation == "graph_adjacency":
        return (
            "Adjacency rows are not independent feature measurements; distributional "
            "tests over rows and graph-native topology distances are different "
            "methodological contracts."
        )
    if feature_representation == "quantile_one_hot":
        return (
            "The simulated source is continuous but the observed benchmark matrix is "
            "quantile-discretized and one-hot encoded."
        )
    if source_family == "stochastic_block_model" and distance_metric is None:
        return "Graph benchmarks require an explicit graph-derived topology distance."
    return "none"


def methodological_contract_metadata(
    *,
    generator: str,
    source_family: str,
    feature_representation: str,
    requires_precomputed_tbs_distance: bool,
    distance_metric: str | None,
) -> dict[str, str]:
    """Return explicit scientific/methodological metadata for a generated case."""
    recommended = _RECOMMENDED_SIMULATION_FAMILIES.get(
        feature_representation,
        "case_specific_model_required",
    )
    if source_family.startswith("phylogenetic") and feature_representation == "categorical_one_hot":
        recommended = "phylogenetic_substitution_indel_sequence_simulator"
    return {
        "simulation_model": _SIMULATION_MODELS.get(source_family, generator),
        "observation_model": _OBSERVATION_MODELS.get(
            feature_representation,
            feature_representation,
        ),
        "benchmark_intent": _benchmark_intent(
            source_family=source_family,
            feature_representation=feature_representation,
            requires_precomputed_tbs_distance=requires_precomputed_tbs_distance,
        ),
        "scientific_caution": _scientific_caution(
            source_family=source_family,
            feature_representation=feature_representation,
            distance_metric=distance_metric,
        ),
        "recommended_simulation_family": recommended,
    }


def case_metadata(
    *,
    test_case: dict,
    n_samples: int,
    n_features: int,
    n_clusters: int,
    noise: float,
    generator: str,
    source_family: str,
    feature_representation: str,
    requires_precomputed_tbs_distance: bool,
    precomputed_distance_matrix: np.ndarray | None = None,
    precomputed_distance_condensed: np.ndarray | None = None,
    distance_metric: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical benchmark metadata shape for generated cases."""
    has_precomputed_distance = (
        precomputed_distance_matrix is not None or precomputed_distance_condensed is not None
    )
    if has_precomputed_distance and not requires_precomputed_tbs_distance:
        raise ValueError(
            "Precomputed TBS tree distances require requires_precomputed_tbs_distance=True."
        )
    if requires_precomputed_tbs_distance and precomputed_distance_condensed is None:
        raise ValueError(
            "Cases requiring a precomputed TBS tree distance must provide "
            "precomputed_distance_condensed."
        )
    metadata: dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_clusters": int(n_clusters),
        "noise": float(noise),
        "name": str(test_case["name"]),
        "generator": generator,
        "source_family": source_family,
        "feature_representation": feature_representation,
        **methodological_contract_metadata(
            generator=generator,
            source_family=source_family,
            feature_representation=feature_representation,
            requires_precomputed_tbs_distance=requires_precomputed_tbs_distance,
            distance_metric=distance_metric,
        ),
        "requires_precomputed_tbs_distance": bool(requires_precomputed_tbs_distance),
        "precomputed_distance_matrix": precomputed_distance_matrix,
        "precomputed_distance_condensed": precomputed_distance_condensed,
    }
    if distance_metric is not None:
        if not distance_metric:
            raise ValueError("distance_metric must be non-empty when provided.")
        metadata["distance_metric"] = distance_metric
    if extra:
        metadata.update(extra)
    return metadata


def one_hot_encode_categorical(
    matrix: np.ndarray,
    n_categories: int,
    sample_names: list[str],
) -> tuple[pd.DataFrame, int, FeatureSpace]:
    """Encode category-index columns into explicit one-hot feature blocks."""
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


def continuous_dataframe_and_metadata(
    matrix: np.ndarray,
    sample_names: list[str],
    feature_names: list[str],
    *,
    tree_distance_metric: str = CONTINUOUS_TREE_DISTANCE_METRIC,
) -> tuple[pd.DataFrame, FeatureSpace, np.ndarray]:
    """Return continuous benchmark data with one empirical-Gaussian block contract."""
    continuous_matrix = np.asarray(matrix, dtype=np.float64)
    data_df = pd.DataFrame(
        continuous_matrix,
        index=sample_names,
        columns=feature_names,
    )
    feature_space = continuous_feature_space_from_columns(tuple(data_df.columns))
    if tree_distance_metric == CONTINUOUS_TREE_DISTANCE_METRIC:
        distance_condensed = continuous_time_distance_condensed(
            continuous_matrix,
            feature_space,
        )
    elif tree_distance_metric == CONTINUOUS_STANDARDIZED_EUCLIDEAN_TREE_DISTANCE_METRIC:
        distance_condensed = standardized_euclidean_distance_condensed(
            continuous_matrix,
            feature_space,
        )
    else:
        raise ValueError(f"Unsupported continuous tree_distance_metric: {tree_distance_metric!r}.")
    return data_df, feature_space, distance_condensed
