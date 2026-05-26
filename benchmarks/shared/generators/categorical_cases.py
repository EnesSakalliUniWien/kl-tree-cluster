"""Benchmark case-data generation for categorical sequence-like families."""

from __future__ import annotations

import numpy as np
from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
    one_hot_encode_categorical,
    require_case_value,
)
from benchmarks.shared.generators.generate_categorical_matrix import (
    generate_categorical_feature_matrix,
)
from benchmarks.shared.generators.generate_phylogenetic import generate_phylogenetic_data
from benchmarks.shared.generators.generate_temporal_evolution import (
    generate_temporal_evolution_data,
)


def _validate_categorical_params(test_case: dict) -> tuple[int, int, int]:
    n_samples = int(require_case_value(test_case, "n_samples", "Categorical"))
    n_features = int(require_case_value(test_case, "n_features", "Categorical"))
    n_categories = int(require_case_value(test_case, "n_categories", "Categorical"))
    return n_samples, n_features, n_categories


def generate_categorical_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate a categorical matrix and one-hot encode feature blocks."""
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
    matrix = np.array([sample_dict[name] for name in original_names], dtype=int)
    data_df, n_binary, feature_space = one_hot_encode_categorical(
        matrix,
        n_categories,
        original_names,
    )
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_binary,
        n_clusters=int(test_case["n_clusters"]),
        noise=float(entropy),
        generator="categorical",
        source_family="categorical_multinomial",
        feature_representation="categorical_one_hot",
        requires_precomputed_kl_distance=False,
        extra={
            "n_features_original": n_features,
            "n_categories": n_categories,
            "feature_space": feature_space,
            "distributions": distributions,
        },
    )

    return data_df, true_labels, matrix.astype(float), metadata


def generate_phylogenetic_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate phylogenetic categorical data and one-hot encode feature blocks."""
    n_taxa = int(require_case_value(test_case, "n_taxa", "Phylogenetic"))
    n_features = int(require_case_value(test_case, "n_features", "Phylogenetic"))
    n_categories = int(require_case_value(test_case, "n_categories", "Phylogenetic"))
    samples_per_taxon = int(require_case_value(test_case, "samples_per_taxon", "Phylogenetic"))
    mutation_rate = float(require_case_value(test_case, "mutation_rate", "Phylogenetic"))
    root_concentration = float(
        require_case_value(test_case, "root_concentration", "Phylogenetic")
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
    data_df, n_binary, feature_space = one_hot_encode_categorical(
        matrix,
        n_categories,
        original_names,
    )
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    metadata = case_metadata(
        test_case=test_case,
        n_samples=len(original_names),
        n_features=n_binary,
        n_clusters=n_taxa,
        noise=mutation_rate,
        generator="phylogenetic",
        source_family="phylogenetic_sequence",
        feature_representation="categorical_one_hot",
        requires_precomputed_kl_distance=False,
        extra={
            "n_features_original": n_features,
            "n_categories": n_categories,
            "feature_space": feature_space,
            "n_taxa": n_taxa,
            "samples_per_taxon": samples_per_taxon,
            "mutation_rate": mutation_rate,
            "distributions": distributions,
            "tree_structure": phylo_meta["tree_structure"],
            "leaf_distributions": phylo_meta["leaf_distributions"],
        },
    )

    return data_df, true_labels, matrix.astype(float), metadata


def generate_temporal_evolution_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate temporal categorical data and one-hot encode feature blocks."""
    n_time_points = int(require_case_value(test_case, "n_time_points", "Temporal evolution"))
    n_features = int(require_case_value(test_case, "n_features", "Temporal evolution"))
    n_categories = int(require_case_value(test_case, "n_categories", "Temporal evolution"))
    samples_per_time = int(require_case_value(test_case, "samples_per_time", "Temporal evolution"))
    mutation_rate = float(require_case_value(test_case, "mutation_rate", "Temporal evolution"))
    shift_strength = require_case_value(test_case, "shift_strength", "Temporal evolution")
    root_concentration = float(
        require_case_value(test_case, "root_concentration", "Temporal evolution")
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
    data_df, n_binary, feature_space = one_hot_encode_categorical(
        matrix,
        n_categories,
        original_names,
    )
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    metadata = case_metadata(
        test_case=test_case,
        n_samples=len(original_names),
        n_features=n_binary,
        n_clusters=n_time_points,
        noise=mutation_rate,
        generator="temporal_evolution",
        source_family="temporal_sequence",
        feature_representation="categorical_one_hot",
        requires_precomputed_kl_distance=False,
        extra={
            "n_features_original": n_features,
            "n_categories": n_categories,
            "feature_space": feature_space,
            "n_time_points": n_time_points,
            "samples_per_time": samples_per_time,
            "mutation_rate": mutation_rate,
            "shift_strength": shift_strength,
            "distributions": distributions,
            "divergence_from_ancestor": evo_meta["divergence_from_ancestor"],
            "divergence_matrix": evo_meta["divergence_matrix"],
        },
    )

    return data_df, true_labels, matrix.astype(float), metadata
