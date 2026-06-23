"""Categorical Dirichlet-multinomial method-proof generators."""

from __future__ import annotations

import numpy as np

from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
    one_hot_encode_categorical,
    require_case_value,
)


def _rare_tail_base(n_categories: int) -> np.ndarray:
    weights = 1.0 / np.arange(1, n_categories + 1, dtype=float) ** 1.4
    return weights / weights.sum()


def generate_categorical_dirichlet_multinomial(
    test_case: dict,
    seed: int | None,
) -> CaseDataResult:
    """Generate high-cardinality categorical data near simplex boundaries."""
    n_samples = int(require_case_value(test_case, "n_samples", "Dirichlet multinomial"))
    n_features = int(require_case_value(test_case, "n_features", "Dirichlet multinomial"))
    n_clusters = int(require_case_value(test_case, "n_clusters", "Dirichlet multinomial"))
    n_categories = int(require_case_value(test_case, "n_categories", "Dirichlet multinomial"))
    overdispersion = float(test_case.get("overdispersion", 0.0))
    rng = np.random.default_rng(seed)

    labels = np.arange(n_samples, dtype=int) % n_clusters
    rng.shuffle(labels)
    base = _rare_tail_base(n_categories)
    cluster_probs = np.tile(base, (n_clusters, n_features, 1))
    for cluster in range(n_clusters):
        promoted = (cluster + np.arange(n_features)) % n_categories
        for feature, category in enumerate(promoted):
            cluster_probs[cluster, feature] *= 0.85
            cluster_probs[cluster, feature, category] += 0.15
            cluster_probs[cluster, feature] /= cluster_probs[cluster, feature].sum()

    matrix = np.empty((n_samples, n_features), dtype=int)
    for row, cluster in enumerate(labels):
        for feature in range(n_features):
            probs = cluster_probs[int(cluster), feature]
            if overdispersion > 0:
                concentration = max((1.0 / overdispersion) - 1.0, 0.5)
                probs = rng.dirichlet(np.maximum(probs * concentration, 1e-4))
            matrix[row, feature] = int(rng.choice(n_categories, p=probs))

    sample_names = [f"S{j}" for j in range(n_samples)]
    data_df, n_binary, feature_space = one_hot_encode_categorical(
        matrix,
        n_categories,
        sample_names,
    )
    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_binary,
        n_clusters=n_clusters,
        noise=float(overdispersion),
        generator="categorical_dirichlet_multinomial",
        source_family="categorical_dirichlet_multinomial",
        feature_representation="categorical_one_hot",
        requires_precomputed_tbs_distance=False,
        extra={
            "feature_space": feature_space,
            "n_features_original": n_features,
            "n_categories": n_categories,
            "concentration_profile": str(test_case.get("concentration_profile", "rare_tail")),
            "overdispersion": overdispersion,
        },
    )
    return data_df, labels, matrix.astype(float), metadata
