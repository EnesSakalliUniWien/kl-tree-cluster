"""Benchmark case-data generation for native binary matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
    require_case_value,
)
from benchmarks.shared.generators.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)


def _validate_binary_params(test_case: dict) -> tuple[int, int]:
    n_samples = int(require_case_value(test_case, "n_samples", "Binary"))
    n_features = int(require_case_value(test_case, "n_features", "Binary"))
    return n_samples, n_features


def generate_binary_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate a native Bernoulli benchmark matrix."""
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
    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=actual_cols,
        n_clusters=int(test_case["n_clusters"]),
        noise=float(entropy),
        generator="binary",
        source_family="binary_template",
        feature_representation="binary",
        requires_precomputed_tbs_distance=False,
        extra={"noise_features": noise_features},
    )

    return data_df, true_labels, matrix.astype(float), metadata
