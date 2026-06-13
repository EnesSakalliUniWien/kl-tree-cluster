"""Generators for barycentric and traversal method-proof benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
    require_case_value,
)


def _cluster_sizes(
    *,
    n_samples: int,
    n_clusters: int,
    child_balance_grid: list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if not child_balance_grid:
        base = np.full(n_clusters, n_samples // n_clusters, dtype=int)
        base[: n_samples - int(base.sum())] += 1
        return base

    if len(child_balance_grid) != 2:
        raise ValueError("child_balance_grid must contain two root proportions.")
    left_fraction = float(child_balance_grid[0])
    right_fraction = float(child_balance_grid[1])
    total = left_fraction + right_fraction
    if total <= 0:
        raise ValueError("child_balance_grid proportions must sum to a positive value.")
    left_fraction /= total

    left_clusters = max(1, n_clusters // 2)
    right_clusters = n_clusters - left_clusters
    left_n = int(round(n_samples * left_fraction))
    left_n = min(max(left_n, left_clusters), n_samples - right_clusters)
    right_n = n_samples - left_n

    left = np.full(left_clusters, left_n // left_clusters, dtype=int)
    left[: left_n - int(left.sum())] += 1
    right = np.full(right_clusters, right_n // right_clusters, dtype=int)
    right[: right_n - int(right.sum())] += 1
    return np.concatenate([left, right])


def _labels_from_sizes(sizes: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.full(int(size), cluster, dtype=int) for cluster, size in enumerate(sizes)]
    )


def _binary_matrix_from_cluster_probabilities(
    *,
    rng: np.random.Generator,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    rows = np.empty((labels.size, probabilities.shape[1]), dtype=int)
    for cluster in np.unique(labels):
        mask = labels == cluster
        rows[mask] = rng.binomial(1, probabilities[int(cluster)], size=(int(mask.sum()), probabilities.shape[1]))
    return rows


def generate_binary_barycentric_template(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate binary clusters with controlled root child balance."""
    n_samples = int(require_case_value(test_case, "n_samples", "Binary barycentric"))
    n_features = int(require_case_value(test_case, "n_features", "Binary barycentric"))
    n_clusters = int(require_case_value(test_case, "n_clusters", "Binary barycentric"))
    rng = np.random.default_rng(seed)
    sizes = _cluster_sizes(
        n_samples=n_samples,
        n_clusters=n_clusters,
        child_balance_grid=test_case.get("child_balance_grid"),
    )
    labels = _labels_from_sizes(sizes)
    rng.shuffle(labels)

    signal = {"weak": 0.12, "moderate": 0.25, "strong": 0.35}.get(
        str(test_case.get("signal_strength", "moderate")),
        0.25,
    )
    probabilities = np.full((n_clusters, n_features), 0.5, dtype=float)
    block_width = max(1, n_features // max(n_clusters, 1))
    for cluster in range(n_clusters):
        start = cluster * block_width
        stop = n_features if cluster == n_clusters - 1 else min(n_features, start + block_width)
        probabilities[cluster, start:stop] = np.clip(0.5 + signal, 0.02, 0.98)
        probabilities[cluster, :start] = np.clip(probabilities[cluster, :start] - signal / 3.0, 0.02, 0.98)

    matrix = _binary_matrix_from_cluster_probabilities(
        rng=rng,
        labels=labels,
        probabilities=probabilities,
    )
    data_df = pd.DataFrame(
        matrix,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        noise=1.0 - signal,
        generator="binary_barycentric_template",
        source_family="binary_template",
        feature_representation="binary",
        requires_precomputed_kl_distance=False,
        extra={
            "child_balance_grid": list(test_case.get("child_balance_grid", [])),
            "signal_strength": str(test_case.get("signal_strength", "moderate")),
        },
    )
    return data_df, labels, matrix.astype(float), metadata


def generate_binary_selected_nonnull_only(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate a strong selected-edge binary case intended to fail support gates."""
    case = {
        **test_case,
        "child_balance_grid": [0.5, 0.5],
        "signal_strength": "strong",
    }
    data_df, labels, matrix, metadata = generate_binary_barycentric_template(case, seed)
    metadata["generator"] = "binary_selected_nonnull_only"
    metadata["source_family"] = "binary_selected_nonnull_only"
    metadata["edge_open_rate_target"] = float(test_case.get("edge_open_rate_target", 0.95))
    return data_df, labels, matrix, metadata


def generate_planted_hierarchy_deep_signal(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate a binary hierarchy with weak root contrast and strong descendants."""
    n_samples = int(require_case_value(test_case, "n_samples", "Planted hierarchy"))
    n_features = int(require_case_value(test_case, "n_features", "Planted hierarchy"))
    n_clusters = int(require_case_value(test_case, "n_clusters", "Planted hierarchy"))
    rng = np.random.default_rng(seed)
    sizes = _cluster_sizes(n_samples=n_samples, n_clusters=n_clusters)
    labels = _labels_from_sizes(sizes)
    rng.shuffle(labels)

    probabilities = np.full((n_clusters, n_features), 0.5, dtype=float)
    root_width = max(1, n_features // 6)
    descendant_width = max(1, (n_features - root_width) // max(n_clusters, 1))
    for cluster in range(n_clusters):
        root_sign = 1.0 if cluster < n_clusters / 2 else -1.0
        probabilities[cluster, :root_width] += 0.05 * root_sign
        start = root_width + cluster * descendant_width
        stop = n_features if cluster == n_clusters - 1 else min(n_features, start + descendant_width)
        probabilities[cluster, start:stop] += 0.32
    probabilities = np.clip(probabilities, 0.02, 0.98)

    matrix = _binary_matrix_from_cluster_probabilities(
        rng=rng,
        labels=labels,
        probabilities=probabilities,
    )
    data_df = pd.DataFrame(
        matrix,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        noise=0.68,
        generator="planted_hierarchy_deep_signal",
        source_family="planted_hierarchy_binary",
        feature_representation="binary",
        requires_precomputed_kl_distance=False,
        extra={
            "root_sibling_effect": str(test_case.get("root_sibling_effect", "weak_same")),
            "descendant_effect": str(test_case.get("descendant_effect", "strong")),
        },
    )
    return data_df, labels, matrix.astype(float), metadata
