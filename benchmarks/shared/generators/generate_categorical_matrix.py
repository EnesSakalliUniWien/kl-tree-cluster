"""
Generates synthetic categorical (multinomial) feature matrices for clustering experiments.

This module provides `generate_categorical_feature_matrix` to create datasets
where each feature has K categories (not just binary 0/1).
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


def _calculate_cluster_sizes(n_rows: int, n_clusters: int, balanced: bool) -> List[int]:
    """Calculates the number of samples to assign to each cluster."""
    if balanced or n_clusters == 1:
        samples_per_cluster, remainder = divmod(n_rows, n_clusters)
        return [
            samples_per_cluster + 1 if i < remainder else samples_per_cluster
            for i in range(n_clusters)
        ]
    else:
        cluster_sizes = [1] * n_clusters
        remaining = n_rows - n_clusters
        for _ in range(remaining):
            idx = np.random.randint(0, n_clusters)
            cluster_sizes[idx] += 1
        return cluster_sizes


def _create_categorical_templates(
    n_clusters: int,
    n_cols: int,
    n_categories: int,
    sparsity: Optional[float] = None,
) -> List[np.ndarray]:
    """Creates distinct categorical distribution templates for each cluster.

    Each template is a (n_cols, n_categories) array where each row is a
    probability simplex (sums to 1).

    Args:
        n_clusters: Number of cluster templates to generate.
        n_cols: Number of features.
        n_categories: Number of categories per feature.
        sparsity: If provided, concentrates probability mass on one category.
            Value in [0.5, 1.0] where 1.0 means all mass on one category.

    Returns:
        List of (n_cols, n_categories) probability arrays.
    """
    templates = []

    for cluster_id in range(n_clusters):
        template = np.zeros((n_cols, n_categories))

        for feat_idx in range(n_cols):
            # Assign each feature a "dominant" category based on cluster
            # This creates cluster-specific patterns
            dominant_cat = (cluster_id + feat_idx) % n_categories

            if sparsity is not None and sparsity > 0.5:
                # Sparse: concentrate mass on dominant category
                probs = np.ones(n_categories) * (1 - sparsity) / (n_categories - 1)
                probs[dominant_cat] = sparsity
            else:
                # Create a gradient: cluster_id shifts the distribution
                # Use Dirichlet with concentration on different categories
                alpha = np.ones(n_categories)
                alpha[dominant_cat] = 3.0  # Higher concentration on dominant
                probs = np.random.dirichlet(alpha)

            template[feat_idx] = probs

        templates.append(template)

    return templates


def _sample_from_categorical(probs: np.ndarray) -> int:
    """Sample a category from a probability distribution."""
    return np.random.choice(len(probs), p=probs)


def _apply_categorical_noise(
    template: np.ndarray,
    noise_prob: float,
    n_categories: int,
) -> np.ndarray:
    """Apply noise to categorical template by mixing with uniform distribution.

    Args:
        template: (n_cols, n_categories) probability array.
        noise_prob: Mixing weight for uniform distribution [0, 1].
        n_categories: Number of categories.

    Returns:
        Noisy probability array.
    """
    uniform = np.ones((template.shape[0], n_categories)) / n_categories
    return (1 - noise_prob) * template + noise_prob * uniform


def generate_categorical_feature_matrix(
    n_rows: int,
    n_cols: int,
    n_categories: int = 3,
    entropy_param: float = 0.5,
    n_clusters: int = 2,
    random_seed: Optional[int] = None,
    balanced_clusters: bool = True,
    category_sparsity: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], np.ndarray]:
    """Generates a random categorical feature matrix with controllable clustering.

    Creates synthetic categorical data where each feature has K categories.
    Returns both the sampled data and the underlying probability distributions.

    Args:
        n_rows: Number of samples (rows) to generate.
        n_cols: Number of features (columns) to generate.
        n_categories: Number of categories per feature (K >= 2).
        entropy_param: Float [0, 1] controlling cluster separation.
            - 0.0: Maximally distinct clusters (no noise).
            - 0.5: Moderate noise/overlap.
            - 1.0: Single uniform cluster (maximum noise).
        n_clusters: Number of clusters to generate.
        random_seed: Optional seed for reproducibility.
        balanced_clusters: If True, clusters have equal sizes.
        category_sparsity: Optional float [0.5, 1.0] to concentrate probability
            mass on dominant categories. Higher = more concentrated.

    Returns:
        Tuple of:
        - sample_dict: Dict mapping sample names to sampled category arrays (n_cols,)
        - cluster_assignments: Dict mapping sample names to cluster IDs
        - distributions: (n_rows, n_cols, n_categories) array of probability distributions
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_clusters = max(1, min(n_clusters, n_rows))
    n_categories = max(2, n_categories)

    sample_dict: Dict[str, np.ndarray] = {}
    cluster_assignments: Dict[str, int] = {}
    distributions: List[np.ndarray] = []

    cluster_sizes = _calculate_cluster_sizes(n_rows, n_clusters, balanced_clusters)

    # Generate cluster templates
    templates = _create_categorical_templates(
        n_clusters, n_cols, n_categories, category_sparsity
    )

    # Apply noise based on entropy_param
    if entropy_param >= 1.0 or n_clusters == 1:
        # All samples from uniform distribution
        uniform_template = np.ones((n_cols, n_categories)) / n_categories
        templates = [uniform_template] * n_clusters

    elif entropy_param > 0.0:
        # Mix templates with uniform based on entropy
        noise_prob = entropy_param
        templates = [
            _apply_categorical_noise(t, noise_prob, n_categories) for t in templates
        ]

    # Generate samples
    sample_idx = 0
    for cluster_id, size in enumerate(cluster_sizes):
        template = templates[cluster_id]

        for _ in range(size):
            # Sample categories from the distribution
            sample = np.array([
                _sample_from_categorical(template[feat_idx])
                for feat_idx in range(n_cols)
            ])

            sample_name = f"L{sample_idx + 1}"
            sample_dict[sample_name] = sample
            cluster_assignments[sample_name] = cluster_id
            distributions.append(template)
            sample_idx += 1

    distributions_array = np.array(distributions)

    return sample_dict, cluster_assignments, distributions_array


__all__ = ["generate_categorical_feature_matrix"]
