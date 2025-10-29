"""
Generates synthetic binary feature matrices for clustering experiments.

This module provides a primary function, `generate_random_feature_matrix`,
to create datasets with controllable properties like the number of clusters,
cluster separation (entropy), and feature diversity.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


# ============================================================================
# HELPER FUNCTIONS - Modularized Logic
# ============================================================================


def _calculate_cluster_sizes(n_rows: int, n_clusters: int, balanced: bool) -> List[int]:
    """Calculates the number of samples to assign to each cluster.

    Args:
        n_rows: The total number of samples to generate.
        n_clusters: The number of clusters to partition the samples into.
        balanced: If True, distributes samples as evenly as possible. If False,
            assigns a random number of samples to each cluster.

    Returns:
        A list where each element is the size of a cluster. The sum of the
        elements will equal `n_rows`.
    """
    if balanced or n_clusters == 1:
        # Distribute samples as evenly as possible
        samples_per_cluster, remainder = divmod(n_rows, n_clusters)
        return [
            samples_per_cluster + 1 if i < remainder else samples_per_cluster
            for i in range(n_clusters)
        ]
    else:
        # Assign a random number of samples to each cluster
        cluster_sizes = [1] * n_clusters
        remaining = n_rows - n_clusters
        for _ in range(remaining):
            idx = np.random.randint(0, n_clusters)
            cluster_sizes[idx] += 1
        return cluster_sizes


def _create_gradient_templates(n_clusters: int, n_cols: int) -> List[List[int]]:
    """Creates distinct binary templates for each cluster.

    The templates are generated on a gradient. For example, for 3 clusters, the
    templates will have features that are approximately 0%, 50%, and 100% ones.

    Args:
        n_clusters: The number of templates to generate.
        n_cols: The number of features (length) of each template.

    Returns:
        A list of binary template vectors.
    """
    if n_clusters == 2:
        # For two clusters, create maximally distinct templates.
        return [[0] * n_cols, [1] * n_cols]
    else:
        templates = []
        for cluster_id in range(n_clusters):
            prob_ones = cluster_id / (n_clusters - 1) if n_clusters > 1 else 0.5
            template = np.random.choice(
                [0, 1], size=n_cols, p=[1 - prob_ones, prob_ones]
            ).tolist()
            templates.append(template)
        return templates


def _apply_bit_flip_noise(template: List[int], noise_prob: float) -> List[int]:
    """Applies bit-flip noise to a binary template.

    Each bit in the template is flipped with a probability of `noise_prob`.

    Args:
        template: The binary template vector (e.g., [0, 1, 0, 1]).
        noise_prob: The probability (0.0 to 1.0) of flipping each bit.

    Returns:
        A new binary vector with noise applied.
    """
    sample = []
    for bit in template:
        if np.random.random() < noise_prob:
            sample.append(1 - bit)  # Flip the bit
        else:
            sample.append(bit)  # Keep the bit
    return sample


def _apply_template_following(template: List[int], follow_prob: float) -> List[int]:
    """Generates a sample that probabilistically follows a template.

    For each position in the template, the output sample will either copy the
    template's bit (with probability `follow_prob`) or be a random bit.

    Args:
        template: The binary template vector to follow.
        follow_prob: The probability (0.0 to 1.0) of following the template
            at each position.

    Returns:
        A new, generated binary vector.
    """
    sample = []
    for bit in template:
        if np.random.random() < follow_prob:
            sample.append(bit)  # Follow the template
        else:
            sample.append(np.random.choice([0, 1]))  # Use a random bit
    return sample


def _ensure_feature_coverage(
    leaf_matrix_dict: Dict[str, List[int]], n_cols: int
) -> None:
    """Ensures all features appear at least once by modifying the data in-place.

    If a feature column is all zeros across all samples, this function will
    randomly pick one sample and set that feature to 1.

    Note:
        This operation can introduce a small amount of noise and potentially
        weaken the original cluster structure for the modified sample.

    Args:
        leaf_matrix_dict: A dictionary mapping sample names to feature vectors.
            This dictionary is modified in-place.
        n_cols: The total number of features.
    """
    feature_sums = np.sum(list(leaf_matrix_dict.values()), axis=0)
    missing_features = np.where(feature_sums == 0)[0]

    if len(missing_features) > 0:
        leaf_names = list(leaf_matrix_dict.keys())
        for feature_idx in missing_features:
            random_sample_name = np.random.choice(leaf_names)
            leaf_matrix_dict[random_sample_name][feature_idx] = 1


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def generate_random_feature_matrix(
    n_rows: int,
    n_cols: int,
    entropy_param: float = 0.5,
    n_clusters: int = 2,
    random_seed: Optional[int] = None,
    balanced_clusters: bool = True,
) -> Tuple[Dict[str, list], Dict[str, int]]:
    """Generates a random binary feature matrix with controllable clustering.

    This function creates synthetic binary data with configurable cluster
    structure, useful for testing and validating clustering algorithms.

    Args:
        n_rows: The number of samples (rows) to generate.
        n_cols: The number of features (columns) to generate.
        entropy_param: A float between 0.0 and 1.0 that controls cluster
            separation.
            - 0.0: Creates maximally distinct, pure clusters.
            - 0.5: Creates clusters with a moderate amount of noise/overlap.
            - 1.0: Creates a single, uniform cluster (minimum separation).
        n_clusters: The number of clusters to generate.
        random_seed: An optional seed for the random number generator to ensure
            reproducibility.
        balanced_clusters: If True, clusters will have nearly equal sizes. If
            False, cluster sizes will be random.

    Returns:
        A tuple containing:
        - leaf_matrix_dict: A dictionary mapping sample names (e.g., 'L1')
          to their binary feature vectors.
        - cluster_assignments: A dictionary mapping sample names to their
          assigned integer cluster ID.

    Examples:
        >>> # Generate 100 samples with 3 well-separated clusters
        >>> data, clusters = generate_random_feature_matrix(
        ...     n_rows=100, n_cols=50, entropy_param=0.1, n_clusters=3
        ... )
        >>> print(f"Generated {len(data)} samples.")

        >>> # Generate highly mixed data with 2 unbalanced clusters
        >>> data, clusters = generate_random_feature_matrix(
        ...     n_rows=50, n_cols=30, entropy_param=0.8, balanced_clusters=False
        ... )
        >>> print(f"Cluster sizes: {list(np.bincount(list(clusters.values())))}")
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Clamp n_clusters to a valid range [1, n_rows]
    n_clusters = max(1, min(n_clusters, n_rows))

    leaf_matrix_dict: Dict[str, List[int]] = {}
    cluster_assignments: Dict[str, int] = {}

    # Step 1: Calculate cluster sizes
    cluster_sizes = _calculate_cluster_sizes(n_rows, n_clusters, balanced_clusters)

    # Step 2: Generate cluster templates and samples based on entropy parameter
    if entropy_param >= 1.0 or n_clusters == 1:
        # Strategy A: All samples are identical (no real cluster structure)
        template = [1] * n_cols
        for i in range(n_rows):
            leaf_matrix_dict[f"L{i + 1}"] = template.copy()
            cluster_assignments[f"L{i + 1}"] = 0

    elif entropy_param <= 0.0:
        # Strategy B: Maximum separation (pure templates, no noise)
        cluster_templates = _create_gradient_templates(n_clusters, n_cols)
        sample_idx = 0
        for cluster_id, size in enumerate(cluster_sizes):
            for _ in range(size):
                leaf_matrix_dict[f"L{sample_idx + 1}"] = cluster_templates[
                    cluster_id
                ].copy()
                cluster_assignments[f"L{sample_idx + 1}"] = cluster_id
                sample_idx += 1

    else:
        # Strategy C: Intermediate entropy (templates with noise or convergence)
        if entropy_param < 0.5:
            # C.1: Low entropy -> Distinct templates + bit-flip noise
            noise_prob = entropy_param * 2  # Scale [0, 0.5) -> [0, 1)
            cluster_templates = _create_gradient_templates(n_clusters, n_cols)
            sample_idx = 0
            for cluster_id, size in enumerate(cluster_sizes):
                base_template = cluster_templates[cluster_id]
                for _ in range(size):
                    sample = _apply_bit_flip_noise(base_template, noise_prob)
                    leaf_matrix_dict[f"L{sample_idx + 1}"] = sample
                    cluster_assignments[f"L{sample_idx + 1}"] = cluster_id
                    sample_idx += 1
        else:
            # C.2: High entropy -> Converging templates + probabilistic following
            follow_prob = (entropy_param - 0.5) * 2  # Scale [0.5, 1) -> [0, 1)
            base_template = np.random.choice([0, 1], size=n_cols).tolist()
            cluster_templates = []
            for _ in range(n_clusters):
                if np.random.random() < follow_prob:
                    cluster_templates.append(base_template.copy())
                else:
                    cluster_templates.append(
                        np.random.choice([0, 1], size=n_cols).tolist()
                    )

            sample_idx = 0
            for cluster_id, size in enumerate(cluster_sizes):
                template = cluster_templates[cluster_id]
                for _ in range(size):
                    sample = _apply_template_following(template, follow_prob)
                    leaf_matrix_dict[f"L{sample_idx + 1}"] = sample
                    cluster_assignments[f"L{sample_idx + 1}"] = cluster_id
                    sample_idx += 1

    # Step 3: Ensure all features appear at least once
    _ensure_feature_coverage(leaf_matrix_dict, n_cols)

    return leaf_matrix_dict, cluster_assignments
