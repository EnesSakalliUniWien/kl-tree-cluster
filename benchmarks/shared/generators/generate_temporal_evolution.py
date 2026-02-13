"""
Generates synthetic temporal evolution data by simulating sequence divergence along a growing branch.

This module provides `generate_temporal_evolution_data` to create datasets that mimic
how sequences evolve over time along a single lineage, with increasing divergence
from the ancestral state.

The simulation:
1. Generates an ancestral distribution for each feature
2. Evolves the distribution forward in time with cumulative mutations
3. Samples sequences at each time point
4. Creates data where adjacent time points are more similar than distant ones

This is useful for testing:
- Temporal clustering (can we recover time points?)
- Evolutionary trajectory reconstruction
- Detecting gradual vs sudden divergence
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, List

from benchmarks.shared.evolution import (
    compute_js_divergence_per_feature,
    generate_dirichlet_distributions,
)


def _evolve_distribution(
    parent_dist: np.ndarray,
    mutation_rate: float,
    shift_strength: Tuple[float, float],
    n_categories: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Evolve distribution by one time step.

    Args:
        parent_dist: (n_features, n_categories) parent distribution
        mutation_rate: Probability of mutation per feature
        shift_strength: (min, max) range for distribution shift
        n_categories: Number of categories
        random_state: Random state

    Returns:
        (n_features, n_categories) evolved distribution
    """
    child_dist = parent_dist.copy()
    n_features = parent_dist.shape[0]

    for f in range(n_features):
        if random_state.random() < mutation_rate:
            # Shift toward a random category
            shift_cat = random_state.randint(n_categories)
            shift_amount = random_state.uniform(*shift_strength)

            new_dist = child_dist[f] * (1 - shift_amount)
            new_dist[shift_cat] += shift_amount
            child_dist[f] = new_dist / new_dist.sum()

    return child_dist


def _sample_from_distribution(
    dist: np.ndarray,
    n_samples: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Sample sequences from a distribution.

    Args:
        dist: (n_features, n_categories) probability distribution
        n_samples: Number of samples to draw
        random_state: Random state

    Returns:
        (n_samples, n_features) array of sampled categories
    """
    n_features = dist.shape[0]
    samples = np.zeros((n_samples, n_features), dtype=int)

    for i in range(n_samples):
        for f in range(n_features):
            samples[i, f] = random_state.choice(dist.shape[1], p=dist[f])

    return samples


def _compute_js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute mean Jensen-Shannon divergence between two distributions."""
    return compute_js_divergence_per_feature(p, q, eps)


def generate_temporal_evolution_data(
    n_time_points: int,
    n_features: int,
    n_categories: int = 4,
    samples_per_time: int = 20,
    mutation_rate: float = 0.3,
    shift_strength: Tuple[float, float] = (0.15, 0.5),
    root_concentration: float = 1.0,
    random_seed: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], np.ndarray, Dict]:
    """Generate temporal evolution data along a growing branch.

    Simulates sequence evolution over time where:
    - Each time point represents a snapshot of the evolving lineage
    - Divergence from ancestor increases monotonically
    - Adjacent time points are more similar than distant ones

    Args:
        n_time_points: Number of time points to sample (becomes n_clusters).
        n_features: Number of features/sites to simulate.
        n_categories: Number of categories per feature (e.g., 4 for DNA).
        samples_per_time: Number of samples to draw at each time point.
        mutation_rate: Probability of mutation per feature per time step.
            Higher = faster divergence.
        shift_strength: (min, max) range for how much a mutation shifts
            the distribution toward a new category.
        root_concentration: Dirichlet concentration for ancestral distribution.
        random_seed: Optional seed for reproducibility.

    Returns:
        Tuple of:
        - sample_dict: Dict mapping sample names to category arrays (n_features,)
        - cluster_assignments: Dict mapping sample names to time point (cluster) IDs
        - distributions: (n_samples, n_features, n_categories) probability arrays
        - metadata: Dict with evolution parameters and divergence metrics

    Example:
        >>> samples, labels, dists, meta = generate_temporal_evolution_data(
        ...     n_time_points=8, n_features=200, n_categories=4,
        ...     samples_per_time=20, mutation_rate=0.35
        ... )
        >>> print(f"Divergence over time: {meta['divergence_from_ancestor']}")
    """
    random_state = np.random.RandomState(random_seed)

    # Generate ancestral distribution
    ancestral_dist = generate_dirichlet_distributions(
        n_features, n_categories, root_concentration, random_state
    )

    # Evolve and sample at each time point
    sample_dict: Dict[str, np.ndarray] = {}
    cluster_assignments: Dict[str, int] = {}
    all_distributions: List[np.ndarray] = []
    distributions_over_time: List[np.ndarray] = []
    divergence_from_ancestor: List[float] = []

    current_dist = ancestral_dist.copy()
    sample_idx = 0

    for t in range(n_time_points):
        # Evolve from previous time point
        current_dist = _evolve_distribution(
            current_dist, mutation_rate, shift_strength, n_categories, random_state
        )
        distributions_over_time.append(current_dist.copy())

        # Compute divergence from ancestor
        js = _compute_js_divergence(ancestral_dist, current_dist)
        divergence_from_ancestor.append(js)

        # Sample sequences at this time point
        samples = _sample_from_distribution(
            current_dist, samples_per_time, random_state
        )

        for i in range(samples_per_time):
            sample_name = f"S{sample_idx}"
            sample_dict[sample_name] = samples[i]
            cluster_assignments[sample_name] = t
            all_distributions.append(current_dist)
            sample_idx += 1

    distributions_array = np.array(all_distributions)

    # Compute pairwise divergence matrix between time points
    divergence_matrix = np.zeros((n_time_points, n_time_points))
    for i in range(n_time_points):
        for j in range(n_time_points):
            divergence_matrix[i, j] = _compute_js_divergence(
                distributions_over_time[i], distributions_over_time[j]
            )

    metadata = {
        "n_time_points": n_time_points,
        "n_samples": len(sample_dict),
        "n_features": n_features,
        "n_categories": n_categories,
        "samples_per_time": samples_per_time,
        "mutation_rate": mutation_rate,
        "shift_strength": shift_strength,
        "root_concentration": root_concentration,
        "divergence_from_ancestor": divergence_from_ancestor,
        "divergence_matrix": divergence_matrix,
        "ancestral_distribution": ancestral_dist,
        "distributions_over_time": distributions_over_time,
    }

    return sample_dict, cluster_assignments, distributions_array, metadata


__all__ = ["generate_temporal_evolution_data"]
