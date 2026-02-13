"""Shared helpers for synthetic data generators."""

from __future__ import annotations

from typing import List

import numpy as np


def calculate_cluster_sizes(n_rows: int, n_clusters: int, balanced: bool) -> List[int]:
    """Calculate per-cluster sample counts that sum to ``n_rows``."""
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_rows < n_clusters:
        raise ValueError("n_rows must be >= n_clusters")

    if balanced or n_clusters == 1:
        samples_per_cluster, remainder = divmod(n_rows, n_clusters)
        return [
            samples_per_cluster + 1 if i < remainder else samples_per_cluster
            for i in range(n_clusters)
        ]

    cluster_sizes = [1] * n_clusters
    remaining = n_rows - n_clusters
    for _ in range(remaining):
        idx = np.random.randint(0, n_clusters)
        cluster_sizes[idx] += 1
    return cluster_sizes


__all__ = ["calculate_cluster_sizes"]
