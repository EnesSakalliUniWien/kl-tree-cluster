"""Diagnostic plots for clustering benchmark."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

def create_k_distance_plot(distance_matrix: np.ndarray, k: int, test_case_num: int, meta: dict):
    """
    Creates and returns a k-distance plot.
    This plot helps in finding a suitable value for the `eps` parameter of DBSCAN.
    The "elbow" or "knee" in the plot is a good candidate for `eps`.
    """
    n_samples = distance_matrix.shape[0]
    if n_samples <= 1 or k >= n_samples:
        # Cannot compute k-distances
        return None

    # Calculate the distance to the k-th nearest neighbor for each point
    kth_distances = np.partition(distance_matrix, k, axis=1)[:, k]
    sorted_kth_distances = np.sort(kth_distances)[::-1]  # Sort in descending order

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(n_samples), sorted_kth_distances, marker='.', linestyle='-')
    ax.set_title(f'K-Distance Plot for Test Case {test_case_num} (k={k})')
    ax.set_xlabel('Points (sorted by distance)')
    ax.set_ylabel(f'{k}-th Nearest Neighbor Distance')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Also plot the estimated eps from the heuristic
    estimated_eps = np.median(kth_distances)
    ax.axhline(y=estimated_eps, color='r', linestyle='--', label=f'Median Heuristic Eps = {estimated_eps:.3f}')
    ax.legend()

    return fig
