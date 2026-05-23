"""HDBSCAN runner for precomputed distance matrices."""

from __future__ import annotations

import numpy as np
from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import _normalize_labels
from benchmarks.shared.util.decomposition import _ok_result_from_labels


def _run_hdbscan_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
) -> MethodRunResult:
    """Run HDBSCAN on a precomputed distance matrix."""
    import hdbscan

    n_samples = distance_matrix.shape[0]
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return _ok_result_from_labels(labels, range(n_samples))

    min_cluster_size = int(params["min_cluster_size"])
    min_samples = int(params["min_samples"])
    cluster_selection_epsilon = float(params["cluster_selection_epsilon"])
    hdbscan_kwargs = {
        k: v
        for k, v in params.items()
        if k not in ["min_cluster_size", "min_samples", "cluster_selection_epsilon"]
    }

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="precomputed",
        **hdbscan_kwargs,
    )
    model.fit(distance_matrix)
    labels = _normalize_labels(model.labels_)
    return _ok_result_from_labels(labels, range(n_samples))
