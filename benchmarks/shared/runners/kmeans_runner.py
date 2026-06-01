"""K-Means runner for benchmark method registry."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import _normalize_labels, _resolve_requested_cluster_count
from benchmarks.shared.util.decomposition import _ok_result_from_labels


def _run_kmeans_method(
    data_matrix: np.ndarray,
    params: dict[str, object],
    seed: int | None = None,
) -> MethodRunResult:
    """Run K-Means on feature matrix data."""
    X = np.asarray(data_matrix, dtype=float)
    n_samples = int(X.shape[0])

    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return _ok_result_from_labels(labels, range(n_samples))

    n_clusters = _resolve_requested_cluster_count(n_samples, params)
    n_init = int(params["n_init"])
    random_state = 42 if seed is None else int(seed)
    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    labels = _normalize_labels(model.fit_predict(X))
    return _ok_result_from_labels(labels, range(n_samples))
