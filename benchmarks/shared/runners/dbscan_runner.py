"""DBSCAN runner for precomputed distance matrices."""

from __future__ import annotations

import numpy as np
from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import _estimate_dbscan_eps, _normalize_labels
from benchmarks.shared.util.decomposition import _ok_result_from_labels
from sklearn.cluster import DBSCAN


def _run_dbscan_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
) -> MethodRunResult:
    """Run DBSCAN on a precomputed distance matrix."""

    n_samples = distance_matrix.shape[0]
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return _ok_result_from_labels(labels, range(n_samples))

    min_samples = int(params.get("min_samples", 5))
    eps = params.get("eps")
    if eps is None:
        eps = _estimate_dbscan_eps(distance_matrix, min_samples)
    model = DBSCAN(metric="precomputed", eps=float(eps), min_samples=min_samples)
    labels = _normalize_labels(model.fit_predict(distance_matrix))
    return _ok_result_from_labels(labels, range(n_samples))
