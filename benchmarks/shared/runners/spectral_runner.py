"""Spectral Clustering runner for benchmark method registry."""

from __future__ import annotations

import numpy as np
from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import _normalize_labels, _resolve_requested_cluster_count
from benchmarks.shared.util.decomposition import _ok_result_from_labels
from sklearn.cluster import SpectralClustering


def _run_spectral_method(
    data_matrix: np.ndarray,
    params: dict[str, object],
    seed: int | None = None,
) -> MethodRunResult:
    """Run Spectral Clustering on feature matrix data."""
    X = np.asarray(data_matrix, dtype=float)
    n_samples = int(X.shape[0])

    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return _ok_result_from_labels(labels, range(n_samples))

    try:
        n_clusters = _resolve_requested_cluster_count(n_samples, params)
        random_state = 42 if seed is None else int(seed)
        affinity = str(params.get("affinity", "nearest_neighbors"))
        assign_labels = str(params.get("assign_labels", "cluster_qr"))

        spectral_kwargs: dict[str, object] = {
            "n_clusters": n_clusters,
            "random_state": random_state,
            "affinity": affinity,
            "assign_labels": assign_labels,
        }
        if affinity == "nearest_neighbors":
            n_neighbors_raw = params.get("n_neighbors")
            if n_neighbors_raw is None:
                n_neighbors = max(2, min(10, n_samples - 1))
            else:
                n_neighbors = max(1, min(int(n_neighbors_raw), n_samples - 1))
            spectral_kwargs["n_neighbors"] = n_neighbors

        model = SpectralClustering(**spectral_kwargs)
        labels = _normalize_labels(model.fit_predict(X))
        return _ok_result_from_labels(labels, range(n_samples))
    except Exception as exc:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status="skip",
            skip_reason=f"Spectral failed: {type(exc).__name__}: {exc}",
        )
