"""HDBSCAN runner for precomputed distance matrices."""

from __future__ import annotations

import numpy as np
from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import _normalize_labels
from benchmarks.shared.util.decomposition import _ok_result_from_labels
from sklearn.cluster import OPTICS


def _run_hdbscan_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
) -> MethodRunResult:
    """Run HDBSCAN on a precomputed distance matrix."""
    # Lazy import hdbscan model
    try:
        import hdbscan
    except ImportError:
        hdbscan = None

    n_samples = distance_matrix.shape[0]
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return _ok_result_from_labels(labels, range(n_samples))

    # Common HDBSCAN parameters
    # HDBSCAN's default min_samples is min_cluster_size
    min_cluster_size = int(params.get("min_cluster_size", 5))
    min_samples_param = params.get("min_samples")
    if min_samples_param is None:
        min_samples = min_cluster_size
    else:
        min_samples = int(min_samples_param)
    cluster_selection_epsilon = float(params.get("cluster_selection_epsilon", 0.0))
    # You can pass other hdbscan parameters via params dictionary
    hdbscan_kwargs = {
        k: v
        for k, v in params.items()
        if k not in ["min_cluster_size", "min_samples", "cluster_selection_epsilon"]
    }

    try:
        if hdbscan is not None:
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric="precomputed",  # Explicitly set metric to precomputed for distance_matrix
                **hdbscan_kwargs,
            )
            model.fit(distance_matrix)
            labels = _normalize_labels(model.labels_)
            return _ok_result_from_labels(labels, range(n_samples))

        # sklearn-only fallback when `hdbscan` is unavailable.
        model = OPTICS(
            metric="precomputed",
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            xi=0.05,
        )
        labels = _normalize_labels(model.fit_predict(distance_matrix))
        return _ok_result_from_labels(
            labels,
            range(n_samples),
            extra={"fallback": "sklearn_optics"},
        )
    except Exception as e:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status="skip",
            skip_reason=f"HDBSCAN/OPTICS fallback failed: {e}",
        )
