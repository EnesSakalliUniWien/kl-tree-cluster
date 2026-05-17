"""OPTICS runner for precomputed distance matrices."""

from __future__ import annotations

import numpy as np
from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import _normalize_labels
from benchmarks.shared.util.decomposition import _ok_result_from_labels
from sklearn.cluster import OPTICS


def _run_optics_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
    seed: int | None = None,
) -> MethodRunResult:
    """Run OPTICS on a precomputed distance matrix."""
    n_samples = int(distance_matrix.shape[0])
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return _ok_result_from_labels(labels, range(n_samples))

    # Kept for the shared runner signature; OPTICS is deterministic for fixed
    # precomputed distances and does not accept a random_state parameter.
    del seed

    try:
        min_samples = int(params.get("min_samples", 5))
        xi = float(params.get("xi", 0.05))
        min_cluster_size = params.get("min_cluster_size", min_samples)
        model = OPTICS(
            metric="precomputed",
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
        )
        labels = _normalize_labels(model.fit_predict(distance_matrix))
        return _ok_result_from_labels(labels, range(n_samples))
    except Exception as exc:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status="skip",
            skip_reason=f"OPTICS failed: {type(exc).__name__}: {exc}",
        )
