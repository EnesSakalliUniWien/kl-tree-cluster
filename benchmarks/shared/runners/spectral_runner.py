"""Spectral Clustering runner for benchmark method registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import _normalize_labels
from benchmarks.shared.util.decomposition import _create_report_dataframe_from_labels


def _resolve_n_clusters(n_samples: int, params: dict[str, object]) -> int:
    """Resolve target K with safe bounds."""
    raw = params.get("n_clusters")
    if raw is None or str(raw).strip().lower() in {"true", "expected", "auto"}:
        raw = max(2, min(10, int(round(np.sqrt(max(n_samples, 2) / 2.0)))))
    n_clusters = int(raw)
    return max(1, min(n_clusters, n_samples))


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
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(
                labels, pd.Index(range(n_samples))
            ),
            status="ok",
            skip_reason=None,
        )

    n_clusters = _resolve_n_clusters(n_samples, params)
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

    try:
        model = SpectralClustering(**spectral_kwargs)
        labels = _normalize_labels(model.fit_predict(X))
        report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
        return MethodRunResult(
            labels=labels,
            found_clusters=int(len({x for x in labels if x >= 0})),
            report_df=report_df,
            status="ok",
            skip_reason=None,
        )
    except Exception as exc:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=pd.DataFrame(),
            status="error",
            skip_reason=f"Spectral failed: {exc}",
        )

