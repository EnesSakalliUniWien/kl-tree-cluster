"""K-Means runner for benchmark method registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import _normalize_labels
from benchmarks.shared.util.decomposition import _create_report_dataframe_from_labels


def _resolve_n_clusters(n_samples: int, params: dict[str, object]) -> int:
    """Resolve target K with safe bounds."""
    raw = params.get("n_clusters")
    if raw is None or str(raw).strip().lower() in {"true", "expected", "auto"}:
        # Fallback heuristic when caller does not provide K.
        raw = max(2, min(10, int(round(np.sqrt(max(n_samples, 2) / 2.0)))))
    n_clusters = int(raw)
    return max(1, min(n_clusters, n_samples))


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
    n_init = int(params.get("n_init", 10))
    random_state = 42 if seed is None else int(seed)

    try:
        model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        )
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
            skip_reason=f"K-Means failed: {exc}",
        )

