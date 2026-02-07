"""DBSCAN runner (moved to benchmarking.runners).

This is the same implementation as before; kept here so the package is self
contained. Runners import pipeline helpers lazily at runtime to avoid circular
imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from benchmarks.shared.types.method_run_result import (
    MethodRunResult,
)
from benchmarks.shared.utils_decomp import (
    _create_report_dataframe_from_labels,
)
from benchmarks.shared.utils import (
    _normalize_labels,
    _estimate_dbscan_eps,
)


def _run_dbscan_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
):
    """Run DBSCAN on a precomputed distance matrix and return a
    `MethodRunResult` (imported lazily to avoid circular imports).
    """

    n_samples = distance_matrix.shape[0]
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

    min_samples = int(params.get("min_samples", 5))
    eps = params.get("eps")
    if eps is None:
        eps = _estimate_dbscan_eps(distance_matrix, min_samples)
    model = DBSCAN(metric="precomputed", eps=float(eps), min_samples=min_samples)
    labels = _normalize_labels(model.fit_predict(distance_matrix))
    report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(len({x for x in labels if x >= 0})),
        report_df=report_df,
        status="ok",
        skip_reason=None,
    )
