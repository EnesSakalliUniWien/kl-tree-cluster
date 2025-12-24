"""OPTICS runner (moved to benchmarking.runners).

Same implementation as before; helpers are imported lazily.
"""

from __future__ import annotations

import numpy as np
from kl_clustering_analysis.benchmarking.types.method_run_result import MethodRunResult
from sklearn.cluster import OPTICS
import pandas as pd


def _run_optics_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
) -> "MethodRunResult":
    """Run OPTICS on a precomputed distance matrix and return a
    `MethodRunResult` (imported lazily to avoid circular imports).
    """
    from kl_clustering_analysis.benchmarking.utils_decomp import (
        _create_report_dataframe_from_labels,
    )
    from kl_clustering_analysis.benchmarking.utils import (
        _normalize_labels,
    )

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
    xi = float(params.get("xi", 0.05))
    min_cluster_size = params.get("min_cluster_size", min_samples)
    model = OPTICS(
        metric="precomputed",
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
    )
    labels = _normalize_labels(model.fit_predict(distance_matrix))
    report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(len({x for x in labels if x >= 0})),
        report_df=report_df,
        status="ok",
        skip_reason=None,
    )
