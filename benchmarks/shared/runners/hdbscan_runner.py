"""HDBSCAN runner (moved to benchmarking.runners).

Same implementation as before; helpers are imported lazily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _run_hdbscan_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
) -> "MethodRunResult":
    """Run HDBSCAN on a precomputed distance matrix and return a
    `MethodRunResult` (imported lazily to avoid circular imports).
    """
    from benchmarks.shared.types.method_run_result import (
        MethodRunResult,
    )
    from benchmarks.shared.utils_decomp import (
        _create_report_dataframe_from_labels,
    )
    from benchmarks.shared.utils import (
        _normalize_labels,
    )

    # Lazy import hdbscan model
    try:
        import hdbscan
    except ImportError:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=pd.DataFrame(),
            status="error",
            skip_reason="hdbscan library not installed. Please install with `pip install hdbscan`.",
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
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="precomputed",  # Explicitly set metric to precomputed for distance_matrix
            **hdbscan_kwargs,
        )
        model.fit(distance_matrix)
        labels = _normalize_labels(model.labels_)
        report_df = _create_report_dataframe_from_labels(
            labels, pd.Index(range(n_samples))
        )
        return MethodRunResult(
            labels=labels,
            found_clusters=int(len({x for x in labels if x >= 0})),
            report_df=report_df,
            status="ok",
            skip_reason=None,
        )
    except Exception as e:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=pd.DataFrame(),
            status="error",
            skip_reason=f"HDBSCAN failed: {e}",
        )
