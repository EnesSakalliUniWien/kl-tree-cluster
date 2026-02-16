"""OPTICS runner (moved to benchmarking.runners).

Same implementation as before; helpers are imported lazily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS

from benchmarks.shared.types.method_run_result import MethodRunResult


def _run_optics_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
    seed: int | None = None,
) -> "MethodRunResult":
    """Run OPTICS on a precomputed distance matrix and return a
    `MethodRunResult` (imported lazily to avoid circular imports).
    """
    from benchmarks.shared.util.core import _normalize_labels
    from benchmarks.shared.util.decomposition import _create_report_dataframe_from_labels

    n_samples = int(distance_matrix.shape[0])
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(labels, pd.Index(range(n_samples))),
            status="ok",
            skip_reason=None,
        )

    # Contract parity with other runners: normalize a deterministic default seed.
    # OPTICS itself is deterministic for fixed distance inputs and does not accept
    # a random_state parameter.
    _random_state = 42 if seed is None else int(seed)
    _ = _random_state

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
            report_df=None,
            status="skip",
            skip_reason=f"OPTICS failed: {type(exc).__name__}: {exc}",
        )
