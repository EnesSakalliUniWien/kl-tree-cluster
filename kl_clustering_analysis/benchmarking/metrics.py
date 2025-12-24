"""Metric helpers for benchmarking pipeline.

Contains helpers for computing clustering metrics (ARI, NMI, Purity).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
)


def _calculate_ari_nmi_purity_metrics(
    num_clusters: int,
    report_df: pd.DataFrame,
    sample_names: pd.Index,
    true_labels: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate clustering metrics (ARI, NMI, Purity).

    This mirrors the implementation that was previously in
    `benchmarking.pipeline` and is exported with the same name for
    backward compatibility.
    """
    if num_clusters > 0 and not report_df.empty:
        # Create a correct mapping from sample name (report_df index) to true cluster label
        true_label_map = {name: label for name, label in zip(sample_names, true_labels)}

        # Work on a copy to avoid side effects
        df_metrics = report_df.copy()
        df_metrics["true_cluster"] = df_metrics.index.map(true_label_map)

        ari = adjusted_rand_score(
            df_metrics["true_cluster"].values, df_metrics["cluster_id"].values
        )

        nmi = normalized_mutual_info_score(
            df_metrics["true_cluster"].values, df_metrics["cluster_id"].values
        )

        purity = homogeneity_score(
            df_metrics["true_cluster"].values, df_metrics["cluster_id"].values
        )
        return ari, nmi, purity

    return 0.0, 0.0, 0.0
