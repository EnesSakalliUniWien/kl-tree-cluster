"""Confirm the report_df index alignment bug in the benchmark pipeline.

Run:
    python debug_scripts/diagnostics/confirm_pipeline_bug.py
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.generators.generate_case_data import generate_case_data
from benchmarks.shared.util.decomposition import _create_report_dataframe
from kl_clustering_analysis.tree.poset_tree import PosetTree


def main() -> None:
    case = {
        "name": "gaussian_clear_1",
        "generator": "blobs",
        "n_samples": 150,
        "n_features": 20,
        "n_clusters": 3,
        "cluster_std": 0.5,
        "seed": 42,
    }

    data_t, y_t, _, _ = generate_case_data(case)

    distance = pdist(data_t.values, metric="hamming")
    Z = linkage(distance, method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_t.index.tolist())
    decomp = tree.decompose(leaf_data=data_t, alpha_local=0.05, sibling_alpha=0.05)
    cluster_assignments = decomp.get("cluster_assignments")
    if not isinstance(cluster_assignments, dict):
        raise TypeError("Decomposition returned non-dict 'cluster_assignments'.")
    report = _create_report_dataframe(cluster_assignments)

    if report.empty:
        raise ValueError("Decomposition produced empty report; cannot test index alignment.")

    truth = pd.Series(np.asarray(y_t), index=data_t.index)

    print("data_t.index[:10]:", data_t.index[:10].tolist())
    print("report.index[:10]:", report.index[:10].tolist())
    print("Same set?", set(report.index) == set(data_t.index))
    print("Same order?", report.index.equals(data_t.index))

    print("\n--- BUG (positional replacement) ---")
    report_bad = report.copy()
    report_bad.index = data_t.index
    y_bad = report_bad.index.to_series().map(truth)
    ari_bad = adjusted_rand_score(y_bad, report_bad["cluster_id"])
    print("ARI (buggy):", ari_bad)

    print("\n--- FIX (.loc reindex) ---")
    report_good = report.loc[data_t.index]
    y_good = report_good.index.to_series().map(truth)
    ari_good = adjusted_rand_score(y_good, report_good["cluster_id"])
    print("ARI (fixed):", ari_good)


if __name__ == "__main__":
    main()
