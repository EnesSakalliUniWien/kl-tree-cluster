"""
Test module for validating the KL-based clustering pipeline.

This test evaluates the performance of the statistical method by comparing
its clustering output against ground truth using label-invariant metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment

from kl_clustering_analysis.core_utils.pipeline_helpers import (
    create_test_case_data,
    build_hierarchical_tree,
    run_statistical_analysis,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def analyze_pipeline_performance():
    """Run the current KL-based pipeline and evaluate clustering quality."""

    # 1) Data + tree
    X, y_true = create_test_case_data(
        n_samples=30, n_features=30, n_clusters=3, noise_level=1.0, seed=42
    )
    tree, _ = build_hierarchical_tree(X)

    # 2) Statistical annotations (no attention rates)
    results_df = run_statistical_analysis(tree, X)

    # 3) Decomposition
    result = tree.decompose(results_df=results_df)
    assert isinstance(tree, PosetTree)
    report = tree.build_sample_cluster_assignments(result)

    # 4) Align predictions to data order and remap labels
    sample_ids = list(X.index)
    y_true_arr = np.array([y_true.iloc[int(sid[1:])] for sid in sample_ids])
    y_pred_raw = np.array([report.loc[sid, "cluster_id"] for sid in sample_ids])

    le_true = LabelEncoder()
    le_pred = LabelEncoder()
    y_true_enc = le_true.fit_transform(y_true_arr)
    y_pred_enc = le_pred.fit_transform(y_pred_raw)
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping_enc = {pred: true for pred, true in zip(col_ind, row_ind)}
    y_pred_remapped_enc = np.array([mapping_enc.get(p, -1) for p in y_pred_enc])

    # Filter out unmapped predictions if any
    valid_mask = y_pred_remapped_enc != -1
    y_true_arr = y_true_arr[valid_mask]
    y_pred_remapped_enc = y_pred_remapped_enc[valid_mask]

    y_pred_remapped = le_true.inverse_transform(y_pred_remapped_enc)

    # 5) Metrics
    ari = float(adjusted_rand_score(y_true_arr, y_pred_remapped))
    nmi = float(normalized_mutual_info_score(y_true_arr, y_pred_remapped))

    tmp = pd.DataFrame({"y_true": y_true_arr, "y_pred": y_pred_raw[valid_mask]})
    purities = [
        tmp[tmp["y_pred"] == c]["y_true"].value_counts().max()
        / len(tmp[tmp["y_pred"] == c])
        for c in sorted(tmp["y_pred"].unique())
    ]
    purity = float(np.mean(purities)) if purities else 0.0

    return {
        "ari": ari,
        "nmi": nmi,
        "purity": purity,
        "found_clusters": len(np.unique(y_pred_raw)),
    }


def test_pipeline_performance():
    """Test that the KL pipeline achieves high performance on a synthetic dataset."""

    results = analyze_pipeline_performance()

    # Assert that the metrics are high, indicating good performance
    assert results["found_clusters"] == 3, (
        f"Expected 3 clusters, but found {results['found_clusters']}"
    )
    assert results["ari"] >= 0.95, f"ARI too low: {results['ari']}"
    assert results["nmi"] >= 0.95, f"NMI too low: {results['nmi']}"
    assert results["purity"] >= 0.95, f"Purity too low: {results['purity']}"

    print("\nValidation of KL Pipeline:")
    print(f"  - Found Clusters: {results['found_clusters']} (Correct)")
    print(f"  - ARI Score: {results['ari']:.4f} (High)")
    print(f"  - NMI Score: {results['nmi']:.4f} (High)")
    print(f"  - Purity Score: {results['purity']:.4f} (High)")
    print("  -> Conclusion: The KL-based pipeline performs excellently on this test case.")
