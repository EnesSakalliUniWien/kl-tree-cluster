"""
Test module for validating the KL-based clustering pipeline.

This test evaluates the performance of the statistical method by comparing
its clustering output against ground truth using label-invariant metrics.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _create_test_case_data(
    n_samples: int = 50,
    n_features: int = 20,
    n_clusters: int = 3,
    noise_level: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic binary test data for end-to-end pipeline tests."""
    from sklearn.datasets import make_blobs

    x_continuous, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=noise_level,
        random_state=seed,
    )
    x_binary = (x_continuous > np.median(x_continuous, axis=0)).astype(int)
    x = pd.DataFrame(
        x_binary,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    return x, pd.Series(y_true)


def _build_hierarchical_tree(
    x: pd.DataFrame,
    linkage_method: str = "complete",
    distance_metric: str = "hamming",
) -> tuple[PosetTree, np.ndarray]:
    """Build a PosetTree from a binary feature matrix."""
    distance_matrix = pdist(x.values, metric=distance_metric)
    linkage_matrix = linkage(distance_matrix, method=linkage_method)
    tree = PosetTree.from_linkage(linkage_matrix, x.index.tolist())
    return tree, linkage_matrix


def _run_statistical_analysis(tree: PosetTree, x: pd.DataFrame) -> pd.DataFrame:
    """Run the production gate-annotation pipeline on a populated tree."""
    tree.populate_node_divergences(x)
    return run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=x,
    ).annotated_df


def _remap_labels(y_true_arr: np.ndarray, y_pred_raw: np.ndarray) -> np.ndarray:
    """Map predicted cluster ids to true labels, allowing extra predicted clusters."""
    le_true = LabelEncoder()
    le_pred = LabelEncoder()
    y_true_enc = le_true.fit_transform(y_true_arr)
    y_pred_enc = le_pred.fit_transform(y_pred_raw)

    cm = contingency_matrix(y_true_enc, y_pred_enc)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping_enc = {pred: true for true, pred in zip(row_ind, col_ind)}
    for pred in range(cm.shape[1]):
        if pred not in mapping_enc:
            mapping_enc[pred] = int(np.argmax(cm[:, pred]))

    y_pred_remapped_enc = np.array([mapping_enc[p] for p in y_pred_enc])
    return le_true.inverse_transform(y_pred_remapped_enc)


def analyze_pipeline_performance():
    """Run the current KL-based pipeline and evaluate clustering quality."""

    # 1) Data + tree
    X, y_true = _create_test_case_data(
        n_samples=90, n_features=60, n_clusters=3, noise_level=1.0, seed=42
    )
    tree, _ = _build_hierarchical_tree(X)

    # 2) Statistical annotations (no attention rates)
    annotations_df = _run_statistical_analysis(tree, X)

    # 3) Decomposition
    result = tree.decompose(annotations_df=annotations_df, leaf_data=X)
    assert isinstance(tree, PosetTree)
    report = tree.build_sample_cluster_assignments(result)

    # 4) Align predictions to data order and remap labels
    sample_ids = list(X.index)
    y_true_arr = np.array([y_true.iloc[int(sid[1:])] for sid in sample_ids])
    y_pred_raw = np.array([report.loc[sid, "cluster_id"] for sid in sample_ids])

    y_pred_remapped = _remap_labels(y_true_arr, y_pred_raw)

    # 5) Metrics
    ari = float(adjusted_rand_score(y_true_arr, y_pred_remapped))
    nmi = float(normalized_mutual_info_score(y_true_arr, y_pred_remapped))

    tmp = pd.DataFrame({"y_true": y_true_arr, "y_pred": y_pred_raw})
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
        "n_clusters_true": len(np.unique(y_true_arr)),
    }


def test_pipeline_performance():
    """Test that the KL pipeline achieves high performance on a synthetic dataset."""
    # Use a moderate edge alpha for this small test case (n=30)
    # to ensure sufficient power for 3-cluster detection.
    old_edge_alpha = config.EDGE_ALPHA
    config.EDGE_ALPHA = 0.01
    try:
        results = analyze_pipeline_performance()
    finally:
        config.EDGE_ALPHA = old_edge_alpha

    # Assert that the metrics are high, indicating good performance
    assert results["n_clusters_true"] <= results["found_clusters"] <= (
        results["n_clusters_true"] + 1
    ), f"Unexpected cluster count: {results['found_clusters']}"
    assert results["ari"] >= 0.95, f"ARI too low: {results['ari']}"
    assert results["nmi"] >= 0.95, f"NMI too low: {results['nmi']}"
    assert results["purity"] >= 0.95, f"Purity too low: {results['purity']}"

    print("\nValidation of KL Pipeline:")
    print(f"  - Found Clusters: {results['found_clusters']}")
    print(f"  - ARI Score: {results['ari']:.4f} (High)")
    print(f"  - NMI Score: {results['nmi']:.4f} (High)")
    print(f"  - Purity Score: {results['purity']:.4f} (High)")
    print("  -> Conclusion: The KL-based pipeline performs excellently on this test case.")
