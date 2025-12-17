"""
Test module for label-invariant cluster assignment validation.

Tests the KL-based decomposition algorithm with proper label remapping
to ensure clustering quality metrics are computed correctly.
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


def analyze_cluster_assignments_label_invariant():
    """Analyze cluster assignments with label remapping for correctness."""

    # 1) Data + tree + stats
    X, y_true = create_test_case_data(
        n_samples=30, n_features=30, n_clusters=3, noise_level=1.0, seed=42
    )
    tree, _ = build_hierarchical_tree(X)
    results_df = run_statistical_analysis(tree, X)

    # 2) Decomposition
    result = tree.decompose(results_df=results_df)
    assert isinstance(tree, PosetTree)
    report = tree.build_sample_cluster_assignments(result)

    # 3) Align predictions to data order
    sample_ids = list(X.index)
    # true labels as array in same order (y_true indexed by position)
    y_true_arr = np.array([y_true.iloc[int(sid[1:])] for sid in sample_ids])
    # raw cluster ids in same order
    y_pred_raw = np.array([report.loc[sid, "cluster_id"] for sid in sample_ids])

    # 4) Label remapping via Hungarian assignment on confusion matrix
    le_true = LabelEncoder()
    le_pred = LabelEncoder()
    y_true_enc = le_true.fit_transform(y_true_arr)
    y_pred_enc = le_pred.fit_transform(y_pred_raw)
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize agreement
    mapping_enc = {pred: true for pred, true in zip(col_ind, row_ind)}
    y_pred_remapped_enc = np.array([mapping_enc[p] for p in y_pred_enc])
    y_pred_remapped = le_true.inverse_transform(y_pred_remapped_enc)

    # 5) Metrics
    ari = float(adjusted_rand_score(y_true_arr, y_pred_remapped))
    nmi = float(normalized_mutual_info_score(y_true_arr, y_pred_remapped))
    acc_raw = float(np.mean(y_true_arr == y_pred_raw))
    acc_remapped = float(np.mean(y_true_arr == y_pred_remapped))

    # Purity (label-invariant by construction)
    tmp = pd.DataFrame({"y_true": y_true_arr, "y_pred": y_pred_raw})
    purities = [
        tmp[tmp["y_pred"] == c]["y_true"].value_counts().max()
        / len(tmp[tmp["y_pred"] == c])
        for c in sorted(tmp["y_pred"].unique())
    ]
    purity = float(np.mean(purities)) if purities else 0.0

    # 6) Build result structure
    sample_assignments = []
    for sid, t, r_raw, r_map in zip(
        sample_ids, y_true_arr, y_pred_raw, y_pred_remapped
    ):
        sample_assignments.append(
            {
                "sample_id": sid,
                "true_cluster": int(t),
                "kl_cluster_raw": int(r_raw),
                "kl_cluster_remapped": int(r_map),
                "correct_raw": bool(t == r_raw),
                "correct_remapped": bool(t == r_map),
            }
        )

    results = {
        "summary": {
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "n_clusters_true": int(len(np.unique(y_true_arr))),
            "n_clusters_kl": int(len(np.unique(y_pred_raw))),
            "noise_level": 1.0,
            "seed": 42,
        },
        "quality_metrics": {
            "ari": ari,
            "nmi": nmi,
            "accuracy_raw": acc_raw,
            "accuracy_remapped": acc_remapped,
            "purity": purity,
        },
        "sample_assignments": sample_assignments,
    }

    # Composition: by predicted cluster (raw)
    comp = {}
    for c in sorted(np.unique(y_pred_raw)):
        mask = y_pred_raw == c
        true_comp = pd.Series(y_true_arr[mask]).value_counts().to_dict()
        comp[f"kl_cluster_{int(c)}"] = {
            "total_samples": int(np.sum(mask)),
            "true_cluster_composition": {str(k): int(v) for k, v in true_comp.items()},
            "is_pure": len(true_comp) == 1,
        }
    results["cluster_composition"] = comp

    return results


def test_cluster_assignment_label_invariant():
    """Test that cluster assignments work correctly with label remapping."""

    results = analyze_cluster_assignments_label_invariant()

    # Basic structure checks
    assert "summary" in results
    assert "quality_metrics" in results
    assert "sample_assignments" in results
    assert "cluster_composition" in results

    summary = results["summary"]
    metrics = results["quality_metrics"]
    assignments = results["sample_assignments"]

    # Summary checks
    assert summary["n_samples"] == 30
    assert summary["n_features"] == 30
    assert summary["n_clusters_true"] == 3
    assert summary["n_clusters_kl"] == 3  # Should find correct number of clusters
    assert summary["noise_level"] == 1.0
    assert summary["seed"] == 42

    # Quality metrics checks - should be very high for this simple case
    assert metrics["ari"] >= 0.95, f"ARI too low: {metrics['ari']}"
    assert metrics["nmi"] >= 0.95, f"NMI too low: {metrics['nmi']}"
    assert metrics["purity"] >= 0.95, f"Purity too low: {metrics['purity']}"
    assert metrics["accuracy_remapped"] >= 0.95, (
        f"Remapped accuracy too low: {metrics['accuracy_remapped']}"
    )

    # Sample assignments checks
    assert len(assignments) == 30  # Should have one assignment per sample
    for assignment in assignments:
        assert "sample_id" in assignment
        assert "true_cluster" in assignment
        assert "kl_cluster_raw" in assignment
        assert "kl_cluster_remapped" in assignment
        assert "correct_raw" in assignment
        assert "correct_remapped" in assignment

        # Check data types
        assert isinstance(assignment["sample_id"], str)
        assert isinstance(assignment["true_cluster"], int)
        assert isinstance(assignment["kl_cluster_raw"], int)
        assert isinstance(assignment["kl_cluster_remapped"], int)
        assert isinstance(assignment["correct_raw"], bool)
        assert isinstance(assignment["correct_remapped"], bool)

    # Cluster composition checks
    composition = results["cluster_composition"]
    assert len(composition) == 3  # Should have 3 clusters

    total_samples_in_composition = 0
    for cluster_key, cluster_info in composition.items():
        assert cluster_key.startswith("kl_cluster_")
        assert "total_samples" in cluster_info
        assert "true_cluster_composition" in cluster_info
        assert "is_pure" in cluster_info

        total_samples_in_composition += cluster_info["total_samples"]

        # Each cluster should be relatively pure (most samples from one true cluster)
        true_comp = cluster_info["true_cluster_composition"]
        max_count = max(true_comp.values())
        cluster_purity = max_count / cluster_info["total_samples"]
        assert cluster_purity >= 0.8, (
            f"Cluster {cluster_key} purity too low: {cluster_purity}"
        )

    assert total_samples_in_composition == 30  # All samples accounted for


def test_cluster_assignment_consistency():
    """Test that the clustering results are consistent across multiple runs."""

    # Run the analysis twice with the same seed
    results1 = analyze_cluster_assignments_label_invariant()

    # Re-run (should be identical due to fixed seed)
    results2 = analyze_cluster_assignments_label_invariant()

    # Results should be identical
    assert results1["summary"] == results2["summary"]
    assert results1["quality_metrics"] == results2["quality_metrics"]

    # Sample assignments should be identical
    assert len(results1["sample_assignments"]) == len(results2["sample_assignments"])
    for i, (assign1, assign2) in enumerate(
        zip(results1["sample_assignments"], results2["sample_assignments"])
    ):
        assert assign1["sample_id"] == assign2["sample_id"]
        assert assign1["true_cluster"] == assign2["true_cluster"]
        assert assign1["kl_cluster_raw"] == assign2["kl_cluster_raw"]
        assert assign1["kl_cluster_remapped"] == assign2["kl_cluster_remapped"]
        assert assign1["correct_raw"] == assign2["correct_raw"]
        assert assign1["correct_remapped"] == assign2["correct_remapped"]
