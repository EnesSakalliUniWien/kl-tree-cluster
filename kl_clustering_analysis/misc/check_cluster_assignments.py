#!/usr/bin/env python3
"""
Cluster Assignment Validation Script (label-sensitive accuracy)

NOTE: This script computes raw accuracy without label remapping.
Cluster IDs are arbitrary; use ARI/NMI/Purity for label-invariant metrics.
For label-invariant evaluation, use the root-level `check_cluster_assignments.py`.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path for imports when running from misc/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from misc.plot_cluster_trees import (
    create_test_case_data,
    build_hierarchical_tree,
    run_statistical_analysis,
)
from hierarchy_analysis.cluster_decomposition import ClusterDecomposer
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def analyze_cluster_assignments():
    """Analyze cluster assignments and save results to file."""

    print("Generating test data...")
    X, y_true = create_test_case_data(
        n_samples=30, n_features=30, n_clusters=3, noise_level=1.0, seed=42
    )

    print("Building hierarchical tree...")
    tree, _ = build_hierarchical_tree(X)

    print("Running statistical analysis...")
    mi_matrix, results_df = run_statistical_analysis(tree, X)

    print("Running KL decomposition...")
    decomposer = ClusterDecomposer(
        tree=tree,
        results_df=results_df,
    )
    result = decomposer.decompose_tree()
    report = decomposer.generate_report(result)

    # Map sample IDs correctly
    y_true_mapped = []
    y_kl_mapped = []
    sample_mapping = []

    for i in range(len(X)):
        sample_id = f"S{i}"
        if sample_id in report.index:
            true_cluster = y_true.iloc[i]
            kl_cluster = report.loc[sample_id, "cluster_id"]
            y_true_mapped.append(true_cluster)
            y_kl_mapped.append(kl_cluster)
            sample_mapping.append(
                {
                    "sample_id": sample_id,
                    "true_cluster": true_cluster,
                    "kl_cluster": kl_cluster,
                    "correct": true_cluster == kl_cluster,
                }
            )

    # Create results dictionary
    results = {
        "summary": {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_clusters_true": len(y_true.unique()),
            "n_clusters_kl": len(report["cluster_id"].unique()),
            "noise_level": 1.0,
            "seed": 42,
        },
        "quality_metrics": {},
        "cluster_composition": {},
        "sample_assignments": sample_mapping,
    }

    # Calculate quality metrics
    if y_true_mapped and y_kl_mapped:
        results["quality_metrics"] = {
            "ari": float(adjusted_rand_score(y_true_mapped, y_kl_mapped)),
            "nmi": float(normalized_mutual_info_score(y_true_mapped, y_kl_mapped)),
            "accuracy": float(np.mean([s["correct"] for s in sample_mapping])),
        }

        # Calculate purity
        contingency = pd.crosstab(pd.Series(y_true_mapped), pd.Series(y_kl_mapped))
        results["quality_metrics"]["purity"] = float(
            np.sum(contingency.max(axis=0)) / np.sum(contingency.values)
        )

        # Confusion matrix
        cm = pd.crosstab(
            pd.Series(y_true_mapped),
            pd.Series(y_kl_mapped),
            rownames=["True"],
            colnames=["KL"],
        )
        results["confusion_matrix"] = cm.to_dict()

    # Analyze cluster composition
    print("\nAnalyzing cluster composition...")
    for kl_cluster in sorted(report["cluster_id"].unique()):
        samples_in_cluster = report[report["cluster_id"] == kl_cluster].index
        true_clusters = [y_true.iloc[int(sid[1:])] for sid in samples_in_cluster]
        unique_true = list(set(true_clusters))
        cluster_sizes = {tc: true_clusters.count(tc) for tc in unique_true}

        results["cluster_composition"][f"kl_cluster_{kl_cluster}"] = {
            "total_samples": len(samples_in_cluster),
            "true_cluster_composition": cluster_sizes,
            "is_pure": len(unique_true) == 1,
        }

    return results


def save_results_to_file(results, filename="cluster_assignment_analysis.json"):
    """Save analysis results to JSON file."""
    import json

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                str(key): convert_to_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {filename}")


def print_summary(results):
    """Print a human-readable summary of the results."""
    print("\n" + "=" * 60)
    print("CLUSTER ASSIGNMENT ANALYSIS SUMMARY")
    print("=" * 60)

    summary = results["summary"]
    metrics = results["quality_metrics"]

    print(f"Dataset: {summary['n_samples']} samples, {summary['n_features']} features")
    print(
        f"Clusters: {summary['n_clusters_true']} true, {summary['n_clusters_kl']} found by KL"
    )
    print(f"Parameters: noise={summary['noise_level']}, seed={summary['seed']}")

    print("\nQuality Metrics:")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    print("\nCluster Composition:")
    for kl_cluster, comp in results["cluster_composition"].items():
        cluster_num = kl_cluster.split("_")[-1]
        print(f"  KL Cluster {cluster_num}: {comp['total_samples']} samples")
        for true_cluster, count in comp["true_cluster_composition"].items():
            print(f"    - {count} samples from true cluster {true_cluster}")
        purity_status = "✅ Pure" if comp["is_pure"] else "❌ Mixed"
        print(f"    {purity_status}")

    # Overall assessment
    ari = metrics.get("ari", 0)
    if ari > 0.8:
        assessment = "✅ EXCELLENT: Clusters match ground truth very closely"
    elif ari > 0.6:
        assessment = "✅ GOOD: Clusters match ground truth reasonably well"
    elif ari > 0.3:
        assessment = "⚠️  MODERATE: Some cluster agreement but significant differences"
    else:
        assessment = "❌ POOR: Clusters do not match ground truth"

    print(f"\nOverall Assessment: {assessment}")


if __name__ == "__main__":
    results = analyze_cluster_assignments()
    save_results_to_file(results)
    print_summary(results)
