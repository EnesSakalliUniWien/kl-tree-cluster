"""
Purpose: Benchmark the current clustering pipeline to verify it finds clusters correctly.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/pipeline_gates/q_current_clustering_pipeline_benchmark__pipeline__benchmark.py
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def generate_well_separated_clusters(
    n_clusters: int = 3,
    samples_per_cluster: int = 30,
    n_features: int = 10,
    separation: float = 2.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate well-separated Gaussian clusters for testing.

    Each cluster has a distinct center with controlled separation.
    Returns (data_df, true_labels).
    """
    np.random.seed(seed)

    all_data = []
    true_labels = []

    # Create cluster centers that are well-separated
    centers = []
    for c in range(n_clusters):
        center = np.zeros(n_features)
        # Each cluster is strong in different features
        start_idx = (c * n_features) // n_clusters
        end_idx = ((c + 1) * n_features) // n_clusters
        center[start_idx:end_idx] = separation
        centers.append(center)

    for cluster_id, center in enumerate(centers):
        for _ in range(samples_per_cluster):
            # Sample around center with unit variance
            sample = center + np.random.normal(0, 0.5, n_features)
            all_data.append(sample)
            true_labels.append(cluster_id)

    sample_names = [f"sample_{i}" for i in range(len(all_data))]
    data_df = pd.DataFrame(all_data, index=sample_names)

    return data_df, np.array(true_labels)


def extract_clusters_from_decomposition(
    decomposition_results: dict, debug: bool = False
) -> dict[str, int]:
    """Extract cluster assignments from decomposition results.

    Uses the cluster_assignments from tree.decompose() directly.
    This is the proper way to get cluster labels - matching quick_start.py.
    """
    cluster_assignments = decomposition_results.get("cluster_assignments", {})

    if debug:
        num_clusters = decomposition_results.get("num_clusters", 0)
        print(f"\nDEBUG: Decomposition found {num_clusters} clusters")

    leaf_to_cluster = {}
    for cluster_id, info in cluster_assignments.items():
        if debug and len(cluster_assignments) <= 10:
            print(
                f"  Cluster {cluster_id} (root: {info['root_node']}): {info['size']} samples"
            )
        for leaf in info["leaves"]:
            leaf_to_cluster[leaf] = cluster_id

    return leaf_to_cluster


def run_benchmark(
    n_clusters: int = 3,
    samples_per_cluster: int = 30,
    n_features: int = 10,
    seed: int = 42,
):
    """Run a single benchmark."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {n_clusters} clusters, {samples_per_cluster} samples each")
    print(f"Features: {n_features}, Seed: {seed}")
    print(f"{'=' * 60}")

    # Generate well-separated data
    data_df, true_labels = generate_well_separated_clusters(
        n_clusters=n_clusters,
        samples_per_cluster=samples_per_cluster,
        n_features=n_features,
        separation=2.0,
        seed=seed,
    )
    print(f"\nData shape: {data_df.shape}")
    print(f"True clusters: {n_clusters} with {samples_per_cluster} samples each")

    # Check cluster separation
    from sklearn.metrics import silhouette_score

    sil = silhouette_score(data_df.values, true_labels)
    print(f"Silhouette score (ground truth): {sil:.4f}")

    # Build tree
    Z = linkage(pdist(data_df.values, metric="euclidean"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    print(f"Tree: {len(tree.nodes)} nodes, {len(tree.edges)} edges")

    # Check branch lengths
    branch_lengths = [d.get("branch_length", 0) for _, _, d in tree.edges(data=True)]
    print(
        f"Branch lengths: min={min(branch_lengths):.4f}, max={max(branch_lengths):.4f}, mean={np.mean(branch_lengths):.4f}"
    )

    # Run decomposition
    print("\nRunning decomposition...")
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
    )

    stats_df = tree.stats_df

    # Summarize test results
    print("\n--- Test Results Summary ---")

    # Child-parent tests
    if "Child_Parent_Divergence_P_Value" in stats_df.columns:
        pvals = stats_df["Child_Parent_Divergence_P_Value"].dropna()
        print(f"Child-Parent p-values: min={pvals.min():.6f}, max={pvals.max():.6f}")
        print(f"  p < 0.01: {(pvals < 0.01).sum()}")
        print(f"  p < 0.05: {(pvals < 0.05).sum()}")
        print(f"  p < 0.10: {(pvals < 0.10).sum()}")

    if "Child_Parent_Divergence_Significant" in stats_df.columns:
        cp_sig = stats_df["Child_Parent_Divergence_Significant"].sum()
        cp_tested = stats_df["Child_Parent_Divergence_P_Value"].notna().sum()
        print(f"Child-Parent significant (after FDR): {cp_sig}/{cp_tested}")

    # Sibling tests
    if "Sibling_BH_Different" in stats_df.columns:
        sib_diff = stats_df["Sibling_BH_Different"].sum()
        sib_tested = stats_df["Sibling_Divergence_P_Value"].notna().sum()
        print(f"Sibling different (after FDR): {sib_diff}/{sib_tested}")

    # Extract predicted clusters using decomposition results (like quick_start.py)
    leaf_to_cluster = extract_clusters_from_decomposition(decomp, debug=True)

    # Map to same order as true labels
    sample_names = data_df.index.tolist()
    pred_labels = [leaf_to_cluster.get(name, 0) for name in sample_names]

    n_predicted = len(set(pred_labels))
    print(f"\nPredicted clusters: {n_predicted}")

    # Compute clustering metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    print(f"\n--- Clustering Quality ---")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")

    # Interpretation
    if ari > 0.8:
        print("✓ Excellent clustering!")
    elif ari > 0.5:
        print("◐ Good clustering")
    elif ari > 0.2:
        print("△ Moderate clustering")
    else:
        print("✗ Poor clustering")

    return {
        "n_clusters_true": n_clusters,
        "n_clusters_pred": n_predicted,
        "ari": ari,
        "nmi": nmi,
        "silhouette": sil,
        "cp_significant": cp_sig
        if "Child_Parent_Divergence_Significant" in stats_df.columns
        else 0,
        "sibling_different": sib_diff
        if "Sibling_BH_Different" in stats_df.columns
        else 0,
    }


def main():
    print("=" * 60)
    print("CLUSTERING BENCHMARK - Current Pipeline")
    print("=" * 60)

    results = []

    # Test with different configurations - well separated clusters
    configs = [
        {"n_clusters": 2, "samples_per_cluster": 40, "n_features": 8, "seed": 1},
        {"n_clusters": 3, "samples_per_cluster": 30, "n_features": 12, "seed": 2},
        {"n_clusters": 4, "samples_per_cluster": 25, "n_features": 16, "seed": 3},
    ]

    for cfg in configs:
        result = run_benchmark(**cfg)
        results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'True K':<8} {'Pred K':<8} {'Silhouette':<12} {'ARI':<10} {'NMI':<10}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['n_clusters_true']:<8} {r['n_clusters_pred']:<8} {r['silhouette']:<12.4f} {r['ari']:<10.4f} {r['nmi']:<10.4f}"
        )

    avg_ari = np.mean([r["ari"] for r in results])
    avg_nmi = np.mean([r["nmi"] for r in results])
    print("-" * 50)
    print(f"{'Average':<8} {'':<8} {'':<12} {avg_ari:<10.4f} {avg_nmi:<10.4f}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
