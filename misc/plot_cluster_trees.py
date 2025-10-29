#!/usr/bin/env python3
"""
Script to plot hierarchical trees with cluster assignments for the KL divergence clustering algorithm.

This script generates synthetic data, builds hierarchical trees, runs cluster decomposition,
and creates visualizations showing how clusters are formed at different levels of the tree.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.generate_random_feature_matrix import generate_random_feature_matrix
from tree.poset_tree import PosetTree
from hierarchy_analysis.cluster_decomposition import ClusterDecomposer
from hierarchy_analysis.statistical_tests import (
    annotate_nodes_with_statistical_significance_tests,
    annotate_local_child_parent_significance,
    annotate_sibling_independence_cmi,
)
from hierarchy_analysis.kl_correlation_analysis import (
    calculate_kl_divergence_mutual_information_matrix,
)
from hierarchy_analysis.divergence_metrics import (
    calculate_hierarchy_kl_divergence,
)
from plot.cluster_tree_visualization import (
    plot_tree_with_clusters,
    plot_cluster_summary,
)


def create_test_case_data(
    n_samples=50, n_features=20, n_clusters=3, noise_level=1.0, seed=42
):
    """Create synthetic test data using the same method as the validation suite."""
    from sklearn.datasets import make_blobs
    import numpy as np

    print(
        f"Generating test data: {n_samples} samples, {n_features} features, {n_clusters} clusters"
    )

    # Use make_blobs like the validation suite, then binarize
    X_continuous, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=noise_level,
        random_state=seed,
    )

    # Binarize by median threshold (same as validation suite)
    X_binary = (X_continuous > np.median(X_continuous, axis=0)).astype(int)

    # Convert to DataFrame
    X = pd.DataFrame(
        X_binary,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    y_true = pd.Series(y_true)

    print(f"Generated data shape: {X.shape}")
    print(f"True cluster sizes: {y_true.value_counts().sort_index()}")

    return X, y_true


def build_hierarchical_tree(X, linkage_method="complete", distance_metric="hamming"):
    """Build hierarchical tree from feature matrix using scipy linkage."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    print("Building hierarchical tree...")

    # Compute pairwise distances
    distance_matrix = pdist(X.values, metric=distance_metric)

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method=linkage_method)

    # Create PosetTree from linkage matrix
    tree = PosetTree.from_linkage(linkage_matrix, X.index.tolist())

    print(f"Tree built with {len(tree.nodes())} nodes")

    return tree, linkage_matrix


def run_statistical_analysis(tree, X):
    """Run statistical tests on the hierarchical tree."""
    print("Running statistical analysis...")

    # Calculate KL divergence statistics
    stats_df = calculate_hierarchy_kl_divergence(tree, X)

    # Compute mutual information matrix
    mi_matrix, _ = calculate_kl_divergence_mutual_information_matrix(tree, stats_df)

    # Run statistical tests
    results_df = annotate_nodes_with_statistical_significance_tests(
        stats_df,
        X.shape[1],
        significance_level_alpha=0.05,
        std_deviation_threshold=2.0,
        include_deviation_test=True,
    )
    results_df = annotate_local_child_parent_significance(
        tree,
        results_df,
        total_number_of_features=X.shape[1],
        significance_level_alpha=0.05,
    )
    results_df = annotate_sibling_independence_cmi(
        tree, results_df, significance_level_alpha=0.05, permutations=75
    )

    print(f"Statistical analysis complete. MI matrix shape: {mi_matrix.shape}")

    return mi_matrix, results_df


def run_cluster_decomposition(tree, results_df, mi_matrix):
    """Run the KL divergence cluster decomposition algorithm."""
    print("Running cluster decomposition...")

    # Initialize decomposer with fixed threshold
    decomposer = ClusterDecomposer(
        tree=tree,
        results_df=results_df,
        alpha_local=0.1,
    )

    # Run decomposition
    decomposition_results = decomposer.decompose_tree()

    print(f"Found {decomposition_results['num_clusters']} clusters")

    return decomposer, decomposition_results


def plot_tree_visualization(
    tree, decomposition_results, test_case_name, save_path=None
):
    """Create and save tree visualization with clusters."""
    print(f"Creating tree visualization for {test_case_name}...")

    # Create the plot
    fig, ax, cluster_colors = plot_tree_with_clusters(
        tree=tree,
        decomposition_results=decomposition_results,
        use_labels=True,
        figsize=(20, 14),
        node_size=2500,
        font_size=9,
        show_cluster_boundaries=True,
        layout="radial",
        title=f"Hierarchical Tree with KL Divergence Clusters\n{test_case_name}",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved tree visualization to {save_path}")

    return fig, ax


def plot_cluster_summary_visualization(
    decomposition_results, test_case_name, save_path=None
):
    """Create and save cluster summary visualization."""
    print(f"Creating cluster summary for {test_case_name}...")

    fig, axes = plot_cluster_summary(decomposition_results, figsize=(14, 6))

    # Add title
    fig.suptitle(
        f"Cluster Analysis Summary - {test_case_name}",
        fontsize=16,
        weight="bold",
        y=0.98,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved cluster summary to {save_path}")

    return fig, axes


def run_complete_analysis(test_cases=None):
    """Run complete analysis pipeline for multiple test cases."""

    if test_cases is None:
        test_cases = [
            {
                "name": "Simple_3_Clusters",
                "n_samples": 30,
                "n_features": 30,
                "n_clusters": 3,
                "noise_level": 1.0,
            },
            {
                "name": "Moderate_4_Clusters",
                "n_samples": 40,
                "n_features": 40,
                "n_clusters": 4,
                "noise_level": 1.5,
            },
            {
                "name": "Complex_5_Clusters",
                "n_samples": 50,
                "n_features": 50,
                "n_clusters": 5,
                "noise_level": 2.0,
            },
        ]

    # Create output directory
    output_dir = Path("cluster_tree_plots")
    output_dir.mkdir(exist_ok=True)

    results_summary = []

    for i, tc in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"TEST CASE {i + 1}: {tc['name']}")
        print(f"{'=' * 60}")

        try:
            # Generate data
            X, y_true = create_test_case_data(
                n_samples=tc["n_samples"],
                n_features=tc["n_features"],
                n_clusters=tc["n_clusters"],
                noise_level=tc["noise_level"],
                seed=42 + i,  # Different seed for each test case
            )

            # Build tree
            tree, linkage_matrix = build_hierarchical_tree(X)

            # Run statistical analysis
            mi_matrix, results_df = run_statistical_analysis(tree, X)

            # Run cluster decomposition
            decomposer, decomposition_results = run_cluster_decomposition(
                tree, results_df, mi_matrix
            )

            # Generate cluster report
            cluster_report = decomposer.generate_report(decomposition_results)

            # Calculate accuracy metrics
            y_pred = cluster_report["cluster_id"]
            from sklearn.metrics import (
                adjusted_rand_score,
                normalized_mutual_info_score,
                homogeneity_score,
            )

            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)
            purity = homogeneity_score(y_true, y_pred)

            print(".3f")
            print(".3f")
            print(".3f")

            # Store results
            results_summary.append(
                {
                    "Test_Case": tc["name"],
                    "True_Clusters": tc["n_clusters"],
                    "Found_Clusters": decomposition_results["num_clusters"],
                    "ARI": ari,
                    "NMI": nmi,
                    "Purity": purity,
                    "Samples": tc["n_samples"],
                    "Features": tc["n_features"],
                    "Noise": tc["noise_level"],
                }
            )

            # Create visualizations
            tree_plot_path = output_dir / f"tree_{tc['name']}.png"
            summary_plot_path = output_dir / f"summary_{tc['name']}.png"

            plot_tree_visualization(
                tree, decomposition_results, tc["name"], tree_plot_path
            )
            plot_cluster_summary_visualization(
                decomposition_results, tc["name"], summary_plot_path
            )

            # Close plots to save memory
            plt.close("all")

        except Exception as e:
            print(f"Error in test case {tc['name']}: {e}")
            import traceback

            traceback.print_exc()

    # Create summary report
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_path = output_dir / "analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'=' * 60}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to {output_dir}/")
        print(f"Summary: {summary_path}")

        # Print summary table
        print("\nSUMMARY OF RESULTS:")
        print(summary_df.to_string(index=False, float_format="%.3f"))

    return results_summary


if __name__ == "__main__":
    # Run the complete analysis
    results = run_complete_analysis()

    print("\n" + "=" * 60)
    print("PLOTTING COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("- cluster_tree_plots/tree_*.png: Tree visualizations with clusters")
    print("- cluster_tree_plots/summary_*.png: Cluster summary plots")
    print("- cluster_tree_plots/analysis_summary.csv: Results summary")
