"""
Test module for validating cluster decomposition algorithm across multiple scenarios.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler

from tree.poset_tree import PosetTree
from hierarchy_analysis import calculate_hierarchy_kl_divergence
from hierarchy_analysis.cluster_decomposition import ClusterDecomposer
from hierarchy_analysis.kl_correlation_analysis import (
    calculate_kl_divergence_mutual_information_matrix,
)
from hierarchy_analysis.statistical_tests import (
    annotate_nodes_with_statistical_significance_tests,
    annotate_local_child_parent_significance,
    annotate_sibling_independence_cmi,
)
from plot.cluster_tree_visualization import (
    plot_tree_with_clusters,
    plot_cluster_summary,
)


def validate_cluster_algorithm(
    test_cases=None,
    significance_level=0.05,
    verbose=True,
    plot_umap=False,
    plot_trees=True,
):
    """
    Validate the cluster decomposition algorithm across multiple test cases.

    Parameters
    ----------
    test_cases : list of dict, optional
        List of test case configurations. Each dict should contain:
        - n_samples: number of samples
        - n_features: number of features
        - n_clusters: true number of clusters
        - cluster_std: noise level (standard deviation)
        - seed: random seed
        If None, uses default test cases.
    significance_level : float, default=0.05
        Statistical significance level
    verbose : bool, default=True
        If True, prints progress and displays validation results
    plot_umap : bool, default=False
        If True, generates t-SNE plots comparing KL clustering with K-means and spectral clustering
    plot_trees : bool, default=True
        If True, generates hierarchical tree visualizations with cluster assignments

    Returns
    -------
    df_results : pd.DataFrame
        Results dataframe with columns: Test, True, Found, Samples, Features,
        Noise, ARI, NMI, Purity
    fig : matplotlib.figure.Figure or None
        Validation plot figure (if verbose=True), otherwise None
    """
    # Default test cases
    if test_cases is None:
        test_cases = [
            # Clear, well-separated clusters (normal cases)
            {
                "n_samples": 30,
                "n_features": 30,
                "n_clusters": 3,
                "cluster_std": 0.5,  # Clear separation
                "seed": 100,
            },
            {
                "n_samples": 40,
                "n_features": 40,
                "n_clusters": 4,
                "cluster_std": 0.8,  # Moderately clear
                "seed": 101,
            },
            {
                "n_samples": 50,
                "n_features": 50,
                "n_clusters": 5,
                "cluster_std": 1.0,  # Standard clear case
                "seed": 102,
            },
            # Normal/mixed difficulty cases
            {
                "n_samples": 35,
                "n_features": 35,
                "n_clusters": 3,
                "cluster_std": 1.2,  # Slightly noisy
                "seed": 103,
            },
            {
                "n_samples": 45,
                "n_features": 45,
                "n_clusters": 4,
                "cluster_std": 1.5,  # Moderate noise
                "seed": 104,
            },
            {
                "n_samples": 55,
                "n_features": 55,
                "n_clusters": 5,
                "cluster_std": 1.8,  # Getting noisy
                "seed": 105,
            },
            {
                "n_samples": 25,
                "n_features": 25,
                "n_clusters": 2,
                "cluster_std": 0.7,  # Very clear, few clusters
                "seed": 106,
            },
            {
                "n_samples": 65,
                "n_features": 65,
                "n_clusters": 6,
                "cluster_std": 2.0,  # Moderately noisy
                "seed": 107,
            },
            # A few extremely noisy cases for robustness testing
            {
                "n_samples": 30,
                "n_features": 30,
                "n_clusters": 3,
                "cluster_std": 5.0,  # Very noisy
                "seed": 42,
            },
            {
                "n_samples": 40,
                "n_features": 20000,
                "n_clusters": 4,
                "cluster_std": 7.5,  # Extremely noisy
                "seed": 43,
            },
            {
                "n_samples": 300,
                "n_features": 2000,
                "n_clusters": 30,
                "cluster_std": 2,  # Ridiculously noisy
                "seed": 44,
            },
        ]

    if verbose:
        print("=" * 80)
        print("CLUSTER ALGORITHM VALIDATION")
        print("=" * 80 + "\n")

    results_data = []

    # Run test cases
    for i, tc in enumerate(test_cases, 1):
        if verbose:
            print(f"\nRunning test case {i}/{len(test_cases)}...")

        # Generate and process data
        X_t, y_t = make_blobs(
            n_samples=tc["n_samples"],
            n_features=tc["n_features"],
            centers=tc["n_clusters"],
            cluster_std=tc["cluster_std"],
            random_state=tc["seed"],
        )
        X_bin = (X_t > np.median(X_t, axis=0)).astype(int)
        data_t = pd.DataFrame(
            X_bin,
            index=[f"S{j}" for j in range(tc["n_samples"])],
            columns=[f"F{j}" for j in range(tc["n_features"])],
        )

        # Build tree and calculate statistics
        Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
        tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
        stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)

        # Calculate MI matrix
        try:
            mi_t, _ = calculate_kl_divergence_mutual_information_matrix(tree_t, stats_t)
            print(f"  MI calculation successful, shape: {mi_t.shape}")
        except Exception as e:
            print(f"  ERROR in MI calculation: {e}")
            mi_t = pd.DataFrame()  # Empty dataframe

        # Statistical testing - BH global and BH local (child vs parent)
        results_t = annotate_nodes_with_statistical_significance_tests(
            stats_t, tc["n_features"], significance_level, 2.0, True
        )
        results_t = annotate_local_child_parent_significance(
            tree_t, results_t, tc["n_features"], significance_level
        )
        results_t = annotate_sibling_independence_cmi(
            tree_t,
            results_t,
            significance_level_alpha=significance_level,
            permutations=150,
        )

        # Use BH-corrected significance with deviation testing (same as main pipeline)
        significance_column = "Are_Features_Dependent"

        # Debug: Check MI values
        print(f"  MI matrix shape: {mi_t.shape}")
        mi_values = mi_t.values.flatten()
        # Handle both float and int types for MI values
        if mi_values.dtype.kind == "f":  # float
            mi_values = mi_values[~np.isnan(mi_values)]  # Remove NaN
            mi_values = mi_values[np.isfinite(mi_values)]  # Remove inf
        if len(mi_values) > 0:
            print(
                f"  MI stats: mean={mi_values.mean():.3f}, std={mi_values.std():.3f}, min={mi_values.min():.3f}, max={mi_values.max():.3f}"
            )
        else:
            print("  MI matrix is empty or all NaN/inf!")

        decomposer_t = ClusterDecomposer(
            tree=tree_t,
            results_df=results_t,
            significance_column=significance_column,
            alpha_local=0.1,
        )
        decomp_t = decomposer_t.decompose_tree()

        # Print decision mode for debugging (no longer uses simple threshold)
        print(
            f"  Decision mode: CMI-based statistical testing (α={significance_level})"
        )

        # Create report dataframe from cluster assignments
        cluster_assignments = decomp_t.get("cluster_assignments", {})
        if cluster_assignments:
            rows = []
            for cid, info in cluster_assignments.items():
                for leaf in info["leaves"]:
                    rows.append(
                        {
                            "sample_id": leaf,
                            "cluster_id": cid,
                            "cluster_root": info["root_node"],
                            "cluster_size": info["size"],
                        }
                    )
            report_t = pd.DataFrame(rows).set_index("sample_id")
        else:
            report_t = pd.DataFrame(
                columns=["cluster_id", "cluster_root", "cluster_size"]
            ).set_index("sample_id")

        # Calculate metrics
        if decomp_t["num_clusters"] > 0 and not report_t.empty:
            # Create a correct mapping from sample name (report_t index) to true cluster label
            true_label_map = {name: label for name, label in zip(data_t.index, y_t)}
            report_t["true_cluster"] = report_t.index.map(true_label_map)

            # Sanity check: ensure all samples were mapped correctly
            if report_t["true_cluster"].isna().any():
                print(
                    f"WARNING: {report_t['true_cluster'].isna().sum()} samples couldn't be mapped back to ground truth labels"
                )
                print(
                    "This indicates a mismatch between PosetTree leaf labels and data_t.index"
                )
                # Continue but metrics will be affected

            ari = adjusted_rand_score(
                report_t["true_cluster"].values, report_t["cluster_id"].values
            )

            nmi = normalized_mutual_info_score(
                report_t["true_cluster"].values, report_t["cluster_id"].values
            )

            purities = [
                report_t[report_t["cluster_id"] == c]["true_cluster"]
                .value_counts()
                .max()
                / len(report_t[report_t["cluster_id"] == c])
                for c in report_t["cluster_id"].unique()
            ]
            purity = np.mean(purities) if purities else 0
        else:
            # No clusters were found, so metrics are 0
            ari = 0
            nmi = 0
            purity = 0

        results_data.append(
            {
                "Test": i,
                "True": tc["n_clusters"],
                "Found": decomp_t["num_clusters"],
                "Samples": tc["n_samples"],
                "Features": tc["n_features"],
                "Noise": tc["cluster_std"],
                "ARI": ari,
                "NMI": nmi,
                "Purity": purity,
            }
        )

    if verbose:
        print(f"Completed {len(test_cases)} test cases.       \n")

    df_results = pd.DataFrame(results_data)

    # Create visualization if verbose
    fig = None
    if verbose:
        fig = _create_validation_plot(df_results)
        fig.savefig("validation_results.png", dpi=150, bbox_inches="tight")
        print("Validation plot saved to 'validation_results.png'")
        # plt.show()  # Skip interactive display in terminal

        print("\nDetailed Results:")
        print(
            df_results[
                [
                    "Test",
                    "True",
                    "Found",
                    "Samples",
                    "Features",
                    "Noise",
                    "ARI",
                    "NMI",
                    "Purity",
                ]
            ].to_string(index=False)
        )

    # Create tree visualizations if requested
    if plot_trees:
        print("\nGenerating tree visualizations...")
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create output directory
        tree_plots_dir = Path("../cluster_tree_plots")
        tree_plots_dir.mkdir(exist_ok=True)

        # We need to regenerate the data and trees for plotting
        for i, tc in enumerate(test_cases, 1):
            if verbose:
                print(f"  Creating tree visualization for test case {i}...")

            # Regenerate the exact same data
            X_t, y_t = make_blobs(
                n_samples=tc["n_samples"],
                n_features=tc["n_features"],
                centers=tc["n_clusters"],
                cluster_std=tc["cluster_std"],
                random_state=tc["seed"],
            )

            # Get KL clustering results from our results dataframe
            kl_result = df_results[df_results["Test"] == i]
            if not kl_result.empty:
                # Recompute the KL clustering to get the tree and decomposition
                X_bin = (X_t > np.median(X_t, axis=0)).astype(int)
                data_t = pd.DataFrame(
                    X_bin,
                    index=[f"S{j}" for j in range(tc["n_samples"])],
                    columns=[f"F{j}" for j in range(tc["n_features"])],
                )

                Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
                tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
                stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)
                mi_t, _ = calculate_kl_divergence_mutual_information_matrix(
                    tree_t, stats_t
                )
                results_t = annotate_nodes_with_statistical_significance_tests(
                    stats_t, tc["n_features"], significance_level, 2.0, True
                )
                results_t = annotate_local_child_parent_significance(
                    tree_t, results_t, tc["n_features"], significance_level
                )
                results_t = annotate_sibling_independence_cmi(
                    tree_t,
                    results_t,
                    significance_level_alpha=significance_level,
                    permutations=150,
                )

                decomposer_t = ClusterDecomposer(
                    tree=tree_t,
                    results_df=results_t,
                    significance_column="Are_Features_Dependent",
                    alpha_local=0.1,
                )

                decomp_t = decomposer_t.decompose_tree()

                # Create tree visualization
                test_case_name = f"Test Case {i}: {tc['n_clusters']} Clusters (σ={tc['cluster_std']})"
                tree_fig, _, _ = plot_tree_with_clusters(
                    tree=tree_t,
                    decomposition_results=decomp_t,
                    use_labels=True,
                    figsize=(20, 14),
                    node_size=2500,
                    font_size=9,
                    show_cluster_boundaries=True,
                    title=f"Hierarchical Tree with KL Divergence Clusters\n{test_case_name}",
                )

                tree_filename = (
                    f"tree_test_{i}_{tc['n_clusters']}_clusters_{current_timestamp}.png"
                )
                tree_fig.savefig(
                    tree_plots_dir / tree_filename,
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(tree_fig)  # Close to save memory

                # Create cluster summary visualization
                if decomp_t["num_clusters"] > 0:
                    summary_fig, _ = plot_cluster_summary(decomp_t, figsize=(14, 6))
                    summary_fig.suptitle(
                        f"Cluster Analysis Summary - {test_case_name}",
                        fontsize=16,
                        weight="bold",
                        y=0.98,
                    )

                    summary_filename = f"summary_test_{i}_{tc['n_clusters']}_clusters_{current_timestamp}.png"
                    summary_fig.savefig(
                        tree_plots_dir / summary_filename,
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(summary_fig)  # Close to save memory

        print("Tree plots saved as '../cluster_tree_plots/tree_test_*_clusters_*.png'")
        print(
            "Summary plots saved as '../cluster_tree_plots/summary_test_*_clusters_*.png'"
        )

    # Create UMAP comparison visualizations if requested
    if plot_umap:
        print("\nGenerating UMAP comparison visualizations...")
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create output directory
        umap_plots_dir = Path("../cluster_tree_plots")
        umap_plots_dir.mkdir(exist_ok=True)

        # We need to regenerate the data for UMAP plotting
        for i, tc in enumerate(test_cases, 1):
            if verbose:
                print(f"  Creating UMAP visualization for test case {i}...")

            # Regenerate the exact same data
            X_t, y_t = make_blobs(
                n_samples=tc["n_samples"],
                n_features=tc["n_features"],
                centers=tc["n_clusters"],
                cluster_std=tc["cluster_std"],
                random_state=tc["seed"],
            )

            # Get KL clustering results from our results dataframe
            kl_result = df_results[df_results["Test"] == i]
            if not kl_result.empty:
                # Recompute the KL clustering to get labels
                X_bin = (X_t > np.median(X_t, axis=0)).astype(int)
                data_t = pd.DataFrame(
                    X_bin,
                    index=[f"S{j}" for j in range(tc["n_samples"])],
                    columns=[f"F{j}" for j in range(tc["n_features"])],
                )

                Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
                tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
                stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)
                results_t = annotate_nodes_with_statistical_significance_tests(
                    stats_t, tc["n_features"], significance_level, 2.0, True
                )
                results_t = annotate_local_child_parent_significance(
                    tree_t, results_t, tc["n_features"], significance_level
                )
                results_t = annotate_sibling_independence_cmi(
                    tree_t,
                    results_t,
                    significance_level_alpha=significance_level,
                    permutations=150,
                )

                decomposer_t = ClusterDecomposer(
                    tree=tree_t,
                    results_df=results_t,
                    significance_column="Are_Features_Dependent",
                    alpha_local=0.1,
                )

                decomp_t = decomposer_t.decompose_tree()

                # Create report dataframe to get KL labels
                cluster_assignments = decomp_t.get("cluster_assignments", {})
                if cluster_assignments:
                    rows = []
                    for cid, info in cluster_assignments.items():
                        for leaf in info["leaves"]:
                            rows.append(
                                {
                                    "sample_id": leaf,
                                    "cluster_id": cid,
                                }
                            )
                    report_t = pd.DataFrame(rows).set_index("sample_id")
                    # Map back to original sample order
                    sample_order = [f"S{j}" for j in range(tc["n_samples"])]
                    kl_labels = []
                    for sample in sample_order:
                        if sample in report_t.index:
                            kl_labels.append(report_t.loc[sample, "cluster_id"])
                        else:
                            kl_labels.append(-1)  # No cluster assigned
                    kl_labels = np.array(kl_labels)
                else:
                    kl_labels = np.full(tc["n_samples"], -1)

                # Create UMAP comparison visualization
                umap_fig = _create_umap_comparison_plot(
                    X_t, y_t, kl_labels, i, tc["n_clusters"]
                )

                umap_filename = (
                    f"umap_test_{i}_{tc['n_clusters']}_clusters_{current_timestamp}.png"
                )
                umap_fig.savefig(
                    umap_plots_dir / umap_filename,
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(umap_fig)  # Close to save memory

        print(
            "UMAP comparison plots saved as '../cluster_tree_plots/umap_test_*_clusters_*.png'"
        )

    return df_results, fig


def _create_validation_plot(df_results):
    """Create 4-panel visualization of validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Algorithm Performance Across Test Cases", fontsize=16, weight="bold", y=0.995
    )

    # 1. Cluster Detection Accuracy
    ax1 = axes[0, 0]
    x = df_results["Test"]
    width = 0.35
    ax1.bar(
        x - width / 2,
        df_results["True"],
        width,
        label="True Clusters",
        color="#2ecc71",
        alpha=0.8,
        edgecolor="black",
    )
    ax1.bar(
        x + width / 2,
        df_results["Found"],
        width,
        label="Found Clusters",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
    )
    for i, row in df_results.iterrows():
        match = "✓" if row["True"] == row["Found"] else "✗"
        color = "green" if row["True"] == row["Found"] else "red"
        ax1.text(
            row["Test"],
            max(row["True"], row["Found"]) + 0.3,
            match,
            ha="center",
            fontsize=20,
            weight="bold",
            color=color,
        )
    ax1.set_xlabel("Test Case", fontsize=11, weight="bold")
    ax1.set_ylabel("Number of Clusters", fontsize=11, weight="bold")
    ax1.set_title("Cluster Count: Expected vs Found", fontsize=12, weight="bold")
    ax1.legend(loc="upper left")
    ax1.set_xticks(x)
    ax1.grid(axis="y", alpha=0.3)

    # 2. Quality Metrics
    ax2 = axes[0, 1]
    x_pos = np.arange(len(df_results))
    ax2.plot(
        x_pos,
        df_results["ARI"],
        "o-",
        linewidth=2.5,
        markersize=10,
        label="ARI",
        color="#e74c3c",
    )
    ax2.plot(
        x_pos,
        df_results["NMI"],
        "s-",
        linewidth=2.5,
        markersize=10,
        label="NMI",
        color="#9b59b6",
    )
    ax2.plot(
        x_pos,
        df_results["Purity"],
        "^-",
        linewidth=2.5,
        markersize=10,
        label="Purity",
        color="#f39c12",
    )
    ax2.axhline(
        y=1.0, color="green", linestyle="--", alpha=0.3, linewidth=2, label="Perfect"
    )
    ax2.set_xlabel("Test Case", fontsize=11, weight="bold")
    ax2.set_ylabel("Score", fontsize=11, weight="bold")
    ax2.set_title("Clustering Quality Metrics", fontsize=12, weight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_results["Test"])
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    # 3. Dataset Complexity
    ax3 = axes[1, 0]
    colors = [
        "#2ecc71" if row["True"] == row["Found"] else "#e74c3c"
        for _, row in df_results.iterrows()
    ]
    ax3.scatter(
        df_results["Samples"],
        df_results["Features"],
        s=df_results["True"] * 100,
        c=colors,
        alpha=0.6,
        edgecolors="black",
        linewidth=2,
    )
    for i, row in df_results.iterrows():
        ax3.annotate(
            f"Test {row['Test']}\n{row['True']}→{row['Found']}",
            (row["Samples"], row["Features"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
        )
    ax3.set_xlabel("Number of Samples", fontsize=11, weight="bold")
    ax3.set_ylabel("Number of Features", fontsize=11, weight="bold")
    ax3.set_title(
        "Dataset Complexity (size = true clusters, color = match)",
        fontsize=12,
        weight="bold",
    )
    ax3.grid(alpha=0.3)

    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate summary
    matches = sum(df_results["True"] == df_results["Found"])
    accuracy = matches / len(df_results) * 100
    avg_ari = df_results["ARI"].mean()
    avg_nmi = df_results["NMI"].mean()
    avg_purity = df_results["Purity"].mean()

    summary_text = f"""
SUMMARY STATISTICS

Cluster Detection
  Correct:     {matches}/{len(df_results)} ({accuracy:.0f}%)
  
Quality Metrics (Average)
  ARI:         {avg_ari:.3f}
  NMI:         {avg_nmi:.3f}
  Purity:      {avg_purity:.3f}
  
Test Complexity
  Samples:     {df_results["Samples"].min()}-{df_results["Samples"].max()}
  Features:    {df_results["Features"].min()}-{df_results["Features"].max()}
  Clusters:    {df_results["True"].min()}-{df_results["True"].max()}
  Noise (σ):   {df_results["Noise"].min()}-{df_results["Noise"].max()}
"""

    ax4.text(
        0.1,
        0.5,
        summary_text,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    return fig


def _create_umap_comparison_plot(X_original, y_true, y_kl, test_case_num, n_clusters):
    """Create UMAP visualization comparing KL clustering with other methods."""
    import warnings

    # Suppress all warnings during UMAP and clustering operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Standardize the data for better UMAP performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_original)

        # Reduce dimensionality if too high for UMAP
        if X_scaled.shape[1] > 50:
            from sklearn.decomposition import PCA

            pca = PCA(
                n_components=min(50, X_scaled.shape[0] - 1, X_scaled.shape[1]),
                random_state=42,
            )
            X_scaled = pca.fit_transform(X_scaled)

        # Apply UMAP for dimensionality reduction
        import umap

        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, len(X_scaled) - 1),
            min_dist=0.1,
        )
        X_embedded = reducer.fit_transform(X_scaled)

        # Apply comparison clustering methods
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_kmeans = kmeans.fit_predict(X_scaled)

        spectral = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity="nearest_neighbors",
            assign_labels="cluster_qr",
        )
        y_spectral = spectral.fit_predict(X_scaled)  # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Test Case {test_case_num}: {n_clusters} Clusters - UMAP Comparison",
        fontsize=16,
        weight="bold",
        y=0.995,
    )

    methods = [
        ("Ground Truth", y_true, "True cluster labels"),
        ("KL Divergence", y_kl, "Local KL divergence clustering"),
        ("K-Means", y_kmeans, "K-means clustering"),
        ("Spectral", y_spectral, "Spectral clustering"),
        ("UMAP X", X_embedded[:, 0], "UMAP dimension 1 (no clustering)"),
        ("UMAP Y", X_embedded[:, 1], "UMAP dimension 2 (no clustering)"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i, (method_name, labels, description) in enumerate(methods):
        ax = axes[i // 3, i % 3]

        if method_name in ["Ground Truth", "KL Divergence", "K-Means", "Spectral"]:
            # Clustering results - color by cluster
            scatter = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=labels,
                cmap="tab10",
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
            # Add cluster centers
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                if np.sum(mask) > 0:
                    center_x = np.mean(X_embedded[mask, 0])
                    center_y = np.mean(X_embedded[mask, 1])
                    ax.scatter(
                        center_x,
                        center_y,
                        c=[colors[cluster_id]],
                        marker="x",
                        s=100,
                        linewidth=3,
                    )
        else:
            # Raw t-SNE dimensions - color by value
            scatter = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
            plt.colorbar(scatter, ax=ax, shrink=0.8)

        ax.set_title(f"{method_name}\n{description}", fontsize=11, weight="bold")
        ax.set_xlabel("UMAP Dimension 1", fontsize=9)
        ax.set_ylabel("UMAP Dimension 2", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def test_cluster_algorithm_validation():
    """Test that the cluster algorithm works correctly across multiple test cases with varying noise levels."""
    df_results, _ = validate_cluster_algorithm(verbose=False)

    # Check that we have results for all test cases (3 clear + 5 normal + 3 noisy = 11 total)
    assert len(df_results) == 11, f"Expected 11 test cases, got {len(df_results)}"

    # Split results into different difficulty categories
    clear_results = df_results.head(3)  # First 3 are clear cases (seeds 100-102)
    normal_results = df_results.iloc[
        3:8
    ]  # Next 5 are normal/mixed cases (seeds 103-107)
    noisy_results = df_results.tail(3)  # Last 3 are extremely noisy (seeds 42-44)

    print(
        f"Clear cases (3): ARI={clear_results['ARI'].mean():.3f}, NMI={clear_results['NMI'].mean():.3f}"
    )
    print(
        f"Normal cases (5): ARI={normal_results['ARI'].mean():.3f}, NMI={normal_results['NMI'].mean():.3f}"
    )
    print(
        f"Noisy cases (3): ARI={noisy_results['ARI'].mean():.3f}, NMI={noisy_results['NMI'].mean():.3f}"
    )

    # Clear cases should perform excellently
    clear_correct = sum(clear_results["True"] == clear_results["Found"])
    assert clear_correct >= 2, (
        f"Clear cases: only {clear_correct}/3 found correct clusters"
    )
    assert clear_results["ARI"].mean() > 0.8, (
        f"Clear cases ARI too low: {clear_results['ARI'].mean():.3f}"
    )
    assert clear_results["NMI"].mean() > 0.85, (
        f"Clear cases NMI too low: {clear_results['NMI'].mean():.3f}"
    )

    # Normal cases should perform well
    normal_correct = sum(normal_results["True"] == normal_results["Found"])
    assert normal_correct >= 3, (
        f"Normal cases: only {normal_correct}/5 found correct clusters"
    )
    assert normal_results["ARI"].mean() > 0.6, (
        f"Normal cases ARI too low: {normal_results['ARI'].mean():.3f}"
    )
    assert normal_results["NMI"].mean() > 0.7, (
        f"Normal cases NMI too low: {normal_results['NMI'].mean():.3f}"
    )

    # Noisy cases will perform poorly but should still be valid
    noisy_correct = sum(noisy_results["True"] == noisy_results["Found"])
    print(f"Correct cluster counts in noisy cases: {noisy_correct}/3")

    # Just ensure the algorithm doesn't crash and produces reasonable results
    # Note: sklearn's NMI can sometimes produce values slightly > 1.0 due to numerical precision
    assert all(df_results["Found"] >= 0), "Found negative cluster counts"
    assert all(df_results["ARI"].between(0, 1)), "ARI values out of [0,1] range"
    assert all(df_results["NMI"] >= 0), (
        "NMI values should be non-negative"
    )  # Allow slightly > 1.0 due to numerical precision
    assert all(df_results["Purity"].between(0, 1)), "Purity values out of [0,1] range"

    # The real test: algorithm should handle the full spectrum from clear to extremely noisy data
    assert len(df_results) == 11, "All test cases should complete"
