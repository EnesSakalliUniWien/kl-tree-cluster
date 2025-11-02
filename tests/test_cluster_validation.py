"""
Test module for validating cluster decomposition algorithm across multiple scenarios.
"""

import re
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
from hierarchy_analysis.local_kl_utils import get_local_kl_series
from hierarchy_analysis.statistical_tests import (
    annotate_nodes_with_statistical_significance_tests,
    annotate_child_parent_divergence,
    annotate_sibling_independence_cmi,
)
from plot.cluster_tree_visualization import (
    plot_tree_with_clusters,
    plot_cluster_summary,
)
from simulation.generate_random_feature_matrix import generate_random_feature_matrix


SMALL_TEST_CASES = [
    {
        "name": "clear",
        "n_samples": 24,
        "n_features": 12,
        "n_clusters": 3,
        "cluster_std": 0.4,
        "seed": 0,
    },
    {
        "name": "moderate",
        "n_samples": 30,
        "n_features": 16,
        "n_clusters": 3,
        "cluster_std": 1.0,
        "seed": 1,
    },
    {
        "name": "noisy",
        "n_samples": 30,
        "n_features": 16,
        "n_clusters": 3,
        "cluster_std": 1.6,
        "seed": 2,
    },
]


def _generate_case_data(
    test_case: dict,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Create a binary dataframe, true labels, original features, and metadata for a test case."""
    generator = test_case.get("generator", "blobs")
    seed = test_case.get("seed")

    if generator == "binary":
        n_rows = test_case.get("n_rows", test_case.get("n_samples"))
        n_cols = test_case.get("n_cols", test_case.get("n_features"))
        if n_rows is None or n_cols is None:
            raise ValueError(
                "Binary generator requires 'n_rows'/'n_cols' or 'n_samples'/'n_features'."
            )
        entropy = test_case.get("entropy_param", 0.5)
        balanced = test_case.get("balanced_clusters", True)

        data_dict, cluster_assignments = generate_random_feature_matrix(
            n_rows=n_rows,
            n_cols=n_cols,
            entropy_param=entropy,
            n_clusters=test_case["n_clusters"],
            random_seed=seed,
            balanced_clusters=balanced,
        )

        original_names = list(data_dict.keys())
        matrix = np.array([data_dict[name] for name in original_names], dtype=int)
        sample_names = [f"S{i}" for i in range(len(original_names))]
        feature_names = [f"F{j}" for j in range(matrix.shape[1])]

        data_df = pd.DataFrame(matrix, index=sample_names, columns=feature_names)
        true_labels = np.array(
            [cluster_assignments[name] for name in original_names],
            dtype=int,
        )

        metadata = {
            "n_samples": n_rows,
            "n_features": n_cols,
            "n_clusters": test_case["n_clusters"],
            "noise": entropy,
            "name": test_case.get("name", f"binary_{n_rows}x{n_cols}"),
            "generator": "binary",
        }

        return data_df, true_labels, matrix.astype(float), metadata

    # Default: Gaussian blobs -> binarize via median threshold
    n_samples = test_case["n_samples"]
    n_features = test_case["n_features"]
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=test_case["n_clusters"],
        cluster_std=test_case["cluster_std"],
        random_state=seed,
    )
    X_bin = (X > np.median(X, axis=0)).astype(int)
    data_df = pd.DataFrame(
        X_bin,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": test_case["n_clusters"],
        "noise": test_case["cluster_std"],
        "name": test_case.get("name", f"blobs_{n_samples}x{n_features}"),
        "generator": "blobs",
    }
    return data_df, y, X, metadata


def _run_pipeline_on_dataframe(
    data_df: pd.DataFrame, significance_level: float = 0.05, permutations: int = 50
) -> tuple[dict, pd.DataFrame]:
    """Execute the full KL-based clustering pipeline on a binary dataframe."""
    parallel_cmi = True
    Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    stats_df = calculate_hierarchy_kl_divergence(tree, data_df)
    stats_df = annotate_nodes_with_statistical_significance_tests(
        stats_df,
        data_df.shape[1],
        significance_level,
        2.0,
        True,
    )
    stats_df = annotate_child_parent_divergence(
        tree, stats_df, data_df.shape[1], significance_level
    )
    stats_df = annotate_sibling_independence_cmi(
        tree,
        stats_df,
        significance_level_alpha=significance_level,
        permutations=permutations,
        parallel=parallel_cmi,
    )

    decomposer = ClusterDecomposer(
        tree=tree,
        results_df=stats_df,
        significance_column="Are_Features_Dependent",
        alpha_local=0.1,
    )
    decomposition = decomposer.decompose_tree()
    return decomposition, stats_df


def _labels_from_decomposition(
    decomposition: dict, sample_index: list[str]
) -> list[int]:
    """Extract cluster labels for each sample from a decomposition result."""
    assignments = {sample: -1 for sample in sample_index}
    for cluster_id, info in decomposition.get("cluster_assignments", {}).items():
        for leaf in info["leaves"]:
            assignments[leaf] = cluster_id
    return [assignments[sample] for sample in sample_index]


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
    cmi_permutations = 50
    project_root = Path(__file__).resolve().parent.parent
    parallel_cmi = True
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
            # Binary feature matrix simulations
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 72,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 40,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 80,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 120,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 140,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 160,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 180,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 220,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 1200,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_balanced_low_entropy",
                "generator": "binary",
                "n_rows": 72,
                "n_cols": 120,
                "n_clusters": 4,
                "entropy_param": 0.25,
                "balanced_clusters": True,
                "seed": 314,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 36,
                "n_clusters": 4,
                "entropy_param": 0.45,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 36,
                "n_clusters": 4,
                "entropy_param": 0.12,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 300,
                "n_clusters": 4,
                "entropy_param": 0.00,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 300,
                "n_clusters": 4,
                "entropy_param": 0.01,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 300,
                "n_clusters": 4,
                "entropy_param": 0.02,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.12,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.17,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.22,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.27,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.28,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.29,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.30,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.31,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.32,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.33,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.34,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 96,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.35,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 180,
                "n_cols": 3000,
                "n_clusters": 4,
                "entropy_param": 0.35,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 360,
                "n_cols": 3000,
                "n_clusters": 8,
                "entropy_param": 0.05,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 480,
                "n_cols": 3000,
                "n_clusters": 8,
                "entropy_param": 0.10,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 480,
                "n_cols": 3000,
                "n_clusters": 12,
                "entropy_param": 0.10,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 480,
                "n_cols": 3000,
                "n_clusters": 12,
                "entropy_param": 0.15,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 480,
                "n_cols": 3000,
                "n_clusters": 12,
                "entropy_param": 0.20,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 480,
                "n_cols": 3000,
                "n_clusters": 12,
                "entropy_param": 0.25,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 480,
                "n_cols": 3000,
                "n_clusters": 8,
                "entropy_param": 0.05,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 360,
                "n_cols": 3000,
                "n_clusters": 8,
                "entropy_param": 0.10,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 360,
                "n_cols": 3000,
                "n_clusters": 12,
                "entropy_param": 0.10,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 360,
                "n_cols": 3000,
                "n_clusters": 12,
                "entropy_param": 0.15,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 360,
                "n_cols": 3000,
                "n_clusters": 12,
                "entropy_param": 0.20,
                "balanced_clusters": False,
                "seed": 2024,
            },
            {
                "name": "binary_unbalanced_high_entropy",
                "generator": "binary",
                "n_rows": 360,
                "n_cols": 3000,
                "n_clusters": 12,
                "entropy_param": 0.25,
                "balanced_clusters": False,
                "seed": 2024,
            },
        ]

    if verbose:
        print("=" * 80)
        print("CLUSTER ALGORITHM VALIDATION")
        print("=" * 80 + "\n")

    results_data = []
    failure_summaries: list[dict] = []
    failure_csv_paths: list[Path] = []

    # Run test cases
    for i, tc in enumerate(test_cases, 1):
        if verbose:
            print(f"\nRunning test case {i}/{len(test_cases)}...")

        case_name = tc.get("name", f"Case {i}")
        if verbose:
            print(f"  -> {case_name}")

        # Generate and process data
        data_t, y_t, X_original, meta = _generate_case_data(tc)

        # Build tree and calculate statistics
        Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
        tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
        stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)

        # Statistical testing - BH global and BH local (child vs parent)
        results_t = annotate_nodes_with_statistical_significance_tests(
            stats_t, meta["n_features"], significance_level, 2.0, True
        )
        results_t = annotate_child_parent_divergence(
            tree_t, results_t, meta["n_features"], significance_level
        )
        results_t = annotate_sibling_independence_cmi(
            tree_t,
            results_t,
            significance_level_alpha=significance_level,
            permutations=cmi_permutations,
            parallel=parallel_cmi,
        )

        # Use BH-corrected significance with deviation testing (same as main pipeline)
        significance_column = "Are_Features_Dependent"

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
                "True": meta["n_clusters"],
                "Found": decomp_t["num_clusters"],
                "Samples": meta["n_samples"],
                "Features": meta["n_features"],
                "Noise": meta["noise"],
                "ARI": ari,
                "NMI": nmi,
                "Purity": purity,
            }
        )
        assigned_fraction = (
            len(report_t) / float(meta["n_samples"]) if meta["n_samples"] else 0.0
        )

        failure_reasons: list[str] = []
        if decomp_t["num_clusters"] != meta["n_clusters"]:
            failure_reasons.append(
                f"cluster_count_mismatch(expected={meta['n_clusters']},found={decomp_t['num_clusters']})"
            )
        if ari < 0.5:
            failure_reasons.append(f"low_ari({ari:.3f})")
        if nmi < 0.5:
            failure_reasons.append(f"low_nmi({nmi:.3f})")
        if purity < 0.7 and decomp_t["num_clusters"] > 0:
            failure_reasons.append(f"low_purity({purity:.3f})")
        if assigned_fraction < 0.75:
            failure_reasons.append(f"low_assignment_fraction({assigned_fraction:.2f})")

        if failure_reasons:
            results_bool = results_t.get("Are_Features_Dependent")
            dependent_nodes = (
                int(results_bool.sum()) if isinstance(results_bool, pd.Series) else 0
            )
            local_bool = results_t.get("Local_Are_Features_Dependent")
            local_dependent = (
                int(local_bool.sum()) if isinstance(local_bool, pd.Series) else 0
            )
            sibling_bool = results_t.get("Sibling_BH_Dependent")
            sibling_dependent = (
                int(sibling_bool.sum()) if isinstance(sibling_bool, pd.Series) else 0
            )
            kl_global = results_t.get("kl_divergence_global", pd.Series(dtype=float))
            kl_global_clean = (
                kl_global.dropna()
                if isinstance(kl_global, pd.Series)
                else pd.Series(dtype=float)
            )
            top_global_nodes = []
            if not kl_global_clean.empty:
                top_global_nodes = [
                    f"{node}:{kl_global_clean.loc[node]:.3f}"
                    for node in kl_global_clean.sort_values(ascending=False)
                    .head(3)
                    .index
                ]
            kl_local = get_local_kl_series(results_t)
            kl_local_clean = kl_local.dropna()
            top_local_nodes = []
            if not kl_local_clean.empty:
                top_local_nodes = [
                    f"{node}:{kl_local_clean.loc[node]:.3f}"
                    for node in kl_local_clean.sort_values(ascending=False)
                    .head(3)
                    .index
                ]

            failure_dir = project_root / "cluster_validation_failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            safe_name = (
                re.sub(r"[^A-Za-z0-9_-]+", "_", case_name.lower()).strip("_")
                or f"case_{i}"
            )
            csv_path = failure_dir / f"failed_test_{i}_{safe_name}.csv"
            enriched_results = results_t.copy()
            enriched_results.insert(0, "Test_Index", i)
            enriched_results.insert(1, "Case_Name", case_name)
            enriched_results.insert(2, "Generator", meta["generator"])
            enriched_results.insert(3, "Failure_Reasons", ";".join(failure_reasons))
            enriched_results["True_Clusters"] = meta["n_clusters"]
            enriched_results["Found_Clusters"] = decomp_t["num_clusters"]
            enriched_results["ARI"] = ari
            enriched_results["NMI"] = nmi
            enriched_results["Purity"] = purity
            enriched_results["Assigned_Fraction"] = assigned_fraction
            enriched_results["Samples"] = meta["n_samples"]
            enriched_results["Features"] = meta["n_features"]
            enriched_results["Noise_Parameter"] = meta["noise"]
            enriched_results["Significance_Level"] = significance_level
            enriched_results["CMI_Permutations"] = cmi_permutations
            enriched_results["Global_Dependent_Count"] = dependent_nodes
            enriched_results["Local_Dependent_Count"] = local_dependent
            enriched_results["Sibling_Dependent_Count"] = sibling_dependent
            enriched_results["Top_Global_KL_Nodes"] = "|".join(top_global_nodes)
            enriched_results["Top_Local_KL_Nodes"] = "|".join(top_local_nodes)
            enriched_results["Total_Nodes"] = len(results_t)
            enriched_results["Failure_CSV_Path"] = str(csv_path)

            summary_info = {
                "Test_Index": i,
                "Case_Name": case_name,
                "Generator": meta["generator"],
                "Failure_Reasons": ";".join(failure_reasons),
                "True_Clusters": meta["n_clusters"],
                "Found_Clusters": decomp_t["num_clusters"],
                "ARI": ari,
                "NMI": nmi,
                "Purity": purity,
                "Assigned_Fraction": assigned_fraction,
                "Samples": meta["n_samples"],
                "Features": meta["n_features"],
                "Noise_Parameter": meta["noise"],
                "Significance_Level": significance_level,
                "CMI_Permutations": cmi_permutations,
                "Global_Dependent_Count": dependent_nodes,
                "Local_Dependent_Count": local_dependent,
                "Sibling_Dependent_Count": sibling_dependent,
                "Total_Nodes": len(results_t),
                "Top_Global_KL_Nodes": "|".join(top_global_nodes),
                "Top_Local_KL_Nodes": "|".join(top_local_nodes),
                "Total_Nodes": len(results_t),
                "Failure_CSV_Path": str(csv_path),
            }
            summary_df = pd.DataFrame([summary_info])
            summary_df.index = ["__summary__"]
            summary_df = summary_df.reindex(columns=enriched_results.columns)
            enriched_results_with_summary = pd.concat(
                [enriched_results, summary_df], axis=0
            )

            enriched_results_with_summary.to_csv(csv_path)

            failure_summary = {
                "Test": i,
                "Name": case_name,
                "Generator": meta["generator"],
                "Reasons": ";".join(failure_reasons),
                "True_Clusters": meta["n_clusters"],
                "Found_Clusters": decomp_t["num_clusters"],
                "ARI": ari,
                "NMI": nmi,
                "Purity": purity,
                "Assigned_Fraction": assigned_fraction,
                "Total_Nodes": len(results_t),
                "Global_Dependent_Nodes": dependent_nodes,
                "Local_Dependent_Nodes": local_dependent,
                "Sibling_Dependent_Nodes": sibling_dependent,
                "Top_Global_KL_Nodes": top_global_nodes,
                "Top_Local_KL_Nodes": top_local_nodes,
                "CSV_Path": csv_path,
            }
            failure_summaries.append(failure_summary)
            failure_csv_paths.append(csv_path)

    if verbose:
        print(f"Completed {len(test_cases)} test cases.       \n")

    df_results = pd.DataFrame(results_data)

    # Report and validate failure statistics
    if failure_summaries:
        required_summary_fields = [
            "Test",
            "Name",
            "Reasons",
            "True_Clusters",
            "Found_Clusters",
            "ARI",
            "NMI",
            "Purity",
            "Assigned_Fraction",
            "Total_Nodes",
            "Global_Dependent_Nodes",
            "Local_Dependent_Nodes",
            "Sibling_Dependent_Nodes",
            "CSV_Path",
        ]
        print("\nFailure Statistics Summary:")
        for summary, csv_path in zip(failure_summaries, failure_csv_paths):
            print(
                f"  Test {summary['Test']} ({summary['Name']} | {summary['Generator']}):"
            )
            print(f"    Reasons: {summary['Reasons']}")
            print(
                f"    Clusters: expected {summary['True_Clusters']}, found {summary['Found_Clusters']}; "
                f"ARI={summary['ARI']:.3f}, NMI={summary['NMI']:.3f}, Purity={summary['Purity']:.3f}"
            )
            print(
                f"    Assignment Fraction: {summary['Assigned_Fraction']:.2f}; "
                f"Nodes total={summary['Total_Nodes']}, global_sig={summary['Global_Dependent_Nodes']}, "
                f"local_sig={summary['Local_Dependent_Nodes']}, sibling_sig={summary['Sibling_Dependent_Nodes']}"
            )
            if summary["Top_Global_KL_Nodes"]:
                print(
                    f"    Top Global KL Nodes: {', '.join(summary['Top_Global_KL_Nodes'])}"
                )
            if summary["Top_Local_KL_Nodes"]:
                print(
                    f"    Top Local KL Nodes: {', '.join(summary['Top_Local_KL_Nodes'])}"
                )
            csv_exists = Path(csv_path).exists()
            missing_fields = [f for f in required_summary_fields if f not in summary]
            validation_pass = csv_exists and not missing_fields
            status_msg = "PASS" if validation_pass else "CHECK FAILED"
            print(
                f"    Validation: {status_msg} (csv_exists={csv_exists}, missing_fields={missing_fields})"
            )
            print(f"    CSV: {csv_path}")
    else:
        print(
            "\nAll test cases met configured thresholds; no failure statistics generated."
        )

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
            data_t, y_t, X_original, meta = _generate_case_data(tc)

            # Get KL clustering results from our results dataframe
            kl_result = df_results[df_results["Test"] == i]
            if not kl_result.empty:
                # Recompute the KL clustering to get the tree and decomposition
                Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
                tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
                stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)
                results_t = annotate_nodes_with_statistical_significance_tests(
                    stats_t, meta["n_features"], significance_level, 2.0, True
                )
                results_t = annotate_child_parent_divergence(
                    tree_t, results_t, meta["n_features"], significance_level
                )
                results_t = annotate_sibling_independence_cmi(
                    tree_t,
                    results_t,
                    significance_level_alpha=significance_level,
                    permutations=cmi_permutations,
                    parallel=parallel_cmi,
                )

                decomposer_t = ClusterDecomposer(
                    tree=tree_t,
                    results_df=results_t,
                    significance_column="Are_Features_Dependent",
                    alpha_local=0.1,
                )

                decomp_t = decomposer_t.decompose_tree()

                # Create tree visualization
                noise_label = "σ" if meta["generator"] == "blobs" else "entropy"
                noise_value = meta["noise"]
                if isinstance(noise_value, (int, float, np.floating, np.integer)):
                    noise_value_str = f"{float(noise_value):.2f}"
                else:
                    noise_value_str = str(noise_value)
                test_case_name = (
                    f"Test Case {i}: {meta['n_clusters']} Clusters "
                    f"({noise_label}={noise_value_str})"
                )
                if meta.get("name"):
                    test_case_name += f" [{meta['name']}]"
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

                tree_filename = f"tree_test_{i}_{meta['n_clusters']}_clusters_{current_timestamp}.png"
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

                    summary_filename = f"summary_test_{i}_{meta['n_clusters']}_clusters_{current_timestamp}.png"
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

            data_t, y_t, X_original, meta = _generate_case_data(tc)

            # Get KL clustering results from our results dataframe
            kl_result = df_results[df_results["Test"] == i]
            if not kl_result.empty:
                # Recompute the KL clustering to get labels
                Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
                tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
                stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)
                results_t = annotate_nodes_with_statistical_significance_tests(
                    stats_t, meta["n_features"], significance_level, 2.0, True
                )
                results_t = annotate_child_parent_divergence(
                    tree_t, results_t, meta["n_features"], significance_level
                )
                results_t = annotate_sibling_independence_cmi(
                    tree_t,
                    results_t,
                    significance_level_alpha=significance_level,
                    permutations=cmi_permutations,
                    parallel=parallel_cmi,
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
                    kl_labels = np.array(
                        [
                            report_t.loc[sample, "cluster_id"]
                            if sample in report_t.index
                            else -1
                            for sample in data_t.index
                        ]
                    )
                else:
                    kl_labels = np.full(meta["n_samples"], -1)

                # Create UMAP comparison visualization
                umap_fig = _create_umap_comparison_plot(
                    X_original, y_t, kl_labels, i, meta["n_clusters"]
                )

                umap_filename = f"umap_test_{i}_{meta['n_clusters']}_clusters_{current_timestamp}.png"
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
    custom_cases = [case.copy() for case in SMALL_TEST_CASES]
    df_results, _ = validate_cluster_algorithm(
        test_cases=custom_cases,
        verbose=False,
        plot_umap=False,
        plot_trees=False,
    )

    assert len(df_results) == len(SMALL_TEST_CASES)
    df_results["case_name"] = [case["name"] for case in SMALL_TEST_CASES]

    clear_case = df_results[df_results["case_name"] == "clear"].iloc[0]
    assert clear_case["Found"] == clear_case["True"]
    assert clear_case["ARI"] > 0.85
    assert clear_case["NMI"] > 0.85
    assert clear_case["Purity"] > 0.9

    moderate_case = df_results[df_results["case_name"] == "moderate"].iloc[0]
    assert moderate_case["ARI"] > 0.6
    assert moderate_case["NMI"] > 0.65
    assert moderate_case["Purity"] > 0.7

    noisy_case = df_results[df_results["case_name"] == "noisy"].iloc[0]
    assert noisy_case["ARI"] >= 0
    assert 0 <= noisy_case["Purity"] <= 1
    assert noisy_case["Found"] >= 0


def test_validate_cluster_algorithm_expected_columns():
    """Ensure the validator returns the expected metrics."""
    df_results, fig = validate_cluster_algorithm(
        test_cases=[SMALL_TEST_CASES[0].copy()],
        verbose=False,
        plot_umap=False,
        plot_trees=False,
    )

    expected_columns = {
        "Test",
        "True",
        "Found",
        "Samples",
        "Features",
        "Noise",
        "ARI",
        "NMI",
        "Purity",
    }
    assert expected_columns.issubset(df_results.columns)
    assert fig is None
    assert (df_results["ARI"].between(0, 1)).all()


def test_validate_cluster_algorithm_handles_empty_cases():
    """Validator should handle an empty case list without errors."""
    df_results, fig = validate_cluster_algorithm(
        test_cases=[],
        verbose=False,
        plot_umap=False,
        plot_trees=False,
    )

    assert df_results.empty
    assert fig is None


def test_complex_random_feature_matrix_balanced_clusters():
    """Synthetic binary data with low entropy should recover most clusters."""
    data_dict, true_clusters = generate_random_feature_matrix(
        n_rows=72,
        n_cols=40,
        entropy_param=0.25,
        n_clusters=4,
        random_seed=314,
        balanced_clusters=True,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)

    decomposition, _ = _run_pipeline_on_dataframe(data_df, significance_level=0.05)
    predicted = _labels_from_decomposition(decomposition, data_df.index.tolist())
    true_labels = [true_clusters[name] for name in data_df.index]

    assigned_mask = np.array(predicted) != -1
    assigned_fraction = float(np.mean(assigned_mask))
    assert assigned_fraction > 0.85, "Too many samples left unassigned"

    ari = adjusted_rand_score(
        np.array(true_labels)[assigned_mask], np.array(predicted)[assigned_mask]
    )

    assert decomposition["num_clusters"] >= 3
    assert ari > 0.7


def test_complex_random_feature_matrix_unbalanced_clusters():
    """Higher entropy and unbalanced clusters should still yield informative groupings."""
    data_dict, true_clusters = generate_random_feature_matrix(
        n_rows=96,
        n_cols=36,
        entropy_param=0.45,
        n_clusters=4,
        random_seed=2024,
        balanced_clusters=False,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)

    decomposition, _ = _run_pipeline_on_dataframe(
        data_df, significance_level=0.05, permutations=40
    )
    predicted = _labels_from_decomposition(decomposition, data_df.index.tolist())
    true_labels = [true_clusters[name] for name in data_df.index]

    assigned_mask = np.array(predicted) != -1
    assigned_fraction = float(np.mean(assigned_mask))
    assert assigned_fraction > 0.6, "Decomposition discarded too many samples"

    if assigned_mask.any():
        ari = adjusted_rand_score(
            np.array(true_labels)[assigned_mask], np.array(predicted)[assigned_mask]
        )
        assert ari > 0.45

    assigned_clusters = {label for label in predicted if label != -1}
    assert len(assigned_clusters) >= 2, "Expected multiple clusters to be detected"
