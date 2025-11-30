"""
Utility functions for cluster validation testing.

This module contains helper functions used across validation tests:
- Data generation for test cases
- Pipeline execution
- Label extraction from decomposition
- Visualization functions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis import compute_node_divergences
from kl_clustering_analysis.hierarchy_analysis.statistics import (
    annotate_root_node_significance,
    annotate_child_parent_divergence,
    annotate_sibling_independence_cmi,
)
from kl_clustering_analysis.plot.validation_visualizations import (
    create_validation_plot,
    create_umap_plots_from_results,
    create_manifold_plots_from_results,
)
from simulation.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis import config


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
    data_df: pd.DataFrame,
    significance_level: float = config.SIGNIFICANCE_ALPHA,
    n_permutations: int = 50,
) -> tuple[dict, pd.DataFrame]:
    """Execute the full KL-based clustering pipeline on a binary dataframe."""
    parallel_cmi = True
    Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    stats_df = compute_node_divergences(tree, data_df)
    stats_df = annotate_child_parent_divergence(
        tree, stats_df, data_df.shape[1], significance_level
    )
    stats_df = annotate_sibling_independence_cmi(
        tree,
        stats_df,
        significance_level_alpha=significance_level,
        n_permutations=n_permutations,
        parallel=parallel_cmi,
    )

    decomposition = tree.decompose(
        results_df=stats_df,
        alpha_local=config.ALPHA_LOCAL,
    )
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
    significance_level=config.SIGNIFICANCE_ALPHA,
    verbose=True,
    plot_umap=False,
    plot_manifold=False,
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
    plot_manifold : bool, default=False
        If True, saves UMAP-vs-Isomap manifold diagnostics for each test case

    Returns
    -------
    df_results : pd.DataFrame
        Results dataframe with columns: Test, True, Found, Samples, Features,
        Noise, ARI, NMI, Purity
    fig : matplotlib.figure.Figure or None
        Validation plot figure (if verbose=True), otherwise None
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Import test cases here to avoid circular dependencies
    try:
        from .test_cases_config import get_default_test_cases
    except ImportError:
        from test_cases_config import get_default_test_cases  # type: ignore

    # Default test cases
    cmi_permutations = 50
    project_root = Path(__file__).resolve().parents[1]
    plots_root = project_root / "cluster_tree_plots"
    parallel_cmi = True
    if test_cases is None:
        test_cases = get_default_test_cases()

    if verbose:
        print("=" * 80)
        print("CLUSTER ALGORITHM VALIDATION")
        print("=" * 80 + "\n")
        print(f"Evaluating {len(test_cases)} test cases.\n")

    results_data = []

    # Store computed results to avoid recalculation
    computed_results = []

    # Run test cases
    total_runs = len(test_cases)
    run_counter = 0

    for i, tc in enumerate(test_cases, 1):
        case_name = tc.get("name", f"Case {i}")
        if verbose:
            print(f"\nRunning test case {i}/{len(test_cases)}...")
            print(f"  -> {case_name}")

        # Generate and process data
        data_t, y_t, X_original, meta = _generate_case_data(tc)

        # Build tree and calculate statistics once per test case
        Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
        tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
        stats_t = compute_node_divergences(tree_t, data_t)

        # Skip root-level significance for speed; use raw stats for downstream gates
        base_results = stats_t.copy()
        run_counter += 1
        if verbose:
            print(f"    Run {run_counter}/{total_runs}")

        base_seed = tc.get("seed", 0) or 0
        mode_seed = int(base_seed) + 1

        # Local child-parent divergence
        base_results_mode = base_results.copy()
        results_t = annotate_child_parent_divergence(
            tree_t,
            base_results_mode,
            meta["n_features"],
            significance_level,
        )
        results_t = annotate_sibling_independence_cmi(
            tree_t,
            results_t,
            significance_level_alpha=significance_level,
            n_permutations=cmi_permutations,
            parallel=parallel_cmi,
            random_state=mode_seed + 7919,
        )

        decomp_t = tree_t.decompose(
            results_df=results_t,
            alpha_local=config.ALPHA_LOCAL,
        )

        if verbose:
            print(
                f"      Decision mode: CMI-based statistical testing "
                f"(α={significance_level})"
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
                            "cluster_size": info["size"],
                        }
                    )
            report_t = pd.DataFrame(rows).set_index("sample_id")
        else:
            report_t = pd.DataFrame(columns=["cluster_id", "cluster_size"]).set_index(
                "sample_id"
            )

        # Store computed results for later use in plotting
        computed_results.append(
            {
                "test_case_num": i,
                "tree": tree_t,
                "decomposition": decomp_t,
                "stats": results_t,
                "data": data_t,
                "meta": meta,
                "X_original": X_original,
                "y_true": y_t,
                "kl_labels": _labels_from_decomposition(
                    decomp_t, data_t.index.tolist()
                ),
            }
        )
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
                "Case_Name": case_name,
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

    if verbose:
        print(
            f"Completed {total_runs} validation runs across {len(test_cases)} test cases.\n"
        )

    df_results = pd.DataFrame(results_data)

    # Create visualization if verbose
    fig = None
    if verbose:
        fig = create_validation_plot(df_results)
        fig.savefig("validation_results.png", dpi=150, bbox_inches="tight")
        print("Validation plot saved to 'validation_results.png'")
        # plt.show()  # Skip interactive display in terminal

        print("\nDetailed Results:")
        print(
            df_results[
                [
                    "Test",
                    "Case_Name",
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

    # Create UMAP comparison visualizations if requested
    if plot_umap:
        print("\nGenerating UMAP comparison visualizations...")
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        umap_plots_dir = plots_root

        # Use the new plotting utility with pre-computed results
        create_umap_plots_from_results(
            computed_results, umap_plots_dir, current_timestamp, verbose
        )

        print(
            f"UMAP comparison plots saved as '{umap_plots_dir}/umap_test_*_clusters_*.png'"
        )

    if plot_manifold:
        print("\nGenerating manifold diagnostics (UMAP vs Isomap)...")
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        manifold_dir = plots_root
        create_manifold_plots_from_results(
            computed_results,
            manifold_dir,
            current_timestamp,
            verbose,
        )
        print(
            f"Manifold plots saved as '{manifold_dir}/manifold_test_*_clusters_*.png'"
        )

    return df_results, fig
