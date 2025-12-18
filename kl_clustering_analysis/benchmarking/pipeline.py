"""
Benchmarking/validation pipeline for the clustering algorithm.

Runs configurable test cases, computes metrics, and optionally produces plots.
The implementation was previously housed in ``tests/validation_utils`` and is
kept here so it can be reused outside the test suite.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
)

try:  # Prefer test config when available (repo usage)
    from tests.test_cases_config import get_default_test_cases  # type: ignore
except ImportError:
    try:  # Fallback when run from within tests package directly
        from test_cases_config import get_default_test_cases  # type: ignore
    except ImportError:  # As a last resort, use an empty default

        def get_default_test_cases():
            return []


from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config
from kl_clustering_analysis.benchmarking.plots import (
    create_validation_plot,
    create_umap_plots_from_results,
    create_manifold_plots_from_results,
    create_tree_plots_from_results,
)

from simulation.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)


# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _log_validation_start(n_cases: int) -> None:
    """Log the start of the validation process."""
    logger.info("=" * 80)
    logger.info("CLUSTER ALGORITHM VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Evaluating {n_cases} test cases.")


def _log_test_case_start(index: int, total: int, name: str) -> None:
    """Log the start of a specific test case."""
    logger.info(f"Running test case {index}/{total}: {name}")


def _log_validation_completion(total_runs: int, n_cases: int) -> None:
    """Log the completion of the validation process."""
    logger.info(f"Completed {total_runs} validation runs across {n_cases} test cases.")


def _log_detailed_results(df_results: pd.DataFrame) -> None:
    """Log the detailed results table."""
    logger.info("Detailed Results:")
    # Convert dataframe to string and log each line to avoid truncation issues in some loggers
    results_str = df_results[
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
    for line in results_str.split("\n"):
        logger.info(line)


def _generate_validation_plots(
    df_results: pd.DataFrame,
    computed_results: list,
    plots_root: Path,
    verbose: bool,
    plot_umap: bool,
    plot_manifold: bool,
):
    """Generate and save validation plots."""
    if df_results.empty:
        return None

    fig = create_validation_plot(df_results)

    if verbose:
        fig.savefig("validation_results.png", dpi=150, bbox_inches="tight")
        logger.info("Validation plot saved to 'validation_results.png'")

        _log_detailed_results(df_results)

        logger.info("Generating tree plots...")
        plots_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        create_tree_plots_from_results(
            computed_results, plots_root, timestamp, verbose=False
        )
        logger.info(f"Tree plots saved to {plots_root}")

    if plot_umap:
        logger.info("Generating UMAP comparison visualizations...")
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        create_umap_plots_from_results(
            computed_results, plots_root, current_timestamp, verbose=verbose
        )

    if plot_manifold:
        logger.info("Generating manifold diagnostics (UMAP vs Isomap)...")
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        create_manifold_plots_from_results(
            computed_results,
            plots_root,
            current_timestamp,
            verbose=verbose,
        )

    return fig


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
        feature_sparsity = test_case.get("feature_sparsity", None)

        data_dict, cluster_assignments = generate_random_feature_matrix(
            n_rows=n_rows,
            n_cols=n_cols,
            entropy_param=entropy,
            n_clusters=test_case["n_clusters"],
            random_seed=seed,
            balanced_clusters=balanced,
            feature_sparsity=feature_sparsity,
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


def _labels_from_decomposition(
    decomposition: dict, sample_index: list[str]
) -> list[int]:
    """Extract cluster labels for each sample from a decomposition result."""
    assignments = {sample: -1 for sample in sample_index}
    for cluster_id, info in decomposition.get("cluster_assignments", {}).items():
        for leaf in info["leaves"]:
            assignments[leaf] = cluster_id
    return [assignments[sample] for sample in sample_index]


def _create_report_dataframe(cluster_assignments: dict) -> pd.DataFrame:
    """Create a report dataframe from cluster assignments."""
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
        return pd.DataFrame(rows).set_index("sample_id")

    return pd.DataFrame(columns=["cluster_id", "cluster_size"]).set_index("sample_id")


def _calculate_ari_nmi_purity_metrics(
    num_clusters: int,
    report_df: pd.DataFrame,
    sample_names: pd.Index,
    true_labels: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate clustering metrics (ARI, NMI, Purity)."""
    if num_clusters > 0 and not report_df.empty:
        # Create a correct mapping from sample name (report_df index) to true cluster label
        true_label_map = {name: label for name, label in zip(sample_names, true_labels)}

        # Work on a copy to avoid side effects
        df_metrics = report_df.copy()
        df_metrics["true_cluster"] = df_metrics.index.map(true_label_map)

        ari = adjusted_rand_score(
            df_metrics["true_cluster"].values, df_metrics["cluster_id"].values
        )

        nmi = normalized_mutual_info_score(
            df_metrics["true_cluster"].values, df_metrics["cluster_id"].values
        )

        purity = homogeneity_score(
            df_metrics["true_cluster"].values, df_metrics["cluster_id"].values
        )
        return ari, nmi, purity

    return 0.0, 0.0, 0.0


def benchmark_cluster_algorithm(
    test_cases=None,
    significance_level=config.SIBLING_ALPHA,
    verbose=True,
    plot_umap=False,
    plot_manifold=False,
):
    """
    Benchmark the cluster decomposition algorithm across multiple test cases.

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
    significance_level : float, default=config.SIBLING_ALPHA
        Significance level used for sibling-independence gating in decomposition
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
    fig : None
        Plotting has been removed; always returns None.
    """

    # Default test cases
    project_root = Path(__file__).resolve().parents[2]
    plots_root = project_root / "cluster_tree_plots"
    if test_cases is None:
        test_cases = get_default_test_cases()

    if verbose:
        _log_validation_start(len(test_cases))

    results_data = []

    # Store computed results to avoid recalculation
    computed_results = []

    # Run test cases
    total_runs = len(test_cases)

    for i, tc in enumerate(test_cases, 1):
        case_name = tc.get("name", f"Case {i}")
        if verbose:
            _log_test_case_start(i, total_runs, case_name)

        # Generate and process data
        data_t, y_t, X_original, meta = _generate_case_data(tc)

        # Build tree and calculate statistics once per test case
        Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
        tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())

        # Use the direct decompose method from PosetTree which handles stats calculation internally
        # if results_df is not provided but leaf_data is.
        decomp_t = tree_t.decompose(
            leaf_data=data_t,
            alpha_local=config.ALPHA_LOCAL,
            sibling_alpha=significance_level,
            # Pass kwargs for internal annotation functions if needed,
            # though decompose() currently hardcodes some config defaults internally.
            # We might need to ensure decompose() uses our significance_level for annotations too.
        )

        # Track the number of clusters actually found so plotting uses the right color mapping.
        meta["found_clusters"] = decomp_t["num_clusters"]

        # Extract the results_df that was computed internally
        results_t = tree_t.stats_df

        # Create report dataframe from cluster assignments
        report_t = _create_report_dataframe(decomp_t.get("cluster_assignments", {}))

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

        ari, nmi, purity = _calculate_ari_nmi_purity_metrics(
            decomp_t["num_clusters"], report_t, data_t.index, y_t
        )

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
        _log_validation_completion(total_runs, len(test_cases))

    df_results = pd.DataFrame(results_data)

    if not verbose and not plot_umap and not plot_manifold:
        return df_results, None

    fig = _generate_validation_plots(
        df_results, computed_results, plots_root, verbose, plot_umap, plot_manifold
    )
    if not verbose:
        return df_results, None
    return df_results, fig


# Backward compatibility: keep old name while promoting the new one.
validate_cluster_algorithm = benchmark_cluster_algorithm

__all__ = [
    "benchmark_cluster_algorithm",
    "validate_cluster_algorithm",
    "_labels_from_decomposition",
]
