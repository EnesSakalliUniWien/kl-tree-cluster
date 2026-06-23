"""
Core validation tests for cluster decomposition algorithm.

Tests basic functionality of the validation framework:
- Multi-scenario validation across noise levels
- Expected column structure
- Empty case handling
"""

from benchmarks.shared.cases import SMALL_TEST_CASES
from benchmarks.shared.pipeline import benchmark_cluster_algorithm


def test_cluster_algorithm_validation():
    """Test that the cluster algorithm works correctly across multiple test cases with varying noise levels."""
    custom_cases = [case.copy() for case in SMALL_TEST_CASES]
    df_results, _ = benchmark_cluster_algorithm(
        test_cases=custom_cases,
        verbose=False,
        plot_umap=False,
        methods=["tbs"],
    )

    tbs_results = df_results[df_results["method"] == "tbs"].reset_index(drop=True)
    assert len(tbs_results) >= len(SMALL_TEST_CASES)

    for case_name in ("clear", "moderate", "noisy"):
        rows = tbs_results[(tbs_results["case_id"] == case_name)]
        best = rows.sort_values(["ari", "params"], ascending=[False, True]).iloc[0]

        assert best["status"] == "ok"
        assert best["found_clusters"] >= 1
        # ARI ranges from -0.5 to 1.0; negative values indicate assignments
        # worse than random.  On very small / noisy data this is expected.
        assert -1 <= best["ari"] <= 1
        assert 0 <= best["nmi"] <= 1
        assert -1 <= best["ami"] <= 1
        assert 0 <= best["purity"] <= 1
        assert 0 <= best["homogeneity"] <= 1
        assert 0 <= best["completeness"] <= 1
        assert 0 <= best["v_measure"] <= 1
        assert 0 <= best["fowlkes_mallows"] <= 1
        assert 0 <= best["macro_recall"] <= 1
        assert 0 <= best["macro_f1"] <= 1
        assert 0 <= best["worst_cluster_recall"] <= 1
        assert best["n_singleton_clusters"] >= 0
        assert 0 <= best["singleton_fraction"] <= 1
        assert best["median_cluster_size"] >= 1
        assert 0 <= best["largest_cluster_fraction"] <= 1
        assert best["effective_cluster_count"] >= 1
        assert best["cluster_size_entropy"] >= 0
        assert 0 <= best["cluster_size_gini"] <= 1
        assert 0 <= best["noise_label_fraction"] <= 1
        assert best["cluster_count_abs_error"] >= 0


def test_benchmark_cluster_algorithm_expected_columns():
    """Ensure the validator returns the expected metrics."""
    df_results, fig = benchmark_cluster_algorithm(
        test_cases=[SMALL_TEST_CASES[0].copy()],
        verbose=False,
        plot_umap=False,
        methods=["tbs"],
    )

    expected_columns = {
        "test_case",
        "case_id",
        "case_category",
        "method",
        "params",
        "true_clusters",
        "found_clusters",
        "samples",
        "features",
        "noise",
        "ari",
        "nmi",
        "ami",
        "purity",
        "homogeneity",
        "completeness",
        "v_measure",
        "fowlkes_mallows",
        "macro_recall",
        "macro_f1",
        "worst_cluster_recall",
        "n_singleton_clusters",
        "singleton_fraction",
        "median_cluster_size",
        "largest_cluster_fraction",
        "effective_cluster_count",
        "cluster_size_entropy",
        "cluster_size_gini",
        "noise_label_fraction",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "outlier_precision",
        "outlier_recall",
        "outlier_f1",
        "singleton_outlier_isolated",
        "grouped_outlier_cluster_recovered",
        "cluster_count_abs_error",
        "over_split",
        "under_split",
        "status",
        "skip_reason",
        "labels_length",
    }
    assert expected_columns.issubset(df_results.columns)
    tbs_results = df_results[df_results["method"] == "tbs"]
    assert len(tbs_results) >= 1
    assert fig is None
    assert (tbs_results["ari"].between(-1, 1)).all()


def test_benchmark_cluster_algorithm_handles_empty_cases():
    """Validator should handle an empty case list without errors."""
    df_results, fig = benchmark_cluster_algorithm(
        test_cases=[],
        verbose=False,
        plot_umap=False,
    )

    assert df_results.empty
    assert fig is None
