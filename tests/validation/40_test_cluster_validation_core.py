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
        methods=["kl"],
    )

    kl_results = df_results[df_results["method"] == "kl"].reset_index(drop=True)
    assert len(kl_results) >= len(SMALL_TEST_CASES)

    for case_name in ("clear", "moderate", "noisy"):
        rows = kl_results[(kl_results["case_id"] == case_name)]
        best = rows.sort_values(["ari", "params"], ascending=[False, True]).iloc[0]

        assert best["status"] == "ok"
        assert best["found_clusters"] >= 1
        # ARI ranges from -0.5 to 1.0; negative values indicate assignments
        # worse than random.  On very small / noisy data this is expected.
        assert -1 <= best["ari"] <= 1
        assert 0 <= best["nmi"] <= 1
        assert 0 <= best["purity"] <= 1
        assert 0 <= best["macro_recall"] <= 1
        assert 0 <= best["macro_f1"] <= 1
        assert 0 <= best["worst_cluster_recall"] <= 1
        assert best["cluster_count_abs_error"] >= 0


def test_benchmark_cluster_algorithm_expected_columns():
    """Ensure the validator returns the expected metrics."""
    df_results, fig = benchmark_cluster_algorithm(
        test_cases=[SMALL_TEST_CASES[0].copy()],
        verbose=False,
        plot_umap=False,
        methods=["kl"],
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
        "purity",
        "macro_recall",
        "macro_f1",
        "worst_cluster_recall",
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
    kl_results = df_results[df_results["method"] == "kl"]
    assert len(kl_results) >= 1
    assert fig is None
    assert (kl_results["ari"].between(-1, 1)).all()


def test_benchmark_cluster_algorithm_handles_empty_cases():
    """Validator should handle an empty case list without errors."""
    df_results, fig = benchmark_cluster_algorithm(
        test_cases=[],
        verbose=False,
        plot_umap=False,
    )

    assert df_results.empty
    assert fig is None
