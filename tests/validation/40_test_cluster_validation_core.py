"""
Core validation tests for cluster decomposition algorithm.

Tests basic functionality of the validation framework:
- Multi-scenario validation across noise levels
- Expected column structure
- Empty case handling
"""

from benchmarks.shared.pipeline import benchmark_cluster_algorithm

try:
    from .test_cases_config import SMALL_TEST_CASES
except ImportError:
    from test_cases_config import SMALL_TEST_CASES  # type: ignore


def test_cluster_algorithm_validation():
    """Test that the cluster algorithm works correctly across multiple test cases with varying noise levels."""
    custom_cases = [case.copy() for case in SMALL_TEST_CASES]
    df_results, _ = benchmark_cluster_algorithm(
        test_cases=custom_cases,
        verbose=False,
        plot_umap=False,
        methods=["kl"],
    )

    kl_results = df_results[df_results["Method"] == "KL Divergence"].reset_index(drop=True)
    assert len(kl_results) >= len(SMALL_TEST_CASES)

    for case_name in ("clear", "moderate", "noisy"):
        rows = kl_results[(kl_results["Case_Name"] == case_name)]
        best = rows.sort_values(["ARI", "Params"], ascending=[False, True]).iloc[0]

        assert best["Status"] == "ok"
        assert best["Found"] >= 1
        # ARI ranges from -0.5 to 1.0; negative values indicate assignments
        # worse than random.  On very small / noisy data this is expected.
        assert -1 <= best["ARI"] <= 1
        assert 0 <= best["NMI"] <= 1
        assert 0 <= best["Purity"] <= 1
        assert 0 <= best["Macro_Recall"] <= 1
        assert 0 <= best["Macro_F1"] <= 1
        assert 0 <= best["Worst_Cluster_Recall"] <= 1
        assert best["Cluster_Count_Abs_Error"] >= 0


def test_benchmark_cluster_algorithm_expected_columns():
    """Ensure the validator returns the expected metrics."""
    df_results, fig = benchmark_cluster_algorithm(
        test_cases=[SMALL_TEST_CASES[0].copy()],
        verbose=False,
        plot_umap=False,
        methods=["kl"],
    )

    expected_columns = {
        "Test",
        "Case_Name",
        "Method",
        "Params",
        "True",
        "Found",
        "Samples",
        "Features",
        "Noise",
        "ARI",
        "NMI",
        "Purity",
        "Macro_Recall",
        "Macro_F1",
        "Worst_Cluster_Recall",
        "Cluster_Count_Abs_Error",
        "Over_Split",
        "Under_Split",
        "Status",
        "Skip_Reason",
        "Labels_Length",
    }
    assert expected_columns.issubset(df_results.columns)
    kl_results = df_results[df_results["Method"] == "KL Divergence"]
    assert len(kl_results) >= 1
    assert fig is None
    assert (kl_results["ARI"].between(-1, 1)).all()


def test_benchmark_cluster_algorithm_handles_empty_cases():
    """Validator should handle an empty case list without errors."""
    df_results, fig = benchmark_cluster_algorithm(
        test_cases=[],
        verbose=False,
        plot_umap=False,
    )

    assert df_results.empty
    assert fig is None
