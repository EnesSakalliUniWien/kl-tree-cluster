"""
Core validation tests for cluster decomposition algorithm.

Tests basic functionality of the validation framework:
- Multi-scenario validation across noise levels
- Expected column structure
- Empty case handling
"""

from tests.validation_utils import validate_cluster_algorithm

try:
    from .test_cases_config import SMALL_TEST_CASES
except ImportError:
    from test_cases_config import SMALL_TEST_CASES  # type: ignore


def test_cluster_algorithm_validation():
    """Test that the cluster algorithm works correctly across multiple test cases with varying noise levels."""
    custom_cases = [case.copy() for case in SMALL_TEST_CASES]
    df_results, _ = validate_cluster_algorithm(
        test_cases=custom_cases,
        verbose=False,
        plot_umap=False,
    )

    assert len(df_results) == len(SMALL_TEST_CASES)

    clear_case = df_results[(df_results["Case_Name"] == "clear")].iloc[0]
    assert clear_case["Found"] == clear_case["True"]
    assert clear_case["ARI"] > 0.85
    assert clear_case["NMI"] > 0.85
    assert clear_case["Purity"] > 0.9

    moderate_case = df_results[(df_results["Case_Name"] == "moderate")].iloc[0]
    assert moderate_case["ARI"] > 0.6
    assert moderate_case["NMI"] > 0.65
    assert moderate_case["Purity"] > 0.7

    noisy_case = df_results[(df_results["Case_Name"] == "noisy")].iloc[0]
    assert noisy_case["ARI"] >= 0
    assert 0 <= noisy_case["Purity"] <= 1
    assert noisy_case["Found"] >= 0


def test_validate_cluster_algorithm_expected_columns():
    """Ensure the validator returns the expected metrics."""
    df_results, fig = validate_cluster_algorithm(
        test_cases=[SMALL_TEST_CASES[0].copy()],
        verbose=False,
        plot_umap=False,
    )

    expected_columns = {
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
    }
    assert expected_columns.issubset(df_results.columns)
    assert len(df_results) == 1
    assert fig is None
    assert (df_results["ARI"].between(0, 1)).all()


def test_validate_cluster_algorithm_handles_empty_cases():
    """Validator should handle an empty case list without errors."""
    df_results, fig = validate_cluster_algorithm(
        test_cases=[],
        verbose=False,
        plot_umap=False,
    )

    assert df_results.empty
    assert fig is None
