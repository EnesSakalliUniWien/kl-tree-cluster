"""
Smoke test for benchmark method adapters.
"""

from kl_clustering_analysis.benchmarking import benchmark_cluster_algorithm

try:
    from .test_cases_config import SMALL_TEST_CASES
except ImportError:
    from test_cases_config import SMALL_TEST_CASES  # type: ignore


def test_benchmark_graph_and_density_methods_smoke():
    """Run one graph method and one density method on a single test case."""
    case = SMALL_TEST_CASES[0].copy()
    df_results, _ = benchmark_cluster_algorithm(
        test_cases=[case],
        verbose=False,
        plot_umap=False,
        methods=["leiden", "dbscan"],
    )

    assert len(df_results) == 2
    assert set(df_results["Method"]) == {"Leiden", "DBSCAN"}

    dbscan_row = df_results[df_results["Method"] == "DBSCAN"].iloc[0]
    assert dbscan_row["Status"] == "ok"
    assert dbscan_row["Labels_Length"] == dbscan_row["Samples"]

    ok_rows = df_results[df_results["Status"] == "ok"]
    assert (ok_rows["Labels_Length"] == ok_rows["Samples"]).all()
