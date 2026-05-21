"""Smoke test for benchmark method runners."""

from benchmarks.shared.cases import SMALL_TEST_CASES
from benchmarks.shared.pipeline import benchmark_cluster_algorithm


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
    assert set(df_results["method"]) == {"leiden", "dbscan"}

    dbscan_row = df_results[df_results["method"] == "dbscan"].iloc[0]
    assert dbscan_row["status"] == "ok"
    assert dbscan_row["labels_length"] == dbscan_row["samples"]

    ok_rows = df_results[df_results["status"] == "ok"]
    assert (ok_rows["labels_length"] == ok_rows["samples"]).all()


def test_benchmark_louvain_and_adaptive_diffusion_methods_smoke():
    """Run the previously nonfunctional methods through one benchmark case."""
    case = SMALL_TEST_CASES[0].copy()
    df_results, _ = benchmark_cluster_algorithm(
        test_cases=[case],
        verbose=False,
        plot_umap=False,
        methods=["louvain", "kl_diffusion_adaptive"],
    )

    assert len(df_results) == 2
    assert set(df_results["method"]) == {"louvain", "kl_diffusion_adaptive"}
    assert set(df_results["status"]) == {"ok"}
    assert (df_results["labels_length"] == df_results["samples"]).all()
