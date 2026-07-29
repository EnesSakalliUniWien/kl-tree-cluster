"""Smoke test for benchmark method runners."""

import pytest
from benchmarks.shared.cases import SMALL_TEST_CASES, get_default_test_cases
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
        methods=["louvain", "tbs_diffusion_adaptive"],
    )

    assert len(df_results) == 2
    assert set(df_results["method"]) == {"louvain", "tbs_diffusion_adaptive"}
    assert set(df_results["status"]) == {"ok"}
    assert (df_results["labels_length"] == df_results["samples"]).all()


def test_benchmark_graphtools_diffusion_method_smoke():
    """Run the optional graphtools diffusion backend when it is installed."""
    pytest.importorskip("graphtools")
    case = SMALL_TEST_CASES[0].copy()
    df_results, _ = benchmark_cluster_algorithm(
        test_cases=[case],
        verbose=False,
        plot_umap=False,
        methods=[
            "tbs_diffusion_graphtools",
            "tbs_diffusion_graphtools_adaptive_nnls",
        ],
    )

    assert set(df_results["method"]) == {
        "tbs_diffusion_graphtools",
        "tbs_diffusion_graphtools_adaptive_nnls",
    }
    assert len(df_results) == 9
    assert df_results["run_id"].nunique() == 9
    assert (
        df_results[df_results["method"] == "tbs_diffusion_graphtools_adaptive_nnls"][
            "benchmark_grid"
        ]
        == "graphtools_adaptive_k_tree_strategy"
    ).all()
    assert set(df_results["status"]) == {"ok"}
    assert (df_results["labels_length"] == df_results["samples"]).all()


def test_hamming_diffusion_rejects_continuous_benchmark_input():
    """The Hamming diffusion method must not silently score continuous cases."""
    case = next(
        case.copy()
        for case in get_default_test_cases()
        if case["name"] == "gauss_clear_medium_continuous"
    )
    with pytest.raises(ValueError, match="requires binary or one-hot"):
        benchmark_cluster_algorithm(
            test_cases=[case],
            verbose=False,
            plot_umap=False,
            methods=["tbs_diffusion"],
        )
