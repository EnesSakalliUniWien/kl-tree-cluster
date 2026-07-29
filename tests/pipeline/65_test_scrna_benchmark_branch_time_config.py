from __future__ import annotations

import importlib
import os

import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kl_te_cluster_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/kl_te_cluster_numba")


def test_scrna_benchmark_branch_time_rows_use_nnls_branch_lengths() -> None:
    pytest.importorskip("anndata")
    pytest.importorskip("scanpy")

    pancreas = importlib.import_module("applications.scrna.pancreas_benchmark")
    configs_by_label = {config.label: config for config in pancreas._method_configs(true_k=8)}

    nnls_rows = {
        "TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001": (
            "euclidean"
        ),
        "TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001": (
            "adaptive_diffusion"
        ),
    }
    for label, distance_source in nnls_rows.items():
        config = configs_by_label[label]
        assert config.distance_source == distance_source
        assert config.params["edge_branch_length_variance_policy"] == ("normalized_branch_length")
        assert config.params["branch_length_optimization_method"] == ("fixed_topology_nnls")
        assert config.params["branch_length_optimization_target_metric"] == (
            "squared_standardized_euclidean"
        )
        assert config.params["branch_length_optimization_pair_sample_size"] == 50_000
        assert config.params["branch_length_optimization_random_state"] == 0
        assert config.params["branch_length_optimization_solver_tolerance"] == 1e-5
        assert config.params["branch_length_optimization_max_iterations"] == 300
        assert "allow_linkage_ultrametric_branch_time" not in config.params

    raw_diagnostic_rows = {
        "TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001": (
            "euclidean"
        ),
        "TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001": (
            "adaptive_diffusion"
        ),
    }
    for label, distance_source in raw_diagnostic_rows.items():
        config = configs_by_label[label]
        assert config.distance_source == distance_source
        assert config.params["edge_branch_length_variance_policy"] == ("normalized_branch_length")
        assert config.params["allow_linkage_ultrametric_branch_time"] is True
        assert "branch_length_optimization_method" not in config.params


def test_scrna_benchmark_does_not_register_distributional_action_configuration() -> None:
    pytest.importorskip("anndata")
    pytest.importorskip("scanpy")

    pancreas = importlib.import_module("applications.scrna.pancreas_benchmark")
    configs = pancreas._method_configs(true_k=8)

    assert all("split-action q50" not in config.label for config in configs)
    assert all(
        not any(str(key).startswith("distributional_action") for key in config.params)
        for config in configs
    )


def test_goncalves_benchmark_reuses_scrna_method_configs() -> None:
    pytest.importorskip("anndata")
    pytest.importorskip("scanpy")

    pancreas = importlib.import_module("applications.scrna.pancreas_benchmark")
    goncalves = importlib.import_module("applications.scrna.goncalves_benchmark")

    assert goncalves._run_benchmarks is pancreas._run_benchmarks
