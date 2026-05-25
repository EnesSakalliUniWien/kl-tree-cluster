"""Integration test for runner failure rows across benchmark surfaces."""

from __future__ import annotations

from benchmarks.branch_length.logic import run_branch_length_benchmark
from benchmarks.multi_split.run import run_multi_split_benchmark
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.types import MethodSpec


def test_forced_runner_failure_records_skip_everywhere(monkeypatch):
    original_kl = METHOD_SPECS["kl"]

    def _raise_runner(*_args, **_kwargs):
        raise RuntimeError("forced integration failure")

    monkeypatch.setitem(
        METHOD_SPECS,
        "kl",
        MethodSpec(
            name=original_kl.name,
            runner=_raise_runner,
            param_grid=original_kl.param_grid,
        ),
    )

    pipeline_case = {
        "name": "forced_fail_case",
        "generator": "blobs",
        "n_samples": 12,
        "n_features": 8,
        "n_clusters": 3,
        "cluster_std": 0.4,
        "seed": 11,
        "category": "forced_failure_contract",
    }

    pipeline_df, _ = benchmark_cluster_algorithm(
        test_cases=[pipeline_case],
        verbose=False,
        methods=["kl"],
        plot_umap=False,
        plot_manifold=False,
    )
    assert pipeline_df.loc[0, "status"] == "skip"
    assert pipeline_df.loc[0, "skip_reason"] == "forced integration failure"

    branch_df = run_branch_length_benchmark(
        n_leaves=20,
        n_features=20,
        branch_lengths=[0.2],
        random_seed=7,
        method="kl",
        verbose=False,
    )
    assert branch_df.loc[0, "status"] == "skip"
    assert branch_df.loc[0, "skip_reason"] == "forced integration failure"

    multi_df = run_multi_split_benchmark(
        n_total_samples=24,
        n_groups_list=[2],
        n_features=20,
        n_replicates=1,
        base_seed=13,
        verbose=False,
    )
    assert multi_df.loc[0, "status"] == "skip"
    assert multi_df.loc[0, "skip_reason"] == "forced integration failure"
