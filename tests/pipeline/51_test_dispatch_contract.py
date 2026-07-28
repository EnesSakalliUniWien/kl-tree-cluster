import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.types.method_spec import MethodSpec
from benchmarks.shared.util.method_sets import TBS_RUNNER_METHODS
from scipy.spatial.distance import pdist, squareform
from tree_break_selection.tree.continuous_distance import (
    continuous_time_distance_condensed,
    standardized_euclidean_distance_condensed,
)
from tree_break_selection.tree.feature_space import continuous_feature_space_from_columns


def _toy_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
                [0.9, 1.0],
            ],
            dtype=float,
        ),
        columns=["f0", "f1"],
    )


def test_dispatch_result_rejects_invalid_runner_params():
    df = _toy_dataframe()
    with pytest.raises(ValueError):
        run_clustering_result(
            data_df=df,
            method_id="kmeans",
            params={"n_clusters": "bad"},
            seed=42,
        )


def test_dispatch_result_rejects_invalid_spectral_params():
    df = _toy_dataframe()
    with pytest.raises(ValueError):
        run_clustering_result(
            data_df=df,
            method_id="spectral",
            params={"n_clusters": "bad"},
            seed=42,
        )


def test_dispatch_result_records_unexpected_exception_as_skip(monkeypatch):
    def _raise_runner(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        METHOD_SPECS,
        "kmeans",
        MethodSpec(
            name="K-Means",
            runner=_raise_runner,
            param_grid=[{"n_clusters": 2}],
        ),
    )

    df = _toy_dataframe()
    result = run_clustering_result(
        data_df=df,
        method_id="kmeans",
        params={"n_clusters": 2},
        seed=42,
    )

    assert result.status == "skip"
    assert result.labels is None
    assert result.skip_reason == "boom"


def test_run_clustering_result_uses_provided_tbs_distance_condensed():
    df = _toy_dataframe()
    dist_condensed = pdist(df.values, metric="euclidean")
    result = run_clustering_result(
        data_df=df,
        method_id="tbs",
        params={"tree_distance_metric": "euclidean", "tree_linkage_method": "average"},
        seed=42,
        distance_condensed=dist_condensed,
    )

    assert result.status in {"ok", "skip"}
    if result.status == "ok":
        assert result.labels is not None
        assert len(result.labels) == len(df)
        assert result.skip_reason is None
    else:
        assert result.labels is None
        assert isinstance(result.skip_reason, str)
        assert result.skip_reason.strip()


def test_run_clustering_result_builds_continuous_mahalanobis_time_distance(monkeypatch):
    captured = {}

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs",
        MethodSpec(
            name="TBS Divergence",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "mahalanobis_time",
                    "tree_linkage_method": "average",
                }
            ],
        ),
    )

    df = _toy_dataframe()
    feature_space = continuous_feature_space_from_columns(tuple(df.columns))
    run_clustering_result(
        data_df=df,
        method_id="tbs",
        params={"tree_distance_metric": "mahalanobis_time", "tree_linkage_method": "average"},
        seed=42,
        feature_space=feature_space,
    )

    np.testing.assert_allclose(
        captured["args"][1],
        continuous_time_distance_condensed(df.values, feature_space),
    )


def test_run_clustering_result_builds_continuous_standardized_euclidean_distance(
    monkeypatch,
):
    captured = {}

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs",
        MethodSpec(
            name="TBS Divergence",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "standardized_euclidean",
                    "tree_linkage_method": "average",
                }
            ],
        ),
    )

    df = _toy_dataframe()
    feature_space = continuous_feature_space_from_columns(tuple(df.columns))
    run_clustering_result(
        data_df=df,
        method_id="tbs",
        params={
            "tree_distance_metric": "standardized_euclidean",
            "tree_linkage_method": "average",
        },
        seed=42,
        feature_space=feature_space,
    )

    np.testing.assert_allclose(
        captured["args"][1],
        standardized_euclidean_distance_condensed(df.values, feature_space),
    )


def test_run_clustering_result_forwards_tbs_gate_profile_params(monkeypatch):
    captured = {}

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs",
        MethodSpec(
            name="TBS Divergence",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "hamming",
                    "tree_linkage_method": "average",
                }
            ],
        ),
    )

    df = _toy_dataframe()
    run_clustering_result(
        data_df=df,
        method_id="tbs",
        params={
            "tree_distance_metric": "euclidean",
            "tree_linkage_method": "average",
            "sibling_gate_profile": "fixed_coordinate_global_passthrough_refined_v1",
            "sibling_gate_method": "fixed_coordinate_bh",
            "sibling_gate_alpha_penalty": 50.0,
            "root_stability_guard_threshold": 0.24,
            "root_stability_subsample_replicates": 12,
            "root_stability_feature_fraction": 0.8,
            "root_stability_seed": 7,
            "root_stability_tree_distance_metric": "jaccard",
            "root_stability_tree_linkage_method": "complete",
            "root_selective_permutation_guard_replicates": 99,
            "root_selective_permutation_guard_seed": 11,
            "root_selective_permutation_guard_alpha": 0.01,
            "root_selective_permutation_guard_scope": (
                "global_sibling_min_passthrough_descendant_refined"
            ),
            "root_selective_permutation_guard_tree_distance_metric": ("rogerstanimoto"),
            "root_selective_permutation_guard_tree_linkage_method": "weighted",
            "spectral_transport_passthrough_guard": True,
            "spectral_transport_max_cost": 0.75,
            "spectral_transport_require_mp_blocks": False,
            "spectral_transport_block_log_tolerance": 0.02,
            "spectral_transport_unmatched_mode_penalty": 1.5,
            "spectral_include_internal_barycenters": True,
            "spectral_internal_distribution_mode": "branch_length_state",
            "continuous_covariance_policy": "guarded_within_child",
            "continuous_covariance_min_child_leaf_count": 8,
            "neighborhood_bandwidth_profile": ("regional_tau_branch_length_support_only_v1"),
            "enforce_internal_support_thresholds": True,
            "passthrough": True,
        },
        seed=42,
        distance_condensed=pdist(df.values, metric="euclidean"),
    )

    assert captured["kwargs"]["sibling_gate_profile"] == (
        "fixed_coordinate_global_passthrough_refined_v1"
    )
    assert captured["kwargs"]["sibling_gate_method"] == "fixed_coordinate_bh"
    assert captured["kwargs"]["sibling_gate_alpha_penalty"] == 50.0
    assert captured["kwargs"]["root_stability_guard_threshold"] == 0.24
    assert captured["kwargs"]["root_stability_subsample_replicates"] == 12
    assert captured["kwargs"]["root_stability_feature_fraction"] == 0.8
    assert captured["kwargs"]["root_stability_seed"] == 7
    assert captured["kwargs"]["root_stability_tree_distance_metric"] == "jaccard"
    assert captured["kwargs"]["root_stability_tree_linkage_method"] == "complete"
    assert captured["kwargs"]["root_selective_permutation_guard_replicates"] == 99
    assert captured["kwargs"]["root_selective_permutation_guard_seed"] == 11
    assert captured["kwargs"]["root_selective_permutation_guard_alpha"] == 0.01
    assert captured["kwargs"]["root_selective_permutation_guard_scope"] == (
        "global_sibling_min_passthrough_descendant_refined"
    )
    assert (
        captured["kwargs"]["root_selective_permutation_guard_tree_distance_metric"]
        == "rogerstanimoto"
    )
    assert captured["kwargs"]["root_selective_permutation_guard_tree_linkage_method"] == "weighted"
    assert captured["kwargs"]["spectral_transport_passthrough_guard"] is True
    assert captured["kwargs"]["spectral_transport_max_cost"] == 0.75
    assert captured["kwargs"]["spectral_transport_require_mp_blocks"] is False
    assert captured["kwargs"]["spectral_transport_block_log_tolerance"] == 0.02
    assert captured["kwargs"]["spectral_transport_unmatched_mode_penalty"] == 1.5
    assert captured["kwargs"]["spectral_include_internal_barycenters"] is True
    assert captured["kwargs"]["spectral_internal_distribution_mode"] == ("branch_length_state")
    assert captured["kwargs"]["continuous_covariance_policy"] == "guarded_within_child"
    assert captured["kwargs"]["continuous_covariance_min_child_leaf_count"] == 8
    assert captured["kwargs"]["neighborhood_bandwidth_profile"] == (
        "regional_tau_branch_length_support_only_v1"
    )
    assert captured["kwargs"]["enforce_internal_support_thresholds"] is True
    assert captured["kwargs"]["passthrough"] is True


def test_run_clustering_result_uses_continuous_sibling_gate_only_for_continuous_blocks(
    monkeypatch,
):
    captured_calls = []

    def _capture_runner(*args, **kwargs):
        captured_calls.append({"args": args, "kwargs": kwargs})
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs",
        MethodSpec(
            name="TBS Adaptive Gate Test",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "euclidean",
                    "tree_linkage_method": "average",
                }
            ],
        ),
    )

    df = _toy_dataframe()
    params = {
        "tree_distance_metric": "euclidean",
        "tree_linkage_method": "average",
        "continuous_sibling_gate_method": "fixed_coordinate_bh",
    }
    distance_condensed = pdist(df.values, metric="euclidean")
    run_clustering_result(
        data_df=df,
        method_id="tbs",
        params=params,
        seed=42,
        distance_condensed=distance_condensed,
        feature_space=None,
    )
    run_clustering_result(
        data_df=df,
        method_id="tbs",
        params=params,
        seed=42,
        distance_condensed=distance_condensed,
        feature_space=continuous_feature_space_from_columns(tuple(df.columns)),
    )

    assert captured_calls[0]["kwargs"]["sibling_gate_method"] == ("projected_wald_inflation")
    assert captured_calls[1]["kwargs"]["sibling_gate_method"] == "fixed_coordinate_bh"


def test_method_registry_exposes_conditional_topology_diagnostic_profile():
    spec = METHOD_SPECS["tbs_conditional_topology_diagnostic"]
    params = spec.param_grid[0]

    assert params["sibling_gate_profile"] == ("fixed_coordinate_conditional_topology_diagnostic_v1")
    assert params["tree_distance_metric"] == "hamming"
    assert params["tree_linkage_method"] == "average"


def test_method_registry_exposes_global_passthrough_refined_profile():
    spec = METHOD_SPECS["tbs_global_passthrough_refined_diagnostic"]
    params = spec.param_grid[0]

    assert params["sibling_gate_profile"] == ("fixed_coordinate_global_passthrough_refined_v1")
    assert params["tree_distance_metric"] == "hamming"
    assert params["tree_linkage_method"] == "average"


def test_method_registry_exposes_fixed_fdr_benchmark_variants():
    expected_methods = {
        "tbs_fixed_coordinate_bh": "fixed_coordinate_bh",
        "tbs_fixed_coordinate_by": "fixed_coordinate_by",
        "tbs_fixed_coordinate_holm": "fixed_coordinate_holm",
        "tbs_fixed_coordinate_bonferroni": "fixed_coordinate_bonferroni",
        "tbs_fixed_block_bh": "fixed_block_bh",
        "tbs_fixed_block_simes_bh": "fixed_block_simes_bh",
    }

    for method_id, sibling_gate_method in expected_methods.items():
        spec = METHOD_SPECS[method_id]
        params = spec.param_grid[0]
        assert method_id in TBS_RUNNER_METHODS
        assert params["sibling_gate_method"] == sibling_gate_method
        assert params["tree_distance_metric"] == "hamming"
        assert params["tree_linkage_method"] == "average"


def test_method_registry_exposes_continuous_guarded_covariance_candidate():
    spec = METHOD_SPECS["tbs_continuous_guarded_within_covariance"]
    params = spec.param_grid[0]

    assert "tbs_continuous_guarded_within_covariance" in TBS_RUNNER_METHODS
    assert params["tree_distance_metric"] == "hamming"
    assert params["tree_linkage_method"] == "average"
    assert params["continuous_covariance_policy"] == "guarded_within_child"
    assert params["continuous_covariance_min_child_leaf_count"] == 8
    assert "sibling_gate_method" not in params
    assert params["continuous_sibling_gate_method"] == "fixed_coordinate_bh"


def test_continuous_guarded_covariance_candidate_runs_non_continuous_input():
    df = _toy_dataframe()
    params = METHOD_SPECS["tbs_continuous_guarded_within_covariance"].param_grid[0]

    result = run_clustering_result(
        data_df=df,
        method_id="tbs_continuous_guarded_within_covariance",
        params=params,
        seed=42,
        distance_condensed=pdist(df.values, metric="euclidean"),
        feature_space=None,
    )

    assert result.status == "ok"
    assert result.labels is not None


def test_method_registry_exposes_spectral_transport_passthrough_profile():
    promoted = METHOD_SPECS["tbs_spectral_transport_passthrough"]
    promoted_params = promoted.param_grid[0]

    assert promoted_params["sibling_gate_profile"] == (
        "fixed_coordinate_spectral_transport_passthrough_v1"
    )
    assert promoted_params["tree_distance_metric"] == "hamming"
    assert promoted_params["tree_linkage_method"] == "average"

    spec = METHOD_SPECS["tbs_spectral_transport_passthrough_diagnostic"]
    params = spec.param_grid[0]

    assert params["sibling_gate_profile"] == (
        "fixed_coordinate_spectral_transport_passthrough_diagnostic_v1"
    )
    assert params["tree_distance_metric"] == "hamming"
    assert params["tree_linkage_method"] == "average"


def test_method_registry_names_diffusion_methods_and_branch_lengths_explicitly():
    nn_diffusion = METHOD_SPECS["tbs_diffusion"]
    nn_params = nn_diffusion.param_grid[0]
    adaptive_diffusion = METHOD_SPECS["tbs_diffusion_adaptive"]
    adaptive_params = adaptive_diffusion.param_grid[0]
    adaptive_nnls_diffusion = METHOD_SPECS["tbs_diffusion_adaptive_nnls"]
    adaptive_nnls_params = adaptive_nnls_diffusion.param_grid[0]
    graphtools_diffusion = METHOD_SPECS["tbs_diffusion_graphtools"]
    graphtools_params = graphtools_diffusion.param_grid[0]
    graphtools_nnls_diffusion = METHOD_SPECS["tbs_diffusion_graphtools_nnls"]
    graphtools_nnls_params = graphtools_nnls_diffusion.param_grid[0]
    graphtools_adaptive_nnls_diffusion = METHOD_SPECS["tbs_diffusion_graphtools_adaptive_nnls"]
    graphtools_adaptive_nnls_params = graphtools_adaptive_nnls_diffusion.param_grid[0]

    assert nn_diffusion.name == "TBS (Hamming NN Diffusion)"
    assert nn_params["diffusion_method"] == "hamming_nn_diffusion"
    assert nn_params["k_neighbors"] == 15
    assert nn_params["tree_linkage_method"] == "average"
    assert nn_params["branch_length_optimization_method"] == "linkage_ultrametric"

    assert adaptive_diffusion.name == "TBS (Adaptive pydiffmap Diffusion)"
    assert adaptive_params["diffusion_method"] == "adaptive_pydiffmap_diffusion"
    assert adaptive_params["bandwidth_type"] == "-1/(d+2)"
    assert adaptive_params["epsilon"] == "median"
    assert adaptive_params["tree_linkage_method"] == "average"
    assert adaptive_params["branch_length_optimization_method"] == "linkage_ultrametric"

    assert adaptive_nnls_diffusion.name == ("TBS (Adaptive pydiffmap Diffusion, NNLS Branch-Time)")
    assert adaptive_nnls_params["diffusion_method"] == "adaptive_pydiffmap_diffusion"
    assert adaptive_nnls_params["bandwidth_type"] == "-1/(d+2)"
    assert adaptive_nnls_params["epsilon"] == "median"
    assert adaptive_nnls_params["edge_branch_length_variance_policy"] == (
        "normalized_branch_length"
    )
    assert adaptive_nnls_params["branch_length_optimization_method"] == "fixed_topology_nnls"
    assert adaptive_nnls_params["branch_length_optimization_target_metric"] == (
        "squared_standardized_euclidean"
    )
    assert adaptive_nnls_params["branch_length_optimization_pair_sample_size"] == 50_000
    assert adaptive_nnls_params["branch_length_optimization_random_state"] == 0
    assert adaptive_nnls_params["branch_length_optimization_solver_tolerance"] == 1e-5
    assert adaptive_nnls_params["branch_length_optimization_max_iterations"] == 300

    assert graphtools_diffusion.name == "TBS (graphtools Kernel Diffusion)"
    assert graphtools_params["diffusion_method"] == "graphtools_kernel_diffusion"
    assert graphtools_params["k_neighbors"] == 10
    assert graphtools_params["decay"] == 40
    assert graphtools_params["kernel_symm"] == "+"
    assert graphtools_params["branch_length_optimization_method"] == "linkage_ultrametric"

    assert graphtools_nnls_diffusion.name == ("TBS (graphtools Kernel Diffusion, NNLS Branch-Time)")
    assert graphtools_nnls_params["diffusion_method"] == "graphtools_kernel_diffusion"
    assert graphtools_nnls_params["k_neighbors"] == 10
    assert graphtools_nnls_params["decay"] == 40
    assert graphtools_nnls_params["kernel_symm"] == "+"
    assert graphtools_nnls_params["edge_branch_length_variance_policy"] == (
        "normalized_branch_length"
    )
    assert graphtools_nnls_params["branch_length_optimization_method"] == ("fixed_topology_nnls")
    assert graphtools_nnls_params["branch_length_optimization_pair_sample_size"] == 50_000

    assert graphtools_adaptive_nnls_diffusion.name == (
        "TBS (graphtools Kernel Diffusion, Adaptive-K NNLS Branch-Time)"
    )
    assert graphtools_adaptive_nnls_params["diffusion_method"] == ("graphtools_kernel_diffusion")
    assert graphtools_adaptive_nnls_params["k_neighbors"] == 10
    assert graphtools_adaptive_nnls_params["adaptive_neighbor_profile"] == ("fragmentation_guard")
    assert graphtools_adaptive_nnls_params["adaptive_neighbor_grid"] == (
        5,
        10,
        15,
        25,
        40,
        80,
        160,
    )
    assert graphtools_adaptive_nnls_params["branch_length_optimization_method"] == (
        "fixed_topology_nnls"
    )
    assert graphtools_adaptive_nnls_params["tree_builder"] == "linkage"
    assert graphtools_adaptive_nnls_params["tree_linkage_method"] == "average"

    tree_strategy_grid = graphtools_adaptive_nnls_diffusion.param_grid
    assert len(tree_strategy_grid) == 8
    assert len({params["benchmark_run_id"] for params in tree_strategy_grid}) == 8
    assert {params["benchmark_class"] for params in tree_strategy_grid} == {"optional_gpl"}
    assert {params["benchmark_grid"] for params in tree_strategy_grid} == {
        "graphtools_adaptive_k_tree_strategy"
    }
    assert {params["benchmark_repeat"] for params in tree_strategy_grid} == {0}

    linkage_methods = {
        params["tree_linkage_method"]
        for params in tree_strategy_grid
        if params["tree_builder"] == "linkage"
    }
    assert linkage_methods == {
        "average",
        "complete",
        "weighted",
        "single",
        "centroid",
        "median",
        "ward",
    }
    for params in tree_strategy_grid:
        assert params["branch_length_optimization_method"] == "fixed_topology_nnls"

    graphtools_adaptive_nnls_neighbor_joining_params = next(
        params for params in tree_strategy_grid if params["tree_builder"] == "neighbor_joining"
    )
    assert graphtools_adaptive_nnls_neighbor_joining_params["tree_builder"] == ("neighbor_joining")
    assert graphtools_adaptive_nnls_neighbor_joining_params["tree_rooting"] == "mad"
    assert (
        graphtools_adaptive_nnls_neighbor_joining_params["branch_length_optimization_method"]
        == "fixed_topology_nnls"
    )


def test_run_clustering_result_forwards_adaptive_nnls_branch_time_params(monkeypatch):
    captured = {}
    original_spec = METHOD_SPECS["tbs_diffusion_adaptive_nnls"]

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs_diffusion_adaptive_nnls",
        MethodSpec(
            name=original_spec.name,
            runner=_capture_runner,
            param_grid=original_spec.param_grid,
        ),
    )

    df = _toy_dataframe()
    result = run_clustering_result(
        data_df=df,
        method_id="tbs_diffusion_adaptive_nnls",
        params=original_spec.param_grid[0],
        seed=42,
    )

    assert result.status == "ok"
    assert captured["kwargs"]["edge_branch_length_variance_policy"] == ("normalized_branch_length")
    assert captured["kwargs"]["branch_length_optimization_method"] == "fixed_topology_nnls"
    assert captured["kwargs"]["branch_length_optimization_target_metric"] == (
        "squared_standardized_euclidean"
    )
    assert captured["kwargs"]["branch_length_optimization_pair_sample_size"] == 50_000
    assert captured["kwargs"]["branch_length_optimization_random_state"] == 0
    assert captured["kwargs"]["branch_length_optimization_solver_tolerance"] == 1e-5
    assert captured["kwargs"]["branch_length_optimization_max_iterations"] == 300


def test_run_clustering_result_forwards_graphtools_adaptive_k_params(monkeypatch):
    captured = {}
    original_spec = METHOD_SPECS["tbs_diffusion_graphtools_adaptive_nnls"]

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs_diffusion_graphtools_adaptive_nnls",
        MethodSpec(
            name=original_spec.name,
            runner=_capture_runner,
            param_grid=original_spec.param_grid,
        ),
    )

    result = run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="tbs_diffusion_graphtools_adaptive_nnls",
        params=original_spec.param_grid[0],
        seed=42,
    )

    assert result.status == "ok"
    assert captured["kwargs"]["adaptive_neighbor_profile"] == "fragmentation_guard"
    assert captured["kwargs"]["adaptive_neighbor_grid"] == (5, 10, 15, 25, 40, 80, 160)
    assert captured["kwargs"]["tree_builder"] == "linkage"
    assert captured["kwargs"]["tree_rooting"] == "linkage_root"
    assert captured["kwargs"]["tree_linkage_method"] == "average"
    assert captured["kwargs"]["edge_branch_length_variance_policy"] == ("normalized_branch_length")
    assert captured["kwargs"]["branch_length_optimization_method"] == ("fixed_topology_nnls")


def test_run_clustering_result_forwards_graphtools_neighbor_joining_tree_params(
    monkeypatch,
):
    captured = {}
    original_spec = METHOD_SPECS["tbs_diffusion_graphtools_adaptive_nnls"]
    neighbor_joining_params = next(
        params
        for params in original_spec.param_grid
        if params["tree_builder"] == "neighbor_joining"
    )

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs_diffusion_graphtools_adaptive_nnls",
        MethodSpec(
            name=original_spec.name,
            runner=_capture_runner,
            param_grid=original_spec.param_grid,
        ),
    )

    result = run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="tbs_diffusion_graphtools_adaptive_nnls",
        params=neighbor_joining_params,
        seed=42,
    )

    assert result.status == "ok"
    assert captured["kwargs"]["tree_builder"] == "neighbor_joining"
    assert captured["kwargs"]["tree_rooting"] == "mad"
    assert captured["kwargs"]["tree_linkage_method"] == "average"
    assert captured["kwargs"]["adaptive_neighbor_profile"] == "fragmentation_guard"


def test_method_registry_exposes_current_support_profiles_without_legacy_ids():
    internal = METHOD_SPECS["tbs_internal_filter_v1"].param_grid[0]
    branch_length = METHOD_SPECS["tbs_internal_filter_branch_length_v1"].param_grid[0]
    bandwidth = METHOD_SPECS["tbs_bandwidth_context_v1"].param_grid[0]

    assert "tbs_legacy_c2ef9a69" not in METHOD_SPECS
    assert "tbs_legacy_internal_spectral_diagnostic" not in METHOD_SPECS
    assert "tbs_rescued_legacy_v1" not in METHOD_SPECS
    assert "tbs_legacy_c2ef9a69" not in TBS_RUNNER_METHODS
    assert "tbs_legacy_internal_spectral_diagnostic" not in TBS_RUNNER_METHODS
    assert "tbs_rescued_legacy_v1" not in TBS_RUNNER_METHODS
    assert internal["spectral_include_internal_barycenters"] is True
    assert internal["spectral_internal_distribution_mode"] == "empirical_barycenter"
    assert internal["enforce_internal_support_thresholds"] is True
    assert branch_length["spectral_internal_distribution_mode"] == ("branch_length_state")
    assert branch_length["enforce_internal_support_thresholds"] is True
    assert bandwidth["neighborhood_bandwidth_profile"] == (
        "regional_tau_branch_length_support_only_v1"
    )


def test_run_clustering_result_dispatches_conditional_topology_as_kl(monkeypatch):
    captured = {}

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs_conditional_topology_diagnostic",
        MethodSpec(
            name="TBS (Conditional Topology Diagnostic)",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "hamming",
                    "tree_linkage_method": "average",
                    "sibling_gate_profile": ("fixed_coordinate_conditional_topology_diagnostic_v1"),
                }
            ],
        ),
    )

    run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="tbs_conditional_topology_diagnostic",
        params={
            "tree_distance_metric": "euclidean",
            "tree_linkage_method": "average",
            "sibling_gate_profile": ("fixed_coordinate_conditional_topology_diagnostic_v1"),
        },
        seed=42,
        distance_condensed=pdist(_toy_dataframe().values, metric="euclidean"),
    )

    assert captured["kwargs"]["sibling_gate_profile"] == (
        "fixed_coordinate_conditional_topology_diagnostic_v1"
    )
    assert captured["kwargs"]["tree_linkage_method"] == "average"


def test_run_clustering_result_dispatches_global_passthrough_refined_as_kl(monkeypatch):
    captured = {}

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs_global_passthrough_refined_diagnostic",
        MethodSpec(
            name="TBS (Global Passthrough Refined Diagnostic)",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "hamming",
                    "tree_linkage_method": "average",
                    "sibling_gate_profile": ("fixed_coordinate_global_passthrough_refined_v1"),
                }
            ],
        ),
    )

    run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="tbs_global_passthrough_refined_diagnostic",
        params={
            "tree_distance_metric": "euclidean",
            "tree_linkage_method": "average",
            "sibling_gate_profile": "fixed_coordinate_global_passthrough_refined_v1",
        },
        seed=42,
        distance_condensed=pdist(_toy_dataframe().values, metric="euclidean"),
    )

    assert captured["kwargs"]["sibling_gate_profile"] == (
        "fixed_coordinate_global_passthrough_refined_v1"
    )
    assert captured["kwargs"]["tree_linkage_method"] == "average"


def test_run_clustering_result_dispatches_spectral_transport_as_kl(monkeypatch):
    captured = {}

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs_spectral_transport_passthrough",
        MethodSpec(
            name="TBS (Spectral Transport Passthrough)",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "hamming",
                    "tree_linkage_method": "average",
                    "sibling_gate_profile": ("fixed_coordinate_spectral_transport_passthrough_v1"),
                }
            ],
        ),
    )

    run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="tbs_spectral_transport_passthrough",
        params={
            "tree_distance_metric": "euclidean",
            "tree_linkage_method": "average",
            "sibling_gate_profile": ("fixed_coordinate_spectral_transport_passthrough_v1"),
        },
        seed=42,
        distance_condensed=pdist(_toy_dataframe().values, metric="euclidean"),
    )

    assert captured["kwargs"]["sibling_gate_profile"] == (
        "fixed_coordinate_spectral_transport_passthrough_v1"
    )
    assert captured["kwargs"]["tree_linkage_method"] == "average"


def test_run_clustering_result_dispatches_iqtree_without_condensed_distance(monkeypatch):
    captured = {}

    def _capture_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setitem(
        METHOD_SPECS,
        "tbs_iqtree3",
        MethodSpec(
            name="TBS (IQ-TREE 3, MAD Root)",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "hamming",
                    "tree_linkage_method": "average",
                    "tree_builder": "iqtree3",
                    "tree_rooting": "mad",
                }
            ],
        ),
    )

    run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="tbs_iqtree3",
        params={
            "tree_distance_metric": "hamming",
            "tree_linkage_method": "average",
            "tree_builder": "iqtree3",
            "tree_rooting": "mad",
        },
        seed=42,
        distance_condensed=pdist(_toy_dataframe().values, metric="euclidean"),
    )

    assert captured["args"][1] is None
    assert captured["kwargs"]["tree_builder"] == "iqtree3"
    assert captured["kwargs"]["tree_rooting"] == "mad"


def test_run_clustering_result_uses_provided_graph_distance_matrix():
    df = _toy_dataframe()
    dist_condensed = pdist(df.values, metric="euclidean")
    dist_matrix = squareform(dist_condensed)
    result = run_clustering_result(
        data_df=df,
        method_id="leiden",
        params={"n_neighbors": 2, "resolution": 1.0},
        seed=42,
        distance_matrix=dist_matrix,
    )

    assert result.status in {"ok", "skip"}
    if result.status == "ok":
        assert result.labels is not None
        assert len(result.labels) == len(df)
        assert result.skip_reason is None
    else:
        assert result.labels is None
        assert isinstance(result.skip_reason, str)
        assert result.skip_reason.strip()
