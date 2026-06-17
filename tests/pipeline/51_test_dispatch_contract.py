import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.types.method_spec import MethodSpec
from benchmarks.shared.util.method_sets import KL_RUNNER_METHODS
from scipy.spatial.distance import pdist, squareform


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


def test_run_clustering_result_uses_provided_kl_distance_condensed():
    df = _toy_dataframe()
    dist_condensed = pdist(df.values, metric="euclidean")
    result = run_clustering_result(
        data_df=df,
        method_id="kl",
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


def test_run_clustering_result_forwards_kl_gate_profile_params(monkeypatch):
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
        "kl",
        MethodSpec(
            name="KL Divergence",
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
        method_id="kl",
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
            "root_selective_permutation_guard_replicates": 99,
            "root_selective_permutation_guard_seed": 11,
            "root_selective_permutation_guard_alpha": 0.01,
            "root_selective_permutation_guard_scope": (
                "global_sibling_min_passthrough_descendant_refined"
            ),
            "spectral_transport_passthrough_guard": True,
            "spectral_transport_max_cost": 0.75,
            "spectral_transport_require_mp_blocks": False,
            "spectral_transport_block_log_tolerance": 0.02,
            "spectral_transport_unmatched_mode_penalty": 1.5,
            "spectral_include_internal_barycenters": True,
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
    assert captured["kwargs"]["root_selective_permutation_guard_replicates"] == 99
    assert captured["kwargs"]["root_selective_permutation_guard_seed"] == 11
    assert captured["kwargs"]["root_selective_permutation_guard_alpha"] == 0.01
    assert captured["kwargs"]["root_selective_permutation_guard_scope"] == (
        "global_sibling_min_passthrough_descendant_refined"
    )
    assert captured["kwargs"]["spectral_transport_passthrough_guard"] is True
    assert captured["kwargs"]["spectral_transport_max_cost"] == 0.75
    assert captured["kwargs"]["spectral_transport_require_mp_blocks"] is False
    assert captured["kwargs"]["spectral_transport_block_log_tolerance"] == 0.02
    assert captured["kwargs"]["spectral_transport_unmatched_mode_penalty"] == 1.5
    assert captured["kwargs"]["spectral_include_internal_barycenters"] is True
    assert captured["kwargs"]["passthrough"] is True


def test_method_registry_exposes_conditional_topology_diagnostic_profile():
    spec = METHOD_SPECS["kl_conditional_topology_diagnostic"]
    params = spec.param_grid[0]

    assert params["sibling_gate_profile"] == (
        "fixed_coordinate_conditional_topology_diagnostic_v1"
    )
    assert params["tree_distance_metric"] == "hamming"
    assert params["tree_linkage_method"] == "average"


def test_method_registry_exposes_global_passthrough_refined_profile():
    spec = METHOD_SPECS["kl_global_passthrough_refined_diagnostic"]
    params = spec.param_grid[0]

    assert params["sibling_gate_profile"] == (
        "fixed_coordinate_global_passthrough_refined_v1"
    )
    assert params["tree_distance_metric"] == "hamming"
    assert params["tree_linkage_method"] == "average"


def test_method_registry_exposes_spectral_transport_passthrough_profile():
    promoted = METHOD_SPECS["kl_spectral_transport_passthrough"]
    promoted_params = promoted.param_grid[0]

    assert promoted_params["sibling_gate_profile"] == (
        "fixed_coordinate_spectral_transport_passthrough_v1"
    )
    assert promoted_params["tree_distance_metric"] == "hamming"
    assert promoted_params["tree_linkage_method"] == "average"

    spec = METHOD_SPECS["kl_spectral_transport_passthrough_diagnostic"]
    params = spec.param_grid[0]

    assert params["sibling_gate_profile"] == (
        "fixed_coordinate_spectral_transport_passthrough_diagnostic_v1"
    )
    assert params["tree_distance_metric"] == "hamming"
    assert params["tree_linkage_method"] == "average"


def test_method_registry_exposes_full_legacy_commit_method():
    spec = METHOD_SPECS["kl_legacy_c2ef9a69"]
    params = spec.param_grid[0]

    assert "kl_legacy_c2ef9a69" in KL_RUNNER_METHODS
    assert params["tree_distance_metric"] == "hamming"
    assert params["tree_linkage_method"] == "average"
    assert params["tree_builder"] == "linkage"
    assert params["tree_rooting"] == "linkage_root"


def test_legacy_commit_package_imports_tree_decomposition():
    from kl_clustering_analysis.legacy_methods.commit_c2ef9a69 import COMMIT
    from kl_clustering_analysis.legacy_methods.commit_c2ef9a69.kl_clustering_analysis.hierarchy_analysis.tree_decomposition import (
        TreeDecomposition,
    )

    assert COMMIT == "c2ef9a69e0888168950bdee4a41ae8ab9996e32f"
    assert TreeDecomposition.__name__ == "TreeDecomposition"


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
        "kl_conditional_topology_diagnostic",
        MethodSpec(
            name="KL (Conditional Topology Diagnostic)",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "hamming",
                    "tree_linkage_method": "average",
                    "sibling_gate_profile": (
                        "fixed_coordinate_conditional_topology_diagnostic_v1"
                    ),
                }
            ],
        ),
    )

    run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="kl_conditional_topology_diagnostic",
        params={
            "tree_distance_metric": "euclidean",
            "tree_linkage_method": "average",
            "sibling_gate_profile": (
                "fixed_coordinate_conditional_topology_diagnostic_v1"
            ),
        },
        seed=42,
        distance_condensed=pdist(_toy_dataframe().values, metric="euclidean"),
    )

    assert captured["kwargs"]["sibling_gate_profile"] == (
        "fixed_coordinate_conditional_topology_diagnostic_v1"
    )
    assert captured["kwargs"]["tree_linkage_method"] == "average"


def test_run_clustering_result_dispatches_full_legacy_commit_as_kl(monkeypatch):
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
        "kl_legacy_c2ef9a69",
        MethodSpec(
            name="KL Legacy Full Method (commit c2ef9a69)",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "hamming",
                    "tree_linkage_method": "average",
                    "tree_builder": "linkage",
                    "tree_rooting": "linkage_root",
                }
            ],
        ),
    )

    run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="kl_legacy_c2ef9a69",
        params={
            "tree_distance_metric": "euclidean",
            "tree_linkage_method": "average",
            "tree_builder": "linkage",
            "tree_rooting": "linkage_root",
        },
        seed=42,
        distance_condensed=pdist(_toy_dataframe().values, metric="euclidean"),
    )

    assert captured["args"][1] is not None
    assert captured["kwargs"]["tree_builder"] == "linkage"
    assert captured["kwargs"]["tree_rooting"] == "linkage_root"
    assert captured["kwargs"]["tree_linkage_method"] == "average"


def test_run_clustering_result_runs_full_legacy_commit_smoke():
    df = _toy_dataframe()

    result = run_clustering_result(
        data_df=df,
        method_id="kl_legacy_c2ef9a69",
        params={
            "tree_distance_metric": "euclidean",
            "tree_linkage_method": "average",
            "tree_builder": "linkage",
            "tree_rooting": "linkage_root",
        },
        seed=42,
        distance_condensed=pdist(df.values, metric="euclidean"),
    )

    assert result.status == "ok"
    assert result.labels is not None
    assert len(result.labels) == len(df)
    assert result.extra["legacy_commit"] == (
        "c2ef9a69e0888168950bdee4a41ae8ab9996e32f"
    )


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
        "kl_global_passthrough_refined_diagnostic",
        MethodSpec(
            name="KL (Global Passthrough Refined Diagnostic)",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "hamming",
                    "tree_linkage_method": "average",
                    "sibling_gate_profile": (
                        "fixed_coordinate_global_passthrough_refined_v1"
                    ),
                }
            ],
        ),
    )

    run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="kl_global_passthrough_refined_diagnostic",
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
        "kl_spectral_transport_passthrough",
        MethodSpec(
            name="KL (Spectral Transport Passthrough)",
            runner=_capture_runner,
            param_grid=[
                {
                    "tree_distance_metric": "hamming",
                    "tree_linkage_method": "average",
                    "sibling_gate_profile": (
                        "fixed_coordinate_spectral_transport_passthrough_v1"
                    ),
                }
            ],
        ),
    )

    run_clustering_result(
        data_df=_toy_dataframe(),
        method_id="kl_spectral_transport_passthrough",
        params={
            "tree_distance_metric": "euclidean",
            "tree_linkage_method": "average",
            "sibling_gate_profile": (
                "fixed_coordinate_spectral_transport_passthrough_v1"
            ),
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
        "kl_iqtree3",
        MethodSpec(
            name="KL (IQ-TREE 3, MAD Root)",
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
        method_id="kl_iqtree3",
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
