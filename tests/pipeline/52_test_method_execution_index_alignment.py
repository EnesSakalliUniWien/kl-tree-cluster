"""Regression tests for index alignment in benchmark method execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.types import MethodRunResult, MethodSpec
from benchmarks.shared.util import method_execution


def _stage_timings(**overrides):
    stage_timings = {
        "tree_build_sec": 0.01,
        "populate_divergences_sec": 0.02,
        "edge_gate_sec": 0.03,
        "edge_gate_contrast_covariance_sec": 0.04,
        "edge_gate_projection_sec": 0.05,
        "edge_gate_wald_statistic_sec": 0.06,
        "edge_gate_tree_bh_sec": 0.07,
        "spectral_context_sec": 0.08,
        "tangent_whitening_sec": 0.09,
        "eigensolve_sec": 0.10,
        "pca_projection_sec": 0.11,
        "sibling_gate_sec": 0.12,
        "sibling_gate_pair_record_collection_sec": 0.13,
        "sibling_gate_inflation_fit_sec": 0.14,
        "sibling_gate_adjusted_tests_sec": 0.15,
        "sibling_gate_fdr_sec": 0.16,
        "traversal_sec": 0.17,
    }
    stage_timings.update(overrides)
    return stage_timings


def _benchmark_meta(**overrides):
    metadata = {
        "name": "regression_case",
        "n_clusters": 2,
        "n_samples": 4,
        "n_features": 2,
        "noise": 0.0,
        "category": "regression",
        "source_family": "binary_template",
        "feature_representation": "binary",
        "requires_precomputed_kl_distance": False,
    }
    metadata.update(overrides)
    return metadata


def test_run_single_method_once_aligns_report_rows_by_sample_id(monkeypatch):
    data_t = pd.DataFrame(
        [[0, 1], [1, 0], [0, 0], [1, 1], [0, 1], [1, 0]],
        index=["S0", "S1", "S2", "S3", "S4", "S5"],
        columns=["F0", "F1"],
    )
    y_t = np.array([0, 0, 1, 1, 2, 2], dtype=int)

    shuffled_index = ["S2", "S0", "S4", "S1", "S5", "S3"]
    cluster_lookup = {"S0": 0, "S1": 0, "S2": 1, "S3": 1, "S4": 2, "S5": 2}
    misordered_report = pd.DataFrame(
        {
            "cluster_id": [cluster_lookup[sample] for sample in shuffled_index],
            "cluster_size": [2, 2, 2, 2, 2, 2],
        },
        index=shuffled_index,
    )
    misordered_report.index.name = "sample_id"

    def _fake_run_clustering_result(**_kwargs):
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1, 2, 2], dtype=int),
            found_clusters=3,
            report_df=misordered_report,
            status="ok",
            skip_reason=None,
            extra={"stage_timings": _stage_timings()},
        )

    monkeypatch.setattr(method_execution, "run_clustering_result", _fake_run_clustering_result)

    spec = MethodSpec(name="KL", runner=lambda **_kwargs: None, param_grid=[{}])
    result_row, computed_result, method_audit = method_execution.run_single_method_once(
        method_id="kl",
        spec=spec,
        params={"tree_distance_metric": "hamming", "tree_linkage_method": "average"},
        case_idx=1,
        case_name="index_alignment_case",
        tc_seed=42,
        significance_level=0.05,
        data_t=data_t,
        y_t=y_t,
        x_original=data_t.values.astype(float),
        meta=_benchmark_meta(
            name="index_alignment_case",
            n_clusters=3,
            n_samples=6,
        ),
        distance_matrix=None,
        distance_condensed=None,
        matrix_audit=False,
    )

    assert np.isclose(result_row.ari, 1.0)
    assert np.isclose(result_row.nmi, 1.0)
    assert np.isclose(result_row.purity, 1.0)
    assert result_row.params_raw["tree_distance_metric"] == "hamming"
    assert result_row.params_raw["tree_distance_source"] == "feature_metric"
    assert result_row.tree_build_sec == 0.01
    assert result_row.edge_gate_sec == 0.03
    assert result_row.edge_gate_contrast_covariance_sec == 0.04
    assert result_row.edge_gate_tree_bh_sec == 0.07
    assert result_row.sibling_gate_sec == 0.12
    assert result_row.sibling_gate_pair_record_collection_sec == 0.13
    assert result_row.sibling_gate_fdr_sec == 0.16
    assert result_row.traversal_sec == 0.17
    assert computed_result is not None
    assert np.isclose(computed_result.ari, 1.0)
    assert computed_result.params["tree_distance_metric"] == "hamming"
    assert computed_result.params["tree_distance_source"] == "feature_metric"
    assert computed_result.meta["stage_timings"]["edge_gate_sec"] == 0.03
    assert method_audit is None


def test_run_single_method_once_records_precomputed_kl_distance_contract(monkeypatch):
    data_t = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]],
        index=["S0", "S1", "S2", "S3"],
        columns=["F0", "F1"],
    )
    y_t = np.array([0, 0, 1, 1], dtype=int)
    precomputed_distance = np.array([0.1, 1.4, 1.3, 1.3, 1.2, 0.1], dtype=float)
    captured_kwargs = {}

    def _fake_run_clustering_result(**kwargs):
        captured_kwargs.update(kwargs)
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={"stage_timings": _stage_timings()},
        )

    monkeypatch.setattr(method_execution, "run_clustering_result", _fake_run_clustering_result)

    spec = MethodSpec(name="KL", runner=lambda **_kwargs: None, param_grid=[{}])
    result_row, computed_result, _method_audit = method_execution.run_single_method_once(
        method_id="kl",
        spec=spec,
        params={"tree_distance_metric": "hamming", "tree_linkage_method": "average"},
        case_idx=1,
        case_name="continuous_case",
        tc_seed=42,
        significance_level=0.05,
        data_t=data_t,
        y_t=y_t,
        x_original=data_t.values.astype(float),
        meta=_benchmark_meta(
            name="continuous_case",
            source_family="gaussian_blobs",
            feature_representation="continuous",
            distance_metric="euclidean",
            requires_precomputed_kl_distance=True,
        ),
        distance_matrix=None,
        distance_condensed=precomputed_distance,
        matrix_audit=False,
    )

    np.testing.assert_allclose(captured_kwargs["distance_condensed"], precomputed_distance)
    assert result_row.params_raw["tree_distance_metric"] == "euclidean"
    assert result_row.params_raw["tree_distance_source"] == "precomputed"
    assert "tree_distance_metric=euclidean" in result_row.params_display
    assert "tree_distance_source=precomputed" in result_row.params_display
    assert computed_result is not None
    assert computed_result.params["tree_distance_metric"] == "euclidean"
    assert computed_result.params["tree_distance_source"] == "precomputed"


def test_run_single_method_once_requires_metric_name_for_precomputed_kl_distance(monkeypatch):
    data_t = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]],
        index=["S0", "S1", "S2", "S3"],
        columns=["F0", "F1"],
    )
    y_t = np.array([0, 0, 1, 1], dtype=int)

    def _fake_run_clustering_result(**_kwargs):
        raise AssertionError("runner should not be called without distance metadata")

    monkeypatch.setattr(method_execution, "run_clustering_result", _fake_run_clustering_result)

    spec = MethodSpec(name="KL", runner=lambda **_kwargs: None, param_grid=[{}])
    with pytest.raises(ValueError, match="distance_metric"):
        method_execution.run_single_method_once(
            method_id="kl",
            spec=spec,
            params={"tree_distance_metric": "hamming", "tree_linkage_method": "average"},
            case_idx=1,
            case_name="broken_precomputed_case",
            tc_seed=42,
            significance_level=0.05,
            data_t=data_t,
            y_t=y_t,
            x_original=data_t.values.astype(float),
            meta=_benchmark_meta(
                name="broken_precomputed_case",
                source_family="gaussian_blobs",
                feature_representation="continuous",
                requires_precomputed_kl_distance=True,
            ),
            distance_matrix=None,
            distance_condensed=np.ones(6, dtype=float),
            matrix_audit=False,
        )


def test_run_single_method_once_requires_kl_stage_timings(monkeypatch):
    data_t = pd.DataFrame(
        [[0, 1], [1, 0], [0, 0], [1, 1]],
        index=["S0", "S1", "S2", "S3"],
        columns=["F0", "F1"],
    )
    y_t = np.array([0, 0, 1, 1], dtype=int)

    def _fake_run_clustering_result(**_kwargs):
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=None,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setattr(method_execution, "run_clustering_result", _fake_run_clustering_result)

    spec = MethodSpec(name="KL", runner=lambda **_kwargs: None, param_grid=[{}])
    with pytest.raises(ValueError, match="stage_timings"):
        method_execution.run_single_method_once(
            method_id="kl",
            spec=spec,
            params={"tree_distance_metric": "hamming", "tree_linkage_method": "average"},
            case_idx=1,
            case_name="missing_timing_case",
            tc_seed=42,
            significance_level=0.05,
            data_t=data_t,
            y_t=y_t,
            x_original=data_t.values.astype(float),
            meta=_benchmark_meta(name="missing_timing_case"),
            distance_matrix=None,
            distance_condensed=None,
            matrix_audit=False,
        )


def test_run_single_method_once_records_runner_exception_as_skip(monkeypatch):
    data_t = pd.DataFrame(
        [[0, 1], [1, 0], [0, 0], [1, 1]],
        index=["S0", "S1", "S2", "S3"],
        columns=["F0", "F1"],
    )
    y_t = np.array([0, 0, 1, 1], dtype=int)

    def _failing_run_clustering_result(**_kwargs):
        raise ValueError("strict calibration support missing")

    monkeypatch.setattr(
        method_execution,
        "run_clustering_result",
        _failing_run_clustering_result,
    )

    spec = MethodSpec(name="KL", runner=lambda **_kwargs: None, param_grid=[{}])
    result_row, computed_result, method_audit = method_execution.run_single_method_once(
        method_id="kl",
        spec=spec,
        params={"tree_distance_metric": "hamming", "tree_linkage_method": "average"},
        case_idx=1,
        case_name="unsupported_calibration_case",
        tc_seed=42,
        significance_level=0.05,
        data_t=data_t,
        y_t=y_t,
        x_original=data_t.values.astype(float),
        meta=_benchmark_meta(name="unsupported_calibration_case"),
        distance_matrix=None,
        distance_condensed=None,
        matrix_audit=False,
    )

    assert result_row.status.value == "skip"
    assert result_row.skip_reason == "strict calibration support missing"
    assert result_row.found_clusters == 0
    assert result_row.labels_length == 0
    assert np.isnan(result_row.ari)
    assert computed_result is None
    assert method_audit is None


def test_run_single_method_once_rejects_report_index_not_sample_ids(monkeypatch):
    data_t = pd.DataFrame(
        [[0, 1], [1, 0], [0, 0], [1, 1]],
        index=["S0", "S1", "S2", "S3"],
        columns=["F0", "F1"],
    )
    y_t = np.array([0, 0, 1, 1], dtype=int)

    # Non-alignable positional index from runner (common in sklearn-style outputs).
    positional_report = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1],
            "cluster_size": [2, 2, 2, 2],
        }
    )

    def _fake_run_clustering_result(**_kwargs):
        return MethodRunResult(
            labels=np.array([0, 0, 1, 1], dtype=int),
            found_clusters=2,
            report_df=positional_report,
            status="ok",
            skip_reason=None,
            extra={},
        )

    monkeypatch.setattr(method_execution, "run_clustering_result", _fake_run_clustering_result)

    spec = MethodSpec(name="K-Means", runner=lambda **_kwargs: None, param_grid=[{}])
    with pytest.raises(ValueError, match="Runner report_df index must match sample ids"):
        method_execution.run_single_method_once(
            method_id="kmeans",
            spec=spec,
            params={"n_clusters": 2, "n_init": 10},
            case_idx=1,
            case_name="positional_index_case",
            tc_seed=42,
            significance_level=0.05,
            data_t=data_t,
            y_t=y_t,
            x_original=data_t.values.astype(float),
            meta=_benchmark_meta(),
            distance_matrix=None,
            distance_condensed=None,
            matrix_audit=False,
        )
