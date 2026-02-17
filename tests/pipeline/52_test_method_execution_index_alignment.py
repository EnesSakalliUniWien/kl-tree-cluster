"""Regression tests for index alignment in benchmark method execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.shared.types import MethodRunResult, MethodSpec
from benchmarks.shared.util import method_execution


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
            extra={},
        )

    monkeypatch.setattr(method_execution, "run_clustering_result", _fake_run_clustering_result)

    spec = MethodSpec(name="KL", runner=lambda **_kwargs: None, param_grid=[{}])
    result_row, computed_result, method_audit = method_execution.run_single_method_once(
        method_id="kl",
        spec=spec,
        params={},
        case_idx=1,
        case_name="index_alignment_case",
        tc_seed=42,
        significance_level=0.05,
        data_t=data_t,
        y_t=y_t,
        x_original=data_t.values.astype(float),
        meta={
            "n_clusters": 3,
            "n_samples": 6,
            "n_features": 2,
            "noise": 0.0,
            "category": "regression",
        },
        distance_matrix=None,
        distance_condensed=None,
        precomputed_distance_condensed=None,
        matrix_audit=False,
    )

    assert np.isclose(result_row.ari, 1.0)
    assert np.isclose(result_row.nmi, 1.0)
    assert np.isclose(result_row.purity, 1.0)
    assert computed_result is not None
    assert np.isclose(computed_result.ari, 1.0)
    assert method_audit is None


def test_run_single_method_once_falls_back_when_report_index_not_sample_ids(monkeypatch):
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
    result_row, computed_result, _ = method_execution.run_single_method_once(
        method_id="kmeans",
        spec=spec,
        params={},
        case_idx=1,
        case_name="positional_index_case",
        tc_seed=42,
        significance_level=0.05,
        data_t=data_t,
        y_t=y_t,
        x_original=data_t.values.astype(float),
        meta={
            "n_clusters": 2,
            "n_samples": 4,
            "n_features": 2,
            "noise": 0.0,
            "category": "regression",
        },
        distance_matrix=None,
        distance_condensed=None,
        precomputed_distance_condensed=None,
        matrix_audit=False,
    )

    assert np.isclose(result_row.ari, 1.0)
    assert np.isclose(result_row.nmi, 1.0)
    assert np.isclose(result_row.purity, 1.0)
    assert computed_result is not None
