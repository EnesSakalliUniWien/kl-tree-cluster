from __future__ import annotations

import math

import numpy as np
import pandas as pd
from benchmarks.shared.result_records import benchmark_rows_to_dataframe
from benchmarks.shared.types import (
    MethodRunResult,
    MethodSpec,
    UnsupportedEvidence,
    UnsupportedReason,
    UnsupportedReasonCode,
)
from benchmarks.shared.util import method_execution


def _unsupported_result() -> MethodRunResult:
    return MethodRunResult(
        labels=None,
        found_clusters=0,
        report_df=None,
        status="unsupported",
        skip_reason=None,
        extra={"stage_timings": {"tree_build_sec": 0.1}},
        unsupported_reason=UnsupportedReason(
            code=UnsupportedReasonCode.EMPIRICAL_NULL_NO_INTERNAL_SUPPORT,
            stage="sibling_calibration",
            message="No admissible internal empirical-null support.",
            evidence=UnsupportedEvidence(3, 0, 3, 6, 6),
        ),
    )


def test_unsupported_run_bypasses_metrics_and_computed_outputs(monkeypatch):
    monkeypatch.setattr(
        method_execution,
        "run_clustering_result",
        lambda **_kwargs: _unsupported_result(),
    )
    data = pd.DataFrame(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        index=["S0", "S1", "S2", "S3"],
        columns=["F0", "F1"],
    )
    meta = {
        "name": "toy",
        "category": "binary",
        "source_family": "binary_template",
        "feature_representation": "binary",
        "simulation_model": "bernoulli_template_latent_class",
        "observation_model": "native_binary_feature_matrix",
        "benchmark_intent": "binary_distributional_recovery",
        "scientific_caution": "none",
        "recommended_simulation_family": "bernoulli_template_or_latent_class_binary_model",
        "n_clusters": 2,
        "n_samples": 4,
        "n_features": 2,
        "noise": 0.0,
        "requires_precomputed_tbs_distance": False,
    }

    row, computed_result, method_audit = method_execution.run_single_method_once(
        method_id="kmeans",
        spec=MethodSpec(name="K-Means", runner=lambda *_a, **_k: _unsupported_result(), param_grid=[]),
        params={"n_clusters": 2},
        case_idx=1,
        case_name="toy",
        tc_seed=1,
        significance_level=0.01,
        edge_alpha=0.001,
        data_t=data,
        y_t=np.array([0, 0, 1, 1]),
        x_original=data.to_numpy(),
        meta=meta,
        distance_matrix=None,
        distance_condensed=None,
        matrix_audit=False,
    )

    assert computed_result is None
    assert method_audit is None
    assert row.status == "unsupported"
    assert row.found_clusters == 0
    assert row.labels_length == 0
    assert row.skip_reason == ""
    assert math.isnan(row.ari)
    assert math.isnan(row.cluster_count_abs_error)
    assert row.unsupported_reason_code == "empirical_null_no_internal_support"
    assert row.unsupported_stage == "sibling_calibration"
    assert row.unsupported_reason == "No admissible internal empirical-null support."
    assert row.unsupported_focal_record_count == 3
    assert row.unsupported_admissible_support_count == 0
    assert row.unsupported_invalid_record_count == 3
    assert row.unsupported_upstream_tested_count == 6
    assert row.unsupported_upstream_rejected_count == 6

    frame = benchmark_rows_to_dataframe([row])
    assert frame.loc[0, "status"] == "unsupported"
    assert frame.loc[0, "unsupported_reason_code"] == (
        "empirical_null_no_internal_support"
    )
    assert math.isnan(frame.loc[0, "ari"])
