import math

import numpy as np
import pytest
from benchmarks.shared.result_records.factory import build_benchmark_result_row
from benchmarks.shared.types import (
    BenchmarkRunStatus,
    MethodRunResult,
    UnsupportedEvidence,
    UnsupportedReason,
    UnsupportedReasonCode,
)


def test_benchmark_run_status_serializes_to_its_stable_value() -> None:
    assert str(BenchmarkRunStatus.OK) == "ok"
    assert str(BenchmarkRunStatus.SKIP) == "skip"
    assert str(BenchmarkRunStatus.UNSUPPORTED) == "unsupported"


def _build_row(status, **overrides):
    kwargs = dict(
        test_case=1,
        case_id="case",
        case_category="cat",
        source_family="binary_template",
        feature_representation="binary",
        method="method",
        run_params={},
        true_clusters=2,
        found_clusters=2,
        samples=8,
        features=4,
        noise=0.1,
        ari=1.0,
        nmi=1.0,
        purity=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        worst_cluster_recall=1.0,
        outlier_precision=1.0,
        outlier_recall=1.0,
        outlier_f1=1.0,
        singleton_outlier_isolated=1.0,
        grouped_outlier_cluster_recovered=1.0,
        cluster_count_abs_error=0.0,
        over_split=0.0,
        under_split=0.0,
        status=status,
        skip_reason=None,
        labels_length=8,
    )
    if status == BenchmarkRunStatus.SKIP or status == "skip":
        kwargs.update(
            found_clusters=0,
            status="skip",
            skip_reason="method unavailable",
            labels_length=0,
        )
    kwargs.update(overrides)
    return build_benchmark_result_row(**kwargs)


def test_build_benchmark_result_row_accepts_ok_and_skip():
    row_ok = _build_row("ok")
    assert row_ok.status == BenchmarkRunStatus.OK

    row_skip = _build_row("skip")
    assert row_skip.status == BenchmarkRunStatus.SKIP


def test_build_benchmark_result_row_accepts_status_enum():
    row = _build_row(BenchmarkRunStatus.OK)
    assert row.status == BenchmarkRunStatus.OK


def _unsupported_reason() -> UnsupportedReason:
    return UnsupportedReason(
        code=UnsupportedReasonCode.EMPIRICAL_NULL_NO_INTERNAL_SUPPORT,
        stage="sibling_calibration",
        message="No admissible internal empirical-null support.",
        evidence=UnsupportedEvidence(
            focal_record_count=39,
            admissible_support_count=0,
            invalid_record_count=39,
            upstream_tested_count=78,
            upstream_rejected_count=78,
        ),
    )


def test_method_run_result_accepts_typed_unsupported_outcome():
    result = MethodRunResult(
        labels=None,
        found_clusters=0,
        report_df=None,
        status="unsupported",
        skip_reason=None,
        extra={"stage_timings": {}},
        unsupported_reason=_unsupported_reason(),
    )

    assert result.status is BenchmarkRunStatus.UNSUPPORTED
    assert result.unsupported_reason == _unsupported_reason()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "labels": np.array([0, 0]),
                "found_clusters": 1,
                "status": "unsupported",
                "unsupported_reason": _unsupported_reason(),
            },
            "must not include labels",
        ),
        (
            {
                "labels": None,
                "found_clusters": 0,
                "status": "unsupported",
                "unsupported_reason": None,
            },
            "requires unsupported_reason",
        ),
        (
            {
                "labels": None,
                "found_clusters": 0,
                "status": "skip",
                "skip_reason": "not available",
                "unsupported_reason": _unsupported_reason(),
            },
            "must not include unsupported_reason",
        ),
        (
            {
                "labels": None,
                "found_clusters": 0,
                "status": "ok",
            },
            "requires labels",
        ),
    ],
)
def test_method_run_result_rejects_inconsistent_states(kwargs, message):
    base = {
        "labels": np.array([0, 1]),
        "found_clusters": 2,
        "report_df": None,
        "status": "ok",
        "skip_reason": None,
        "extra": None,
        "unsupported_reason": None,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        MethodRunResult(**base)


def test_unsupported_evidence_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        UnsupportedEvidence(
            focal_record_count=1,
            admissible_support_count=0,
            invalid_record_count=-1,
            upstream_tested_count=2,
            upstream_rejected_count=2,
        )


def test_build_benchmark_result_row_requires_explicit_unknown_cluster_count():
    with pytest.raises(ValueError, match="true_clusters must be an integer"):
        _build_row("ok", true_clusters=None)


def test_build_benchmark_result_row_preserves_missing_noise_as_nan():
    row = _build_row("ok", noise=math.nan)
    assert math.isnan(row.noise)


def test_build_benchmark_result_row_records_stage_timings():
    row = _build_row(
        "ok",
        stage_timings={
            "tree_build_sec": 0.1,
            "edge_gate_sec": 0.2,
            "traversal_sec": 0.3,
        },
    )

    assert row.tree_build_sec == 0.1
    assert row.edge_gate_sec == 0.2
    assert row.traversal_sec == 0.3
    assert math.isnan(row.sibling_gate_sec)


def test_build_benchmark_result_row_rejects_missing_noise_value():
    with pytest.raises(ValueError, match="noise must be a float"):
        _build_row("ok", noise=None)


def test_build_benchmark_result_row_rejects_invalid_stage_timing():
    with pytest.raises(ValueError, match="Stage timing"):
        _build_row("ok", stage_timings={"edge_gate_sec": -0.1})


@pytest.mark.parametrize("status", ["error", "skipped", "unknown", "", "OKAY"])
def test_build_benchmark_result_row_rejects_invalid_status(status):
    with pytest.raises(ValueError, match="Invalid benchmark status"):
        _build_row(status)
