import math

import pytest
from benchmarks.shared.result_records.factory import build_benchmark_result_row
from benchmarks.shared.result_records.models import BenchmarkRunStatus


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
