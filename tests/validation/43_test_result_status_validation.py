import pytest

from benchmarks.shared.results.factory import build_benchmark_result_row
from benchmarks.shared.results.models import BenchmarkRunStatus


def _build_row(status):
    return build_benchmark_result_row(
        case_idx=1,
        case_name="case",
        case_category="cat",
        method_name="Method",
        run_params={},
        true_clusters=2,
        found_clusters=2,
        samples=8,
        features=4,
        noise=0.1,
        ari=1.0,
        nmi=1.0,
        purity=1.0,
        status=status,
        skip_reason=None,
        labels_length=8,
    )


def test_build_benchmark_result_row_accepts_ok_and_skip():
    row_ok = _build_row("ok")
    assert row_ok.status == BenchmarkRunStatus.OK

    row_skip = _build_row("skip")
    assert row_skip.status == BenchmarkRunStatus.SKIP


def test_build_benchmark_result_row_accepts_status_enum():
    row = _build_row(BenchmarkRunStatus.OK)
    assert row.status == BenchmarkRunStatus.OK


@pytest.mark.parametrize("status", ["error", "skipped", "unknown", "", "OKAY"])
def test_build_benchmark_result_row_rejects_legacy_or_invalid_status(status):
    with pytest.raises(ValueError, match="Invalid benchmark status"):
        _build_row(status)
