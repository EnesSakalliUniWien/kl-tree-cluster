from datetime import datetime, timezone
from pathlib import Path

from benchmarks.shared.util.pdf.session import resolve_pdf_output_path as _resolve_pdf_output_path


def test_resolve_pdf_output_path_defaults_to_timestamped_name_under_plots_root():
    plots_root = Path("/tmp/results/plots")
    started_at = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    out = _resolve_pdf_output_path(None, plots_root=plots_root, started_at=started_at)
    assert out == plots_root / "benchmark_plots_20250102_030405Z.pdf"


def test_resolve_pdf_output_path_adds_pdf_suffix_when_missing():
    plots_root = Path("/tmp/results/plots")
    started_at = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    out = _resolve_pdf_output_path("custom_name", plots_root=plots_root, started_at=started_at)
    assert out == Path("custom_name.pdf")


def test_resolve_pdf_output_path_preserves_explicit_suffix():
    plots_root = Path("/tmp/results/plots")
    started_at = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    out = _resolve_pdf_output_path(
        "custom_name.v1.pdf", plots_root=plots_root, started_at=started_at
    )
    assert out == Path("custom_name.v1.pdf")
