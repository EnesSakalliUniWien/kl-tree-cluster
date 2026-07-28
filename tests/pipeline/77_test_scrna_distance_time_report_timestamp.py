"""Tests for scRNA distance/time model report timestamps."""

import importlib
import sys


def _load_pancreas_module():
    module_name = "applications.scrna.pancreas_benchmark"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_distance_time_model_report_records_generated_timestamp(tmp_path):
    module = _load_pancreas_module()

    module._write_distance_time_model_analysis(
        output_dir=tmp_path,
        tree_summaries=[],
        adaptive_metadata=None,
        length_sensitivity_rows=[],
        generated_at="2026-06-24T20:16:00+02:00",
    )

    report = (tmp_path / "edge_gate_distance_time_model_analysis.md").read_text(encoding="utf-8")
    assert report.startswith(
        "# Edge gate distance/time model analysis\n\nGenerated at: 2026-06-24T20:16:00+02:00\n"
    )
