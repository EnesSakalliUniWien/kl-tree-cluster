"""Tests for scRNA distance/time model report timestamps."""

import importlib.util
import sys
from pathlib import Path


def _load_pancreas_module():
    script_path = Path(__file__).resolve().parents[2] / "applications/scrna/pancreas_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "applications.scrna.pancreas_benchmark", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_distance_time_model_report_records_generated_timestamp(tmp_path):
    module = _load_pancreas_module()

    module._write_distance_time_model_analysis(
        output_dir=tmp_path,
        tree_summaries=[],
        adaptive_metadata=None,
        length_sensitivity_rows=[],
        generated_at="2026-06-24T20:16:00+02:00",
    )

    report = (tmp_path / "edge_gate_distance_time_model_analysis.md").read_text(
        encoding="utf-8"
    )
    assert report.startswith(
        "# Edge gate distance/time model analysis\n\n"
        "Generated at: 2026-06-24T20:16:00+02:00\n"
    )
