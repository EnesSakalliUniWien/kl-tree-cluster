"""Tests for scRNA plot manifest metadata."""

import importlib.util
import json
import re
from pathlib import Path


def _load_scrna_plot_pipeline_module():
    script_path = Path(__file__).resolve().parents[2] / "applications/scrna/plot_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_scrna_plot_pipeline", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_manifest_json_records_generated_timestamp(tmp_path):
    pipeline = _load_scrna_plot_pipeline_module()
    rows = [
        {
            field: ""
            for field in pipeline.MANIFEST_FIELDS
        }
    ]
    rows[0].update(
        {
            "dataset": "pancreas",
            "plot_id": "plot",
            "path": str(tmp_path / "plot.png"),
            "format": "png",
            "stage": "unit",
            "role": "canonical",
            "canonical": "true",
            "status": "present",
        }
    )

    summary = pipeline.write_manifest(
        dataset="pancreas",
        output_dir=tmp_path,
        rows=rows,
        commands=[],
    )
    written = json.loads((tmp_path / "plot_manifest.json").read_text(encoding="utf-8"))

    assert re.match(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}",
        summary["generated_at"],
    )
    assert written["generated_at"] == summary["generated_at"]
    assert (tmp_path / "plot_manifest.csv").exists()


def test_pipeline_commands_resolve_relocated_application_entrypoints(tmp_path):
    pipeline = _load_scrna_plot_pipeline_module()

    for dataset in ("pancreas", "goncalves"):
        commands = pipeline.pipeline_commands(dataset, tmp_path)
        assert commands
        for command in commands:
            script_path = Path(command[1])
            assert script_path.is_file(), command
            assert "scripts" not in script_path.parts
