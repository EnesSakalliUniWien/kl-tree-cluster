"""Tests for scRNA plot manifest metadata."""

import importlib
import json
import re
from pathlib import Path


def _load_scrna_plot_pipeline_module():
    return importlib.import_module("applications.scrna.plot_pipeline")


def test_plot_manifest_json_records_generated_timestamp(tmp_path):
    pipeline = _load_scrna_plot_pipeline_module()
    rows = [{field: "" for field in pipeline.MANIFEST_FIELDS}]
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
