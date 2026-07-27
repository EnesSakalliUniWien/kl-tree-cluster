"""Tests for discrete covariance diagnostic manifest timestamps."""

import importlib.util
import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _load_diagnostic_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/diagnostics/calibration/discrete_covariance_guard.py"
    )
    spec = importlib.util.spec_from_file_location("discrete_covariance_guard", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_discrete_covariance_diagnostic_writes_manifest_timestamp(monkeypatch, tmp_path):
    module = _load_diagnostic_module()
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            output_dir=tmp_path,
            cases=["case_a"],
            case_set="default",
            root_only=True,
        ),
    )
    monkeypatch.setattr(
        module,
        "_benchmark_cases_by_name",
        lambda: {"case_a": {"name": "case_a"}},
    )
    monkeypatch.setattr(
        module,
        "_prepare_benchmark_case",
        lambda _case: SimpleNamespace(case={"name": "case_a"}),
    )
    monkeypatch.setattr(
        module,
        "_diagnose_case",
        lambda _prepared, root_only: [{"case": "case_a", "root_only": root_only}],
    )
    monkeypatch.setattr(
        module,
        "_summarize",
        lambda _rows: pd.DataFrame([{"case": "case_a", "node_count": 1}]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["discrete_covariance_guard.py", "--root-only"],
    )

    module.main()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}",
        manifest["generated_at"],
    )
    assert manifest["command"] == "discrete_covariance_guard.py --root-only"
    assert manifest["cases"] == ["case_a"]
    assert manifest["root_only"] is True
    assert manifest["node_rows"] == 1
    assert manifest["summary_rows"] == 1
