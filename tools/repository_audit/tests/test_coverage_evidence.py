import json
from pathlib import Path

from tbs_repo_audit.coverage_evidence import compare_coverage


def _write_coverage(path: Path, files: dict[str, object]) -> None:
    path.write_text(json.dumps({"files": files}), encoding="utf-8")


def test_compare_coverage_reports_only_calibration_exclusive_lines(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration.json"
    baseline = tmp_path / "baseline.json"
    _write_coverage(
        calibration,
        {
            "tree_break_selection/gate.py": {
                "executed_lines": [1, 2, 3],
                "contexts": {"2": ["test_panel"], "3": ["test_panel"]},
            },
            "tests/test_panel.py": {"executed_lines": [1]},
        },
    )
    _write_coverage(
        baseline,
        {
            "tree_break_selection/gate.py": {
                "executed_lines": [1, 2],
                "contexts": {"2": ["test_contract"]},
            }
        },
    )

    result = compare_coverage(calibration, baseline)

    assert result["production_line_count"] == 1
    assert result["benchmark_line_count"] == 0
    assert result["files"][0]["calibration_only_lines"] == [3]
    assert result["files"][0]["contexts"] == {"3": ["test_panel"]}
    assert result["raw_reports_retained"] is False


def test_compare_coverage_separates_benchmark_and_production_counts(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration.json"
    baseline = tmp_path / "baseline.json"
    _write_coverage(
        calibration,
        {
            "tree_break_selection/gate.py": {"executed_lines": [1]},
            "benchmarks/diagnostics/panel.py": {"executed_lines": [2, 3]},
        },
    )
    _write_coverage(baseline, {})

    result = compare_coverage(calibration, baseline)

    assert result["line_count"] == 3
    assert result["production_line_count"] == 1
    assert result["benchmark_line_count"] == 2
    benchmark = next(item for item in result["files"] if item["surface"] == "benchmark")
    assert benchmark["contexts"] == {}
