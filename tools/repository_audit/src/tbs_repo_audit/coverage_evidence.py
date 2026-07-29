"""Comparison of calibration and non-calibration coverage evidence."""

from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_coverage(calibration_path: Path, baseline_path: Path) -> dict[str, object]:
    """Return lines exercised by calibration tests but not the remaining suite."""
    calibration = _load(calibration_path)
    baseline = _load(baseline_path)
    calibration_files = calibration.get("files", {})
    baseline_files = baseline.get("files", {})
    comparison: list[dict[str, object]] = []

    for path, calibration_info in sorted(calibration_files.items()):
        if not path.startswith(("benchmarks/", "tree_break_selection/")):
            continue
        baseline_info = baseline_files.get(path, {})
        calibration_lines = set(calibration_info.get("executed_lines", []))
        baseline_lines = set(baseline_info.get("executed_lines", []))
        exclusive_lines = sorted(calibration_lines - baseline_lines)
        if not exclusive_lines:
            continue
        contexts = calibration_info.get("contexts", {})
        surface = (
            "production"
            if path.startswith("tree_break_selection/")
            else "benchmark"
        )
        comparison.append(
            {
                "path": path,
                "surface": surface,
                "calibration_only_lines": exclusive_lines,
                "line_count": len(exclusive_lines),
                "contexts": (
                    {
                        str(line): contexts.get(str(line), [])
                        for line in exclusive_lines
                    }
                    if surface == "production"
                    else {}
                ),
            }
        )

    return {
        "raw_reports_retained": False,
        "file_count": len(comparison),
        "line_count": sum(item["line_count"] for item in comparison),
        "production_line_count": sum(
            item["line_count"] for item in comparison if item["surface"] == "production"
        ),
        "benchmark_line_count": sum(
            item["line_count"] for item in comparison if item["surface"] == "benchmark"
        ),
        "files": comparison,
    }
