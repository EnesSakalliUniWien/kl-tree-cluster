#!/usr/bin/env python3
"""Generate relationship analysis artifacts for an existing benchmark run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from benchmarks.shared.relationship_analysis import analyze_benchmark_relationships

_SCRIPT_PATH = Path(__file__).resolve()
BENCHMARKS_ROOT = next(parent for parent in _SCRIPT_PATH.parents if parent.name == "benchmarks")
_SUITE_RUN_RE = re.compile(r"^run_\d{8}_\d{6}Z_(?P<suite>.+)$")


def _comparison_csv_candidates(run_dir: Path) -> list[Path]:
    """Return benchmark-comparison CSVs under a run directory."""
    return sorted(run_dir.glob("*_benchmark_comparison.csv"))


def _resolve_comparison_csv(run_dir: Path) -> Path:
    """Resolve the comparison CSV for a benchmark run directory."""
    preferred = run_dir / "full_benchmark_comparison.csv"
    if preferred.exists():
        return preferred

    match = _SUITE_RUN_RE.match(run_dir.name)
    if match is not None:
        suite_path = run_dir / f"{match.group('suite')}_benchmark_comparison.csv"
        if suite_path.exists():
            return suite_path

    candidates = _comparison_csv_candidates(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No benchmark CSV found under {run_dir}")
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Multiple benchmark CSVs found under {run_dir}: "
        f"{[path.name for path in candidates]!r}. Keep one comparison CSV or "
        "use a suite-suffixed run directory."
    )


def _resolve_run_dir(run_dir: str | None) -> Path:
    results_dir = BENCHMARKS_ROOT / "results"
    if run_dir is not None:
        candidate = Path(run_dir)
        if not candidate.is_absolute():
            candidate = results_dir / candidate
        return candidate

    runs = sorted(
        path
        for path in results_dir.glob("run_*")
        if path.is_dir() and _comparison_csv_candidates(path)
    )
    if not runs:
        raise FileNotFoundError(f"No benchmark run directories found under {results_dir}")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark result relationships")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Run directory name or absolute path. Defaults to the latest run_* "
            "directory with a benchmark CSV."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the relationship PDF and only write CSV/markdown artifacts.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    csv_path = _resolve_comparison_csv(run_dir)

    df = pd.read_csv(csv_path)
    artifacts = analyze_benchmark_relationships(
        df,
        run_dir,
        source_path=csv_path,
        include_plots=not args.no_plots,
    )

    print(f"Analyzed run: {run_dir}")
    for key, value in artifacts.as_dict().items():
        if value is not None:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
