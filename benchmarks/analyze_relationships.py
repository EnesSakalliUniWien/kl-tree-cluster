#!/usr/bin/env python3
"""Generate relationship analysis artifacts for an existing benchmark run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmarks.shared.relationship_analysis import analyze_benchmark_relationships


def _resolve_run_dir(run_dir: str | None) -> Path:
    results_dir = Path(__file__).resolve().parent / "results"
    if run_dir is not None:
        candidate = Path(run_dir)
        if not candidate.is_absolute():
            candidate = results_dir / candidate
        return candidate

    runs = sorted(path for path in results_dir.glob("run_*Z") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No benchmark run directories found under {results_dir}")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark result relationships")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory name or absolute path. Defaults to the latest run_*Z directory.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the relationship PDF and only write CSV/markdown artifacts.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    csv_path = run_dir / "full_benchmark_comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No benchmark CSV found at {csv_path}")

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
