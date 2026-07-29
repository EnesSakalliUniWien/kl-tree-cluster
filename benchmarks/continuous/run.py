#!/usr/bin/env python3
"""Run the CI/local continuous benchmark gate."""

from __future__ import annotations

from benchmarks.shared.benchmark_runs.regression_gate_cli import run_regression_gate_cli


def main() -> None:
    run_regression_gate_cli(
        description="Run the continuously maintained benchmark gate.",
        case_help_label="continuous gate",
        completion_label="Continuous benchmark gate",
    )


if __name__ == "__main__":
    main()
