#!/usr/bin/env python3
"""Run the fast regression-gate benchmark suite.

The gate is a fixed, historically sensitive subset of the full benchmark and
is intended for fast regression detection rather than exhaustive evaluation.
"""

from __future__ import annotations

from benchmarks.shared.benchmark_runs.regression_gate_cli import run_regression_gate_cli


def main() -> None:
    run_regression_gate_cli(
        description="Run the fixed regression-gate benchmark suite.",
        case_help_label="regression gate",
        completion_label="Regression gate",
    )


if __name__ == "__main__":
    main()
