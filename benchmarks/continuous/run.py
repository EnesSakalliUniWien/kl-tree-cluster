#!/usr/bin/env python3
"""Run the CI/local continuous benchmark gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.shared.benchmark_runs.regression_gate import (
    parse_methods,
    print_regression_summary,
    resolve_case_list,
    run_regression_gate,
)
from benchmarks.shared.cases.regression_gate import get_regression_gate_case_names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the continuously maintained benchmark gate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to benchmarks/results/regression_gate_<timestamp>/",
    )
    parser.add_argument(
        "--methods",
        default="tbs",
        help="Comma-separated method ids. Defaults to 'tbs'.",
    )
    parser.add_argument(
        "--case-names",
        default="",
        help="Optional comma-separated override of case names from the continuous gate.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the continuous-gate case names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_cases:
        for name in get_regression_gate_case_names():
            print(name)
        return

    methods = parse_methods(args.methods)
    test_cases = resolve_case_list(args.case_names)
    result = run_regression_gate(
        methods=methods,
        test_cases=test_cases,
        output_dir=args.output_dir,
    )

    print(f"Continuous benchmark gate complete in {result.elapsed_sec:.2f}s")
    print(f"Results: {result.results_csv}")
    print(f"Metadata: {result.metadata_json}")
    print_regression_summary(result.results, methods)


if __name__ == "__main__":
    main()
