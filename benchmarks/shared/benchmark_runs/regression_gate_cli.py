"""Shared CLI helpers for regression-style benchmark gates."""

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


def build_regression_gate_parser(
    *,
    description: str,
    case_help_label: str,
) -> argparse.ArgumentParser:
    """Build a parser for a regression-case benchmark gate command."""
    parser = argparse.ArgumentParser(description=description)
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
        help=f"Optional comma-separated override of case names from the {case_help_label}.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help=f"Print the {case_help_label} case names and exit.",
    )
    return parser


def run_regression_gate_cli(
    *,
    description: str,
    case_help_label: str,
    completion_label: str,
) -> None:
    """Parse arguments, run a regression-case gate, and print its summary."""
    args = build_regression_gate_parser(
        description=description,
        case_help_label=case_help_label,
    ).parse_args()

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

    print(f"{completion_label} complete in {result.elapsed_sec:.2f}s")
    print(f"Results: {result.results_csv}")
    print(f"Metadata: {result.metadata_json}")
    print_regression_summary(result.results, methods)


__all__ = ["build_regression_gate_parser", "run_regression_gate_cli"]
