"""Command-line contracts shared by traversal diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from benchmarks.validation.statistics.selected_edge_type1_geometry import parse_names


def parse_traversal_audit_args(
    *,
    description: str | None,
    default_case_names: tuple[str, ...],
    default_methods: tuple[str, ...],
    default_significance_level: float,
    default_edge_alpha: float,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the common suite, case, method, and alpha traversal contract."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--case-names", type=parse_names, default=default_case_names)
    parser.add_argument("--methods", type=parse_names, default=default_methods)
    parser.add_argument(
        "--significance-level",
        type=float,
        default=default_significance_level,
    )
    parser.add_argument("--edge-alpha", type=float, default=default_edge_alpha)
    return parser.parse_args(argv)


__all__ = ["parse_traversal_audit_args"]
