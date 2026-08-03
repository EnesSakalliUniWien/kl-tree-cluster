"""Command-line contract for selected-hierarchy diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

TARGET_MODES = (
    "root",
    "strongest",
    "median_parent_size",
    "non_root_strongest",
    "non_root_median_parent_size",
)
CONTEXT_MATCH_MODES = (
    "any",
    "projection",
    "projection_and_parent_size",
    "projection_parent_size_depth",
)


def parse_selected_hierarchy_args(
    argv: Sequence[str] | None = None,
    *,
    description: str | None,
    default_case_names: tuple[str, ...],
    default_seed: int,
) -> argparse.Namespace:
    """Parse the common selected-hierarchy simulation contract."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--case-names", default=",".join(default_case_names))
    parser.add_argument("--n-replicates", type=int, default=25)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="root")
    parser.add_argument(
        "--context-match",
        choices=CONTEXT_MATCH_MODES,
        default="projection",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


__all__ = ["CONTEXT_MATCH_MODES", "TARGET_MODES", "parse_selected_hierarchy_args"]
