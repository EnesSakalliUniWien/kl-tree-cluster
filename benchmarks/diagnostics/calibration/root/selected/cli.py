"""Command-line contracts shared by selected-root diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_action_support_panel_args(
    *,
    description: str | None,
    default_rows_path: Path,
    default_population_law_status: str,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the common selected-root action-support contract."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joined-feasibility-rows-path",
        type=Path,
        default=default_rows_path,
    )
    parser.add_argument(
        "--h-u-population-law-status",
        default=default_population_law_status,
    )
    return parser.parse_args(argv)


__all__ = ["parse_action_support_panel_args"]
