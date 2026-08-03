"""Command-line contracts shared by root tie-rank diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_proposal_panel_args(
    *,
    description: str | None,
    default_rows_path: Path,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the common proposal-feasibility input/output contract."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--proposal-feasibility-rows-path",
        type=Path,
        default=default_rows_path,
    )
    return parser.parse_args(argv)


__all__ = ["parse_proposal_panel_args"]
