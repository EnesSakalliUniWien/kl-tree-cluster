"""Command-line contracts shared across calibration diagnostic families."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_null_calibration_panel_args(
    argv: Sequence[str] | None = None,
    *,
    description: str | None,
) -> argparse.Namespace:
    """Parse the common null-role panel validation contract."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--min-rows", type=int, default=30)
    return parser.parse_args(argv)


__all__ = ["parse_null_calibration_panel_args"]
