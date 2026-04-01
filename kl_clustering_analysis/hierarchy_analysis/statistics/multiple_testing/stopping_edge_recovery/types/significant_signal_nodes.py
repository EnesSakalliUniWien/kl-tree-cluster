from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _SignificantSignalNodes:
    """Tested-significant node ids aligned with their corrected p-values."""

    node_ids: tuple[str, ...]
    p_values: tuple[float, ...]
