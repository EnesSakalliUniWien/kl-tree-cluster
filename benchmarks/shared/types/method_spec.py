"""MethodSpec dataclass for benchmarking (moved to types package)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MethodSpec:
    name: str
    runner: Callable[..., "MethodRunResult"]
    param_grid: list[dict[str, object]]
