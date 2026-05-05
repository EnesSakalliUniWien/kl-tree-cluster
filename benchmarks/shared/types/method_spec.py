"""MethodSpec dataclass for benchmarking (moved to types package)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .method_run_result import MethodRunResult


@dataclass(frozen=True)
class MethodSpec:
    name: str
    runner: Callable[..., "MethodRunResult"]
    param_grid: list[dict[str, object]]
