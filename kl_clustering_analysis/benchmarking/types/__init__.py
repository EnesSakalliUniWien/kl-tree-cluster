"""Benchmarking types package.

Re-export small type dataclasses for convenient imports.
"""

from __future__ import annotations

from .method_spec import MethodSpec
from .method_run_result import MethodRunResult

__all__ = ["MethodSpec", "MethodRunResult"]
