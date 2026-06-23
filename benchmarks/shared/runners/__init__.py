"""Benchmarking runner package.

Important: do not import optional-dependency runners (igraph/leidenalg/etc.) at
import time.

The benchmarking pipeline loads runners lazily via ``importlib``. Importing
optional runners here makes importing *any* runner (even TBS) fail when optional
dependencies aren't installed, because Python executes this package
``__init__.py`` before importing submodules (e.g. ``.tbs_runner``).
"""

from __future__ import annotations

__all__: list[str] = []
