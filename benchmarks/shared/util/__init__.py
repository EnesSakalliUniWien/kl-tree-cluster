"""Utility package for shared benchmark helpers.

This module intentionally avoids eager imports to prevent circular-import
side effects during subprocess initialization (notably in runner registration).
Submodules are loaded lazily via ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_SUBMODULES = {
    "core",
    "decomposition",
    "case_inputs",
    "time",
    "method_selection",
    "case_execution",
    "case_run",
    "method_execution",
    "pdf",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_LAZY_SUBMODULES)
