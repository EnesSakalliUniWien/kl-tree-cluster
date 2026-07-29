"""Execution parameter parsing helpers for benchmark runners."""

from __future__ import annotations

from typing import Any


def coerce_bool_param(value: Any, *, name: str) -> bool:
    """Parse bool-like benchmark params without treating non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value; got {value!r}.")


__all__ = ["coerce_bool_param"]
