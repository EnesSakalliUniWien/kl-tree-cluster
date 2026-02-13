"""Environment variable parsing helpers for benchmark runners/plots."""

from __future__ import annotations

import os


def get_env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean environment flags like 1/true/yes/on."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_env_int(name: str, default: int) -> int:
    """Parse integer environment variables with a safe default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


__all__ = ["get_env_bool", "get_env_int"]
