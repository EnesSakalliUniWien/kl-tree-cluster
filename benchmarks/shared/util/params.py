"""Parameter formatting helpers shared across benchmark modules."""

from __future__ import annotations


def format_params_for_display(params: dict[str, object]) -> str:
    """Create a stable human-readable parameter string."""
    if not params:
        return ""
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


__all__ = ["format_params_for_display"]
