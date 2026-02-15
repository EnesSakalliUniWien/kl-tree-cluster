"""Method selection helpers for benchmark runner scripts."""

from __future__ import annotations

import os
from collections.abc import Mapping


def resolve_methods_from_env(
    method_specs: Mapping[str, object],
    *,
    env_var: str = "KL_TE_METHODS",
) -> list[str]:
    """Resolve method ids from an env var, defaulting to all available methods."""
    raw = (os.getenv(env_var) or "all").strip()
    if raw.lower() == "all":
        return list(method_specs.keys())

    selected = [m.strip() for m in raw.split(",") if m.strip()]
    if not selected:
        raise ValueError(
            f"{env_var} is empty. Provide comma-separated method ids or 'all'."
        )

    unknown = [m for m in selected if m not in method_specs]
    if unknown:
        available = ", ".join(sorted(method_specs.keys()))
        raise ValueError(
            f"Unknown methods in {env_var}: {unknown}. Available: {available}"
        )
    return selected


__all__ = ["resolve_methods_from_env"]
