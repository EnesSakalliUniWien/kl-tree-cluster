"""Method selection helpers for benchmark runner scripts."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any


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


def resolve_selected_methods_and_param_sets(
    *,
    methods: list[str] | None,
    method_params: dict[str, list[dict[str, object]]] | None,
    default_methods: Sequence[str],
    method_specs: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[dict[str, object]]]]:
    """Validate selected methods and resolve each method's parameter grid."""
    selected_methods = list(methods) if methods is not None else list(default_methods)
    params_by_method = method_params or {}
    for method_id in selected_methods:
        if method_id not in method_specs:
            raise ValueError(f"Unknown method: {method_id}")

    param_sets = {
        method_id: (params_by_method.get(method_id) or method_specs[method_id].param_grid)
        for method_id in selected_methods
    }
    return selected_methods, param_sets


__all__ = [
    "resolve_methods_from_env",
    "resolve_selected_methods_and_param_sets",
]
