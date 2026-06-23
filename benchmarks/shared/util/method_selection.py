"""Method selection helpers for benchmark runner scripts."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any


def resolve_methods_from_env(
    method_specs: Mapping[str, object],
    *,
    env_var: str = "TBS_METHODS",
    default_methods: Sequence[str] | None = None,
) -> list[str]:
    """Resolve method ids from an env var.

    When the env var is unset, defaults to ``default_methods`` when provided,
    otherwise all available methods. The special value ``all`` expands to the
    same default set.
    """
    raw = (os.getenv(env_var) or "").strip()
    if not raw or raw.lower() == "all":
        if default_methods is not None:
            return list(default_methods)
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
    if not selected_methods:
        raise ValueError("At least one benchmark method must be selected.")
    for method_id in selected_methods:
        if method_id not in method_specs:
            raise ValueError(f"Unknown method: {method_id}")

    param_sets: dict[str, list[dict[str, object]]] = {}
    for method_id in selected_methods:
        if method_id in params_by_method:
            explicit_params = params_by_method[method_id]
            if not explicit_params:
                raise ValueError(
                    f"Method {method_id!r} was given an empty parameter grid."
                )
            param_sets[method_id] = explicit_params
        else:
            param_sets[method_id] = method_specs[method_id].param_grid
    return selected_methods, param_sets


__all__ = [
    "resolve_methods_from_env",
    "resolve_selected_methods_and_param_sets",
]
