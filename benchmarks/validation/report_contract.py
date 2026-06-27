"""Shared validation-report contract helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def validate_common_run_inputs(
    *,
    n_replicates: int,
    alpha: float,
    base_seed: int,
    code_commit: str,
    git_worktree_status: Sequence[str],
    run_command: str,
    ridge: float,
) -> None:
    """Validate common simulation-run metadata used by validation reports."""
    if n_replicates < 2:
        raise ValueError("n_replicates must be at least 2.")
    if not math.isfinite(float(alpha)) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must be in (0, 1). Got {alpha!r}.")
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative. Got {base_seed!r}.")
    if not code_commit:
        raise ValueError("code_commit must be non-empty.")
    if isinstance(git_worktree_status, str) or not isinstance(
        git_worktree_status,
        Sequence,
    ):
        raise ValueError("git_worktree_status must be a sequence of status lines.")
    if any(not isinstance(line, str) for line in git_worktree_status):
        raise ValueError("git_worktree_status entries must be strings.")
    if not run_command:
        raise ValueError("run_command must be non-empty.")
    if not math.isfinite(float(ridge)) or ridge < 0.0:
        raise ValueError(f"ridge must be finite and non-negative. Got {ridge!r}.")


def completed_target_entry(
    target: Any,
    results: Sequence[Mapping[str, Any]],
    *,
    extra_grids: Mapping[str, Any],
    primary_endpoint: str,
) -> dict[str, Any]:
    """Return the common completed-target envelope for validation evidence."""
    return {
        "target_id": target.target_id,
        "display_name": target.display_name,
        "validation_question": target.validation_question,
        "required_output_fields": list(target.required_output_fields),
        "evidence_status": "complete",
        "evidence": {
            "status": "complete",
            "source_path": None,
            "metrics": {
                "simulation_grid": [result["setting"] for result in results],
                "primary_endpoint": primary_endpoint,
                "results": list(results),
                "limitations": list(target.limitations),
                **dict(extra_grids),
            },
            "missing_required_fields": [],
        },
    }


def validate_complete_report_context(
    report: Mapping[str, Any],
    errors: list[str],
    *,
    primary_endpoint: str,
) -> None:
    """Validate common top-level fields required by complete validation reports."""
    required_top_level_types = {
        "created_utc": str,
        "code_commit": str,
        "git_worktree_status": list,
        "run_command": str,
        "random_seed_policy": str,
        "base_seed": int,
        "n_replicates": int,
        "alpha": float,
        "primary_endpoint": str,
    }
    for key, expected_type in required_top_level_types.items():
        if key not in report:
            errors.append(f"{key} is required when evidence is complete")
            continue
        value = report[key]
        if key == "alpha":
            if not isinstance(value, int | float):
                errors.append("alpha must be numeric when evidence is complete")
            elif not 0.0 < float(value) < 1.0:
                errors.append("alpha must be in (0, 1) when evidence is complete")
            continue
        if not isinstance(value, expected_type):
            errors.append(
                f"{key} must be {expected_type.__name__} when evidence is complete"
            )
    if report.get("primary_endpoint") != primary_endpoint:
        errors.append(f"primary_endpoint must be {primary_endpoint!r}")


__all__ = [
    "completed_target_entry",
    "validate_common_run_inputs",
    "validate_complete_report_context",
]
