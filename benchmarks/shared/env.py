"""Environment variable parsing helpers for benchmark runners/plots."""

from __future__ import annotations

import os


def get_env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean environment flags like 1/true/yes/on."""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean flag (1/true/yes/on or 0/false/no/off), got {{raw!r}}."
    )


def get_env_int(name: str, default: int) -> int:
    """Parse integer environment variables."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


def resolve_aws_batch_shard_index(
    explicit_index: int | None,
    environ: dict[str, str] | None = None,
) -> int:
    """Resolve a zero-based shard index from CLI input or AWS Batch array state."""
    if explicit_index is not None:
        if explicit_index < 0:
            raise ValueError(f"shard_index must be non-negative; got {explicit_index!r}.")
        return int(explicit_index)

    environment = os.environ if environ is None else environ
    raw_index = environment.get("AWS_BATCH_JOB_ARRAY_INDEX")
    if raw_index is None:
        raise ValueError(
            "Shard index is required. Pass --shard-index outside AWS Batch, or "
            "run as an AWS Batch array job with AWS_BATCH_JOB_ARRAY_INDEX."
        )
    shard_index = int(raw_index)
    if shard_index < 0:
        raise ValueError(f"AWS_BATCH_JOB_ARRAY_INDEX must be non-negative; got {raw_index!r}.")
    return shard_index


__all__ = ["get_env_bool", "get_env_int", "resolve_aws_batch_shard_index"]
