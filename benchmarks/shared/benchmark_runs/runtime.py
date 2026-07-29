"""Runtime defaults shared by benchmark entrypoints."""

from __future__ import annotations

import os

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_single_thread_runtime_defaults() -> dict[str, str | None]:
    """Default numeric runtimes to one worker for reproducible benchmark gates."""
    for env_var in THREAD_ENV_VARS:
        os.environ.setdefault(env_var, "1")
    os.environ.setdefault("TBS_N_JOBS", "1")
    return {env_var: os.environ.get(env_var) for env_var in THREAD_ENV_VARS}


__all__ = ["THREAD_ENV_VARS", "apply_single_thread_runtime_defaults"]
