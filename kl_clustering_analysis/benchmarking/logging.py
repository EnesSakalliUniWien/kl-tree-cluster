"""Small logging helpers for the benchmarking pipeline.

Functions are kept separate so they can be imported and reused elsewhere
without pulling in the entire `pipeline` module.
"""

from __future__ import annotations

import logging


def _default_pipeline_logger() -> logging.Logger:
    # Keep using the pipeline logger name to preserve existing logging
    # configuration and tests.
    return logging.getLogger("kl_clustering_analysis.benchmarking.pipeline")


def log_validation_start(n_cases: int, logger: logging.Logger | None = None) -> None:
    """Log the start of the validation process."""
    logger = logger or _default_pipeline_logger()
    logger.info("%s", "=" * 80)
    logger.info("CLUSTER ALGORITHM VALIDATION")
    logger.info("%s", "=" * 80)
    logger.info("Evaluating %d test cases.", n_cases)


def log_test_case_start(
    index: int, total: int, name: str, logger: logging.Logger | None = None
) -> None:
    """Log the start of a specific test case."""
    logger = logger or _default_pipeline_logger()
    logger.info("Running test case %d/%d: %s", index, total, name)


def log_validation_completion(
    total_runs: int, n_cases: int, logger: logging.Logger | None = None
) -> None:
    """Log the completion of the validation process."""
    logger = logger or _default_pipeline_logger()
    logger.info(
        "Completed %d validation runs across %d test cases.", total_runs, n_cases
    )
