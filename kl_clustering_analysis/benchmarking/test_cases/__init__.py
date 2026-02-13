"""
Benchmark test case configurations.

This package contains all test case configurations organized by category.
Import `get_default_test_cases()` to get all cases, or import specific
category modules for targeted testing.
"""

from benchmarks.shared.cases.gaussian import GAUSSIAN_CASES
from benchmarks.shared.cases.binary import BINARY_CASES
from benchmarks.shared.cases.sbm import SBM_CASES
from benchmarks.shared.cases.categorical import CATEGORICAL_CASES
from benchmarks.phylogenetic.cases import PHYLOGENETIC_CASES
from .temporal import TEMPORAL_CASES
from benchmarks.shared.cases.overlapping import OVERLAPPING_CASES

# Small test cases for quick smoke tests
SMALL_TEST_CASES = [
    {
        "name": "clear",
        "n_samples": 24,
        "n_features": 12,
        "n_clusters": 3,
        "cluster_std": 0.4,
        "seed": 0,
    },
    {
        "name": "moderate",
        "n_samples": 30,
        "n_features": 16,
        "n_clusters": 3,
        "cluster_std": 1.0,
        "seed": 1,
    },
    {
        "name": "noisy",
        "n_samples": 30,
        "n_features": 16,
        "n_clusters": 3,
        "cluster_std": 1.6,
        "seed": 2,
    },
]

# All test case categories
ALL_CASE_CATEGORIES = {
    **GAUSSIAN_CASES,
    **BINARY_CASES,
    **SBM_CASES,
    **CATEGORICAL_CASES,
    **PHYLOGENETIC_CASES,
    **TEMPORAL_CASES,
    **OVERLAPPING_CASES,
}


def get_default_test_cases() -> list[dict]:
    """Flatten all test case groups into a single list."""
    return [case.copy() for group in ALL_CASE_CATEGORIES.values() for case in group]


def get_test_cases_by_category(category: str) -> list[dict]:
    """Get test cases for a specific category."""
    if category not in ALL_CASE_CATEGORIES:
        raise ValueError(
            f"Unknown category: {category}. Available: {list(ALL_CASE_CATEGORIES.keys())}"
        )
    return [case.copy() for case in ALL_CASE_CATEGORIES[category]]


def list_categories() -> list[str]:
    """List all available test case categories."""
    return list(ALL_CASE_CATEGORIES.keys())
