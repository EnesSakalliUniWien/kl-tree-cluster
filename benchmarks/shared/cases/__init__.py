"""
Benchmark test case configurations.

This package contains all test case configurations organized by category.
Import `get_default_test_cases()` to get all cases, or import specific
category modules for targeted testing.
"""

from .gaussian import GAUSSIAN_CASES
from .binary import BINARY_CASES
from .sbm import SBM_CASES
from .categorical import CATEGORICAL_CASES
from benchmarks.phylogenetic.cases import PHYLOGENETIC_CASES
from .overlapping import OVERLAPPING_CASES
from .real_data import REAL_DATA_CASES

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
    **OVERLAPPING_CASES,
    **REAL_DATA_CASES,
}


def _normalize_case_list(cases: list[dict], category: str) -> list[dict]:
    """Attach stable metadata and ensure names are present/unique in a category."""
    normalized: list[dict] = []
    seen_names: dict[str, int] = {}

    for idx, case in enumerate(cases, start=1):
        item = case.copy()
        item.setdefault("category", category)

        base_name = item.get("name")
        if not base_name:
            base_name = f"{category}_{idx}"

        seen_names[base_name] = seen_names.get(base_name, 0) + 1
        if seen_names[base_name] > 1:
            item["name"] = f"{base_name}__{seen_names[base_name]}"
        else:
            item["name"] = base_name

        normalized.append(item)

    return normalized


def get_default_test_cases() -> list[dict]:
    """Flatten all test case groups into a single list with unique case names."""
    flattened: list[dict] = []
    seen_global: dict[str, int] = {}

    for category, group in ALL_CASE_CATEGORIES.items():
        for case in _normalize_case_list(group, category):
            name = str(case["name"])
            seen_global[name] = seen_global.get(name, 0) + 1
            if seen_global[name] > 1:
                case["name"] = f"{name}__{seen_global[name]}"
            flattened.append(case)

    return flattened


def get_test_cases_by_category(category: str) -> list[dict]:
    """Get test cases for a specific category."""
    if category not in ALL_CASE_CATEGORIES:
        raise ValueError(
            f"Unknown category: {category}. Available: {list(ALL_CASE_CATEGORIES.keys())}"
        )
    return _normalize_case_list(ALL_CASE_CATEGORIES[category], category)


def list_categories() -> list[str]:
    """List all available test case categories."""
    return list(ALL_CASE_CATEGORIES.keys())
