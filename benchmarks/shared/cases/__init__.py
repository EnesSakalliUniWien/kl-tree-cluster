"""
Benchmark test case configurations.

This package contains all test case configurations organized by category.
Import `get_default_test_cases()` to get all cases, or import specific
category modules for targeted testing.
"""

from .binary import BINARY_CASES
from .categorical import CATEGORICAL_CASES
from .dimensionality import DIMENSIONALITY_CASES
from .gaussian import GAUSSIAN_CASES
from .outliers import OUTLIER_CASES
from .overlapping import OVERLAPPING_CASES
from .phylogenetic import PHYLOGENETIC_CASES
from .sbm import SBM_CASES

BINARY_BENCHMARK_GENERATORS = frozenset({"binary"})
DISCRETIZED_GAUSSIAN_BENCHMARK_GENERATORS = frozenset(
    {
        "blobs",
        "blobs_quantile",
        "dimensional_gaussian",
        "gaussian_outliers",
    }
)
CONTINUOUS_BENCHMARK_GENERATORS = frozenset(
    {
        "blobs_continuous",
        "dimensional_gaussian_continuous",
        "gaussian_outliers_continuous",
    }
)
CATEGORICAL_BENCHMARK_GENERATORS = frozenset(
    {
        "categorical",
        "blobs_quantile",
        "phylogenetic",
        "temporal_evolution",
    }
)
GRAPH_BENCHMARK_GENERATORS = frozenset({"sbm"})

BENCHMARK_CASE_SUITES = frozenset(
    {
        "full",
        "binary",
        "categorical",
        "continuous",
        "discretized_gaussian",
        "graph",
    }
)

# Small test cases for quick smoke tests
SMALL_TEST_CASES = [
    {
        "name": "clear",
        "generator": "blobs",
        "n_samples": 24,
        "n_features": 12,
        "n_clusters": 3,
        "cluster_std": 0.4,
        "seed": 0,
        "category": "small_smoke",
    },
    {
        "name": "moderate",
        "generator": "blobs",
        "n_samples": 30,
        "n_features": 16,
        "n_clusters": 3,
        "cluster_std": 1.0,
        "seed": 1,
        "category": "small_smoke",
    },
    {
        "name": "noisy",
        "generator": "blobs",
        "n_samples": 30,
        "n_features": 16,
        "n_clusters": 3,
        "cluster_std": 1.6,
        "seed": 2,
        "category": "small_smoke",
    },
]

# All test case categories
ALL_CASE_CATEGORIES = {
    **GAUSSIAN_CASES,
    **DIMENSIONALITY_CASES,
    **OUTLIER_CASES,
    **BINARY_CASES,
    **SBM_CASES,
    **CATEGORICAL_CASES,
    **PHYLOGENETIC_CASES,
    **OVERLAPPING_CASES,
}


def _normalize_case_list(cases: list[dict], category: str) -> list[dict]:
    """Attach stable metadata and ensure names are present/unique in a category."""
    normalized: list[dict] = []
    seen_names: dict[str, int] = {}

    for idx, case in enumerate(cases, start=1):
        item = case.copy()
        item["category"] = category
        base_name = item["name"]

        if base_name not in seen_names:
            seen_names[base_name] = 0
        seen_names[base_name] += 1
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
            if name not in seen_global:
                seen_global[name] = 0
            seen_global[name] += 1
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


def get_test_cases_by_suite(suite: str) -> list[dict]:
    """Return benchmark cases for a mathematical input-contract suite."""
    normalized_suite = str(suite).strip().lower()
    if normalized_suite not in BENCHMARK_CASE_SUITES:
        raise ValueError(
            f"Unknown benchmark suite: {suite!r}. "
            f"Available: {sorted(BENCHMARK_CASE_SUITES)}"
        )

    all_cases = get_default_test_cases()
    if normalized_suite == "full":
        return all_cases

    if normalized_suite == "binary":
        allowed_generators = BINARY_BENCHMARK_GENERATORS
    elif normalized_suite == "categorical":
        allowed_generators = CATEGORICAL_BENCHMARK_GENERATORS
    elif normalized_suite == "continuous":
        allowed_generators = CONTINUOUS_BENCHMARK_GENERATORS
    elif normalized_suite == "discretized_gaussian":
        allowed_generators = DISCRETIZED_GAUSSIAN_BENCHMARK_GENERATORS
    elif normalized_suite == "graph":
        allowed_generators = GRAPH_BENCHMARK_GENERATORS
    else:
        raise AssertionError(f"Unhandled benchmark suite: {normalized_suite!r}")

    return [
        case
        for case in all_cases
        if str(case["generator"]) in allowed_generators
    ]
