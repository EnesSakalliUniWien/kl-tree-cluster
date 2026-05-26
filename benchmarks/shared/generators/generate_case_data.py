"""Dispatch benchmark case recipes to the matching data generator."""

from __future__ import annotations

from benchmarks.shared.generators.binary_cases import generate_binary_case
from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    CaseGenerator,
    require_case_value,
)
from benchmarks.shared.generators.categorical_cases import (
    generate_categorical_case,
    generate_phylogenetic_case,
    generate_temporal_evolution_case,
)
from benchmarks.shared.generators.gaussian_cases import (
    generate_blobs_case,
    generate_blobs_continuous_case,
    generate_blobs_quantile_case,
    generate_dimensional_gaussian_case,
    generate_dimensional_gaussian_continuous_case,
    generate_gaussian_outlier_case,
    generate_gaussian_outlier_continuous_case,
)
from benchmarks.shared.generators.graph_cases import generate_sbm_case
from benchmarks.shared.generators.preloaded_cases import generate_preloaded_case

GENERATOR_DISPATCH: dict[str, CaseGenerator] = {
    "binary": generate_binary_case,
    "blobs": generate_blobs_case,
    "blobs_continuous": generate_blobs_continuous_case,
    "blobs_quantile": generate_blobs_quantile_case,
    "dimensional_gaussian": generate_dimensional_gaussian_case,
    "dimensional_gaussian_continuous": generate_dimensional_gaussian_continuous_case,
    "gaussian_outliers": generate_gaussian_outlier_case,
    "gaussian_outliers_continuous": generate_gaussian_outlier_continuous_case,
    "sbm": generate_sbm_case,
    "categorical": generate_categorical_case,
    "phylogenetic": generate_phylogenetic_case,
    "temporal_evolution": generate_temporal_evolution_case,
    "preloaded": generate_preloaded_case,
}


def generate_case_data(test_case: dict) -> CaseDataResult:
    """Create matrix data, true labels, original features, and canonical metadata."""
    require_case_value(test_case, "name", "Benchmark case")
    generator = str(require_case_value(test_case, "generator", "Benchmark case"))
    if generator not in GENERATOR_DISPATCH:
        raise ValueError(f"Unknown generator: {generator}")
    generator_fn = GENERATOR_DISPATCH[generator]

    seed = None
    if generator != "preloaded":
        seed = int(require_case_value(test_case, "seed", generator))
    data_df, y, x_original, metadata = generator_fn(test_case, seed)

    if "category" in test_case:
        metadata["category"] = test_case["category"]
    if "baseline_case_name" in test_case:
        metadata["baseline_case_name"] = str(test_case["baseline_case_name"])
    if "representation_role" in test_case:
        metadata["representation_role"] = str(test_case["representation_role"])

    return data_df, y, x_original, metadata
