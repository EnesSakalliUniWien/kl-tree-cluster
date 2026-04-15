"""Fixed case list for the fast regression-gate benchmark.

This suite is intentionally smaller than the full benchmark. It focuses on
historically regression-sensitive KL cases so it can serve as a reliable,
fast gate in local development and CI.
"""

from __future__ import annotations

from benchmarks.phylogenetic.cases import PHYLOGENETIC_CASES

from .binary import BINARY_CASES
from .categorical import CATEGORICAL_CASES
from .dimensionality import DIMENSIONALITY_CASES
from .gaussian import GAUSSIAN_CASES
from .outliers import OUTLIER_CASES
from .overlapping import OVERLAPPING_CASES
from .sbm import SBM_CASES

REGRESSION_GATE_CASE_NAMES: tuple[str, ...] = (
    "gauss_extreme_noise_3c",
    "gauss_extreme_noise_highd",
    "dim_consolidated_4c_24f",
    "dim_consolidated_4c_272f",
    "dim_diffuse_6c_36f",
    "sbm_moderate",
    "sbm_hard",
    "cat_highcard_20cat_4c",
    "cat_overlap_3cat_4c",
    "phylo_dna_8taxa_med_mut",
    "phylo_protein_4taxa",
    "phylo_protein_12taxa",
    "phylo_conserved_4taxa",
    "phylo_conserved_8taxa",
    "overlap_heavy_4c_med_feat",
    "overlap_extreme_4c",
    "overlap_unbal_4c_small",
)

_ALL_CASE_CATEGORIES = {
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
    """Attach stable metadata and ensure names are present within a category."""
    normalized: list[dict] = []
    seen_names: dict[str, int] = {}

    for idx, case in enumerate(cases, start=1):
        item = case.copy()
        item.setdefault("category", category)

        base_name = item.get("name") or f"{category}_{idx}"
        seen_names[base_name] = seen_names.get(base_name, 0) + 1
        if seen_names[base_name] > 1:
            item["name"] = f"{base_name}__{seen_names[base_name]}"
        else:
            item["name"] = base_name

        normalized.append(item)

    return normalized


def _build_case_index() -> dict[str, dict]:
    """Return all benchmark cases keyed by normalized unique case name."""
    indexed: dict[str, dict] = {}
    seen_global: dict[str, int] = {}

    for category, group in _ALL_CASE_CATEGORIES.items():
        for case in _normalize_case_list(group, category):
            name = str(case["name"])
            seen_global[name] = seen_global.get(name, 0) + 1
            if seen_global[name] > 1:
                case["name"] = f"{name}__{seen_global[name]}"
            indexed[str(case["name"])] = case

    return indexed


def get_regression_gate_case_names() -> list[str]:
    """Return the stable ordered case names for the regression gate."""
    return list(REGRESSION_GATE_CASE_NAMES)


def get_regression_gate_test_cases() -> list[dict]:
    """Resolve the ordered regression-gate case list from the shared registry."""
    case_index = _build_case_index()
    missing = [name for name in REGRESSION_GATE_CASE_NAMES if name not in case_index]
    if missing:
        raise ValueError(
            "Regression gate references unknown benchmark cases: "
            + ", ".join(missing)
        )

    return [case_index[name].copy() for name in REGRESSION_GATE_CASE_NAMES]


__all__ = [
    "REGRESSION_GATE_CASE_NAMES",
    "get_regression_gate_case_names",
    "get_regression_gate_test_cases",
]
