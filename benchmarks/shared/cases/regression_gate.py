"""Fixed case list for the fast regression-gate benchmark.

This suite is intentionally smaller than the full benchmark. It focuses on
historically regression-sensitive KL cases so it can serve as a reliable,
fast gate in local development and CI.
"""

from __future__ import annotations

from . import get_default_test_cases

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

def _build_case_index() -> dict[str, dict]:
    """Return all benchmark cases keyed by normalized unique case name."""
    return {str(case["name"]): case for case in get_default_test_cases()}


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
