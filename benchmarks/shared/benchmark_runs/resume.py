"""Resume/grid coverage helpers for benchmark run orchestrators."""

from __future__ import annotations

import pandas as pd

from benchmarks.shared.benchmark_grid import benchmark_run_id


def compute_resume_coverage(
    existing_results: pd.DataFrame,
    expected_run_ids: list[str],
) -> tuple[set[int], dict[int, list[str]], int]:
    """Compute fully-complete and partial case coverage from an existing CSV."""
    if existing_results.empty:
        return set(), {}, 0

    if "test_case" not in existing_results.columns:
        return set(), {}, 0

    if "run_id" not in existing_results.columns:
        return set(), {}, 0

    expected_set = set(expected_run_ids)

    case_keys = pd.to_numeric(existing_results["test_case"], errors="coerce")
    run_ids_raw = existing_results["run_id"].astype(str)
    run_ids_norm = run_ids_raw.where(run_ids_raw.isin(expected_set))
    progress = pd.DataFrame({"case_key": case_keys, "run_id": run_ids_norm})
    progress = progress.dropna(subset=["case_key", "run_id"])
    if progress.empty:
        return set(), {}, 0

    progress["case_key"] = progress["case_key"].astype(int)

    run_ids_by_case = (
        progress.groupby("case_key")["run_id"].agg(lambda s: set(s.tolist())).to_dict()
    )

    completed_cases: set[int] = set()
    missing_run_ids_by_case: dict[int, list[str]] = {}
    for case_key, seen_run_ids in run_ids_by_case.items():
        missing = [run_id for run_id in expected_run_ids if run_id not in seen_run_ids]
        if not missing:
            completed_cases.add(int(case_key))
        else:
            missing_run_ids_by_case[int(case_key)] = missing

    return completed_cases, missing_run_ids_by_case, len(run_ids_by_case)


def run_ids_for_params(
    methods: list[str],
    param_sets: dict[str, list[dict[str, object]]],
) -> list[str]:
    """Return canonical benchmark run IDs for selected methods and grids."""
    return [
        benchmark_run_id(method_id, params)
        for method_id in methods
        for params in param_sets[method_id]
    ]


def filter_methods_and_params_by_run_ids(
    *,
    methods: list[str],
    param_sets: dict[str, list[dict[str, object]]],
    run_ids: set[str],
) -> tuple[list[str], dict[str, list[dict[str, object]]]]:
    """Restrict methods and grids to the requested run IDs."""
    filtered_methods: list[str] = []
    filtered_params: dict[str, list[dict[str, object]]] = {}
    for method_id in methods:
        selected_params = [
            params
            for params in param_sets[method_id]
            if benchmark_run_id(method_id, params) in run_ids
        ]
        if selected_params:
            filtered_methods.append(method_id)
            filtered_params[method_id] = selected_params
    return filtered_methods, filtered_params


__all__ = [
    "compute_resume_coverage",
    "filter_methods_and_params_by_run_ids",
    "run_ids_for_params",
]
