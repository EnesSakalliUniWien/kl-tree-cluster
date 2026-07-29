from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.shared.benchmark_runs.regression_gate import (
    parse_methods,
    resolve_case_list,
    sort_regression_results,
)
from benchmarks.shared.benchmark_runs.smoke import SMOKE_SUBSET_NAMES, select_smoke_cases
from benchmarks.shared.cases import get_default_test_cases


def test_smoke_module_selects_declared_cases_in_default_order() -> None:
    all_cases = get_default_test_cases()
    subset = select_smoke_cases(all_cases)

    assert {case["name"] for case in subset} == SMOKE_SUBSET_NAMES
    assert [case["name"] for case in subset] == [
        case["name"] for case in all_cases if case["name"] in SMOKE_SUBSET_NAMES
    ]


def test_regression_gate_module_validates_methods_and_cases() -> None:
    assert parse_methods("tbs,kmeans") == ["tbs", "kmeans"]
    with pytest.raises(ValueError, match="Unknown methods"):
        parse_methods("not_a_method")

    cases = resolve_case_list("gauss_extreme_noise_3c,overlap_extreme_4c")
    assert [case["name"] for case in cases] == [
        "gauss_extreme_noise_3c",
        "overlap_extreme_4c",
    ]
    with pytest.raises(ValueError, match="Unknown regression-gate case names"):
        resolve_case_list("gauss_clear_medium,not_a_case")


def test_regression_gate_sort_keeps_case_and_method_order() -> None:
    cases = [{"name": "case_b"}, {"name": "case_a"}]
    methods = ["tbs", "kmeans"]
    rows = pd.DataFrame(
        [
            {"case_id": "case_a", "method": "kmeans", "test_case": 2},
            {"case_id": "case_b", "method": "kmeans", "test_case": 1},
            {"case_id": "case_a", "method": "tbs", "test_case": 2},
            {"case_id": "case_b", "method": "tbs", "test_case": 1},
        ]
    )

    sorted_rows = sort_regression_results(rows, cases, methods)

    assert list(zip(sorted_rows["case_id"], sorted_rows["method"], strict=True)) == [
        ("case_b", "tbs"),
        ("case_b", "kmeans"),
        ("case_a", "tbs"),
        ("case_a", "kmeans"),
    ]
