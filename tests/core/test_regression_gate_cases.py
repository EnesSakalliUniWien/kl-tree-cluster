from benchmarks.shared.cases.regression_gate import (
    REGRESSION_GATE_CASE_NAMES,
    get_regression_gate_test_cases,
)


def test_regression_gate_cases_resolve_in_stable_order():
    cases = get_regression_gate_test_cases()

    assert [case["name"] for case in cases] == list(REGRESSION_GATE_CASE_NAMES)
    assert len(cases) == 17
    assert len({case["name"] for case in cases}) == len(cases)


def test_regression_gate_cases_include_category_metadata():
    cases = get_regression_gate_test_cases()

    assert all(case.get("category") for case in cases)
