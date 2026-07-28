from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.sibling.nulls.runner_support import (
    prepare_sibling_diagnostic,
    select_sibling_target_rows,
)
from benchmarks.diagnostics.runner_support import (
    PROJECT_ROOT,
    configure_serial_runtime,
    create_result_directory,
    parse_csv_values,
    resolve_classified_cases,
)
from benchmarks.shared.cases.regression_gate import get_regression_gate_test_cases


def test_runner_values_and_serial_defaults_have_one_contract() -> None:
    assert parse_csv_values(" first, ,second ") == ("first", "second")

    environ = {"OMP_NUM_THREADS": "4"}
    configure_serial_runtime(environ)

    assert environ["OMP_NUM_THREADS"] == "4"
    assert environ["OPENBLAS_NUM_THREADS"] == "1"
    assert environ["TBS_N_JOBS"] == "1"


def test_resolve_classified_cases_joins_validated_records(tmp_path: Path) -> None:
    case_name = str(get_regression_gate_test_cases()[0]["name"])
    classification_path = tmp_path / "classification.csv"
    pd.DataFrame.from_records(
        [
            {
                "case_id": case_name,
                "failure_class": "gate_under_split",
                "tbs_ari": 0.25,
            },
            {
                "case_id": "not_selected",
                "failure_class": "solved",
                "tbs_ari": 1.0,
            },
        ]
    ).to_csv(classification_path, index=False)

    selection = resolve_classified_cases(
        suite="regression_gate",
        classification_csv=classification_path,
        failure_classes=("gate_under_split",),
        case_names=(case_name,),
        required_columns=("tbs_ari",),
    )

    assert selection.classification_path == classification_path
    assert len(selection.cases) == 1
    assert selection.cases[0].case["name"] == case_name
    assert selection.cases[0].classification["tbs_ari"] == pytest.approx(0.25)


def test_resolve_classified_cases_rejects_missing_contract_columns(
    tmp_path: Path,
) -> None:
    classification_path = tmp_path / "classification.csv"
    pd.DataFrame.from_records(
        [{"case_id": "case", "failure_class": "gate_under_split"}]
    ).to_csv(classification_path, index=False)

    with pytest.raises(ValueError, match="tbs_ari"):
        resolve_classified_cases(
            suite="regression_gate",
            classification_csv=classification_path,
            failure_classes=("gate_under_split",),
            case_names=(),
            required_columns=("tbs_ari",),
        )


def test_result_directory_uses_the_project_results_root(tmp_path: Path) -> None:
    assert PROJECT_ROOT == Path(__file__).resolve().parents[5]

    output_dir = create_result_directory(
        None,
        study_slug="sibling_null_test",
        project_root=tmp_path,
        timestamp="20260728_120000Z",
    )

    assert output_dir == (
        tmp_path
        / "benchmarks"
        / "results"
        / "sibling_null_test_20260728_120000Z"
    )
    assert output_dir.is_dir()


def test_prepare_sibling_diagnostic_preserves_observable_table_contract() -> None:
    case = get_regression_gate_test_cases()[0]
    prepared = prepare_sibling_diagnostic(
        case,
        {"failure_class": "gate_under_split", "tbs_ari": 0.0},
        max_contributors=2,
        contributors_for_all_crossings=False,
    )

    expected_prefix = [
        "case_id",
        "failure_class",
        "tree_distance_metric",
        "tree_distance_source",
        "tree_linkage_method",
    ]
    assert prepared.tables.targets.columns[:5].tolist() == expected_prefix
    assert prepared.tables.summary.columns[:5].tolist() == expected_prefix
    assert not prepared.tables.targets.empty
    assert len(prepared.tables.summary) == 1
    assert prepared.tables.targets["case_id"].unique().tolist() == [case["name"]]


def test_target_mode_selection_is_explicit() -> None:
    targets = pd.DataFrame.from_records(
        [
            {
                "parent": "N0",
                "blocker_candidate": True,
                "current_blocks_at_alpha": False,
            },
            {
                "parent": "N1",
                "blocker_candidate": False,
                "current_blocks_at_alpha": True,
            },
        ]
    )

    blockers = select_sibling_target_rows(targets, target_mode="blockers")
    current = select_sibling_target_rows(targets, target_mode="current_blocks")
    all_focal = select_sibling_target_rows(targets, target_mode="all_focal")

    assert blockers["parent"].tolist() == ["N0"]
    assert current["parent"].tolist() == ["N1"]
    assert all_focal["parent"].tolist() == ["N0", "N1"]
    with pytest.raises(ValueError, match="Unknown target_mode"):
        select_sibling_target_rows(targets, target_mode="implicit")
