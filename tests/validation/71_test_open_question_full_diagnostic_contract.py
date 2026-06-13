from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.open_questions.full_diagnostic_contract import (
    CONTRACT_STATUS,
    REQUIRED_FIELDS,
    STUDY_ROLE,
    diagnostic_contract_summary,
    diagnostic_contract_table,
    run_open_question_full_diagnostic_contract,
    validate_diagnostic_contracts,
)


def test_diagnostic_contract_table_covers_all_open_questions() -> None:
    table = diagnostic_contract_table()

    assert table.shape[0] == 44
    assert set(table["question_id"]) == {f"Q{index}" for index in range(1, 45)}
    assert list(table.columns) == list(REQUIRED_FIELDS)
    assert table["diagnostic_contract_status"].eq(CONTRACT_STATUS).all()
    assert table["study_role"].eq(STUDY_ROLE).all()
    assert validate_diagnostic_contracts(table) == []


def test_diagnostic_contracts_are_decision_complete() -> None:
    table = diagnostic_contract_table()

    decision_columns = [
        "diagnostic_family",
        "diagnostic_scale",
        "existing_entrypoint",
        "required_inputs",
        "required_outputs",
        "acceptance_criteria",
        "blocks_method_claim_until",
    ]
    for column in decision_columns:
        assert not table[column].astype(str).str.strip().eq("").any(), column

    assert table["acceptance_criteria"].str.contains("Pass").all()
    assert table["blocks_method_claim_until"].str.len().min() > 20


def test_diagnostic_summary_reports_full_contract_status() -> None:
    summary = diagnostic_contract_summary()

    status_row = summary[
        summary["summary_type"].eq("diagnostic_contract_status")
        & summary["value"].eq(CONTRACT_STATUS)
    ].iloc[0]
    assert int(status_row["n_questions"]) == 44
    assert summary["study_role"].eq(STUDY_ROLE).all()


def test_run_open_question_full_diagnostic_contract_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "out"

        outputs = run_open_question_full_diagnostic_contract(output_dir=output_dir)

        assert set(outputs) == {"contract", "summary", "manifest"}
        contract = pd.read_csv(output_dir / "open_question_full_diagnostic_contract.csv")
        summary = pd.read_csv(output_dir / "open_question_full_diagnostic_summary.csv")
        assert contract.shape[0] == 44
        assert not summary.empty
        assert (output_dir / "manifest.json").exists()
