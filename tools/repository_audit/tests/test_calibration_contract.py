from pathlib import Path

import pytest
from tbs_repo_audit.calibration_contract import (
    build_calibration_contract_report,
    write_calibration_contract_report,
)

HEADER = (
    "source_case_id,parent,left,right,degrees_of_freedom,"
    "sibling_null_weight,is_edge_blocked,is_role_supported\n"
)


def _records(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "records.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_support_is_read_from_the_stamp_rather_than_re_derived(
    tmp_path: Path,
) -> None:
    records = _records(tmp_path, "toy,p,a,b,1,1,false,false\n")

    summary = build_calibration_contract_report(records)["summary"]

    assert summary["role_supported_record_count"] == 0


def test_repeated_parent_within_a_group_is_rejected(tmp_path: Path) -> None:
    records = _records(
        tmp_path,
        "toy,zeta,a,b,1,1,true,true\ntoy,zeta,c,d,1,1,false,true\n",
    )

    with pytest.raises(ValueError, match="zeta"):
        build_calibration_contract_report(records)


def test_nonfinite_numeric_records_are_counted_rather_than_dropped(
    tmp_path: Path,
) -> None:
    records = _records(
        tmp_path,
        "toy,p,a,b,nan,1,true,true\ntoy,q,c,d,1,1,true,true\n",
    )

    summary = build_calibration_contract_report(records)["summary"]

    assert summary["raw_record_count"] == 2
    assert summary["role_supported_record_count"] == 1
    assert summary["nonfinite_record_count"] == 1


def test_roles_partition_supported_records_and_name_each_parent(
    tmp_path: Path,
) -> None:
    records = _records(
        tmp_path,
        "toy,root,stopped,root_leaf,1,1,false,true\n"
        "toy,stopped,nested,frontier_leaf,1,1,true,true\n"
        "toy,nested,left_leaf,right_leaf,1,1,true,true\n",
    )

    group = build_calibration_contract_report(records)["groups"][0]

    assert group["tested_null_parents"] == ["root"]
    assert group["stopped_frontier_parents"] == ["stopped"]
    assert group["nested_blocked_parents"] == ["nested"]
    assert (
        group["tested_null_record_count"]
        + group["stopped_frontier_record_count"]
        + group["nested_blocked_record_count"]
        == group["role_supported_record_count"]
    )
    assert group["structural_calibration_record_count"] == 2


def test_markdown_reports_every_role_count(tmp_path: Path) -> None:
    records = _records(
        tmp_path,
        "toy,root,stopped,root_leaf,1,1,false,true\n"
        "toy,stopped,nested,frontier_leaf,1,1,true,true\n"
        "toy,nested,left_leaf,right_leaf,1,1,true,true\n",
    )
    report = build_calibration_contract_report(records)

    write_calibration_contract_report(
        report,
        output=tmp_path / "report.json",
        markdown_output=tmp_path / "report.md",
    )

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "- Tested-null records: `1`" in text
    assert "- Stopped-subtree frontiers: `1`" in text
    assert "- Nested blocked descendants: `1`" in text
    assert "- Structural calibration records: `2`" in text
    assert "- Non-finite records: `0`" in text


def test_parse_error_names_the_true_csv_line(tmp_path: Path) -> None:
    records = _records(
        tmp_path,
        "g1,p,a,b,1,1,false,true\n"
        "g2,q,c,d,1,1,false,true\n"
        "g1,r,e,f,1,1,false,MAYBE\n",
    )

    with pytest.raises(ValueError, match="line=4"):
        build_calibration_contract_report(records)


def test_unparsable_numeric_names_its_column_and_line(tmp_path: Path) -> None:
    records = _records(tmp_path, "toy,p,a,b,,1,true,true\n")

    with pytest.raises(ValueError, match="degrees_of_freedom.*line=2"):
        build_calibration_contract_report(records)


def test_header_only_records_report_no_groups(tmp_path: Path) -> None:
    records = _records(tmp_path, "")

    report = build_calibration_contract_report(records)

    assert report["groups"] == []
    assert report["summary"]["group_count"] == 0
    assert report["summary"]["raw_record_count"] == 0


def test_repeated_parent_across_groups_is_allowed(tmp_path: Path) -> None:
    records = _records(
        tmp_path,
        "case_a,zeta,a,b,1,1,true,true\ncase_b,zeta,c,d,1,1,true,true\n",
    )

    report = build_calibration_contract_report(records)

    assert report["summary"]["group_count"] == 2


def test_records_missing_the_stamp_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_text(
        "source_case_id,parent,left,right,degrees_of_freedom,"
        "sibling_null_weight,is_null_like,is_edge_blocked\n"
        "toy,p,a,b,1,1,true,false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="is_role_supported"):
        build_calibration_contract_report(path)
