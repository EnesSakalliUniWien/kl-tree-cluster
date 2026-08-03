import json
from pathlib import Path

import pytest
import tbs_repo_audit.cli as cli
from tbs_repo_audit.cli import (
    CHECK_COMMANDS,
    CheckResult,
    _coverage_command,
    _coverage_json_command,
    _mutation_target,
    _repo_root,
    _run_generator_geometry_audit,
    _run_mutation,
    main,
)


def test_repo_root_accepts_explicit_repository(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    assert _repo_root(tmp_path) == tmp_path.resolve()


def test_repo_root_rejects_non_repository(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Not a Git repository root"):
        _repo_root(tmp_path)


def test_coverage_command_records_test_contexts(tmp_path: Path) -> None:
    command = _coverage_command(["tests/validation/calibration"])

    assert command[:6] == ["uv", "run", "--no-sync", "--with", "pytest-cov", "pytest"]
    assert "--cov-context=test" in command
    assert "tests/validation/calibration" in command


def test_coverage_json_command_preserves_contexts(tmp_path: Path) -> None:
    output = tmp_path / "coverage.json"

    command = _coverage_json_command(output)

    assert "--show-contexts" in command
    assert command[-1] == str(output)


def test_dead_fixture_check_uses_repository_application_environment() -> None:
    command = CHECK_COMMANDS["fixtures"]

    assert command[:6] == [
        "uv",
        "run",
        "--no-sync",
        "--with",
        "pytest-deadfixtures",
        "pytest",
    ]
    assert command[-2:] == ["--dead-fixtures", "-q"]


def test_mutation_target_must_be_explicit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires --mutation-target"):
        _mutation_target(tmp_path, None)


def test_mutation_target_cannot_escape_repository(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="inside the repository"):
        _mutation_target(tmp_path, outside)


def test_mutation_adapter_uses_and_removes_transient_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "tree_break_selection"
    target.mkdir()

    def fake_run_check(name: str, command: list[str], repo: Path) -> CheckResult:
        config = (repo / "setup.cfg").read_text(encoding="utf-8")
        assert "source_paths=tree_break_selection" in config
        return CheckResult(name, command, 0, "", "")

    monkeypatch.setattr(cli, "_run_check", fake_run_check)

    result = _run_mutation(tmp_path, Path("tree_break_selection"))

    assert result.returncode == 0
    assert not (tmp_path / "setup.cfg").exists()


def test_map_mode_writes_report_through_public_interface(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "tree_break_selection/example.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")

    result = main(
        [
            "--repo",
            str(tmp_path),
            "--mode",
            "map",
            "--output",
            "reports/audit.json",
        ]
    )

    assert result == 0
    assert (tmp_path / "reports/audit.json").exists()


def test_fields_mode_writes_json_markdown_and_graphml(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "tree_break_selection/result.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'def build():\n'
        "    row = {}\n"
        '    row["lineage_field"] = 1\n'
        '    return row["lineage_field"]\n',
        encoding="utf-8",
    )

    result = main(
        [
            "--repo",
            str(tmp_path),
            "--mode",
            "fields",
            "--field-output",
            "reports/fields.json",
            "--field-graphml-output",
            "reports/fields.graphml",
            "--field-markdown-output",
            "reports/fields.md",
        ]
    )

    assert result == 0
    lineage = json.loads((tmp_path / "reports/fields.json").read_text(encoding="utf-8"))
    assert lineage["adapter"] == "libcst"
    assert any(field["key"] == "lineage_field" for field in lineage["fields"])
    assert (tmp_path / "reports/fields.graphml").exists()
    assert (tmp_path / "reports/fields.md").exists()


def test_quick_mode_records_filtered_field_lineage_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "tree_break_selection/result.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'def build():\n'
        "    row = {}\n"
        '    row["dead_field"] = 1\n'
        "    return row\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "CHECK_COMMANDS", {})

    result = main(
        [
            "--repo",
            str(tmp_path),
            "--mode",
            "quick",
            "--output",
            "reports/audit.json",
        ]
    )

    assert result == 0
    inventory = json.loads((tmp_path / "reports/audit.json").read_text(encoding="utf-8"))
    assert inventory["field_lineage_summary"]["dead_write_candidate_count"] == 1


def test_duplicates_mode_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "applications").mkdir()

    def fake_build_duplicate_report(repo: Path, scopes: list[Path]) -> dict[str, object]:
        assert repo == tmp_path.resolve()
        assert scopes == [Path("applications")]
        return {
            "scope": ["applications"],
            "summary": {
                "clone_group_count": 0,
                "duplicated_lines": 0,
                "duplicated_percent": 0,
                "classification_counts": {},
            },
            "groups": [],
        }

    monkeypatch.setattr(cli, "build_duplicate_report", fake_build_duplicate_report)

    result = main(
        [
            "--repo",
            str(tmp_path),
            "--mode",
            "duplicates",
            "--scope",
            "applications",
            "--output",
            "reports/duplicates.json",
            "--duplicates-markdown-output",
            "reports/duplicates.md",
        ]
    )

    assert result == 0
    assert (tmp_path / "reports/duplicates.json").exists()
    assert (tmp_path / "reports/duplicates.md").exists()


def test_generator_geometry_audit_runs_in_repository_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[str], Path]] = []

    def fake_run_check(name: str, command: list[str], repo: Path) -> CheckResult:
        calls.append((name, command, repo))
        return CheckResult(name, command, 0, "ok\n", "")

    monkeypatch.setattr(cli, "_run_check", fake_run_check)

    result = _run_generator_geometry_audit(
        tmp_path,
        output=tmp_path / "reports/generator.csv",
        markdown_output=tmp_path / "reports/generator.md",
    )

    assert result.returncode == 0
    name, command, repo = calls[0]
    assert name == "generator-geometry"
    assert repo == tmp_path
    assert command[:4] == ["uv", "run", "python", "-m"]
    assert "benchmarks.diagnostics.generators.case_geometry_audit" in command
    assert "--output" in command
    assert "--markdown-output" in command


def test_generators_mode_forwards_audit_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()

    def fake_generator_audit(
        repo: Path,
        *,
        output: Path,
        markdown_output: Path,
    ) -> CheckResult:
        assert repo == tmp_path.resolve()
        assert output == tmp_path.resolve() / "reports/generator.csv"
        assert markdown_output == tmp_path.resolve() / "reports/generator.md"
        return CheckResult("generator-geometry", ["uv"], 0, "ok\n", "")

    monkeypatch.setattr(cli, "_run_generator_geometry_audit", fake_generator_audit)

    result = main(
        [
            "--repo",
            str(tmp_path),
            "--mode",
            "generators",
            "--generator-geometry-output",
            "reports/generator.csv",
            "--generator-geometry-markdown-output",
            "reports/generator.md",
        ]
    )

    assert result == 0


def test_calibration_contract_mode_reports_stopped_frontier_multiplicity(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    records = tmp_path / "records.csv"
    records.write_text(
        "source_case_id,parent,left,right,degrees_of_freedom,"
        "sibling_null_weight,is_edge_blocked,is_role_supported\n"
        "toy,root,stopped_frontier,root_leaf,1,1,false,true\n"
        "toy,stopped_frontier,nested_stopped,frontier_leaf,2,1,true,true\n"
        "toy,nested_stopped,left_leaf,right_leaf,1,1,true,true\n",
        encoding="utf-8",
    )

    result = main(
        [
            "--repo",
            str(tmp_path),
            "--mode",
            "calibration-contract",
            "--calibration-records",
            str(records),
            "--output",
            "reports/calibration-contract.json",
            "--calibration-markdown-output",
            "reports/calibration-contract.md",
        ]
    )

    assert result == 0
    report = json.loads(
        (tmp_path / "reports/calibration-contract.json").read_text(encoding="utf-8")
    )
    assert report["summary"] == {
        "group_count": 1,
        "raw_record_count": 3,
        "nonfinite_record_count": 0,
        "role_supported_record_count": 3,
        "tested_null_record_count": 1,
        "stopped_frontier_record_count": 1,
        "nested_blocked_record_count": 1,
        "structural_calibration_record_count": 2,
    }
    assert (tmp_path / "reports/calibration-contract.md").exists()
