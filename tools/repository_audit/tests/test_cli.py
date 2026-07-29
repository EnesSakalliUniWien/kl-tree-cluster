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
