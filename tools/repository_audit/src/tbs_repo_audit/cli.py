"""Command-line interface for repository hygiene evidence."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .coverage_evidence import compare_coverage
from .field_lineage import build_field_lineage, write_field_lineage_outputs
from .inventory import build_inventory, write_inventory

DEFAULT_OUTPUT = Path("reports/audits/generated/repository-hygiene.json")
DEFAULT_FIELD_OUTPUT = Path("reports/audits/generated/field-function-lineage.json")
DEFAULT_FIELD_GRAPHML_OUTPUT = Path(
    "reports/audits/generated/field-function-lineage.graphml"
)
DEFAULT_FIELD_MARKDOWN_OUTPUT = Path(
    "reports/audits/generated/field-function-lineage.md"
)
CHECK_COMMANDS = {
    "ruff": ["ruff", "check", "."],
    "vulture": [
        "vulture",
        "applications",
        "benchmarks",
        "scripts",
        "tree_break_selection",
        "--min-confidence",
        "80",
        "--sort-by-size",
    ],
    "duplicates": [
        "jscpd",
        "applications",
        "benchmarks",
        "scripts",
        "tests",
        "tree_break_selection",
        "--min-lines",
        "8",
        "--min-tokens",
        "70",
        "--reporters",
        "console",
        "--silent",
    ],
    "fixtures": [
        "uv",
        "run",
        "--no-sync",
        "--with",
        "pytest-deadfixtures",
        "pytest",
        "--dead-fixtures",
        "-q",
    ],
}
TOOL_NAMES = (
    "git",
    "jscpd",
    "uv",
)
PYTHON_ADAPTERS = {
    "coverage": "coverage",
    "grimp": "grimp",
    "libcst": "libcst",
    "mutmut": "mutmut",
    "networkx": "networkx",
    "pytest": "pytest",
    "pytest-cov": "pytest_cov",
    "pytest-deadfixtures": "pytest_deadfixtures",
    "pytest-testmon": "testmon",
    "ruff": "ruff",
    "semgrep": "semgrep",
    "vulture": "vulture",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def _repo_root(argument: Path | None) -> Path:
    if argument is not None:
        candidate = argument
    elif configured := os.environ.get("TBS_AUDIT_REPO"):
        candidate = Path(configured)
    else:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise SystemExit(
                "Cannot locate a repository. Run inside one, pass --repo, or set "
                "TBS_AUDIT_REPO."
            )
        candidate = Path(result.stdout.strip())
    candidate = candidate.expanduser().resolve()
    if not (candidate / ".git").exists():
        raise SystemExit(f"Not a Git repository root: {candidate}")
    return candidate


def _doctor() -> int:
    missing = []
    for name in TOOL_NAMES:
        path = shutil.which(name)
        if path:
            print(f"{name:12} {path}")
        else:
            print(f"{name:12} MISSING")
            missing.append(name)
    for name, module in PYTHON_ADAPTERS.items():
        available = importlib.util.find_spec(module) is not None
        print(f"{name:20} {'installed' if available else 'MISSING'}")
        if not available:
            missing.append(name)
    return 1 if missing else 0


def _environment_executable(name: str) -> str:
    sibling = Path(sys.executable).parent / name
    if sibling.exists():
        return str(sibling)
    executable = shutil.which(name)
    if executable:
        return executable
    raise RuntimeError(f"Audit adapter is not installed: {name}")


def _run_check(
    name: str,
    command: list[str],
    repo: Path,
    *,
    env: dict[str, str] | None = None,
) -> CheckResult:
    command = [_environment_executable(command[0]), *command[1:]]
    result = subprocess.run(
        command,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return CheckResult(name, command, result.returncode, result.stdout, result.stderr)


def _coverage_command(test_arguments: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "--with",
        "pytest-cov",
        "pytest",
        *test_arguments,
        "--cov=tree_break_selection",
        "--cov=benchmarks",
        "--cov-context=test",
        "--cov-report=term",
    ]


def _coverage_json_command(output: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "--with",
        "coverage",
        "coverage",
        "json",
        "--show-contexts",
        "-o",
        str(output),
    ]


def _run_coverage(
    name: str,
    repo: Path,
    test_arguments: list[str],
    output: Path,
) -> CheckResult:
    environment = dict(os.environ)
    environment["COVERAGE_FILE"] = str(output.with_suffix(".data"))
    pytest_result = _run_check(
        name,
        _coverage_command(test_arguments),
        repo,
        env=environment,
    )
    if pytest_result.returncode:
        return pytest_result
    json_result = _run_check(
        f"{name}-json",
        _coverage_json_command(output),
        repo,
        env=environment,
    )
    return CheckResult(
        name=name,
        command=pytest_result.command + ["&&"] + json_result.command,
        returncode=json_result.returncode,
        stdout=pytest_result.stdout + json_result.stdout,
        stderr=pytest_result.stderr + json_result.stderr,
    )


def _mutation_command() -> list[str]:
    return ["mutmut", "run"]


def _mutation_target(repo: Path, argument: Path | None) -> Path:
    if argument is None:
        raise SystemExit(
            "Mutation mode requires --mutation-target so the tool never guesses "
            "which production code to mutate."
        )
    target = (repo / argument).resolve() if not argument.is_absolute() else argument.resolve()
    try:
        relative = target.relative_to(repo)
    except ValueError as exc:
        raise SystemExit("Mutation target must be inside the repository.") from exc
    if not target.exists():
        raise SystemExit(f"Mutation target does not exist: {relative}")
    return relative


def _run_mutation(repo: Path, target: Path) -> CheckResult:
    config = repo / "setup.cfg"
    created_config = not config.exists()
    if created_config:
        config.write_text(
            "[mutmut]\n"
            f"source_paths={target}\n"
            "mutate_only_covered_lines=true\n"
            "pytest_add_cli_args=-q\n",
            encoding="utf-8",
        )
    try:
        return _run_check("mutation", _mutation_command(), repo)
    finally:
        if created_config:
            config.unlink(missing_ok=True)


def _summary(inventory: dict[str, object], output: Path) -> None:
    calibration = inventory["calibration"]
    fields = inventory["fields"]
    print(f"Audit inventory: {output}")
    print(
        "Calibration modules: "
        f"{calibration['module_count']} total, "
        f"{calibration['test_only_count']} test-only, "
        f"{calibration['statically_unimported_count']} statically unimported, "
        f"{calibration['unresolved_count']} unresolved"
    )
    print(
        "String fields: "
        f"{fields['key_count']} total, "
        f"{fields['diagnostic_only_count']} diagnostic-only, "
        f"{fields['write_only_count']} write-only"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tbs-audit",
        description=(
            "Build evidence for dead-code, calibration-study, and diagnostic-field "
            "cleanup decisions."
        ),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        help="repository root; defaults to the current Git repository",
    )
    parser.add_argument(
        "--mode",
        choices=("map", "fields", "quick", "evidence", "mutation"),
        default="map",
        help=(
            "map: static evidence; fields: LibCST field/function lineage; "
            "quick: add linters/clones/fixtures; "
            "evidence: add calibration coverage contexts; mutation: add mutmut"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON inventory path, relative to the repository",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="show whether every audit executable is reachable",
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=DEFAULT_FIELD_OUTPUT,
        help="JSON field-lineage path, relative to the repository",
    )
    parser.add_argument(
        "--field-graphml-output",
        type=Path,
        default=DEFAULT_FIELD_GRAPHML_OUTPUT,
        help="GraphML field-lineage path, relative to the repository",
    )
    parser.add_argument(
        "--field-markdown-output",
        type=Path,
        default=DEFAULT_FIELD_MARKDOWN_OUTPUT,
        help="Markdown field-lineage path, relative to the repository",
    )
    parser.add_argument(
        "--mutation-target",
        type=Path,
        help=(
            "production file or directory to mutate; required in mutation mode "
            "to prevent an accidental whole-repository run"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.doctor:
        return _doctor()

    repo = _repo_root(args.repo)
    if args.mode == "fields":
        field_output = (
            args.field_output if args.field_output.is_absolute() else repo / args.field_output
        )
        graphml_output = (
            args.field_graphml_output
            if args.field_graphml_output.is_absolute()
            else repo / args.field_graphml_output
        )
        markdown_output = (
            args.field_markdown_output
            if args.field_markdown_output.is_absolute()
            else repo / args.field_markdown_output
        )
        lineage = build_field_lineage(repo)
        write_field_lineage_outputs(
            lineage,
            output=field_output,
            graphml_output=graphml_output,
            markdown_output=markdown_output,
        )
        print(f"Field lineage: {field_output}")
        print(
            "Fields: "
            f"{lineage['field_count']} total; "
            f"{lineage['reuse_counts'].get('no_reader_dead_candidate', 0)} "
            "no-reader candidates"
        )
        return 0

    mutation_target = (
        _mutation_target(repo, args.mutation_target)
        if args.mode == "mutation"
        else None
    )
    output = args.output if args.output.is_absolute() else repo / args.output
    inventory = build_inventory(repo)

    checks: list[CheckResult] = []
    if args.mode in {"quick", "evidence", "mutation"}:
        for name, command in CHECK_COMMANDS.items():
            checks.append(_run_check(name, command, repo))
    if args.mode in {"evidence", "mutation"}:
        calibration_output = output.with_name("calibration-coverage.json")
        baseline_output = output.with_name("noncalibration-coverage.json")
        calibration_check = _run_coverage(
            "calibration-coverage-contexts",
            repo,
            ["tests/validation/calibration"],
            calibration_output,
        )
        baseline_check = _run_coverage(
            "noncalibration-coverage-contexts",
            repo,
            ["tests", "--ignore=tests/validation/calibration"],
            baseline_output,
        )
        checks.extend([calibration_check, baseline_check])
        if not calibration_check.returncode and not baseline_check.returncode:
            inventory["coverage_comparison"] = compare_coverage(
                calibration_output, baseline_output
            )
            for artifact in (
                calibration_output,
                calibration_output.with_suffix(".data"),
                baseline_output,
                baseline_output.with_suffix(".data"),
            ):
                artifact.unlink(missing_ok=True)
    if args.mode == "mutation":
        assert mutation_target is not None
        checks.append(_run_mutation(repo, mutation_target))

    inventory["checks"] = [asdict(check) for check in checks]
    write_inventory(inventory, output)
    _summary(inventory, output)

    failed = [check for check in checks if check.returncode]
    for check in failed:
        print(
            f"{check.name} reported findings or failed (exit {check.returncode})",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
