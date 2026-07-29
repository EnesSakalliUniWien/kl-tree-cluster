"""Duplicate-cleanup reporting built on jscpd clone groups."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CloneEndpoint:
    path: str
    start: int
    end: int
    full_path: str


@dataclass(frozen=True)
class CloneGroup:
    lines: int
    tokens: int | None
    first: CloneEndpoint
    second: CloneEndpoint
    classification: str
    risk: str
    recommendation: str


DEFAULT_DUPLICATE_SCOPES = ("applications",)
JSPCD_MIN_LINES = 8
JSPCD_MIN_TOKENS = 70


def _jscpd_executable() -> str:
    executable = shutil.which("jscpd")
    if executable is None:
        raise RuntimeError("jscpd is not installed or not on PATH.")
    return executable


def _scope_paths(repo: Path, scopes: list[Path]) -> list[Path]:
    resolved = []
    for scope in scopes:
        path = scope if scope.is_absolute() else repo / scope
        path = path.resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise SystemExit(f"Duplicate scope must be inside the repository: {scope}") from exc
        if not path.exists():
            raise SystemExit(f"Duplicate scope does not exist: {scope}")
        resolved.append(path)
    return resolved


def _relative_name(repo: Path, scope_paths: list[Path], name: str) -> str:
    candidate = Path(name)
    if candidate.is_absolute():
        try:
            return str(candidate.resolve().relative_to(repo))
        except ValueError:
            return str(candidate)
    direct = repo / candidate
    if direct.exists():
        return str(candidate)
    for scope in scope_paths:
        scoped = scope / candidate
        if scoped.exists():
            return str(scoped.relative_to(repo))
    return str(candidate)


def _endpoint(repo: Path, scope_paths: list[Path], raw: dict[str, object]) -> CloneEndpoint:
    path = str(raw["name"])
    return CloneEndpoint(
        path=path,
        start=int(raw.get("start", 0)),
        end=int(raw.get("end", 0)),
        full_path=_relative_name(repo, scope_paths, path),
    )


def _path_parts(path: str) -> tuple[str, ...]:
    return Path(path).parts


def _is_r_bootstrap(first: CloneEndpoint, second: CloneEndpoint) -> bool:
    return (
        first.full_path.endswith(".R")
        and second.full_path.endswith(".R")
        and first.start <= 20
        and second.start <= 20
    )


def classify_duplicate(first: CloneEndpoint, second: CloneEndpoint) -> tuple[str, str, str]:
    """Classify one duplicate group for cleanup triage."""

    paths = (first.full_path, second.full_path)
    parts = tuple(_path_parts(path) for path in paths)
    if _is_r_bootstrap(first, second):
        return (
            "bootstrap_structural",
            "low",
            "Usually leave unless a shared dispatcher is introduced.",
        )
    if any("tests" in item for item in parts):
        return (
            "test_fixture_or_test_helper",
            "low",
            "Extract only when it improves test intent; otherwise tolerate.",
        )
    if all(path.startswith("tree_break_selection/") for path in paths):
        return (
            "production_production",
            "high",
            "Inspect first; extract only with direct regression tests.",
        )
    if any("plots" in item or "reports" in item for item in parts):
        return (
            "plot_report",
            "low",
            "Good cleanup candidate when rendering output is unchanged.",
        )
    if any("analysis" in item for item in parts):
        return (
            "analysis_methodological",
            "medium",
            "Inspect semantics before extraction; avoid hiding methodological differences.",
        )
    if any(path.startswith("benchmarks/") for path in paths):
        return (
            "benchmark",
            "medium",
            "Extract shared run/report mechanics; keep benchmark intent explicit.",
        )
    if any(path.startswith("applications/") for path in paths):
        return (
            "application",
            "medium",
            "Prefer application-local helpers over global utilities.",
        )
    return (
        "uncategorized",
        "medium",
        "Inspect manually before changing.",
    )


def _clone_group(repo: Path, scope_paths: list[Path], raw: dict[str, object]) -> CloneGroup:
    first = _endpoint(repo, scope_paths, raw["firstFile"])  # type: ignore[arg-type]
    second = _endpoint(repo, scope_paths, raw["secondFile"])  # type: ignore[arg-type]
    classification, risk, recommendation = classify_duplicate(first, second)
    return CloneGroup(
        lines=int(raw.get("lines", 0)),
        tokens=int(raw["tokens"]) if raw.get("tokens") is not None else None,
        first=first,
        second=second,
        classification=classification,
        risk=risk,
        recommendation=recommendation,
    )


def _jscpd_command(scope_paths: list[Path], output_dir: Path) -> list[str]:
    return [
        _jscpd_executable(),
        *[str(path) for path in scope_paths],
        "--min-lines",
        str(JSPCD_MIN_LINES),
        "--min-tokens",
        str(JSPCD_MIN_TOKENS),
        "--reporters",
        "json",
        "--output",
        str(output_dir),
        "--silent",
    ]


def build_duplicate_report(repo: Path, scopes: list[Path]) -> dict[str, object]:
    """Run jscpd for scopes and return a classified cleanup report."""

    repo = repo.resolve()
    scope_paths = _scope_paths(repo, scopes)
    with tempfile.TemporaryDirectory(prefix="tbs-audit-duplicates.") as tmp:
        output_dir = Path(tmp)
        command = _jscpd_command(scope_paths, output_dir)
        result = subprocess.run(
            command,
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        report_path = output_dir / "jscpd-report.json"
        if not report_path.exists():
            raise RuntimeError(
                "jscpd did not write jscpd-report.json. "
                f"Exit {result.returncode}: {result.stderr or result.stdout}"
            )
        raw_report = json.loads(report_path.read_text(encoding="utf-8"))

    groups = [
        _clone_group(repo, scope_paths, raw)
        for raw in raw_report.get("duplicates", [])
    ]
    groups.sort(key=lambda group: (-group.lines, group.risk, group.first.full_path))
    total = raw_report.get("statistics", {}).get("total", {})
    classification_counts = Counter(group.classification for group in groups)
    risk_counts = Counter(group.risk for group in groups)
    return {
        "adapter": "jscpd",
        "scope": [str(path.relative_to(repo)) for path in scope_paths],
        "parameters": {
            "min_lines": JSPCD_MIN_LINES,
            "min_tokens": JSPCD_MIN_TOKENS,
        },
        "summary": {
            "clone_group_count": len(groups),
            "duplicated_lines": total.get("duplicatedLines"),
            "duplicated_percent": total.get("percentage"),
            "classification_counts": dict(sorted(classification_counts.items())),
            "risk_counts": dict(sorted(risk_counts.items())),
        },
        "groups": [
            {
                "lines": group.lines,
                "tokens": group.tokens,
                "classification": group.classification,
                "risk": group.risk,
                "recommendation": group.recommendation,
                "first": group.first.__dict__,
                "second": group.second.__dict__,
            }
            for group in groups
        ],
    }


def write_duplicate_report(report: dict[str, object], *, output: Path, markdown_output: Path) -> None:
    """Write duplicate cleanup JSON and Markdown reports."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(duplicate_report_markdown(report), encoding="utf-8")


def duplicate_report_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]  # type: ignore[index]
    groups = report["groups"]  # type: ignore[index]
    lines = [
        "# Duplicate Cleanup Report",
        "",
        f"- Scope: `{', '.join(report['scope'])}`",  # type: ignore[index]
        f"- Clone groups: `{summary['clone_group_count']}`",  # type: ignore[index]
        f"- Duplicated lines: `{summary['duplicated_lines']}`",  # type: ignore[index]
        f"- Duplicated percent: `{summary['duplicated_percent']}`",  # type: ignore[index]
        "",
        "## Classification counts",
        "",
    ]
    for key, value in sorted(summary["classification_counts"].items()):  # type: ignore[index, union-attr]
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Ranked clone groups", ""])
    for index, group in enumerate(groups, 1):  # type: ignore[assignment]
        first = group["first"]
        second = group["second"]
        lines.extend(
            [
                f"### {index}. {group['classification']} ({group['risk']} risk)",
                "",
                f"- Lines: `{group['lines']}`",
                f"- First: `{first['full_path']}:{first['start']}-{first['end']}`",
                f"- Second: `{second['full_path']}:{second['start']}-{second['end']}`",
                f"- Recommendation: {group['recommendation']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
