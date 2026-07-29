"""Static evidence collectors for repository hygiene decisions."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import grimp

SOURCE_ROOTS = (
    "applications",
    "benchmarks",
    "scripts",
    "tests",
    "tree_break_selection",
)
CALIBRATION_ROOT = Path("benchmarks/diagnostics/calibration")
IGNORED_PARTS = {".git", ".venv", "__pycache__", "vendor"}
EVIDENCE_ROOTS = ("docs", "manuscript", "reports", "wiki")
EVIDENCE_SUFFIXES = {".json", ".md", ".tex", ".txt", ".yaml", ".yml"}


@dataclass(frozen=True)
class ImportUse:
    importer: str
    line: int


@dataclass(frozen=True)
class FieldUse:
    key: str
    operation: str
    path: str
    line: int
    surface: str


def _python_files(repo: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in SOURCE_ROOTS:
        root = repo / root_name
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*.py")
            if not IGNORED_PARTS.intersection(path.relative_to(repo).parts)
        )
    return sorted(set(files))


def _module_name(repo: Path, path: Path) -> str:
    relative = path.relative_to(repo).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _surface(path: Path) -> str:
    first = path.parts[0]
    if first == "tests":
        return "test"
    if first == "benchmarks":
        return "benchmark"
    if first == "scripts":
        return "maintenance"
    if first == "applications":
        return "application"
    if first == "tree_break_selection":
        return "production"
    return "other"


def _imports(tree: ast.AST) -> Iterable[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported_name in node.names:
                yield imported_name.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module, node.lineno
            for imported_name in node.names:
                if imported_name.name != "*":
                    yield f"{node.module}.{imported_name.name}", node.lineno


def _string_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _field_uses(tree: ast.AST, relative: Path) -> Iterable[FieldUse]:
    surface = _surface(relative)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            key = _string_key(node.slice)
            if key is not None:
                operation = {
                    ast.Load: "read",
                    ast.Store: "write",
                    ast.Del: "delete",
                }.get(type(node.ctx), "unknown")
                yield FieldUse(key, operation, str(relative), node.lineno, surface)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop", "setdefault"}
            and node.args
        ):
            key = _string_key(node.args[0])
            if key is not None:
                operation = "read" if node.func.attr == "get" else node.func.attr
                yield FieldUse(key, operation, str(relative), node.lineno, surface)


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        comparison = node.test
        if not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1:
            continue
        if not isinstance(comparison.ops[0], ast.Eq) or len(comparison.comparators) != 1:
            continue
        left = comparison.left
        right = comparison.comparators[0]
        if (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
        ):
            return True
    return False


def _evidence_references(repo: Path, stems: set[str]) -> dict[str, list[str]]:
    references: dict[str, list[str]] = defaultdict(list)
    files = [repo / "README.md", repo / "CHANGELOG.md"]
    for root_name in EVIDENCE_ROOTS:
        root = repo / root_name
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in EVIDENCE_SUFFIXES
                and "generated" not in path.relative_to(repo).parts
            )
    for path in sorted(set(files)):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower())
        for stem in stems:
            normalized_stem = re.sub(r"[^a-z0-9]+", " ", stem.lower())
            if stem in text or normalized_stem in normalized_text:
                references[stem].append(str(path.relative_to(repo)))
    return references


def _git_evidence(repo: Path, relative: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "git",
            "log",
            "--follow",
            "--format=%H%x09%aI%x09%s",
            "--",
            str(relative),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line]
    if not lines:
        return {"commit_count": 0, "last_commit": None}
    commit, date, subject = lines[0].split("\t", 2)
    return {
        "commit_count": len(lines),
        "last_commit": {"commit": commit, "date": date, "subject": subject},
    }


def _grimp_graph(repo: Path) -> tuple[object | None, str | None]:
    packages = [
        name
        for name in ("applications", "benchmarks", "tree_break_selection")
        if (repo / name / "__init__.py").exists()
    ]
    if not packages:
        return None, "no importable top-level packages"
    sys.path.insert(0, str(repo))
    try:
        return grimp.build_graph(*packages), None
    except Exception as exc:  # Grimp can reject malformed or namespace packages.
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        sys.path.remove(str(repo))


def build_inventory(repo: Path) -> dict[str, object]:
    """Build static import, field-use, and history evidence for a repository."""
    repo = repo.resolve()
    files = _python_files(repo)
    module_paths: dict[str, Path] = {}
    importer_map: dict[str, list[ImportUse]] = defaultdict(list)
    fields: list[FieldUse] = []
    main_guards: dict[str, bool] = {}
    parse_errors: list[dict[str, object]] = []

    for path in files:
        relative = path.relative_to(repo)
        module = _module_name(repo, path)
        module_paths[module] = relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append({"path": str(relative), "error": str(exc)})
            continue
        main_guards[module] = _has_main_guard(tree)
        for imported, line in _imports(tree):
            importer_map[imported].append(ImportUse(module, line))
        fields.extend(_field_uses(tree, relative))

    calibration_modules = {
        module: relative
        for module, relative in module_paths.items()
        if relative.is_relative_to(CALIBRATION_ROOT) and relative.name != "__init__.py"
    }
    evidence_references = _evidence_references(
        repo, {relative.stem for relative in calibration_modules.values()}
    )
    graph, graph_error = _grimp_graph(repo)
    calibration: list[dict[str, object]] = []
    for module, relative in sorted(calibration_modules.items()):
        uses: list[ImportUse] = []
        for imported, imported_uses in importer_map.items():
            if imported == module or imported.startswith(f"{module}."):
                uses.extend(imported_uses)
        importers = sorted({use.importer for use in uses})
        grimp_importers = (
            sorted(graph.find_modules_that_directly_import(module))
            if graph is not None and module in graph.modules
            else []
        )
        importers = sorted(set(importers) | set(grimp_importers))
        non_test_importers = [
            importer for importer in importers if not importer.startswith("tests.")
        ]
        documented_by = evidence_references.get(relative.stem, [])
        has_main_guard = main_guards.get(module, False)
        statically_unimported = not importers
        if non_test_importers:
            disposition = "non_test_imported"
        elif importers and documented_by:
            disposition = "test_only_documented"
        elif importers:
            disposition = "test_only_undocumented"
        elif has_main_guard and documented_by:
            disposition = "standalone_documented"
        elif has_main_guard:
            disposition = "standalone_undocumented"
        elif documented_by:
            disposition = "unimported_documented"
        else:
            disposition = "unresolved_unimported"
        calibration.append(
            {
                "module": module,
                "path": str(relative),
                "importers": importers,
                "ast_importers": sorted({use.importer for use in uses}),
                "grimp_importers": grimp_importers,
                "non_test_importers": non_test_importers,
                "test_only": bool(importers) and not non_test_importers,
                "statically_unimported": statically_unimported,
                "has_main_guard": has_main_guard,
                "documented_by": documented_by,
                "disposition": disposition,
                "history": _git_evidence(repo, relative),
            }
        )

    field_groups: dict[str, list[FieldUse]] = defaultdict(list)
    for use in fields:
        field_groups[use.key].append(use)
    field_summary: list[dict[str, object]] = []
    for key, uses in sorted(field_groups.items()):
        operation_counts = Counter(use.operation for use in uses)
        surface_counts = Counter(use.surface for use in uses)
        read_surfaces = sorted(
            {use.surface for use in uses if use.operation in {"read", "pop", "setdefault"}}
        )
        field_summary.append(
            {
                "key": key,
                "operations": dict(sorted(operation_counts.items())),
                "surfaces": dict(sorted(surface_counts.items())),
                "read_surfaces": read_surfaces,
                "diagnostic_only": bool(read_surfaces)
                and set(read_surfaces) <= {"benchmark", "test"},
                "write_only": not read_surfaces,
                "uses": [asdict(use) for use in uses],
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "files_scanned": len(files),
        "parse_errors": parse_errors,
        "import_graph": {
            "adapter": "grimp",
            "available": graph is not None,
            "error": graph_error,
            "module_count": len(graph.modules) if graph is not None else 0,
        },
        "calibration": {
            "module_count": len(calibration),
            "test_only_count": sum(bool(item["test_only"]) for item in calibration),
            "statically_unimported_count": sum(
                bool(item["statically_unimported"]) for item in calibration
            ),
            "unresolved_count": sum(
                item["disposition"] == "unresolved_unimported" for item in calibration
            ),
            "modules": calibration,
        },
        "fields": {
            "key_count": len(field_summary),
            "diagnostic_only_count": sum(
                bool(item["diagnostic_only"]) for item in field_summary
            ),
            "write_only_count": sum(bool(item["write_only"]) for item in field_summary),
            "keys": field_summary,
        },
    }


def write_inventory(inventory: dict[str, object], output: Path) -> None:
    """Write an inventory atomically enough for a local audit artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
