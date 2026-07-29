"""LibCST field/function lineage and NetworkX graph export."""

from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst
import networkx as nx
from libcst.metadata import MetadataWrapper, ParentNodeProvider, PositionProvider

SOURCE_ROOTS = (
    "applications",
    "benchmarks",
    "scripts",
    "tests",
    "tree_break_selection",
)
IGNORED_PARTS = {".git", ".venv", "__pycache__", "vendor"}
READ_OPERATIONS = {"read", "pop", "setdefault"}
FIELD_READ_FUNCTIONS = {
    "_annotation_bool",
    "_annotation_float",
    "_annotation_int",
    "_annotation_str",
    "_annotation_value",
    "annotation_bool",
    "column_present_and_true",
}
ENV_READ_FUNCTIONS = {"getenv"}
CONFIG_ROOT_NAMES = {
    "config",
    "metadata",
    "params",
    "payload",
    "recorded_run_params",
    "run_params",
}
GRAPH_ROOT_NAMES = {
    "attrs",
    "edge_attrs",
    "graph",
    "rooted",
    "tree",
    "weighted",
}
GRAPH_KEYS = {"weight", "length", "branch_length", "linkage_branch_length"}


@dataclass(frozen=True)
class FieldFunctionUse:
    key: str
    operation: str
    access_kind: str
    path: str
    line: int
    scope: str
    surface: str


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


def _literal_string(node: cst.CSTNode) -> str | None:
    if isinstance(node, cst.SimpleString):
        try:
            value = ast.literal_eval(node.value)
        except (SyntaxError, ValueError):
            return None
        return value if isinstance(value, str) else None
    return None


def _schema_literal_strings(node: cst.CSTNode) -> Iterable[str]:
    if isinstance(node, (cst.List, cst.Tuple, cst.Set)):
        for element in node.elements:
            value = _literal_string(element.value)
            if value is not None:
                yield value


def _subscript_literal_strings(node: cst.Subscript) -> Iterable[str]:
    for element in node.slice:
        index = element.slice
        if not isinstance(index, cst.Index):
            continue
        key = _literal_string(index.value)
        if key is not None:
            yield key


def _call_name(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return node.attr.value
    return None


def _expression_name(node: cst.CSTNode) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        base = _expression_name(node.value)
        return f"{base}.{node.attr.value}" if base else node.attr.value
    return None


def _root_name(expression_name: str | None) -> str:
    return "" if not expression_name else expression_name.split(".", 1)[0]


def _is_schema_assignment_name(name: str) -> bool:
    lowered = name.lower()
    return (
        name.endswith("COLUMNS")
        or name.endswith("_COLS")
        or name.endswith("_FIELDS")
        or name.endswith("_SCHEMA")
        or lowered
        in {
            "display_cols",
            "display_columns",
            "export_columns",
            "keep_columns",
            "output_cols",
            "output_columns",
            "required_columns",
        }
    )


def _access_kind_for_expression(expression_name: str | None) -> str:
    root = _root_name(expression_name)
    if expression_name in {"os.environ"} or expression_name == "os.environ":
        return "environment"
    if expression_name and expression_name.endswith(".environ"):
        return "environment"
    if expression_name and (".graph" in expression_name or expression_name.endswith(".es")):
        return "graph"
    if root in GRAPH_ROOT_NAMES:
        return "graph"
    if root in CONFIG_ROOT_NAMES or (expression_name and "config" in expression_name.lower()):
        return "configuration"
    return "field"


class _FieldFunctionVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    def __init__(self, relative: Path) -> None:
        self.relative = relative
        self.surface = _surface(relative)
        self.scope_stack: list[str] = []
        self.uses: list[FieldFunctionUse] = []

    def _scope(self) -> str:
        return ".".join(self.scope_stack) if self.scope_stack else "<module>"

    def _line(self, node: cst.CSTNode) -> int:
        return int(self.get_metadata(PositionProvider, node).start.line)

    def _record(
        self,
        *,
        key: str,
        operation: str,
        node: cst.CSTNode,
        access_kind: str = "field",
    ) -> None:
        self.uses.append(
            FieldFunctionUse(
                key=key,
                operation=operation,
                access_kind=access_kind,
                path=str(self.relative),
                line=self._line(node),
                scope=self._scope(),
                surface=self.surface,
            )
        )

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.scope_stack.append(node.name.value)
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.scope_stack.append(node.name.value)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.scope_stack.pop()

    def visit_Assign(self, node: cst.Assign) -> bool | None:
        target_names = [
            target.target.value
            for target in node.targets
            if isinstance(target.target, cst.Name)
        ]
        if any(_is_schema_assignment_name(name) for name in target_names):
            for key in _schema_literal_strings(node.value):
                self._record(key=key, operation="schema", node=node, access_kind="schema")
        return True

    def visit_Dict(self, node: cst.Dict) -> bool | None:
        for element in node.elements:
            if not isinstance(element, cst.DictElement):
                continue
            key = _literal_string(element.key)
            if key is not None:
                self._record(key=key, operation="schema", node=element, access_kind="schema")
        return True

    def visit_Subscript(self, node: cst.Subscript) -> bool | None:
        keys = tuple(_subscript_literal_strings(node))
        if not keys:
            return True
        operation = "read"
        access_kind = _access_kind_for_expression(_expression_name(node.value))
        parent = self.get_metadata(ParentNodeProvider, node, None)
        if isinstance(parent, cst.AssignTarget) and parent.target is node:
            operation = "write"
        elif isinstance(parent, cst.AugAssign) and parent.target is node:
            operation = "write"
        elif isinstance(parent, cst.AnnAssign) and parent.target is node:
            operation = "write"
        elif isinstance(parent, cst.Del):
            operation = "delete"
        for key in keys:
            self._record(key=key, operation=operation, node=node, access_kind=access_kind)
        return True

    def visit_Call(self, node: cst.Call) -> bool | None:
        method = _call_name(node.func)
        if method in ENV_READ_FUNCTIONS and node.args:
            key = _literal_string(node.args[0].value)
            if key is not None:
                self._record(
                    key=key,
                    operation="read",
                    node=node,
                    access_kind="environment",
                )
            return True
        if method in FIELD_READ_FUNCTIONS:
            for argument in node.args:
                key = _literal_string(argument.value)
                if key is not None:
                    self._record(key=key, operation="read", node=node)
            return True

        if not isinstance(node.func, cst.Attribute):
            return True
        method = node.func.attr.value
        if method not in {"get", "pop", "setdefault"} or not node.args:
            return True
        key = _literal_string(node.args[0].value)
        if key is not None:
            operation = "read" if method == "get" else method
            access_kind = _access_kind_for_expression(_expression_name(node.func.value))
            self._record(key=key, operation=operation, node=node, access_kind=access_kind)
        return True


def _parse_field_uses(repo: Path) -> tuple[list[FieldFunctionUse], list[dict[str, object]]]:
    uses: list[FieldFunctionUse] = []
    parse_errors: list[dict[str, object]] = []
    for path in _python_files(repo):
        relative = path.relative_to(repo)
        try:
            module = cst.parse_module(path.read_text(encoding="utf-8"))
            wrapper = MetadataWrapper(module)
            visitor = _FieldFunctionVisitor(relative)
            wrapper.visit(visitor)
        except Exception as exc:
            parse_errors.append({"path": str(relative), "error": str(exc)})
            continue
        uses.extend(visitor.uses)
    return uses, parse_errors


def _field_reuse(uses: list[FieldFunctionUse]) -> str:
    writers = [use for use in uses if use.operation == "write"]
    readers = [use for use in uses if use.operation in READ_OPERATIONS]
    schemas = [use for use in uses if use.operation == "schema"]
    writer_scopes = {use.scope for use in writers}
    reader_scopes = {use.scope for use in readers}
    if readers and writer_scopes & reader_scopes:
        return "same_function_reader"
    if readers:
        return "cross_function_reader"
    if schemas:
        return "schema_only_export_candidate"
    if writers:
        return "no_reader_dead_candidate"
    return "read_only_or_schema"


def _cleanup_classification(uses: list[FieldFunctionUse]) -> str:
    writers = [use for use in uses if use.operation == "write"]
    readers = [use for use in uses if use.operation in READ_OPERATIONS]
    schemas = [use for use in uses if use.operation == "schema"]
    access_kinds = {use.access_kind for use in uses if use.access_kind != "schema"}
    key = uses[0].key if uses else ""

    if schemas:
        return "output_schema_column"
    if access_kinds and access_kinds <= {"environment"}:
        return "environment_key"
    if access_kinds and access_kinds <= {"configuration"}:
        return "configuration_key"
    if access_kinds and access_kinds <= {"graph"}:
        return "graph_key"
    if key in GRAPH_KEYS and any(use.access_kind == "graph" for use in uses):
        return "graph_key"
    if writers and not readers:
        return "dead_write_candidate"
    if readers:
        return "used_field"
    return "review_required"


def build_field_lineage(repo: Path) -> dict[str, object]:
    """Build LibCST-backed field/function lineage evidence."""
    repo = repo.resolve()
    uses, parse_errors = _parse_field_uses(repo)
    grouped: dict[str, list[FieldFunctionUse]] = defaultdict(list)
    for use in uses:
        grouped[use.key].append(use)

    fields: list[dict[str, object]] = []
    for key, key_uses in sorted(grouped.items()):
        writers = [use for use in key_uses if use.operation == "write"]
        readers = [use for use in key_uses if use.operation in READ_OPERATIONS]
        schemas = [use for use in key_uses if use.operation == "schema"]
        fields.append(
            {
                "key": key,
                "reuse": _field_reuse(key_uses),
                "cleanup_classification": _cleanup_classification(key_uses),
                "operations": dict(sorted(Counter(use.operation for use in key_uses).items())),
                "access_kinds": dict(
                    sorted(Counter(use.access_kind for use in key_uses).items())
                ),
                "surfaces": dict(sorted(Counter(use.surface for use in key_uses).items())),
                "writer_scopes": sorted(
                    {f"{use.path}::{use.scope}" for use in writers}
                ),
                "reader_scopes": sorted(
                    {f"{use.path}::{use.scope}" for use in readers}
                ),
                "schema_scopes": sorted(
                    {f"{use.path}::{use.scope}" for use in schemas}
                ),
                "uses": [asdict(use) for use in key_uses],
            }
        )

    return {
        "schema_version": 1,
        "adapter": "libcst",
        "field_count": len(fields),
        "parse_errors": parse_errors,
        "reuse_counts": dict(sorted(Counter(field["reuse"] for field in fields).items())),
        "cleanup_classification_counts": dict(
            sorted(Counter(field["cleanup_classification"] for field in fields).items())
        ),
        "fields": fields,
    }


def build_field_lineage_graph(lineage: dict[str, object]) -> nx.DiGraph:
    """Represent field/function lineage as a NetworkX directed graph."""
    graph = nx.DiGraph()
    graph.add_node("fields", kind="root", label=f"Fields\\n{lineage['field_count']}")
    for field in lineage["fields"]:
        assert isinstance(field, dict)
        key = str(field["key"])
        field_id = f"field:{key}"
        graph.add_node(
            field_id,
            kind="field",
            label=key,
            reuse=str(field["reuse"]),
            cleanup_classification=str(field["cleanup_classification"]),
            surfaces=json.dumps(field["surfaces"], sort_keys=True),
        )
        graph.add_edge("fields", field_id, kind="contains")
        for scope in field["writer_scopes"]:
            scope_id = f"scope:{scope}"
            graph.add_node(scope_id, kind="scope", label=str(scope))
            graph.add_edge(scope_id, field_id, kind="writes")
        for scope in field["reader_scopes"]:
            scope_id = f"scope:{scope}"
            graph.add_node(scope_id, kind="scope", label=str(scope))
            graph.add_edge(field_id, scope_id, kind="read_by")
        for scope in field["schema_scopes"]:
            scope_id = f"scope:{scope}"
            graph.add_node(scope_id, kind="scope", label=str(scope))
            graph.add_edge(scope_id, field_id, kind="declares_schema")
    return graph


def write_field_lineage_outputs(
    lineage: dict[str, object],
    *,
    output: Path,
    graphml_output: Path,
    markdown_output: Path,
) -> None:
    """Write JSON, GraphML, and compact Markdown field-lineage evidence."""
    output.parent.mkdir(parents=True, exist_ok=True)
    graphml_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    nx.write_graphml(build_field_lineage_graph(lineage), graphml_output)

    fields = list(lineage["fields"])
    production_dead = [
        field
        for field in fields
        if field["reuse"] == "no_reader_dead_candidate"
        and "production" in field["surfaces"]
    ]
    production_cleanup_candidates = [
        field
        for field in production_dead
        if field["cleanup_classification"] == "dead_write_candidate"
    ]
    lines = [
        "# Field Function Lineage",
        "",
        "Static source: LibCST. Graph model/export: NetworkX GraphML.",
        "",
        f"- Fields: `{lineage['field_count']}`",
        f"- Production no-reader raw candidates: `{len(production_dead)}`",
        (
            "- Production cleanup candidates excluding graph/env/config/schema: "
            f"`{len(production_cleanup_candidates)}`"
        ),
        "",
        "## Reuse Counts",
        "",
    ]
    for reuse, count in lineage["reuse_counts"].items():
        lines.append(f"- `{reuse}`: `{count}`")
    lines.extend(["", "## Cleanup Classifications", ""])
    for cleanup_classification, count in lineage["cleanup_classification_counts"].items():
        lines.append(f"- `{cleanup_classification}`: `{count}`")
    lines.extend(["", "## Production Cleanup Candidates", ""])
    for field in production_cleanup_candidates:
        lines.append(f"- `{field['key']}`")
        for scope in field["writer_scopes"]:
            lines.append(f"  - writes: `{scope}`")
    lines.extend(["", "## Production No-Reader Raw Candidates", ""])
    for field in production_dead:
        lines.append(f"- `{field['key']}` — `{field['cleanup_classification']}`")
        for scope in field["writer_scopes"]:
            lines.append(f"  - writes: `{scope}`")
        for scope in field["schema_scopes"]:
            lines.append(f"  - schema: `{scope}`")
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "build_field_lineage",
    "build_field_lineage_graph",
    "write_field_lineage_outputs",
]
