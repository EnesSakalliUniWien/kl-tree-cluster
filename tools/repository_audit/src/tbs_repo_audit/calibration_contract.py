"""Calibration-support contract evidence from exported sibling records."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REQUIRED_COLUMNS = frozenset(
    {
        "parent",
        "left",
        "right",
        "degrees_of_freedom",
        "sibling_null_weight",
        "is_role_supported",
        "is_edge_blocked",
    }
)
LINE_NUMBER_KEY = "_line_number"
DEFAULT_GROUP_COLUMNS = (
    "source_case_id",
    "geometry_method",
    "tree_builder",
    "tree_linkage_method",
    "branch_source",
    "spectral_context",
)


def _parse_bool(value: str, *, column: str, line_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"{column} must contain true or false; line={line_number}, value={value!r}."
    )


def _parse_float(value: str, *, column: str, line_number: int) -> float:
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"{column} must contain a number; line={line_number}, value={value!r}."
        ) from None


def _summarize_group(rows: list[dict[str, str]]) -> dict[str, object]:
    seen_parents: set[str] = set()
    for row in rows:
        parent = row["parent"]
        if parent in seen_parents:
            raise ValueError(
                "Calibration records must hold one row per parent within a group; "
                f"parent={parent!r} repeats."
            )
        seen_parents.add(parent)

    supported: list[dict[str, object]] = []
    nonfinite_count = 0
    for row in rows:
        line_number = int(row[LINE_NUMBER_KEY])
        degrees_of_freedom = _parse_float(
            row["degrees_of_freedom"],
            column="degrees_of_freedom",
            line_number=line_number,
        )
        sibling_null_weight = _parse_float(
            row["sibling_null_weight"],
            column="sibling_null_weight",
            line_number=line_number,
        )
        if not (
            math.isfinite(degrees_of_freedom) and math.isfinite(sibling_null_weight)
        ):
            nonfinite_count += 1
            continue
        is_role_supported = _parse_bool(
            row["is_role_supported"],
            column="is_role_supported",
            line_number=line_number,
        )
        is_edge_blocked = _parse_bool(
            row["is_edge_blocked"],
            column="is_edge_blocked",
            line_number=line_number,
        )
        if degrees_of_freedom > 0.0 and sibling_null_weight > 0.0 and is_role_supported:
            supported.append({**row, "is_edge_blocked": is_edge_blocked})

    parent_by_child: dict[str, str] = {}
    for row in supported:
        parent = str(row["parent"])
        parent_by_child[str(row["left"])] = parent
        parent_by_child[str(row["right"])] = parent

    blocked_parents = {
        str(row["parent"]) for row in supported if bool(row["is_edge_blocked"])
    }
    tested_null: list[str] = []
    stopped_frontier: list[str] = []
    nested_blocked: list[str] = []
    for row in supported:
        parent = str(row["parent"])
        if not bool(row["is_edge_blocked"]):
            tested_null.append(parent)
        elif parent_by_child.get(parent) in blocked_parents:
            nested_blocked.append(parent)
        else:
            stopped_frontier.append(parent)

    return {
        "raw_record_count": len(rows),
        "nonfinite_record_count": nonfinite_count,
        "role_supported_record_count": len(supported),
        "tested_null_record_count": len(tested_null),
        "stopped_frontier_record_count": len(stopped_frontier),
        "nested_blocked_record_count": len(nested_blocked),
        "structural_calibration_record_count": (
            len(tested_null) + len(stopped_frontier)
        ),
        "tested_null_parents": sorted(tested_null),
        "stopped_frontier_parents": sorted(stopped_frontier),
        "nested_blocked_parents": sorted(nested_blocked),
    }


def build_calibration_contract_report(records_path: Path) -> dict[str, object]:
    """Return stopped-frontier multiplicity evidence for a sibling-record CSV."""
    with records_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS.difference(columns))
        if missing:
            raise ValueError(
                f"Calibration records are missing required columns: {missing!r}."
            )
        rows = []
        for row in reader:
            row[LINE_NUMBER_KEY] = str(reader.line_num)
            rows.append(row)

    group_columns = [column for column in DEFAULT_GROUP_COLUMNS if column in columns]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[column] for column in group_columns)].append(row)

    groups: list[dict[str, object]] = []
    for group_key, group_rows in sorted(grouped.items()):
        groups.append(
            {
                "group": dict(zip(group_columns, group_key, strict=True)),
                **_summarize_group(group_rows),
            }
        )

    count_fields = (
        "raw_record_count",
        "nonfinite_record_count",
        "role_supported_record_count",
        "tested_null_record_count",
        "stopped_frontier_record_count",
        "nested_blocked_record_count",
        "structural_calibration_record_count",
    )
    summary = {
        "group_count": len(groups),
        **{
            field: sum(int(group[field]) for group in groups)
            for field in count_fields
        },
    }
    return {
        "schema_version": 1,
        "adapter": "calibration_contract",
        "records_path": str(records_path),
        "group_columns": group_columns,
        "summary": summary,
        "groups": groups,
    }


def write_calibration_contract_report(
    report: dict[str, object],
    *,
    output: Path,
    markdown_output: Path,
) -> None:
    """Write JSON and compact Markdown calibration-contract evidence."""
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = report["summary"]
    assert isinstance(summary, dict)
    markdown_output.write_text(
        "\n".join(
            [
                "# Calibration support contract",
                "",
                f"- Groups: `{summary['group_count']}`",
                f"- Raw records: `{summary['raw_record_count']}`",
                f"- Non-finite records: `{summary['nonfinite_record_count']}`",
                (
                    "- Role-supported records: "
                    f"`{summary['role_supported_record_count']}`"
                ),
                f"- Tested-null records: `{summary['tested_null_record_count']}`",
                (
                    "- Stopped-subtree frontiers: "
                    f"`{summary['stopped_frontier_record_count']}`"
                ),
                (
                    "- Nested blocked descendants: "
                    f"`{summary['nested_blocked_record_count']}`"
                ),
                (
                    "- Structural calibration records: "
                    f"`{summary['structural_calibration_record_count']}`"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = [
    "build_calibration_contract_report",
    "write_calibration_contract_report",
]
