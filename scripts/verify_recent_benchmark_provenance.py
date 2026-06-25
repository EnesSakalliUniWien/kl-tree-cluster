"""Verify timestamp/provenance coverage for recent benchmark-result folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = Path("raw/assets/benchmark-results")
DEFAULT_DATE_PATTERN = r"2026062[34]"
TEXT_TIMESTAMP_RE = re.compile(
    r"(Generated at|generated_at|provenance_timestamp|Provenance timestamp)"
)
TEXT_TIMESTAMP_VALUE_RE = re.compile(
    r"(Generated at|generated_at|provenance_timestamp|Provenance timestamp)"
    r"(?:\s+added at)?\s*[:=]\s*"
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
)
MANIFEST_TIMESTAMP_FIELDS = ("generated_at", "provenance_timestamp", "run_at_utc")
CSV_TIMESTAMP_FIELDS = ("generated_at", "provenance_timestamp")
MANIFEST_REFERENCE_SUFFIXES = ("_csv", "_md", "_json")
MANIFEST_SOURCE_FIELDS = ("source_script", "source_scripts", "source_inputs")
PLOT_EXISTING_FILE_STATUSES = {
    "present",
    "orphaned_existing_file",
    "unclassified_existing_file",
}
TEXT_ARTIFACT_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class FolderCoverage:
    path: Path
    file_count: int
    manifest_files: tuple[str, ...]
    has_text_timestamp: bool
    has_csv_timestamp: bool

    @property
    def has_coverage(self) -> bool:
        return bool(self.manifest_files) or self.has_text_timestamp or self.has_csv_timestamp


def _is_manifest(path: Path) -> bool:
    return path.name in {"manifest.json", "plot_manifest.json"} or path.name.endswith(
        "_manifest.json"
    )


def _has_text_timestamp(path: Path) -> bool:
    if path.suffix.lower() not in {".md", ".html"}:
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")[:20000]
    return bool(TEXT_TIMESTAMP_RE.search(text))


def _has_csv_timestamp(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()
    return "generated_at" in header or "provenance_timestamp" in header


def _plain_text_for_timestamp_scan(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")[:50000]
    if path.suffix.lower() == ".html":
        return re.sub(r"<[^>]+>", " ", text)
    return text


def scan_recent_folders(
    root: Path = DEFAULT_ROOT,
    *,
    date_pattern: str = DEFAULT_DATE_PATTERN,
) -> list[FolderCoverage]:
    compiled_pattern = re.compile(date_pattern)
    rows: list[FolderCoverage] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if not compiled_pattern.search(folder.name):
            continue
        files = sorted(path for path in folder.iterdir() if path.is_file())
        rows.append(
            FolderCoverage(
                path=folder,
                file_count=len(files),
                manifest_files=tuple(path.name for path in files if _is_manifest(path)),
                has_text_timestamp=any(_has_text_timestamp(path) for path in files),
                has_csv_timestamp=any(_has_csv_timestamp(path) for path in files),
            )
        )
    return rows


def coverage_gaps(rows: list[FolderCoverage]) -> list[FolderCoverage]:
    return [row for row in rows if not row.has_coverage]


def _iter_manifest_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if _is_manifest(path))


def _has_manifest_timestamp(manifest: dict) -> bool:
    return any(isinstance(manifest.get(field), str) for field in MANIFEST_TIMESTAMP_FIELDS)


def _is_iso_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _is_timestamp_like_field(field: str) -> bool:
    if field.endswith("_source"):
        return False
    return (
        field in MANIFEST_TIMESTAMP_FIELDS
        or "timestamp" in field
        or "generated_at" in field
    )


def _timestamp_value_errors(manifest_path: Path, field: str, raw_value: object) -> list[str]:
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, list):
        values = raw_value
    else:
        return [f"{manifest_path}: invalid timestamp field {field}"]
    if not all(isinstance(value, str) for value in values):
        return [f"{manifest_path}: invalid timestamp field {field}"]
    return [
        f"{manifest_path}: invalid timestamp {field} {value}"
        for value in values
        if not _is_iso_timestamp(value)
    ]


def _manifest_timestamp_errors(manifest_path: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    for field, raw_value in manifest.items():
        if not _is_timestamp_like_field(field):
            continue
        errors.extend(_timestamp_value_errors(manifest_path, field, raw_value))
    return errors


def _manifest_path_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return []


def _manifest_path_value_errors(manifest_path: Path, field: str, value: object) -> list[str]:
    if value is None or isinstance(value, str):
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return []
    return [f"{manifest_path}: invalid path field {field}"]


def _is_external_reference(path_text: str) -> bool:
    return "://" in path_text


def _split_manifest_list(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def _resolve_artifact_path(path_text: str, project_root: Path, manifest_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    root_relative = project_root / path
    if root_relative.exists():
        return root_relative
    return manifest_dir / path


def _csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return rows, len(header)


def _csv_unique_values(path: Path, column: str) -> list[str] | None:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            return None
        return sorted({str(row[column]) for row in reader if row.get(column) not in {None, ""}})


def _is_integer_metadata(value: object) -> bool:
    return type(value) is int


def _manifest_row_value(row: object, field: str) -> object:
    if isinstance(row, dict):
        return row.get(field)
    return None


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_count_map(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and _is_integer_metadata(count) for key, count in value.items()
    )


def validate_static_artifact_manifest(
    manifest_path: Path,
    *,
    project_root: Path = Path("."),
) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema_version") != "static_artifact_provenance/v1":
        return []

    errors: list[str] = []
    if not _has_manifest_timestamp(manifest):
        errors.append(f"{manifest_path}: static manifest has no top-level timestamp")
    else:
        errors.extend(_manifest_timestamp_errors(manifest_path, manifest))
    for field in MANIFEST_SOURCE_FIELDS:
        errors.extend(_manifest_path_value_errors(manifest_path, field, manifest.get(field)))
        for path_text in _manifest_path_values(manifest.get(field)):
            if _is_external_reference(path_text):
                continue
            path = _resolve_artifact_path(path_text, project_root, manifest_path.parent)
            if not path.exists():
                errors.append(f"{manifest_path}: missing source {field} {path_text}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{manifest_path}: static manifest has no artifacts list")
        return errors

    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"{manifest_path}: artifact {index} is not an object")
            continue
        path_text = artifact.get("path")
        if not isinstance(path_text, str):
            errors.append(f"{manifest_path}: artifact {index} has no path")
            continue
        path = _resolve_artifact_path(path_text, project_root, manifest_path.parent)
        if not path.exists():
            errors.append(f"{manifest_path}: missing artifact {path_text}")
            continue

        for field, raw_value in artifact.items():
            if _is_timestamp_like_field(field) and not field.endswith("_values"):
                errors.extend(_timestamp_value_errors(manifest_path, field, raw_value))
        if "bytes" not in artifact:
            errors.append(f"{manifest_path}: missing bytes for {path_text}")
        elif not _is_integer_metadata(artifact["bytes"]):
            errors.append(f"{manifest_path}: invalid bytes for {path_text}")
        elif path.stat().st_size != artifact["bytes"]:
            errors.append(f"{manifest_path}: byte mismatch for {path_text}")
        if "sha256" not in artifact:
            errors.append(f"{manifest_path}: missing sha256 for {path_text}")
        elif not isinstance(artifact["sha256"], str) or not artifact["sha256"]:
            errors.append(f"{manifest_path}: invalid sha256 for {path_text}")
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact["sha256"]:
                errors.append(f"{manifest_path}: sha256 mismatch for {path_text}")
        if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES and "lines" not in artifact:
            errors.append(f"{manifest_path}: missing lines for {path_text}")
        elif "lines" in artifact and not _is_integer_metadata(artifact["lines"]):
            errors.append(f"{manifest_path}: invalid lines for {path_text}")
        elif "lines" in artifact:
            line_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
            if line_count != artifact["lines"]:
                errors.append(f"{manifest_path}: line-count mismatch for {path_text}")
        if path.suffix.lower() == ".csv":
            rows, columns = _csv_shape(path)
            if "rows" not in artifact:
                errors.append(f"{manifest_path}: missing CSV rows for {path_text}")
            elif not _is_integer_metadata(artifact["rows"]):
                errors.append(f"{manifest_path}: invalid CSV rows for {path_text}")
            elif rows != artifact["rows"]:
                errors.append(f"{manifest_path}: CSV row mismatch for {path_text}")
            if "columns" not in artifact:
                errors.append(f"{manifest_path}: missing CSV columns for {path_text}")
            elif not _is_integer_metadata(artifact["columns"]):
                errors.append(f"{manifest_path}: invalid CSV columns for {path_text}")
            elif columns != artifact["columns"]:
                errors.append(f"{manifest_path}: CSV column mismatch for {path_text}")
            for timestamp_field in CSV_TIMESTAMP_FIELDS:
                actual_values = _csv_unique_values(path, timestamp_field)
                if actual_values is None:
                    continue
                expected_key = f"{timestamp_field}_values"
                if expected_key not in artifact:
                    errors.append(f"{manifest_path}: missing {expected_key} for {path_text}")
                    continue
                if not _is_string_list(artifact[expected_key]):
                    errors.append(f"{manifest_path}: invalid {expected_key} for {path_text}")
                    continue
                if actual_values != artifact[expected_key]:
                    errors.append(f"{manifest_path}: {expected_key} mismatch for {path_text}")
    return errors


def validate_static_manifests(
    root: Path = DEFAULT_ROOT,
    *,
    project_root: Path = Path("."),
) -> list[str]:
    errors: list[str] = []
    for manifest_path in _iter_manifest_files(root):
        errors.extend(
            validate_static_artifact_manifest(
                manifest_path,
                project_root=project_root,
            )
        )
    return errors


def validate_plot_manifest(
    manifest_path: Path,
    *,
    project_root: Path = Path("."),
) -> list[str]:
    if manifest_path.name != "plot_manifest.json":
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not isinstance(manifest.get("generated_at"), str):
        errors.append(f"{manifest_path}: plot manifest has no generated_at timestamp")
    elif not _is_iso_timestamp(manifest["generated_at"]):
        errors.append(f"{manifest_path}: invalid timestamp generated_at {manifest['generated_at']}")

    rows = manifest.get("rows")
    if not isinstance(rows, list):
        return [*errors, f"{manifest_path}: plot manifest has no rows list"]

    if not _is_integer_metadata(manifest.get("plot_count")):
        errors.append(f"{manifest_path}: invalid plot_count")
    elif manifest.get("plot_count") != len(rows):
        errors.append(f"{manifest_path}: plot_count does not match rows")

    status_counts = dict(
        sorted(Counter(str(_manifest_row_value(row, "status")) for row in rows).items())
    )
    if not _is_count_map(manifest.get("status_counts")):
        errors.append(f"{manifest_path}: invalid status_counts")
    elif manifest.get("status_counts") != status_counts:
        errors.append(f"{manifest_path}: status_counts does not match rows")

    role_counts = dict(
        sorted(Counter(str(_manifest_row_value(row, "role")) for row in rows).items())
    )
    if not _is_count_map(manifest.get("role_counts")):
        errors.append(f"{manifest_path}: invalid role_counts")
    elif manifest.get("role_counts") != role_counts:
        errors.append(f"{manifest_path}: role_counts does not match rows")

    manifest_csv = manifest.get("manifest_csv")
    if not isinstance(manifest_csv, str):
        errors.append(f"{manifest_path}: plot manifest has no manifest_csv path")
    else:
        csv_path = _resolve_artifact_path(manifest_csv, project_root, manifest_path.parent)
        if not csv_path.exists():
            errors.append(f"{manifest_path}: missing manifest_csv {manifest_csv}")
        else:
            if "manifest_csv_bytes" not in manifest:
                errors.append(f"{manifest_path}: missing manifest_csv_bytes")
            elif not _is_integer_metadata(manifest["manifest_csv_bytes"]):
                errors.append(f"{manifest_path}: invalid manifest_csv_bytes")
            elif csv_path.stat().st_size != manifest["manifest_csv_bytes"]:
                errors.append(f"{manifest_path}: manifest_csv byte mismatch")
            if "manifest_csv_sha256" not in manifest:
                errors.append(f"{manifest_path}: missing manifest_csv_sha256")
            elif not isinstance(manifest["manifest_csv_sha256"], str) or not manifest[
                "manifest_csv_sha256"
            ]:
                errors.append(f"{manifest_path}: invalid manifest_csv_sha256")
            else:
                digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
                if digest != manifest["manifest_csv_sha256"]:
                    errors.append(f"{manifest_path}: manifest_csv sha256 mismatch")
            with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            if len(csv_rows) != len(rows):
                errors.append(f"{manifest_path}: manifest_csv row count does not match rows")
            else:
                for index, (row, csv_row) in enumerate(zip(rows, csv_rows, strict=True)):
                    if not isinstance(row, dict):
                        continue
                    for field in csv_row:
                        if str(row.get(field, "")) != csv_row.get(field, ""):
                            errors.append(
                                f"{manifest_path}: manifest_csv row {index} {field} mismatch"
                            )
                            break

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"{manifest_path}: row {index} is not an object")
            continue
        for script in _split_manifest_list(row.get("generator_script")):
            if script == "unknown" or _is_external_reference(script):
                continue
            script_path = _resolve_artifact_path(script, project_root, manifest_path.parent)
            if not script_path.exists():
                errors.append(f"{manifest_path}: missing generator_script {script}")
        if row.get("status") == "present":
            for input_table in _split_manifest_list(row.get("input_tables")):
                if (
                    input_table == "unknown"
                    or "*" in input_table
                    or _is_external_reference(input_table)
                ):
                    continue
                input_path = _resolve_artifact_path(
                    input_table,
                    project_root,
                    manifest_path.parent,
                )
                if not input_path.exists():
                    errors.append(f"{manifest_path}: missing input_table {input_table}")
        status = row.get("status")
        if status not in PLOT_EXISTING_FILE_STATUSES:
            continue
        path_text = row.get("path")
        if not isinstance(path_text, str):
            errors.append(f"{manifest_path}: row {index} has no path")
            continue
        path = _resolve_artifact_path(path_text, project_root, manifest_path.parent)
        if not path.exists():
            errors.append(f"{manifest_path}: missing plot {path_text}")
            continue
        if not _is_integer_metadata(row.get("bytes")):
            errors.append(f"{manifest_path}: invalid bytes for {path_text}")
        elif path.stat().st_size != row.get("bytes"):
            errors.append(f"{manifest_path}: byte mismatch for {path_text}")
        sha256 = row.get("sha256")
        if "sha256" not in row:
            errors.append(f"{manifest_path}: missing sha256 for {path_text}")
            continue
        if not isinstance(sha256, str) or not sha256:
            errors.append(f"{manifest_path}: invalid sha256 for {path_text}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            errors.append(f"{manifest_path}: sha256 mismatch for {path_text}")
    return errors


def validate_manifests(
    root: Path = DEFAULT_ROOT,
    *,
    project_root: Path = Path("."),
) -> list[str]:
    errors: list[str] = []
    for manifest_path in _iter_manifest_files(root):
        errors.extend(
            validate_static_artifact_manifest(
                manifest_path,
                project_root=project_root,
            )
        )
        errors.extend(
            validate_plot_manifest(
                manifest_path,
                project_root=project_root,
            )
        )
    return errors


def validate_recent_manifest_timestamps(
    root: Path = DEFAULT_ROOT,
    *,
    date_pattern: str = DEFAULT_DATE_PATTERN,
) -> list[str]:
    compiled_pattern = re.compile(date_pattern)
    errors: list[str] = []
    for manifest_path in _iter_manifest_files(root):
        if not compiled_pattern.search(str(manifest_path)):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not _has_manifest_timestamp(manifest):
            errors.append(f"{manifest_path}: recent manifest has no top-level timestamp")
        else:
            errors.extend(_manifest_timestamp_errors(manifest_path, manifest))
    return errors


def validate_recent_manifest_references(
    root: Path = DEFAULT_ROOT,
    *,
    date_pattern: str = DEFAULT_DATE_PATTERN,
    project_root: Path = Path("."),
) -> list[str]:
    compiled_pattern = re.compile(date_pattern)
    errors: list[str] = []
    for manifest_path in _iter_manifest_files(root):
        if not compiled_pattern.search(str(manifest_path)):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            continue
        for key, value in manifest.items():
            if not key.endswith(MANIFEST_REFERENCE_SUFFIXES):
                continue
            errors.extend(_manifest_path_value_errors(manifest_path, key, value))
            for path_text in _manifest_path_values(value):
                if _is_external_reference(path_text):
                    continue
                path = _resolve_artifact_path(path_text, project_root, manifest_path.parent)
                if not path.exists():
                    errors.append(f"{manifest_path}: missing referenced {key} {path_text}")
    return errors


def validate_recent_manifest_sources(
    root: Path = DEFAULT_ROOT,
    *,
    date_pattern: str = DEFAULT_DATE_PATTERN,
    project_root: Path = Path("."),
) -> list[str]:
    compiled_pattern = re.compile(date_pattern)
    errors: list[str] = []
    for manifest_path in _iter_manifest_files(root):
        if not compiled_pattern.search(str(manifest_path)):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            continue
        for field in MANIFEST_SOURCE_FIELDS:
            errors.extend(_manifest_path_value_errors(manifest_path, field, manifest.get(field)))
            for path_text in _manifest_path_values(manifest.get(field)):
                if _is_external_reference(path_text):
                    continue
                path = _resolve_artifact_path(path_text, project_root, manifest_path.parent)
                if not path.exists():
                    errors.append(f"{manifest_path}: missing source {field} {path_text}")
    return errors


def validate_recent_csv_timestamps(
    root: Path = DEFAULT_ROOT,
    *,
    date_pattern: str = DEFAULT_DATE_PATTERN,
) -> list[str]:
    compiled_pattern = re.compile(date_pattern)
    errors: list[str] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if not compiled_pattern.search(folder.name):
            continue
        for csv_path in sorted(folder.rglob("*.csv")):
            with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    continue
                timestamp_fields = [
                    field for field in CSV_TIMESTAMP_FIELDS if field in reader.fieldnames
                ]
                if not timestamp_fields:
                    continue
                seen_values = {field: False for field in timestamp_fields}
                for row in reader:
                    for field in timestamp_fields:
                        value = str(row.get(field, "")).strip()
                        if not value:
                            continue
                        seen_values[field] = True
                        if not _is_iso_timestamp(value):
                            errors.append(f"{csv_path}: invalid timestamp {field} {value}")
                for field, seen in seen_values.items():
                    if not seen:
                        errors.append(f"{csv_path}: timestamp column {field} has no values")
    return errors


def validate_recent_text_timestamps(
    root: Path = DEFAULT_ROOT,
    *,
    date_pattern: str = DEFAULT_DATE_PATTERN,
) -> list[str]:
    compiled_pattern = re.compile(date_pattern)
    errors: list[str] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if not compiled_pattern.search(folder.name):
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in {".md", ".html"} or not path.is_file():
                continue
            text = _plain_text_for_timestamp_scan(path)
            if not TEXT_TIMESTAMP_RE.search(text):
                continue
            values = [match.group(2) for match in TEXT_TIMESTAMP_VALUE_RE.finditer(text)]
            if not values:
                errors.append(f"{path}: timestamp label has no parseable value")
                continue
            for value in values:
                if not _is_iso_timestamp(value):
                    errors.append(f"{path}: invalid text timestamp {value}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify recent benchmark-result timestamp/provenance coverage."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--date-pattern", default=DEFAULT_DATE_PATTERN)
    parser.add_argument(
        "--skip-manifest-validation",
        "--skip-static-manifest-validation",
        dest="skip_manifest_validation",
        action="store_true",
        help="Only check coverage signals; do not validate manifest contents.",
    )
    args = parser.parse_args()

    rows = scan_recent_folders(args.root, date_pattern=args.date_pattern)
    gaps = coverage_gaps(rows)
    if gaps:
        for gap in gaps:
            print(f"MISSING {gap.path} files={gap.file_count}")
        return 1
    if not args.skip_manifest_validation:
        manifest_errors = validate_recent_manifest_timestamps(
            args.root,
            date_pattern=args.date_pattern,
        )
        manifest_errors.extend(
            validate_recent_manifest_references(
                args.root,
                date_pattern=args.date_pattern,
            )
        )
        manifest_errors.extend(
            validate_recent_manifest_sources(
                args.root,
                date_pattern=args.date_pattern,
            )
        )
        manifest_errors.extend(
            validate_recent_csv_timestamps(
                args.root,
                date_pattern=args.date_pattern,
            )
        )
        manifest_errors.extend(
            validate_recent_text_timestamps(
                args.root,
                date_pattern=args.date_pattern,
            )
        )
        manifest_errors.extend(
            validate_manifests(
                args.root,
            )
        )
        if manifest_errors:
            for error in manifest_errors:
                print(error)
            return 1
    print(f"recent benchmark provenance coverage ok: {len(rows)} folders checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
