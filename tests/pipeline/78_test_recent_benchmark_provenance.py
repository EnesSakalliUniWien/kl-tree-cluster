"""Tests for recent benchmark-result provenance coverage scanning."""

import hashlib
from pathlib import Path

from scripts.verify_recent_benchmark_provenance import (
    coverage_gaps,
    scan_recent_folders,
    validate_plot_manifest,
    validate_recent_csv_timestamps,
    validate_recent_manifest_references,
    validate_recent_manifest_sources,
    validate_recent_manifest_timestamps,
    validate_recent_text_timestamps,
    validate_static_artifact_manifest,
    validate_static_manifests,
)


def _plot_manifest_csv_fields(csv_path: Path) -> str:
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    return (
        f'  "manifest_csv_bytes": {csv_path.stat().st_size},\n'
        f'  "manifest_csv_sha256": "{digest}",\n'
    )


def test_recent_provenance_scan_accepts_manifest_text_or_csv_timestamp(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    root.mkdir()
    manifest_dir = root / "with_manifest_20260624"
    text_dir = root / "with_text_20260624"
    csv_dir = root / "with_csv_20260624"
    old_dir = root / "old_uncovered_20260601"
    for folder in [manifest_dir, text_dir, csv_dir, old_dir]:
        folder.mkdir()

    (manifest_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (text_dir / "report.md").write_text("Generated at: 2026-06-24T20:00:00+02:00\n")
    (csv_dir / "summary.csv").write_text("generated_at,value\n2026-06-24T20:00:00+02:00,1\n")
    (old_dir / "summary.csv").write_text("value\n1\n")

    rows = scan_recent_folders(root)

    assert {row.path.name for row in rows} == {
        "with_manifest_20260624",
        "with_text_20260624",
        "with_csv_20260624",
    }
    assert coverage_gaps(rows) == []


def test_recent_provenance_scan_reports_uncovered_recent_folder(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    uncovered = root / "uncovered_20260624"
    uncovered.mkdir(parents=True)
    (uncovered / "summary.csv").write_text("value\n1\n", encoding="utf-8")

    rows = scan_recent_folders(root)
    gaps = coverage_gaps(rows)

    assert [gap.path.name for gap in gaps] == ["uncovered_20260624"]


def test_recent_manifest_timestamp_validation_accepts_run_at_utc(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    with_run_at = root / "with_run_at_20260624"
    missing = root / "missing_timestamp_20260624"
    invalid = root / "invalid_timestamp_20260624"
    invalid_list = root / "invalid_timestamp_list_20260624"
    invalid_shape = root / "invalid_timestamp_shape_20260624"
    old = root / "missing_timestamp_20260601"
    for folder in [with_run_at, missing, invalid, invalid_list, invalid_shape, old]:
        folder.mkdir(parents=True)
    (with_run_at / "manifest.json").write_text(
        '{"run_at_utc": "2026-06-24T18:00:00+00:00"}',
        encoding="utf-8",
    )
    (missing / "manifest.json").write_text('{"rows": []}', encoding="utf-8")
    (invalid / "manifest.json").write_text(
        '{"generated_at": "not-a-timestamp"}',
        encoding="utf-8",
    )
    (invalid_list / "manifest.json").write_text(
        '{"generated_at": "2026-06-24T18:00:00+00:00", "generated_artifact_timestamps": ["bad"]}',
        encoding="utf-8",
    )
    (invalid_shape / "manifest.json").write_text(
        '{"generated_at": "2026-06-24T18:00:00+00:00", "generated_artifact_timestamps": null}',
        encoding="utf-8",
    )
    (old / "manifest.json").write_text('{"rows": []}', encoding="utf-8")

    errors = validate_recent_manifest_timestamps(root)

    assert errors == [
        f"{invalid / 'manifest.json'}: invalid timestamp generated_at not-a-timestamp",
        f"{invalid_list / 'manifest.json'}: invalid timestamp generated_artifact_timestamps bad",
        f"{invalid_shape / 'manifest.json'}: invalid timestamp field generated_artifact_timestamps",
        f"{missing / 'manifest.json'}: recent manifest has no top-level timestamp",
    ]


def test_recent_csv_timestamp_validation_rejects_malformed_values(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    current = root / "csv_timestamp_20260624"
    old = root / "csv_timestamp_20260601"
    current.mkdir(parents=True)
    old.mkdir(parents=True)
    (current / "summary.csv").write_text(
        "generated_at,value\nnot-a-timestamp,1\n",
        encoding="utf-8",
    )
    (old / "summary.csv").write_text(
        "generated_at,value\nnot-a-timestamp,1\n",
        encoding="utf-8",
    )

    errors = validate_recent_csv_timestamps(root)

    assert errors == [f"{current / 'summary.csv'}: invalid timestamp generated_at not-a-timestamp"]


def test_recent_csv_timestamp_validation_rejects_empty_timestamp_columns(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    current = root / "csv_empty_timestamp_20260624"
    current.mkdir(parents=True)
    (current / "summary.csv").write_text("generated_at,value\n,1\n", encoding="utf-8")

    errors = validate_recent_csv_timestamps(root)

    assert errors == [f"{current / 'summary.csv'}: timestamp column generated_at has no values"]


def test_recent_text_timestamp_validation_rejects_label_without_parseable_value(
    tmp_path: Path,
):
    root = tmp_path / "benchmark-results"
    current = root / "text_timestamp_20260624"
    old = root / "text_timestamp_20260601"
    current.mkdir(parents=True)
    old.mkdir(parents=True)
    (current / "report.md").write_text("Generated at: not-a-timestamp\n", encoding="utf-8")
    (old / "report.md").write_text("Generated at: not-a-timestamp\n", encoding="utf-8")

    errors = validate_recent_text_timestamps(root)

    assert errors == [f"{current / 'report.md'}: timestamp label has no parseable value"]


def test_recent_text_timestamp_validation_accepts_html_generated_at(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    current = root / "html_timestamp_20260624"
    current.mkdir(parents=True)
    (current / "index.html").write_text(
        "<p><strong>Generated at:</strong> 2026-06-24T21:21:03+02:00</p>",
        encoding="utf-8",
    )

    assert validate_recent_text_timestamps(root) == []


def test_recent_manifest_reference_validation_checks_declared_csv_paths(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    current = root / "missing_reference_20260624"
    old = root / "missing_reference_20260601"
    for folder in [current, old]:
        folder.mkdir(parents=True)
    (current / "manifest.json").write_text(
        '{"generated_at": "2026-06-24T20:00:00+02:00", "results_csv": "missing.csv"}',
        encoding="utf-8",
    )
    (old / "manifest.json").write_text(
        '{"generated_at": "2026-06-01T20:00:00+02:00", "results_csv": "missing.csv"}',
        encoding="utf-8",
    )

    errors = validate_recent_manifest_references(root, project_root=root)

    assert errors == [f"{current / 'manifest.json'}: missing referenced results_csv missing.csv"]


def test_recent_manifest_reference_validation_checks_declared_csv_path_lists(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    current = root / "missing_reference_list_20260624"
    current.mkdir(parents=True)
    (current / "present.csv").write_text("value\n1\n", encoding="utf-8")
    (current / "manifest.json").write_text(
        (
            '{"generated_at": "2026-06-24T20:00:00+02:00", '
            '"results_csv": ["present.csv", "missing.csv"]}'
        ),
        encoding="utf-8",
    )

    errors = validate_recent_manifest_references(root, project_root=root)

    assert errors == [f"{current / 'manifest.json'}: missing referenced results_csv missing.csv"]


def test_recent_manifest_reference_validation_rejects_malformed_path_field(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    current = root / "bad_reference_field_20260624"
    current.mkdir(parents=True)
    (current / "manifest.json").write_text(
        '{"generated_at": "2026-06-24T20:00:00+02:00", "results_csv": ["ok.csv", 3]}',
        encoding="utf-8",
    )

    errors = validate_recent_manifest_references(root, project_root=root)

    assert errors == [f"{current / 'manifest.json'}: invalid path field results_csv"]


def test_recent_manifest_source_validation_checks_declared_source_paths(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    current = root / "missing_source_20260624"
    old = root / "missing_source_20260601"
    for folder in [current, old]:
        folder.mkdir(parents=True)
    (current / "manifest.json").write_text(
        '{"generated_at": "2026-06-24T20:00:00+02:00", "source_script": "missing.py"}',
        encoding="utf-8",
    )
    (old / "manifest.json").write_text(
        '{"generated_at": "2026-06-01T20:00:00+02:00", "source_script": "missing.py"}',
        encoding="utf-8",
    )

    errors = validate_recent_manifest_sources(root, project_root=root)

    assert errors == [f"{current / 'manifest.json'}: missing source source_script missing.py"]


def test_recent_manifest_source_validation_rejects_malformed_path_field(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    current = root / "bad_source_field_20260624"
    current.mkdir(parents=True)
    (current / "manifest.json").write_text(
        '{"generated_at": "2026-06-24T20:00:00+02:00", "source_inputs": ["input.csv", 3]}',
        encoding="utf-8",
    )

    errors = validate_recent_manifest_sources(root, project_root=root)

    assert errors == [f"{current / 'manifest.json'}: invalid path field source_inputs"]


def test_static_manifest_validation_checks_declared_hash_and_csv_shape(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value,label\n1,a\n2,b\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 20,
      "sha256": "34a690584c43949769dc90be70bdf9fe22622251b54ce98ce00794d3fe22e395",
      "rows": 2,
      "columns": 2
    }
  ]
}
""",
        encoding="utf-8",
    )

    assert validate_static_artifact_manifest(manifest, project_root=tmp_path) == []


def test_static_manifest_validation_reports_stale_hash(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 8,
      "sha256": "not-a-real-hash",
      "rows": 1,
      "columns": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: sha256 mismatch for summary.csv"]


def test_static_manifest_validation_requires_artifact_hash(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 8,
      "rows": 1,
      "columns": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing sha256 for summary.csv"]


def test_static_manifest_validation_rejects_non_string_artifact_hash(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 8,
      "sha256": true,
      "rows": 1,
      "columns": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: invalid sha256 for summary.csv"]


def test_static_manifest_validation_requires_artifact_bytes(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "sha256": "1a80986111952a11d02e84dbed98ae00f279469aad0615d17fa81911f8a6b428",
      "rows": 1,
      "columns": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing bytes for summary.csv"]


def test_static_manifest_validation_rejects_non_integer_artifact_bytes(tmp_path: Path):
    artifact = tmp_path / "plot.png"
    artifact.write_bytes(b"x")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        f"""{{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {{
      "path": "plot.png",
      "bytes": true,
      "sha256": "{digest}"
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: invalid bytes for plot.png"]


def test_static_manifest_validation_requires_text_lines(tmp_path: Path):
    artifact = tmp_path / "report.md"
    artifact.write_text("Generated at: 2026-06-24T20:00:00+02:00\nbody\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "report.md",
      "bytes": 45,
      "sha256": "f577f6992ebdfb5036a92517e7979bdb9bb2a4303a5b883cc5abd191b5647bd0"
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing lines for report.md"]


def test_static_manifest_validation_requires_csv_rows(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 8,
      "sha256": "1a80986111952a11d02e84dbed98ae00f279469aad0615d17fa81911f8a6b428",
      "columns": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing CSV rows for summary.csv"]


def test_static_manifest_validation_requires_csv_columns(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 8,
      "sha256": "1a80986111952a11d02e84dbed98ae00f279469aad0615d17fa81911f8a6b428",
      "rows": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing CSV columns for summary.csv"]


def test_static_manifest_validation_rejects_non_integer_csv_shape(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        f"""{{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {{
      "path": "summary.csv",
      "bytes": 8,
      "sha256": "{digest}",
      "rows": true,
      "columns": true
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [
        f"{manifest}: invalid CSV rows for summary.csv",
        f"{manifest}: invalid CSV columns for summary.csv",
    ]


def test_static_manifest_validation_checks_declared_generated_at_values(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("generated_at,value\n2026-06-24T20:00:00+02:00,1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 47,
      "sha256": "6315e2f4c9d9816b622e5f3d399d5922dbe423229f3977c3d9cf2152be615aae",
      "rows": 1,
      "columns": 2,
      "generated_at_values": ["2026-06-24T19:00:00+02:00"]
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: generated_at_values mismatch for summary.csv"]


def test_static_manifest_validation_requires_declared_csv_timestamp_values(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("generated_at,value\n2026-06-24T20:00:00+02:00,1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 47,
      "sha256": "6315e2f4c9d9816b622e5f3d399d5922dbe423229f3977c3d9cf2152be615aae",
      "rows": 1,
      "columns": 2
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing generated_at_values for summary.csv"]


def test_static_manifest_validation_rejects_malformed_csv_timestamp_values(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("generated_at,value\n2026-06-24T20:00:00+02:00,1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 47,
      "sha256": "6315e2f4c9d9816b622e5f3d399d5922dbe423229f3977c3d9cf2152be615aae",
      "rows": 1,
      "columns": 2,
      "generated_at_values": ["2026-06-24T20:00:00+02:00", 3]
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: invalid generated_at_values for summary.csv"]


def test_static_manifest_validation_rejects_malformed_artifact_timestamp_field(
    tmp_path: Path,
):
    artifact = tmp_path / "report.md"
    artifact.write_text("Generated at: 2026-06-24T20:00:00+02:00\nbody\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        f"""{{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {{
      "path": "report.md",
      "bytes": 45,
      "sha256": "{digest}",
      "generated_at": null,
      "lines": 2
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: invalid timestamp field generated_at"]


def test_static_manifest_validation_requires_top_level_timestamp(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 8,
      "sha256": "1a80986111952a11d02e84dbed98ae00f279469aad0615d17fa81911f8a6b428",
      "rows": 1,
      "columns": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: static manifest has no top-level timestamp"]


def test_static_manifest_validation_rejects_malformed_timestamp(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "generated_at": "not-a-timestamp",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 8,
      "sha256": "1a80986111952a11d02e84dbed98ae00f279469aad0615d17fa81911f8a6b428",
      "rows": 1,
      "columns": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: invalid timestamp generated_at not-a-timestamp"]


def test_static_manifest_validation_checks_declared_source_paths(tmp_path: Path):
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "selected_report_manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "source_scripts": ["missing_report_generator.py"],
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 8,
      "sha256": "1a80986111952a11d02e84dbed98ae00f279469aad0615d17fa81911f8a6b428",
      "rows": 1,
      "columns": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_artifact_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing source source_scripts missing_report_generator.py"]


def test_static_manifest_validation_includes_root_level_named_manifest(tmp_path: Path):
    root = tmp_path / "benchmark-results"
    root.mkdir()
    artifact = root / "summary.csv"
    artifact.write_text("value\n1\n2\n", encoding="utf-8")
    manifest = root / "selected_report_manifest.json"
    manifest.write_text(
        """{
  "manifest_schema_version": "static_artifact_provenance/v1",
  "provenance_timestamp": "2026-06-24T20:00:00+02:00",
  "artifacts": [
    {
      "path": "summary.csv",
      "bytes": 10,
      "sha256": "423259c2ffc87996610b685b3ecdfbeb0b7b79f2250a166202de07fb4fe7f311",
      "rows": 1,
      "columns": 1
    }
  ]
}
""",
        encoding="utf-8",
    )

    errors = validate_static_manifests(root, project_root=root)

    assert errors == [f"{manifest}: CSV row mismatch for summary.csv"]


def test_plot_manifest_validation_checks_existing_plot_hash_and_counts(tmp_path: Path):
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"plot")
    csv_path = tmp_path / "plot_manifest.csv"
    wrong_sha = hashlib.sha256(b"other").hexdigest()
    csv_path.write_text(
        f"plot_id,path,role,status,sha256,bytes\nplot,{plot},canonical,present,{wrong_sha},4\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
{_plot_manifest_csv_fields(csv_path)}
  "plot_count": 1,
  "role_counts": {{"canonical": 1}},
  "status_counts": {{"missing": 1}},
  "rows": [
    {{
      "bytes": 4,
      "path": "{plot}",
      "plot_id": "plot",
      "role": "canonical",
      "sha256": "{wrong_sha}",
      "status": "present"
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [
        f"{manifest}: status_counts does not match rows",
        f"{manifest}: sha256 mismatch for {plot}",
    ]


def test_plot_manifest_validation_rejects_non_integer_count_maps(tmp_path: Path):
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"plot")
    digest = hashlib.sha256(plot.read_bytes()).hexdigest()
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text(
        f"plot_id,path,role,status,sha256,bytes\nplot,{plot},canonical,present,{digest},4\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
{_plot_manifest_csv_fields(csv_path)}
  "plot_count": 1,
  "role_counts": {{"canonical": true}},
  "status_counts": {{"present": true}},
  "rows": [
    {{
      "bytes": 4,
      "path": "{plot}",
      "plot_id": "plot",
      "role": "canonical",
      "sha256": "{digest}",
      "status": "present"
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [
        f"{manifest}: invalid status_counts",
        f"{manifest}: invalid role_counts",
    ]


def test_plot_manifest_validation_rejects_malformed_generated_at(tmp_path: Path):
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text(
        "plot_id,path,role,status,sha256,bytes\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "not-a-timestamp",
  "manifest_csv": "{csv_path}",
{_plot_manifest_csv_fields(csv_path)}
  "plot_count": 0,
  "role_counts": {{}},
  "status_counts": {{}},
  "rows": []
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: invalid timestamp generated_at not-a-timestamp"]


def test_plot_manifest_validation_requires_manifest_csv_bytes(tmp_path: Path):
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text("plot_id,path,role,status,sha256,bytes\n", encoding="utf-8")
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
  "manifest_csv_sha256": "{csv_digest}",
  "plot_count": 0,
  "role_counts": {{}},
  "status_counts": {{}},
  "rows": []
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing manifest_csv_bytes"]


def test_plot_manifest_validation_rejects_non_integer_plot_count_and_csv_bytes(
    tmp_path: Path,
):
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text("plot_id,path,role,status,sha256,bytes\n", encoding="utf-8")
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
  "manifest_csv_bytes": true,
  "manifest_csv_sha256": "{csv_digest}",
  "plot_count": true,
  "role_counts": {{}},
  "status_counts": {{}},
  "rows": []
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [
        f"{manifest}: invalid plot_count",
        f"{manifest}: invalid manifest_csv_bytes",
    ]


def test_plot_manifest_validation_checks_manifest_csv_hash(tmp_path: Path):
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text("plot_id,path,role,status,sha256,bytes\n", encoding="utf-8")
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
  "manifest_csv_bytes": {csv_path.stat().st_size},
  "manifest_csv_sha256": "not-a-real-hash",
  "plot_count": 0,
  "role_counts": {{}},
  "status_counts": {{}},
  "rows": []
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: manifest_csv sha256 mismatch"]


def test_plot_manifest_validation_rejects_non_string_manifest_csv_hash(tmp_path: Path):
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text("plot_id,path,role,status,sha256,bytes\n", encoding="utf-8")
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
  "manifest_csv_bytes": {csv_path.stat().st_size},
  "manifest_csv_sha256": true,
  "plot_count": 0,
  "role_counts": {{}},
  "status_counts": {{}},
  "rows": []
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: invalid manifest_csv_sha256"]


def test_plot_manifest_validation_reports_non_object_rows(tmp_path: Path):
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text(
        "plot_id,path,role,status,sha256,bytes\n,,,,,\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
{_plot_manifest_csv_fields(csv_path)}
  "plot_count": 1,
  "role_counts": {{"None": 1}},
  "status_counts": {{"None": 1}},
  "rows": [3]
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: row 0 is not an object"]


def test_plot_manifest_validation_checks_manifest_csv_row_fields(tmp_path: Path):
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"plot")
    digest = hashlib.sha256(b"plot").hexdigest()
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text(
        "plot_id,path,role,status,sha256,bytes,input_tables\n"
        f"plot,{plot},canonical,present,{digest},4,stale_input.csv\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
{_plot_manifest_csv_fields(csv_path)}
  "plot_count": 1,
  "role_counts": {{"canonical": 1}},
  "status_counts": {{"present": 1}},
  "rows": [
    {{
      "bytes": 4,
      "input_tables": "source_input.csv",
      "path": "{plot}",
      "plot_id": "plot",
      "role": "canonical",
      "sha256": "{digest}",
      "status": "present"
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [
        f"{manifest}: manifest_csv row 0 input_tables mismatch",
        f"{manifest}: missing input_table source_input.csv",
    ]


def test_plot_manifest_validation_checks_generator_script_paths(tmp_path: Path):
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"plot")
    digest = hashlib.sha256(b"plot").hexdigest()
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text(
        f"plot_id,path,role,status,sha256,bytes\nplot,{plot},canonical,present,{digest},4\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
{_plot_manifest_csv_fields(csv_path)}
  "plot_count": 1,
  "role_counts": {{"canonical": 1}},
  "status_counts": {{"present": 1}},
  "rows": [
    {{
      "bytes": 4,
      "generator_script": "missing_plot_generator.py",
      "path": "{plot}",
      "plot_id": "plot",
      "role": "canonical",
      "sha256": "{digest}",
      "status": "present"
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing generator_script missing_plot_generator.py"]


def test_plot_manifest_validation_checks_present_input_tables(tmp_path: Path):
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"plot")
    digest = hashlib.sha256(b"plot").hexdigest()
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text(
        f"plot_id,path,role,status,sha256,bytes\nplot,{plot},canonical,present,{digest},4\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
{_plot_manifest_csv_fields(csv_path)}
  "plot_count": 1,
  "role_counts": {{"canonical": 1}},
  "status_counts": {{"present": 1}},
  "rows": [
    {{
      "bytes": 4,
      "input_tables": "missing_input.csv",
      "path": "{plot}",
      "plot_id": "plot",
      "role": "canonical",
      "sha256": "{digest}",
      "status": "present"
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: missing input_table missing_input.csv"]


def test_plot_manifest_validation_rejects_non_integer_plot_bytes(tmp_path: Path):
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"x")
    digest = hashlib.sha256(plot.read_bytes()).hexdigest()
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text(
        f"plot_id,path,role,status,sha256,bytes\nplot,{plot},canonical,present,{digest},True\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
{_plot_manifest_csv_fields(csv_path)}
  "plot_count": 1,
  "role_counts": {{"canonical": 1}},
  "status_counts": {{"present": 1}},
  "rows": [
    {{
      "bytes": true,
      "path": "{plot}",
      "plot_id": "plot",
      "role": "canonical",
      "sha256": "{digest}",
      "status": "present"
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: invalid bytes for {plot}"]


def test_plot_manifest_validation_rejects_non_string_plot_hash(tmp_path: Path):
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"plot")
    csv_path = tmp_path / "plot_manifest.csv"
    csv_path.write_text(
        f"plot_id,path,role,status,sha256,bytes\nplot,{plot},canonical,present,True,4\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        f"""{{
  "dataset": "unit",
  "generated_at": "2026-06-24T20:00:00+02:00",
  "manifest_csv": "{csv_path}",
{_plot_manifest_csv_fields(csv_path)}
  "plot_count": 1,
  "role_counts": {{"canonical": 1}},
  "status_counts": {{"present": 1}},
  "rows": [
    {{
      "bytes": 4,
      "path": "{plot}",
      "plot_id": "plot",
      "role": "canonical",
      "sha256": true,
      "status": "present"
    }}
  ]
}}
""",
        encoding="utf-8",
    )

    errors = validate_plot_manifest(manifest, project_root=tmp_path)

    assert errors == [f"{manifest}: invalid sha256 for {plot}"]
