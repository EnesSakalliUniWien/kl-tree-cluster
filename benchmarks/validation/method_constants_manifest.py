#!/usr/bin/env python3
"""Create and validate method-constant evidence manifests.

The manifest is intentionally conservative: it records benchmark output paths
as candidate sources, but it does not extract or infer metrics from them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "method_constant_validation_manifest/v1"
GENERATED_BY = "benchmarks.validation.method_constants_manifest"
EVIDENCE_STATUSES = ("missing", "partial", "complete")
RECOGNIZED_OUTPUT_NAMES = (
    "batch_summary.csv",
    "cluster_assignments.csv",
    "cluster_sizes.csv",
    "failure_report.md",
    "full_benchmark_comparison.csv",
    "reference_endotype_alignment.csv",
    "reference_endotype_confusion_matrix.csv",
    "summary.json",
)
COMMON_REQUIRED_OUTPUT_FIELDS = (
    "validation_design",
    "source_artifact_paths",
    "code_commit",
    "run_command",
    "random_seed_policy",
    "n_replicates",
    "baseline_setting",
    "candidate_setting",
    "primary_endpoint",
    "effect_estimate",
    "confidence_interval",
    "decision_rule",
    "limitations",
)

CONSTANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "constant_id": "edge_alpha",
        "display_name": "Edge alpha",
        "default_value": 0.001,
        "validation_question": (
            "Does the child-parent edge alpha control null edge discoveries while "
            "retaining planted split power?"
        ),
        "additional_required_output_fields": (
            "null_edge_discovery_rate",
            "planted_edge_power",
            "alpha_grid",
        ),
    },
    {
        "constant_id": "sibling_alpha",
        "display_name": "Sibling alpha",
        "default_value": 0.01,
        "validation_question": (
            "Does the sibling alpha control focal sibling decisions after the "
            "empirical-null inflation step?"
        ),
        "additional_required_output_fields": (
            "null_sibling_discovery_rate",
            "planted_sibling_power",
            "alpha_grid",
        ),
    },
    {
        "constant_id": "mp_upper_edge_threshold",
        "display_name": "Marchenko-Pastur upper-edge threshold",
        "default_value": "upper-edge dimension rule",
        "validation_question": (
            "Does the Marchenko-Pastur upper-edge rule select projection dimensions "
            "that preserve calibration and power?"
        ),
        "additional_required_output_fields": (
            "dimension_selection_distribution",
            "null_calibration_summary",
            "planted_signal_recovery_summary",
        ),
    },
    {
        "constant_id": "min_spectral_dimension",
        "display_name": "Minimum spectral dimension",
        "default_value": 2,
        "validation_question": (
            "Does the spectral-dimension floor avoid degenerate tests without "
            "inflating false discoveries?"
        ),
        "additional_required_output_fields": (
            "dimension_floor_grid",
            "floor_activation_rate",
            "calibration_by_floor_setting",
        ),
    },
    {
        "constant_id": "sibling_projection_dimension_rule",
        "display_name": "Sibling projection dimension",
        "default_value": "geometric mean of child edge dimensions",
        "validation_question": (
            "Does the sibling projection-dimension rule balance sibling-test "
            "calibration and planted sibling contrast power?"
        ),
        "additional_required_output_fields": (
            "dimension_rule_grid",
            "sibling_dimension_distribution",
            "calibration_by_dimension_rule",
        ),
    },
    {
        "constant_id": "empirical_null_weight_rule",
        "display_name": "Empirical-null weight rule",
        "default_value": "edge-adjusted-p-value calibration weight",
        "validation_question": (
            "Does the empirical-null weight rule produce stable inflation estimates "
            "without treating the weight as a posterior null probability?"
        ),
        "additional_required_output_fields": (
            "weight_rule_grid",
            "effective_null_weight_summary",
            "inflation_estimate_stability",
        ),
    },
    {
        "constant_id": "context_bandwidth_rule",
        "display_name": "Context bandwidth",
        "default_value": "bandwidth over log projection dimension",
        "validation_question": (
            "Is the context bandwidth robust when calibration records are sparse or "
            "projection dimensions vary?"
        ),
        "additional_required_output_fields": (
            "bandwidth_grid",
            "records_per_context_summary",
            "inflation_sensitivity_summary",
        ),
    },
    {
        "constant_id": "pass_through_traversal",
        "display_name": "Pass-through traversal",
        "default_value": True,
        "validation_question": (
            "Does pass-through traversal recover descendant signal without increasing "
            "false final-cluster discoveries under null structure?"
        ),
        "additional_required_output_fields": (
            "traversal_modes",
            "final_cluster_count_control",
            "descendant_signal_recovery",
        ),
    },
)

CONSTANT_IDS = tuple(spec["constant_id"] for spec in CONSTANT_SPECS)
REQUIRED_OUTPUT_FIELDS = {
    spec["constant_id"]: list(
        COMMON_REQUIRED_OUTPUT_FIELDS + tuple(spec["additional_required_output_fields"])
    )
    for spec in CONSTANT_SPECS
}

MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "manifest_schema_version",
        "created_utc",
        "generated_by",
        "source_paths",
        "required_output_fields",
        "constants",
    ],
    "properties": {
        "manifest_schema_version": {"const": SCHEMA_VERSION},
        "created_utc": {"type": "string"},
        "generated_by": {"const": GENERATED_BY},
        "source_paths": {"type": "array"},
        "required_output_fields": {"type": "object"},
        "constants": {"type": "array"},
    },
}


def create_manifest(
    source_paths: Iterable[str | Path],
    *,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Return a manifest skeleton for method-constant validation evidence."""

    return {
        "manifest_schema_version": SCHEMA_VERSION,
        "created_utc": created_utc or _utc_now(),
        "generated_by": GENERATED_BY,
        "source_paths": [_describe_source_path(Path(path)) for path in source_paths],
        "required_output_fields": _copy_required_output_fields(),
        "constants": [_constant_entry(spec) for spec in CONSTANT_SPECS],
        "notes": (
            "Skeleton only. Benchmark paths are candidate sources; metrics remain "
            "empty until a validation run explicitly supplies every required field."
        ),
    }


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return schema errors for a manifest-like mapping."""

    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return ["manifest must be an object"]

    for key in MANIFEST_SCHEMA["required"]:
        if key not in manifest:
            errors.append(f"missing required top-level field: {key}")

    if manifest.get("manifest_schema_version") != SCHEMA_VERSION:
        errors.append(
            "manifest_schema_version must be "
            f"{SCHEMA_VERSION!r}, got {manifest.get('manifest_schema_version')!r}"
        )
    if manifest.get("generated_by") != GENERATED_BY:
        errors.append(f"generated_by must be {GENERATED_BY!r}")

    _validate_source_paths(manifest.get("source_paths"), errors)
    _validate_required_output_fields(manifest.get("required_output_fields"), errors)
    _validate_constants(manifest.get("constants"), errors)
    return errors


def write_manifest(
    manifest: Mapping[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and write a manifest JSON file."""

    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid manifest:\n" + "\n".join(f"- {error}" for error in errors))

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _constant_entry(spec: Mapping[str, Any]) -> dict[str, Any]:
    constant_id = str(spec["constant_id"])
    return {
        "constant_id": constant_id,
        "display_name": spec["display_name"],
        "default_value": spec["default_value"],
        "validation_question": spec["validation_question"],
        "required_output_fields": list(REQUIRED_OUTPUT_FIELDS[constant_id]),
        "evidence_status": "missing",
        "evidence": _missing_evidence(constant_id),
    }


def _missing_evidence(constant_id: str) -> dict[str, Any]:
    return {
        "status": "missing",
        "source_path": None,
        "source_artifacts": [],
        "metrics": {},
        "missing_required_fields": list(REQUIRED_OUTPUT_FIELDS[constant_id]),
        "notes": "No validation evidence has been attached for this constant.",
    }


def _describe_source_path(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    exists = expanded.exists()
    if not exists:
        kind = "missing"
        recognized_outputs: list[str] = []
    elif expanded.is_dir():
        kind = "directory"
        recognized_outputs = _recognized_directory_outputs(expanded)
    elif expanded.is_file():
        kind = "file"
        recognized_outputs = [expanded.name] if expanded.name in RECOGNIZED_OUTPUT_NAMES else []
    else:
        kind = "other"
        recognized_outputs = []

    return {
        "path": str(path),
        "exists": exists,
        "kind": kind,
        "recognized_outputs": recognized_outputs,
        "used_as_evidence": False,
        "notes": (
            "Candidate source only; this scaffold does not read metrics from benchmark "
            "outputs."
        ),
    }


def _recognized_directory_outputs(path: Path) -> list[str]:
    try:
        children = list(path.iterdir())
    except OSError:
        return []
    return sorted(child.name for child in children if child.name in RECOGNIZED_OUTPUT_NAMES)


def _copy_required_output_fields() -> dict[str, list[str]]:
    return {constant_id: list(fields) for constant_id, fields in REQUIRED_OUTPUT_FIELDS.items()}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_source_paths(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("source_paths must be a list")
        return
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            errors.append(f"source_paths[{index}] must be an object")
            continue
        _require_type(entry, "path", str, f"source_paths[{index}]", errors)
        _require_type(entry, "exists", bool, f"source_paths[{index}]", errors)
        _require_type(entry, "kind", str, f"source_paths[{index}]", errors)
        _require_type(entry, "recognized_outputs", list, f"source_paths[{index}]", errors)
        _require_type(entry, "used_as_evidence", bool, f"source_paths[{index}]", errors)
        kind = entry.get("kind")
        if isinstance(kind, str) and kind not in {"directory", "file", "missing", "other"}:
            errors.append(f"source_paths[{index}].kind has unsupported value: {kind!r}")
        if entry.get("used_as_evidence") is not False:
            errors.append(f"source_paths[{index}].used_as_evidence must be false in skeletons")


def _validate_required_output_fields(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append("required_output_fields must be an object")
        return
    if dict(value) != REQUIRED_OUTPUT_FIELDS:
        errors.append("required_output_fields must match the built-in method-constant contract")


def _validate_constants(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("constants must be a list")
        return
    ids = [entry.get("constant_id") for entry in value if isinstance(entry, Mapping)]
    if ids != list(CONSTANT_IDS):
        errors.append(
            "constant_id set and order must match the built-in method-constant contract"
        )

    seen: set[str] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            errors.append(f"constants[{index}] must be an object")
            continue
        context = f"constants[{index}]"
        _require_type(entry, "constant_id", str, context, errors)
        _require_type(entry, "display_name", str, context, errors)
        if "default_value" not in entry:
            errors.append(f"{context}.default_value is required")
        _require_type(entry, "validation_question", str, context, errors)
        _require_type(entry, "required_output_fields", list, context, errors)
        _require_type(entry, "evidence_status", str, context, errors)
        constant_id = entry.get("constant_id")
        if not isinstance(constant_id, str):
            continue
        if constant_id in seen:
            errors.append(f"{context}.constant_id duplicates {constant_id!r}")
        seen.add(constant_id)
        if constant_id not in REQUIRED_OUTPUT_FIELDS:
            errors.append(f"{context}.constant_id is unknown: {constant_id!r}")
            continue
        if entry.get("required_output_fields") != REQUIRED_OUTPUT_FIELDS[constant_id]:
            errors.append(f"{context}.required_output_fields does not match contract")
        status = entry.get("evidence_status")
        if status not in EVIDENCE_STATUSES:
            errors.append(f"{context}.evidence_status has unsupported value: {status!r}")
        _validate_evidence(entry.get("evidence"), constant_id, status, context, errors)


def _validate_evidence(
    evidence: Any,
    constant_id: str,
    status: Any,
    context: str,
    errors: list[str],
) -> None:
    if not isinstance(evidence, Mapping):
        errors.append(f"{context}.evidence must be an object")
        return
    evidence_context = f"{context}.evidence"
    _require_type(evidence, "status", str, evidence_context, errors)
    _require_type(evidence, "source_artifacts", list, evidence_context, errors)
    _require_type(evidence, "metrics", dict, evidence_context, errors)
    _require_type(evidence, "missing_required_fields", list, evidence_context, errors)
    if evidence.get("status") != status:
        errors.append(f"{evidence_context}.status must match {context}.evidence_status")
    if status == "missing":
        if evidence.get("source_path") is not None:
            errors.append(f"{evidence_context}.source_path must be null when evidence is missing")
        if evidence.get("source_artifacts") != []:
            errors.append(
                f"{evidence_context}.source_artifacts must be empty when evidence is missing"
            )
        if evidence.get("metrics") != {}:
            errors.append(f"{evidence_context}.metrics must be empty when evidence is missing")
        expected_missing = REQUIRED_OUTPUT_FIELDS[constant_id]
        if evidence.get("missing_required_fields") != expected_missing:
            errors.append(
                f"{evidence_context}.missing_required_fields must list every required field"
            )
    elif status == "complete" and evidence.get("missing_required_fields") != []:
        errors.append(f"{evidence_context}.missing_required_fields must be empty when complete")


def _require_type(
    mapping: Mapping[str, Any],
    key: str,
    expected_type: type,
    context: str,
    errors: list[str],
) -> None:
    if key not in mapping:
        errors.append(f"{context}.{key} is required")
        return
    if not isinstance(mapping[key], expected_type):
        errors.append(
            f"{context}.{key} must be {expected_type.__name__}, "
            f"got {type(mapping[key]).__name__}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or validate method-constant validation manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create",
        help="Create a manifest skeleton from benchmark output paths.",
    )
    create_parser.add_argument(
        "source_paths",
        nargs="*",
        help="Existing benchmark output files or directories to record as candidate sources.",
    )
    create_parser.add_argument(
        "-o",
        "--output",
        default="benchmarks/validation/manifests/method_constant_validation_manifest.json",
        help="Manifest JSON path to write.",
    )
    create_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest file.",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate a manifest JSON file.")
    validate_parser.add_argument("manifest", help="Manifest JSON path to validate.")

    subparsers.add_parser("constants", help="Print the constants requiring validation.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "create":
        manifest = create_manifest(args.source_paths)
        try:
            path = write_manifest(manifest, args.output, overwrite=args.overwrite)
        except (FileExistsError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote manifest skeleton: {path}")
        return 0

    if args.command == "validate":
        try:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: unable to read manifest: {exc}", file=sys.stderr)
            return 1
        errors = validate_manifest(manifest)
        if errors:
            print("Manifest is invalid:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("Manifest is valid.")
        return 0

    if args.command == "constants":
        for constant_id in CONSTANT_IDS:
            spec = next(spec for spec in CONSTANT_SPECS if spec["constant_id"] == constant_id)
            print(f"{constant_id}: {spec['display_name']}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
