"""
Purpose: Validate debug script naming and header standards.
Inputs: Python files under debug_scripts/ plus optional CLI flags.
Outputs: Console report of violations and a non-zero exit code on failure.
Expected runtime: ~1-5 seconds.
How to run: python debug_scripts/_shared/validate_debug_scripts.py [--include-archive]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUG_ROOT = REPO_ROOT / "debug_scripts"
SCRIPT_NAME_RE = re.compile(r"^q_[a-z0-9_]+__[a-z0-9_]+__[a-z0-9_]+\.py$")
ACTIVE_DIRS = (
    "smoke",
    "pipeline_gates",
    "sibling_calibration",
    "branch_length",
    "tree_construction",
    "sbm",
    "projection_power",
    "case_studies",
    "diagnostics",
)
REQUIRED_HEADER_FIELDS = (
    "Purpose:",
    "Inputs:",
    "Outputs:",
    "Expected runtime:",
    "How to run:",
)


def _collect_targets(include_archive: bool, include_root_legacy: bool) -> list[Path]:
    targets = []
    for path in sorted(DEBUG_ROOT.rglob("*.py")):
        rel_parts = path.relative_to(DEBUG_ROOT).parts
        if not rel_parts:
            continue
        top = rel_parts[0]
        if top == "_shared":
            continue
        if top not in ACTIVE_DIRS and top not in {"archive"}:
            # Legacy root-level scripts (or unknown dirs) are ignored unless requested.
            if not include_root_legacy:
                continue
        if top == "archive" and not include_archive:
            continue
        targets.append(path)
    return targets


def _parse_docstring(path: Path) -> tuple[str | None, str | None]:
    try:
        mod = ast.parse(path.read_text())
    except Exception as exc:
        return None, f"parse_error: {exc}"
    doc = ast.get_docstring(mod)
    if not doc:
        return None, "missing_module_docstring"
    return doc, None


def validate(include_archive: bool, include_root_legacy: bool) -> tuple[int, list[str]]:
    errors: list[str] = []
    targets = _collect_targets(
        include_archive=include_archive, include_root_legacy=include_root_legacy
    )

    if not targets:
        errors.append("No target scripts found under debug_scripts/.")
        return 1, errors

    for path in targets:
        rel = path.relative_to(REPO_ROOT)
        rel_parts = path.relative_to(DEBUG_ROOT).parts
        name = path.name

        if not SCRIPT_NAME_RE.match(name):
            errors.append(f"{rel}: invalid filename (must match q_<question>__<method>__<scope>.py)")

        # Enforce dedicated branch-length methods directory.
        if rel_parts and rel_parts[0] == "branch_length":
            if len(rel_parts) < 3 or rel_parts[1] != "methods":
                errors.append(
                    f"{rel}: branch-length scripts must live under debug_scripts/branch_length/methods/"
                )

        doc, parse_err = _parse_docstring(path)
        if parse_err:
            errors.append(f"{rel}: {parse_err}")
            continue

        assert doc is not None
        missing = [field for field in REQUIRED_HEADER_FIELDS if field not in doc]
        if missing:
            errors.append(f"{rel}: missing header fields: {', '.join(missing)}")

    return (1 if errors else 0), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate debug script naming and header standards.")
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Also validate scripts under debug_scripts/archive.",
    )
    parser.add_argument(
        "--include-root-legacy",
        action="store_true",
        help="Also validate legacy root-level scripts during migration.",
    )
    args = parser.parse_args()

    code, errors = validate(
        include_archive=args.include_archive,
        include_root_legacy=args.include_root_legacy,
    )
    if code == 0:
        checked = len(
            _collect_targets(
                include_archive=args.include_archive,
                include_root_legacy=args.include_root_legacy,
            )
        )
        print(f"OK: validated {checked} debug scripts.")
        return 0

    print("Validation failed with the following issues:")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
