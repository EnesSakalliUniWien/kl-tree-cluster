"""Shared execution support for classified benchmark diagnostics."""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.cases.regression_gate import get_regression_gate_test_cases

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_CLASSIFICATION_GLOB = (
    "benchmarks/results/oracle_tree_recoverability_*/oracle_tree_recoverability.csv"
)


@dataclass(frozen=True)
class ClassifiedBenchmarkCase:
    """One benchmark case paired with its oracle-classification record."""

    case: dict[str, object]
    classification: Mapping[str, object]


@dataclass(frozen=True)
class ClassifiedCaseSelection:
    """Resolved classification source and selected benchmark cases."""

    classification_path: Path
    cases: tuple[ClassifiedBenchmarkCase, ...]


def configure_serial_runtime(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Set deterministic single-process defaults without overriding caller choices."""

    target = os.environ if environ is None else environ
    for env_var in _THREAD_ENV_VARS:
        target.setdefault(env_var, "1")
    target.setdefault("TBS_N_JOBS", "1")


def parse_csv_values(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated command value into non-empty stripped values."""

    return tuple(part.strip() for part in raw.split(",") if part.strip())


def resolve_classified_cases(
    *,
    suite: str,
    classification_csv: Path | None,
    failure_classes: Sequence[str],
    case_names: Sequence[str],
    required_columns: Collection[str] = (),
    project_root: Path = PROJECT_ROOT,
) -> ClassifiedCaseSelection:
    """Load, validate, and join oracle classifications to benchmark case definitions."""

    cases = _load_benchmark_suite(suite)
    classification_path = (
        _latest_classification_csv(project_root)
        if classification_csv is None
        else classification_csv
    )
    classification_df = pd.read_csv(classification_path)
    required = {"case_id", "failure_class", *required_columns}
    missing = required - set(classification_df.columns)
    if missing:
        raise ValueError(
            f"Classification CSV {classification_path} is missing columns: {sorted(missing)}."
        )

    selected = classification_df[
        classification_df["failure_class"].astype(str).isin(failure_classes)
    ].copy()
    if case_names:
        selected = selected[selected["case_id"].astype(str).isin(set(case_names))]
    if selected.empty:
        raise ValueError("No cases matched the requested failure class/name filters.")

    case_by_name = {str(case["name"]): case for case in cases}
    missing_cases = [
        case_id
        for case_id in selected["case_id"].astype(str)
        if case_id not in case_by_name
    ]
    if missing_cases:
        raise ValueError(
            f"Selected cases are not present in the {len(cases)}-case suite: "
            f"{missing_cases}."
        )

    selected_cases = tuple(
        ClassifiedBenchmarkCase(
            case=case_by_name[str(record["case_id"])].copy(),
            classification=dict(record),
        )
        for record in selected.to_dict(orient="records")
    )
    return ClassifiedCaseSelection(
        classification_path=classification_path,
        cases=selected_cases,
    )


def create_result_directory(
    explicit_output_dir: Path | None,
    *,
    study_slug: str,
    project_root: Path = PROJECT_ROOT,
    timestamp: str | None = None,
) -> Path:
    """Create an explicit or timestamped benchmark result directory."""

    if not study_slug or Path(study_slug).name != study_slug:
        raise ValueError(f"study_slug must be one path-safe name; got {study_slug!r}.")
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    output_dir = (
        explicit_output_dir
        if explicit_output_dir is not None
        else project_root / "benchmarks" / "results" / f"{study_slug}_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_benchmark_suite(suite: str) -> list[dict[str, object]]:
    if suite == "regression_gate":
        return get_regression_gate_test_cases()
    if suite == "full":
        return get_default_test_cases()
    raise ValueError(f"Unknown suite {suite!r}.")


def _latest_classification_csv(project_root: Path) -> Path:
    candidates = sorted(project_root.glob(_CLASSIFICATION_GLOB))
    for candidate in reversed(candidates):
        if "failure_class" in pd.read_csv(candidate, nrows=0).columns:
            return candidate
    raise FileNotFoundError(
        "No oracle_tree_recoverability CSV with a failure_class column was found."
    )


__all__ = [
    "ClassifiedBenchmarkCase",
    "ClassifiedCaseSelection",
    "PROJECT_ROOT",
    "configure_serial_runtime",
    "create_result_directory",
    "parse_csv_values",
    "resolve_classified_cases",
]
