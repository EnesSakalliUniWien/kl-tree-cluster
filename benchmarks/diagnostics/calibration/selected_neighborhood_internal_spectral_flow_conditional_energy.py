"""Condition internal-barycenter spectral-flow energy by root/topology context.

This postprocess asks the rescue question explicitly. Internal-barycenter
energy is not treated as valid support merely because it smooths a signal row;
it must also pass selected-root validity, selected-root tail support, strict
shared MP transport, and paired selected-null checks.

The output is diagnostic-only and does not change traversal or p-values.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "selected_neighborhood_internal_spectral_flow_conditional_energy/v1"
STUDY_ROLE = (
    "diagnostic_internal_barycenter_conditional_energy_not_calibration"
)
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "selected_neighborhood_internal_spectral_flow_conditional_energy"
)

DEFAULT_RESULT_ROOT = Path(
    "raw/assets/benchmark-results/specific_small_method_benchmark_20260615"
)
DEFAULT_INTERNAL_ENERGY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "selected_neighborhood_internal_spectral_flow_overlap_seven_case"
    / "selected_neighborhood_internal_spectral_flow_neighborhood_energy.csv"
)
DEFAULT_ROOT_VALIDITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_validity_replay_overlap_seven_signal_v1"
    / "root_selected_validity_replay_rows.csv"
)

ROWS_OUTPUT = (
    "selected_neighborhood_internal_spectral_flow_conditional_energy_rows.csv"
)
SUMMARY_OUTPUT = (
    "selected_neighborhood_internal_spectral_flow_conditional_energy_summary.csv"
)
MANIFEST_OUTPUT = "manifest.json"

DEFAULT_HARD_NEGATIVE_CASES = ("overlap_extreme_4c",)
SMOOTHING_STATUS = "diagnostic_internal_smooths_strict_shared_transport"


@dataclass(frozen=True)
class ConditionalInternalEnergyConfig:
    """Input/output contract for conditional internal-energy diagnostics."""

    output_dir: Path
    internal_energy_rows_path: Path = DEFAULT_INTERNAL_ENERGY_ROWS
    root_validity_rows_path: Path | None = DEFAULT_ROOT_VALIDITY_ROWS
    hard_negative_cases: tuple[str, ...] = DEFAULT_HARD_NEGATIVE_CASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--internal-energy-rows-path",
        type=Path,
        default=DEFAULT_INTERNAL_ENERGY_ROWS,
    )
    parser.add_argument(
        "--root-validity-rows-path",
        type=Path,
        default=DEFAULT_ROOT_VALIDITY_ROWS,
    )
    parser.add_argument(
        "--hard-negative-case",
        action="append",
        default=None,
        help=(
            "Case id that must remain blocked. May be supplied multiple times; "
            "defaults to overlap_extreme_4c."
        ),
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, ConditionalInternalEnergyConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _finite_int(value: object) -> int:
    numeric = _finite_float(value)
    return int(numeric) if math.isfinite(numeric) else 0


def _root_validity_lookup(root_validity_rows: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if root_validity_rows is None or root_validity_rows.empty:
        return {}
    if "target_case_id" not in root_validity_rows:
        raise ValueError("root validity rows must contain target_case_id.")
    return {
        str(row["target_case_id"]): row.to_dict()
        for _, row in root_validity_rows.iterrows()
    }


def _selected_null_lookup(energy_rows: pd.DataFrame) -> dict[tuple[str, str, int], dict[str, object]]:
    selected_null = energy_rows[energy_rows["data_role"].astype(str).eq("selected_null")]
    records: dict[tuple[str, str, int], dict[str, object]] = {}
    for _, row in selected_null.iterrows():
        key = (
            str(row["case_id"]),
            str(row["method_id"]),
            _finite_int(row["replicate"]),
        )
        records[key] = row.to_dict()
    return records


def _root_context(
    case_id: str,
    validity_by_case: dict[str, dict[str, object]],
) -> tuple[str, str, str, bool, bool]:
    record = validity_by_case.get(str(case_id), {})
    validity_status = str(
        record.get("root_validity_status", "root_validity_unmeasured")
    )
    tail_status = str(
        record.get("root_tail_inference_status", "root_tail_unmeasured")
    )
    usability_status = str(
        record.get(
            "selected_root_usability_status",
            "selected_root_usability_unmeasured",
        )
    )
    validity_pass = validity_status.startswith("root_validity_supported")
    tail_pass = tail_status == "calibrated_selected_root_spectral_tail_available"
    return validity_status, tail_status, usability_status, validity_pass, tail_pass


def _strict_energy_pass(row: pd.Series) -> bool:
    strict_count = _finite_int(row.get("strict_shared_mp_supported_edge_count", 0))
    return (
        strict_count > 0
        and str(row.get("neighborhood_energy_status", "")) == SMOOTHING_STATUS
    )


def _classify_conditional_energy(
    *,
    row: pd.Series,
    hard_negative_control: bool,
    root_validity_pass: bool,
    root_tail_pass: bool,
    paired_selected_null_warning: bool,
    paired_selected_null_missing: bool,
) -> tuple[bool, bool, str, str]:
    data_role = str(row.get("data_role", ""))
    internal_only = _finite_int(row.get("internal_only_mp_supported_edge_count", 0))
    strict_count = _finite_int(row.get("strict_shared_mp_supported_edge_count", 0))
    strict_pass = _strict_energy_pass(row)

    if data_role == "selected_null":
        if internal_only > 0:
            return (
                False,
                True,
                "selected_null_internal_energy_warning",
                "fail_closed_selected_null_internal_support",
            )
        return (
            False,
            False,
            "selected_null_energy_clean",
            "selected_null_control_clean",
        )

    if hard_negative_control:
        return (
            False,
            False,
            "fail_closed_hard_negative_control",
            "root_or_energy_rescue_disallowed_for_hard_negative",
        )
    if not root_validity_pass:
        return (
            False,
            False,
            "fail_closed_root_validity_missing_or_failed",
            "root_validity_required_before_internal_energy_rescue",
        )
    if not root_tail_pass:
        return (
            False,
            False,
            "fail_closed_root_tail_support_missing",
            "root_tail_required_before_internal_energy_rescue",
        )
    if strict_count <= 0:
        return (
            False,
            False,
            "fail_closed_no_strict_shared_mp_transport",
            "strict_shared_mp_transport_required",
        )
    if not strict_pass:
        return (
            False,
            False,
            "fail_closed_strict_shared_energy_not_smoothing",
            "internal_energy_does_not_improve_strict_shared_transport",
        )
    if paired_selected_null_missing:
        return (
            False,
            False,
            "fail_closed_paired_selected_null_energy_missing",
            "paired_selected_null_control_required",
        )
    if paired_selected_null_warning:
        return (
            False,
            False,
            "fail_closed_paired_selected_null_internal_energy_warning",
            "paired_selected_null_internal_support_blocks_rescue",
        )
    return (
        True,
        False,
        "conditional_internal_energy_rescue_candidate",
        "diagnostic_candidate_requires_future_selective_p_value",
    )


def build_conditional_internal_energy_rows(
    internal_energy_rows: pd.DataFrame,
    *,
    root_validity_rows: pd.DataFrame | None = None,
    hard_negative_cases: tuple[str, ...] = DEFAULT_HARD_NEGATIVE_CASES,
) -> pd.DataFrame:
    """Classify internal-barycenter energy under root and null controls."""
    required = {
        "case_id",
        "data_role",
        "method_id",
        "replicate",
        "strict_shared_mp_supported_edge_count",
        "internal_only_mp_supported_edge_count",
        "delta_strict_shared_mp_joint_transport_energy",
        "internal_only_mp_joint_transport_energy",
        "neighborhood_energy_status",
    }
    missing = required - set(internal_energy_rows.columns)
    if missing:
        raise ValueError(
            f"internal energy rows missing required columns: {sorted(missing)!r}."
        )

    validity_by_case = _root_validity_lookup(root_validity_rows)
    null_by_key = _selected_null_lookup(internal_energy_rows)
    hard_negative_set = {str(case) for case in hard_negative_cases}
    records: list[dict[str, object]] = []

    for _, row in internal_energy_rows.iterrows():
        case_id = str(row["case_id"])
        method_id = str(row["method_id"])
        replicate = _finite_int(row["replicate"])
        data_role = str(row["data_role"])
        validity_status, tail_status, usability_status, validity_pass, tail_pass = (
            _root_context(case_id, validity_by_case)
        )

        null_key = (case_id, method_id, replicate)
        null_record = null_by_key.get(null_key)
        paired_null_missing = bool(data_role == "signal" and null_record is None)
        paired_null_internal_only = (
            _finite_int(null_record.get("internal_only_mp_supported_edge_count", 0))
            if null_record
            else 0
        )
        paired_null_warning = bool(
            data_role == "signal" and paired_null_internal_only > 0
        )
        hard_negative_control = bool(case_id in hard_negative_set and data_role == "signal")
        rescue_candidate, selected_null_warning, status, action = (
            _classify_conditional_energy(
                row=row,
                hard_negative_control=hard_negative_control,
                root_validity_pass=validity_pass,
                root_tail_pass=tail_pass,
                paired_selected_null_warning=paired_null_warning,
                paired_selected_null_missing=paired_null_missing,
            )
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": case_id,
            "data_role": data_role,
            "method_id": method_id,
            "replicate": replicate,
            "hard_negative_control": hard_negative_control,
            "root_validity_status": validity_status,
            "root_tail_inference_status": tail_status,
            "selected_root_usability_status": usability_status,
            "root_validity_pass": bool(validity_pass),
            "root_tail_pass": bool(tail_pass),
            "edge_count": _finite_int(row.get("edge_count", 0)),
            "leaf_mp_supported_edge_count": _finite_int(
                row.get("leaf_mp_supported_edge_count", 0)
            ),
            "internal_mp_supported_edge_count": _finite_int(
                row.get("internal_mp_supported_edge_count", 0)
            ),
            "strict_shared_mp_supported_edge_count": _finite_int(
                row.get("strict_shared_mp_supported_edge_count", 0)
            ),
            "internal_only_mp_supported_edge_count": _finite_int(
                row.get("internal_only_mp_supported_edge_count", 0)
            ),
            "delta_strict_shared_mp_joint_transport_energy": _finite_float(
                row.get("delta_strict_shared_mp_joint_transport_energy")
            ),
            "internal_only_mp_joint_transport_energy": _finite_float(
                row.get("internal_only_mp_joint_transport_energy")
            ),
            "neighborhood_energy_status": str(
                row.get("neighborhood_energy_status", "")
            ),
            "strict_shared_energy_pass": bool(_strict_energy_pass(row)),
            "paired_selected_null_missing": paired_null_missing,
            "paired_selected_null_internal_only_mp_supported_edge_count": (
                paired_null_internal_only
            ),
            "paired_selected_null_internal_energy_warning": paired_null_warning,
            "conditional_internal_energy_rescue_candidate": bool(rescue_candidate),
            "selected_null_internal_energy_warning": bool(selected_null_warning),
            "conditional_energy_status": status,
            "method_action": action,
        }
        records.append(record)

    return pd.DataFrame.from_records(records)


def summarize_conditional_internal_energy(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize conditional internal-energy rescue outcomes."""
    if rows.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "row_count": 0,
                    "case_count": 0,
                    "rescue_candidate_count": 0,
                    "selected_null_warning_count": 0,
                    "summary_status": "no_rows",
                }
            ]
        )
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(len(rows)),
                "case_count": int(rows["case_id"].nunique()),
                "signal_row_count": int(rows["data_role"].eq("signal").sum()),
                "selected_null_row_count": int(
                    rows["data_role"].eq("selected_null").sum()
                ),
                "rescue_candidate_count": int(
                    rows["conditional_internal_energy_rescue_candidate"].sum()
                ),
                "selected_null_warning_count": int(
                    rows["selected_null_internal_energy_warning"].sum()
                ),
                "paired_selected_null_warning_count": int(
                    rows["paired_selected_null_internal_energy_warning"].sum()
                ),
                "root_validity_blocked_count": int(
                    rows["conditional_energy_status"]
                    .eq("fail_closed_root_validity_missing_or_failed")
                    .sum()
                ),
                "root_tail_blocked_count": int(
                    rows["conditional_energy_status"]
                    .eq("fail_closed_root_tail_support_missing")
                    .sum()
                ),
                "strict_energy_blocked_count": int(
                    rows["conditional_energy_status"]
                    .eq("fail_closed_strict_shared_energy_not_smoothing")
                    .sum()
                ),
                "hard_negative_blocked_count": int(
                    rows["conditional_energy_status"]
                    .eq("fail_closed_hard_negative_control")
                    .sum()
                ),
                "summary_status": (
                    "conditional_internal_energy_rescue_candidates_observed"
                    if bool(
                        rows["conditional_internal_energy_rescue_candidate"].any()
                    )
                    else "conditional_internal_energy_rescue_fail_closed"
                ),
            }
        ]
    )


def run_conditional_internal_energy_panel(
    config: ConditionalInternalEnergyConfig,
) -> dict[str, Path]:
    """Read energy/root rows, write conditional rows, summary, and manifest."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    energy_rows = pd.read_csv(config.internal_energy_rows_path)
    root_validity_rows = (
        None
        if config.root_validity_rows_path is None
        else pd.read_csv(config.root_validity_rows_path)
    )
    rows = build_conditional_internal_energy_rows(
        energy_rows,
        root_validity_rows=root_validity_rows,
        hard_negative_cases=tuple(config.hard_negative_cases),
    )
    summary = summarize_conditional_internal_energy(rows)

    paths = {
        "rows": config.output_dir / ROWS_OUTPUT,
        "summary": config.output_dir / SUMMARY_OUTPUT,
        "manifest": config.output_dir / MANIFEST_OUTPUT,
    }
    rows.to_csv(paths["rows"], index=False)
    summary.to_csv(paths["summary"], index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "parameters": config,
        "outputs": paths,
        "n_rows": int(len(rows)),
        "n_summary_rows": int(len(summary)),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    args = parse_args()
    hard_negative_cases = (
        tuple(args.hard_negative_case)
        if args.hard_negative_case is not None
        else DEFAULT_HARD_NEGATIVE_CASES
    )
    run_conditional_internal_energy_panel(
        ConditionalInternalEnergyConfig(
            output_dir=args.output_dir,
            internal_energy_rows_path=args.internal_energy_rows_path,
            root_validity_rows_path=args.root_validity_rows_path,
            hard_negative_cases=hard_negative_cases,
        )
    )


if __name__ == "__main__":
    main()
