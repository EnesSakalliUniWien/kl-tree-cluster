"""Plug-in deformed-MP edge diagnostic for selected-root spectra.

This panel computes a diagnostic local deformed Marchenko--Pastur edge from
captured root bulk spectra. It does not produce production p-values and does
not rescue root splits. Its job is to turn the current H_u task into an
explicit row-level edge calculation that can later be conditioned on inside
the selected-root spectral-tail law.
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

from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
    _finite_float,
)

SCHEMA_VERSION = "root_selected_deformed_mp_edge_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_deformed_mp_edge_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.selected.root_selected_deformed_mp_edge_panel"
)

DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_importance_external_null_topology_join_mild_accumulated"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)
DEFAULT_H_U_OBSERVABILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_h_u_observability_mild_accumulated"
    / "root_selected_h_u_observability_rows.csv"
)

ROWS_OUTPUT = "root_selected_deformed_mp_edge_rows.csv"
SUPPORT_ROWS_OUTPUT = "root_selected_deformed_mp_edge_support_rows.csv"
SUMMARY_OUTPUT = "root_selected_deformed_mp_edge_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "h_u_observability_status",
    "root_active_feature_count",
    "root_mp_threshold_rows",
    "root_aspect_ratio",
    "root_full_eigenvalue_count",
    "bulk_excluded_top_eigenvalue_count",
    "bulk_eigenvalue_count",
    "identity_mp_upper_edge",
    "deformed_mp_upper_edge",
    "deformed_mp_edge_multiplier",
    "selected_root_eigenvalue",
    "selected_root_identity_ratio",
    "selected_root_deformed_ratio",
    "s_root_identity_excess_log",
    "s_root_deformed_excess_log",
    "deformed_mp_edge_status",
    "production_inference_status",
    "next_mathematical_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "deformed_edge_computed_count",
    "exact_tail_h_u_optional_count",
    "support_missing_count",
    "missing_input_count",
    "median_deformed_mp_edge_multiplier",
    "max_deformed_mp_edge_multiplier",
    "median_deformed_excess_log",
    "max_deformed_excess_log",
    "summary_status",
)

SUPPORT_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "calibration_role",
    "proposal_family",
    "conditioning_target_case_id",
    "root_active_feature_count",
    "root_mp_threshold_rows",
    "root_aspect_ratio",
    "root_full_eigenvalue_count",
    "bulk_excluded_top_eigenvalue_count",
    "bulk_eigenvalue_count",
    "identity_mp_upper_edge",
    "deformed_mp_upper_edge",
    "deformed_mp_edge_multiplier",
    "selected_root_eigenvalue",
    "selected_root_identity_ratio",
    "selected_root_deformed_ratio",
    "s_root_identity_excess_log",
    "s_root_deformed_excess_log",
    "support_deformed_mp_edge_status",
    "production_inference_status",
)


@dataclass(frozen=True)
class RootSelectedDeformedMPEdgeConfig:
    """Input/output contract for plug-in deformed MP edge diagnostics."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    h_u_observability_rows_path: Path = DEFAULT_H_U_OBSERVABILITY_ROWS
    minimum_bulk_eigenvalue_count: int = 3
    derivative_tolerance: float = 1e-10
    maximum_bisection_iterations: int = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joined-feasibility-rows-path",
        type=Path,
        default=DEFAULT_JOINED_FEASIBILITY_ROWS,
    )
    parser.add_argument(
        "--h-u-observability-rows-path",
        type=Path,
        default=DEFAULT_H_U_OBSERVABILITY_ROWS,
    )
    parser.add_argument("--minimum-bulk-eigenvalue-count", type=int, default=3)
    parser.add_argument("--derivative-tolerance", type=float, default=1e-10)
    parser.add_argument("--maximum-bisection-iterations", type=int, default=100)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedDeformedMPEdgeConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def _string_value(
    row: pd.Series | dict[str, object],
    column: str,
    default: str = "",
) -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _is_observed_target(row: pd.Series) -> bool:
    return (
        _string_value(row, "proposal_family") == "observed_target"
        or _string_value(row, "calibration_role") == "observed_target_not_null_support"
        or _string_value(row, "data_role") == "observed_target"
    )


def _positive_log(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return float(max(math.log(max(value, 1e-12)), 0.0))


def _parse_spectrum(value: object) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(value, float) and math.isnan(value):
        return np.zeros(0, dtype=np.float64)
    if isinstance(value, np.ndarray):
        parsed = value
    elif isinstance(value, (list, tuple)):
        parsed = np.asarray(value, dtype=np.float64)
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return np.zeros(0, dtype=np.float64)
        try:
            parsed = np.asarray(json.loads(text), dtype=np.float64)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = np.asarray(
                [part for part in text.replace("[", "").replace("]", "").split(",")],
                dtype=np.float64,
            )
    parsed = np.asarray(parsed, dtype=np.float64).reshape(-1)
    return parsed[np.isfinite(parsed) & (parsed > 0.0)]


def _identity_mp_upper_edge(*, aspect_ratio: float) -> float:
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        return math.nan
    return float((1.0 + math.sqrt(aspect_ratio)) ** 2)


def _deformed_mp_z(v_value: float, spectrum: np.ndarray, aspect_ratio: float) -> float:
    return float(-1.0 / v_value + aspect_ratio * np.mean(spectrum / (1.0 + spectrum * v_value)))


def _deformed_mp_edge_derivative(
    v_value: float,
    spectrum: np.ndarray,
    aspect_ratio: float,
) -> float:
    return float(
        1.0 / (v_value * v_value)
        - aspect_ratio * np.mean((spectrum * spectrum) / ((1.0 + spectrum * v_value) ** 2))
    )


def deformed_mp_upper_edge(
    population_spectrum: np.ndarray,
    *,
    aspect_ratio: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Compute the right support edge from the Silverstein-Choi inverse map.

    The input spectrum is treated as a discrete estimate of H. The returned
    value solves z'(v)=0 on the right-edge interval (-1 / max(H), 0).
    """
    spectrum = np.asarray(population_spectrum, dtype=np.float64).reshape(-1)
    spectrum = spectrum[np.isfinite(spectrum) & (spectrum > 0.0)]
    if spectrum.size == 0:
        return math.nan
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        return math.nan

    max_eigenvalue = float(np.max(spectrum))
    left = -1.0 / max_eigenvalue + max(1e-12, tolerance)
    right = -max(1e-12, tolerance)
    if not left < right < 0.0:
        return math.nan
    left_value = _deformed_mp_edge_derivative(left, spectrum, aspect_ratio)
    right_value = _deformed_mp_edge_derivative(right, spectrum, aspect_ratio)
    if not (math.isfinite(left_value) and math.isfinite(right_value)):
        return math.nan
    if left_value > 0.0 or right_value < 0.0:
        return math.nan

    lo = left
    hi = right
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        mid_value = _deformed_mp_edge_derivative(mid, spectrum, aspect_ratio)
        if not math.isfinite(mid_value):
            return math.nan
        if abs(mid_value) <= tolerance or abs(hi - lo) <= tolerance:
            return _deformed_mp_z(mid, spectrum, aspect_ratio)
        if mid_value < 0.0:
            lo = mid
        else:
            hi = mid
    return _deformed_mp_z(0.5 * (lo + hi), spectrum, aspect_ratio)


def _bulk_spectrum(
    full_spectrum: np.ndarray,
    *,
    raw_mp_signal_count: float,
) -> tuple[np.ndarray, int]:
    sorted_spectrum = np.sort(np.asarray(full_spectrum, dtype=np.float64))[::-1]
    sorted_spectrum = sorted_spectrum[np.isfinite(sorted_spectrum) & (sorted_spectrum > 0.0)]
    raw_count = _finite_float(raw_mp_signal_count)
    if not math.isfinite(raw_count):
        raw_count = 0.0
    excluded = int(max(raw_count, 0.0))
    excluded = min(excluded, max(int(sorted_spectrum.size) - 1, 0))
    return sorted_spectrum[excluded:], excluded


def _edge_status(
    *,
    h_u_status: str,
    has_inputs: bool,
    bulk_count: int,
    minimum_bulk_eigenvalue_count: int,
    deformed_edge: float,
) -> str:
    if h_u_status == "exact_tail_support_available_h_u_optional":
        return "exact_tail_support_h_u_optional"
    if h_u_status == "support_missing_before_h_u_estimation":
        return "support_missing_before_deformed_edge"
    if not has_inputs or bulk_count < int(minimum_bulk_eigenvalue_count):
        return "deformed_mp_edge_missing_inputs"
    if not math.isfinite(deformed_edge) or deformed_edge <= 0.0:
        return "deformed_mp_edge_numerical_failure"
    return "deformed_mp_edge_computed_diagnostic_only"


def _next_step(status: str) -> str:
    if status == "deformed_mp_edge_computed_diagnostic_only":
        return "join_deformed_edge_into_selected_root_tail_conditioning"
    if status == "exact_tail_support_h_u_optional":
        return "use_exact_root_tail_panel"
    if status == "support_missing_before_deformed_edge":
        return "generate_support_or_external_law_before_tail_calibration"
    if status == "deformed_mp_edge_numerical_failure":
        return "inspect_silverstein_edge_solver_and_bulk_spectrum"
    return "capture_root_bulk_spectrum_active_feature_count_and_mp_rows"


def _deformed_edge_components(
    row: pd.Series,
    *,
    minimum_bulk_eigenvalue_count: int,
    derivative_tolerance: float,
    maximum_bisection_iterations: int,
) -> dict[str, object]:
    full_spectrum = _parse_spectrum(row["root_full_component_eigenvalues_json"])
    bulk, excluded = _bulk_spectrum(
        full_spectrum,
        raw_mp_signal_count=row.get("root_raw_mp_signal_count", 0),
    )
    active_feature_count = _finite_float(row["root_active_feature_count"])
    mp_rows = _finite_float(row["root_mp_threshold_rows"])
    aspect_ratio = (
        float(active_feature_count / mp_rows)
        if math.isfinite(active_feature_count)
        and math.isfinite(mp_rows)
        and active_feature_count > 0.0
        and mp_rows > 0.0
        else math.nan
    )
    identity_edge = _finite_float(row.get("root_mp_upper_bound", math.nan))
    if not math.isfinite(identity_edge):
        identity_edge = _identity_mp_upper_edge(aspect_ratio=aspect_ratio)
    has_inputs = (
        full_spectrum.size >= int(minimum_bulk_eigenvalue_count)
        and math.isfinite(aspect_ratio)
        and math.isfinite(identity_edge)
        and identity_edge > 0.0
    )
    deformed_edge = (
        deformed_mp_upper_edge(
            bulk,
            aspect_ratio=aspect_ratio,
            tolerance=float(derivative_tolerance),
            max_iterations=int(maximum_bisection_iterations),
        )
        if has_inputs and bulk.size >= int(minimum_bulk_eigenvalue_count)
        else math.nan
    )
    identity_ratio = _finite_float(row["root_selected_eigenvalue_over_mp_upper_bound"])
    selected_eigenvalue = (
        identity_ratio * identity_edge
        if math.isfinite(identity_ratio) and math.isfinite(identity_edge)
        else math.nan
    )
    deformed_ratio = (
        selected_eigenvalue / deformed_edge
        if math.isfinite(selected_eigenvalue)
        and math.isfinite(deformed_edge)
        and deformed_edge > 0.0
        else math.nan
    )
    return {
        "full_spectrum": full_spectrum,
        "bulk": bulk,
        "excluded": excluded,
        "active_feature_count": active_feature_count,
        "mp_rows": mp_rows,
        "aspect_ratio": aspect_ratio,
        "identity_edge": identity_edge,
        "has_inputs": has_inputs,
        "deformed_edge": deformed_edge,
        "identity_ratio": identity_ratio,
        "selected_eigenvalue": selected_eigenvalue,
        "deformed_ratio": deformed_ratio,
    }


def build_root_selected_deformed_mp_edge_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    h_u_observability_rows: pd.DataFrame,
    minimum_bulk_eigenvalue_count: int = 3,
    derivative_tolerance: float = 1e-10,
    maximum_bisection_iterations: int = 100,
) -> pd.DataFrame:
    """Build plug-in deformed MP edge rows for observed selected roots."""
    _require_columns(
        joined_feasibility_rows,
        {
            "case_id",
            "data_role",
            "calibration_role",
            "proposal_family",
            "root_selected_eigenvalue_over_mp_upper_bound",
            "root_mp_upper_bound",
            "root_mp_threshold_rows",
            "root_active_feature_count",
            "root_raw_mp_signal_count",
            "root_full_component_eigenvalues_json",
        },
        "joined feasibility rows",
    )
    _require_columns(
        h_u_observability_rows,
        {
            "target_case_id",
            "h_u_observability_status",
            "production_inference_status",
        },
        "H_u observability rows",
    )
    h_u_by_case = {
        _string_value(row, "target_case_id"): row for _, row in h_u_observability_rows.iterrows()
    }
    targets = joined_feasibility_rows[
        joined_feasibility_rows.apply(_is_observed_target, axis=1)
    ].copy()
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        case_id = _string_value(target, "case_id")
        h_u_row = h_u_by_case.get(case_id)
        h_u_status = (
            _string_value(h_u_row, "h_u_observability_status")
            if h_u_row is not None
            else "h_u_observability_missing"
        )
        components = _deformed_edge_components(
            target,
            minimum_bulk_eigenvalue_count=int(minimum_bulk_eigenvalue_count),
            derivative_tolerance=float(derivative_tolerance),
            maximum_bisection_iterations=int(maximum_bisection_iterations),
        )
        status = _edge_status(
            h_u_status=h_u_status,
            has_inputs=bool(components["has_inputs"]),
            bulk_count=int(components["bulk"].size),
            minimum_bulk_eigenvalue_count=int(minimum_bulk_eigenvalue_count),
            deformed_edge=float(components["deformed_edge"]),
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": case_id,
                "h_u_observability_status": h_u_status,
                "root_active_feature_count": components["active_feature_count"],
                "root_mp_threshold_rows": components["mp_rows"],
                "root_aspect_ratio": components["aspect_ratio"],
                "root_full_eigenvalue_count": int(components["full_spectrum"].size),
                "bulk_excluded_top_eigenvalue_count": components["excluded"],
                "bulk_eigenvalue_count": int(components["bulk"].size),
                "identity_mp_upper_edge": components["identity_edge"],
                "deformed_mp_upper_edge": components["deformed_edge"],
                "deformed_mp_edge_multiplier": (
                    components["deformed_edge"] / components["identity_edge"]
                    if math.isfinite(float(components["deformed_edge"]))
                    and math.isfinite(float(components["identity_edge"]))
                    and float(components["identity_edge"]) > 0.0
                    else math.nan
                ),
                "selected_root_eigenvalue": components["selected_eigenvalue"],
                "selected_root_identity_ratio": components["identity_ratio"],
                "selected_root_deformed_ratio": components["deformed_ratio"],
                "s_root_identity_excess_log": _positive_log(float(components["identity_ratio"])),
                "s_root_deformed_excess_log": _positive_log(float(components["deformed_ratio"])),
                "deformed_mp_edge_status": status,
                "production_inference_status": (
                    "exact_support_available_defer_to_root_tail_panel"
                    if status == "exact_tail_support_h_u_optional"
                    else "fail_closed_deformed_mp_edge_diagnostic_only"
                ),
                "next_mathematical_step": _next_step(status),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def build_root_support_deformed_mp_edge_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    minimum_bulk_eigenvalue_count: int = 3,
    derivative_tolerance: float = 1e-10,
    maximum_bisection_iterations: int = 100,
) -> pd.DataFrame:
    """Build plug-in deformed MP edge rows for non-target support/proposal roots."""
    _require_columns(
        joined_feasibility_rows,
        {
            "case_id",
            "data_role",
            "calibration_role",
            "proposal_family",
            "root_selected_eigenvalue_over_mp_upper_bound",
            "root_mp_upper_bound",
            "root_mp_threshold_rows",
            "root_active_feature_count",
            "root_raw_mp_signal_count",
            "root_full_component_eigenvalues_json",
        },
        "joined feasibility rows",
    )
    candidates = joined_feasibility_rows[
        ~joined_feasibility_rows.apply(_is_observed_target, axis=1)
    ].copy()
    records: list[dict[str, object]] = []
    for _, row in candidates.sort_values("case_id").iterrows():
        components = _deformed_edge_components(
            row,
            minimum_bulk_eigenvalue_count=int(minimum_bulk_eigenvalue_count),
            derivative_tolerance=float(derivative_tolerance),
            maximum_bisection_iterations=int(maximum_bisection_iterations),
        )
        if not components["has_inputs"] or int(components["bulk"].size) < int(
            minimum_bulk_eigenvalue_count
        ):
            status = "support_deformed_mp_edge_missing_inputs"
        elif (
            not math.isfinite(float(components["deformed_edge"]))
            or float(components["deformed_edge"]) <= 0.0
        ):
            status = "support_deformed_mp_edge_numerical_failure"
        else:
            status = "support_deformed_mp_edge_computed_diagnostic_only"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": _string_value(row, "case_id"),
                "data_role": _string_value(row, "data_role"),
                "calibration_role": _string_value(row, "calibration_role"),
                "proposal_family": _string_value(row, "proposal_family"),
                "conditioning_target_case_id": _string_value(
                    row,
                    "conditioning_target_case_id",
                ),
                "root_active_feature_count": components["active_feature_count"],
                "root_mp_threshold_rows": components["mp_rows"],
                "root_aspect_ratio": components["aspect_ratio"],
                "root_full_eigenvalue_count": int(components["full_spectrum"].size),
                "bulk_excluded_top_eigenvalue_count": components["excluded"],
                "bulk_eigenvalue_count": int(components["bulk"].size),
                "identity_mp_upper_edge": components["identity_edge"],
                "deformed_mp_upper_edge": components["deformed_edge"],
                "deformed_mp_edge_multiplier": (
                    components["deformed_edge"] / components["identity_edge"]
                    if math.isfinite(float(components["deformed_edge"]))
                    and math.isfinite(float(components["identity_edge"]))
                    and float(components["identity_edge"]) > 0.0
                    else math.nan
                ),
                "selected_root_eigenvalue": components["selected_eigenvalue"],
                "selected_root_identity_ratio": components["identity_ratio"],
                "selected_root_deformed_ratio": components["deformed_ratio"],
                "s_root_identity_excess_log": _positive_log(float(components["identity_ratio"])),
                "s_root_deformed_excess_log": _positive_log(float(components["deformed_ratio"])),
                "support_deformed_mp_edge_status": status,
                "production_inference_status": (
                    "diagnostic_only_not_selected_null_calibration"
                    if _string_value(row, "calibration_role") != "external_null_support"
                    else "external_null_support_requires_same_stratum_tail_join"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUPPORT_ROW_COLUMNS)


def summarize_root_selected_deformed_mp_edge_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize plug-in deformed MP edge diagnostics."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    status = rows["deformed_mp_edge_status"].astype(str)
    computed = rows.loc[status.eq("deformed_mp_edge_computed_diagnostic_only")]
    multipliers = pd.to_numeric(
        computed["deformed_mp_edge_multiplier"],
        errors="coerce",
    )
    excess = pd.to_numeric(computed["s_root_deformed_excess_log"], errors="coerce")
    computed_count = int(computed.shape[0])
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "deformed_edge_computed_count": computed_count,
                "exact_tail_h_u_optional_count": int(
                    status.eq("exact_tail_support_h_u_optional").sum()
                ),
                "support_missing_count": int(
                    status.eq("support_missing_before_deformed_edge").sum()
                ),
                "missing_input_count": int(status.eq("deformed_mp_edge_missing_inputs").sum()),
                "median_deformed_mp_edge_multiplier": float(multipliers.median())
                if computed_count
                else math.nan,
                "max_deformed_mp_edge_multiplier": float(multipliers.max())
                if computed_count
                else math.nan,
                "median_deformed_excess_log": float(excess.median())
                if computed_count
                else math.nan,
                "max_deformed_excess_log": float(excess.max()) if computed_count else math.nan,
                "summary_status": (
                    "deformed_mp_edges_computed_for_estimable_roots"
                    if computed_count > 0
                    else "deformed_mp_edges_not_computed"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_deformed_mp_edge_panel(
    config: RootSelectedDeformedMPEdgeConfig,
) -> dict[str, pd.DataFrame]:
    joined = pd.read_csv(config.joined_feasibility_rows_path, low_memory=False)
    h_u_rows = pd.read_csv(config.h_u_observability_rows_path)
    rows = build_root_selected_deformed_mp_edge_rows(
        joined_feasibility_rows=joined,
        h_u_observability_rows=h_u_rows,
        minimum_bulk_eigenvalue_count=config.minimum_bulk_eigenvalue_count,
        derivative_tolerance=config.derivative_tolerance,
        maximum_bisection_iterations=config.maximum_bisection_iterations,
    )
    support_rows = build_root_support_deformed_mp_edge_rows(
        joined_feasibility_rows=joined,
        minimum_bulk_eigenvalue_count=config.minimum_bulk_eigenvalue_count,
        derivative_tolerance=config.derivative_tolerance,
        maximum_bisection_iterations=config.maximum_bisection_iterations,
    )
    summary = summarize_root_selected_deformed_mp_edge_rows(rows)
    return {"rows": rows, "support_rows": support_rows, "summary": summary}


def run_root_selected_deformed_mp_edge_panel(
    config: RootSelectedDeformedMPEdgeConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_deformed_mp_edge_panel(config)
    rows_path = config.output_dir / ROWS_OUTPUT
    support_rows_path = config.output_dir / SUPPORT_ROWS_OUTPUT
    summary_path = config.output_dir / SUMMARY_OUTPUT
    manifest_path = config.output_dir / MANIFEST_OUTPUT
    tables["rows"].to_csv(rows_path, index=False)
    tables["support_rows"].to_csv(support_rows_path, index=False)
    tables["summary"].to_csv(summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "outputs": {
            "rows": str(rows_path),
            "support_rows": str(support_rows_path),
            "summary": str(summary_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": rows_path,
        "support_rows": support_rows_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def main() -> None:
    args = parse_args()
    config = RootSelectedDeformedMPEdgeConfig(
        output_dir=args.output_dir,
        joined_feasibility_rows_path=args.joined_feasibility_rows_path,
        h_u_observability_rows_path=args.h_u_observability_rows_path,
        minimum_bulk_eigenvalue_count=args.minimum_bulk_eigenvalue_count,
        derivative_tolerance=args.derivative_tolerance,
        maximum_bisection_iterations=args.maximum_bisection_iterations,
    )
    outputs = run_root_selected_deformed_mp_edge_panel(config)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
