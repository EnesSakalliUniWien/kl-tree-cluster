"""Finite-support feasibility for selected-root conditional tilts.

The external-law target panel identifies the conditional law that is missing:

    Q_theta proportional to P0 exp(theta^T phi_root)

conditioned on the selected root event, bandwidth/topology status B, and a
measured local spectral population law H_u. This diagnostic asks the finite
support question before any production calibration is attempted:

Can the current external-null support rows be reweighted to match the target
moments phi_root=(T,A,E,S_Hu)?

If a target moment vector is outside the convex hull of the support moment
vectors, no finite exponential tilt over the current support can match it. The
panel is therefore diagnostic-only and fail-closed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
    _finite_float,
    _is_calibration_support,
    _is_observed_target,
    _safe_log1p,
    _string_value,
)

SCHEMA_VERSION = "root_selected_conditional_tilt_feasibility_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_conditional_tilt_feasibility_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.root.selected.root_selected_conditional_tilt_feasibility_panel"

DEFAULT_EXTERNAL_LAW_TARGET_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_deformed_external_law_target_mild_replay_v3_smoke"
    / "root_selected_deformed_external_law_target_rows.csv"
)
DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_importance_external_null_topology_join_mild_hu_replay_v3_smoke"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)
DEFAULT_DEFORMED_SUPPORT_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_deformed_mp_edge_mild_hu_replay_v3_smoke"
    / "root_selected_deformed_mp_edge_support_rows.csv"
)

ROWS_OUTPUT = "root_selected_conditional_tilt_feasibility_rows.csv"
SUMMARY_OUTPUT = "root_selected_conditional_tilt_feasibility_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ALL_AXES = ("T", "A", "E", "S_Hu")
NUMERIC_TOLERANCE = 1e-8
INTERIOR_WEIGHT_TOLERANCE = 1e-6

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_status",
    "target_root_tail_stratum_key",
    "support_pool_scope",
    "required_tilt_axes",
    "checked_moment_axes",
    "support_pool_count",
    "finite_support_hull_residual_l2",
    "moment_range_contains_target",
    "convex_hull_moment_feasible",
    "relative_interior_witness_available",
    "tilt_feasibility_status",
    "target_moment_vector_json",
    "support_min_moment_vector_json",
    "support_max_moment_vector_json",
    "projected_moment_vector_json",
    "moment_residual_vector_json",
    "max_support_weight",
    "min_positive_support_weight",
    "effective_support_size",
    "next_mathematical_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "full_phi_feasible_count",
    "required_axes_feasible_count",
    "full_phi_missing_support_count",
    "required_axes_missing_support_count",
    "full_phi_outside_hull_count",
    "required_axes_outside_hull_count",
    "max_full_phi_residual_l2",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedConditionalTiltFeasibilityConfig:
    """Input/output contract for selected-root tilt feasibility."""

    output_dir: Path
    external_law_target_rows_path: Path = DEFAULT_EXTERNAL_LAW_TARGET_ROWS
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    deformed_support_rows_path: Path = DEFAULT_DEFORMED_SUPPORT_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--external-law-target-rows-path",
        type=Path,
        default=DEFAULT_EXTERNAL_LAW_TARGET_ROWS,
    )
    parser.add_argument(
        "--joined-feasibility-rows-path",
        type=Path,
        default=DEFAULT_JOINED_FEASIBILITY_ROWS,
    )
    parser.add_argument(
        "--deformed-support-rows-path",
        type=Path,
        default=DEFAULT_DEFORMED_SUPPORT_ROWS,
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedConditionalTiltFeasibilityConfig):
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


def _finite_json(values: dict[str, float]) -> str:
    clean = {
        key: (float(value) if math.isfinite(float(value)) else None)
        for key, value in values.items()
    }
    return json.dumps(clean, sort_keys=True)


def _parse_moment_json(value: object) -> dict[str, float]:
    if isinstance(value, str) and value:
        raw = json.loads(value)
    elif isinstance(value, dict):
        raw = value
    else:
        raw = {}
    return {axis: _finite_float(raw.get(axis, math.nan)) for axis in ALL_AXES}


def _axis_tuple(value: object) -> tuple[str, ...]:
    text = str(value or "")
    if text == "none":
        return ALL_AXES
    axes = tuple(axis for axis in text.split(",") if axis in ALL_AXES)
    return axes or ALL_AXES


def _support_deformed_lookup(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        return {}
    _require_columns(rows, {"case_id", "s_root_deformed_excess_log"}, "support rows")
    lookup: dict[str, float] = {}
    for _, row in rows.iterrows():
        value = _finite_float(row.get("s_root_deformed_excess_log", math.nan))
        if math.isfinite(value):
            lookup[_string_value(row, "case_id")] = value
    return lookup


def _support_pool(
    *,
    joined_feasibility_rows: pd.DataFrame,
    deformed_support_rows: pd.DataFrame,
    target_bandwidth: str,
    scope: Literal["same_B_Hu", "all_B_Hu"],
) -> pd.DataFrame:
    support_mask = joined_feasibility_rows.apply(_is_calibration_support, axis=1)
    target_mask = joined_feasibility_rows.apply(_is_observed_target, axis=1)
    support = joined_feasibility_rows.loc[support_mask & ~target_mask].copy()
    support_s = _support_deformed_lookup(deformed_support_rows)
    support["S_Hu"] = support["case_id"].astype(str).map(support_s)
    support = support.loc[pd.to_numeric(support["S_Hu"], errors="coerce").notna()].copy()
    if scope == "same_B_Hu":
        support = support.loc[
            support["root_bandwidth_reopen_band"].astype(str).eq(str(target_bandwidth))
        ].copy()
    support["T"] = pd.to_numeric(
        support["root_tie_rank_median_fraction"],
        errors="coerce",
    )
    support["A"] = support["root_sibling_selected_ratio"].map(_safe_log1p)
    support["E"] = support["root_edge_path_statistic_margin"].map(_safe_log1p)
    support["S_Hu"] = pd.to_numeric(support["S_Hu"], errors="coerce")
    finite = np.isfinite(support[list(ALL_AXES)].to_numpy(dtype=float)).all(axis=1)
    return support.loc[finite].copy()


def _target_bandwidth(stratum_key: str) -> str:
    parts = str(stratum_key).split("|")
    return parts[4] if len(parts) >= 5 else ""


def _effective_support_size(weights: np.ndarray) -> float:
    squared = float(np.sum(weights * weights))
    return float(1.0 / squared) if squared > 0.0 else 0.0


def _project_to_hull(
    support: pd.DataFrame,
    target: dict[str, float],
    axes: tuple[str, ...],
) -> dict[str, object]:
    if support.empty:
        empty = {axis: math.nan for axis in axes}
        return {
            "support_count": 0,
            "range_contains": False,
            "feasible": False,
            "interior": False,
            "residual_l2": math.nan,
            "projected": empty,
            "residual": empty,
            "min_values": empty,
            "max_values": empty,
            "weights": np.asarray([], dtype=float),
        }
    matrix = support[list(axes)].to_numpy(dtype=float)
    target_vec = np.asarray([target[axis] for axis in axes], dtype=float)
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    range_contains = bool(
        np.all(target_vec >= mins - NUMERIC_TOLERANCE)
        and np.all(target_vec <= maxs + NUMERIC_TOLERANCE)
    )
    n_rows = int(matrix.shape[0])
    initial = np.full(n_rows, 1.0 / n_rows, dtype=float)

    def objective(weights: np.ndarray) -> float:
        diff = weights @ matrix - target_vec
        return float(np.dot(diff, diff))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_rows,
        constraints=({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},),
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    weights = np.asarray(result.x if result.success else initial, dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    weight_sum = float(np.sum(weights))
    if weight_sum > 0.0:
        weights = weights / weight_sum
    projected_vec = weights @ matrix
    residual_vec = target_vec - projected_vec
    residual_l2 = float(np.linalg.norm(residual_vec))
    feasible = bool(residual_l2 <= 1e-6)
    positive = weights[weights > INTERIOR_WEIGHT_TOLERANCE]
    interior = bool(feasible and positive.size == n_rows)
    return {
        "support_count": n_rows,
        "range_contains": range_contains,
        "feasible": feasible,
        "interior": interior,
        "residual_l2": residual_l2,
        "projected": dict(zip(axes, projected_vec, strict=True)),
        "residual": dict(zip(axes, residual_vec, strict=True)),
        "min_values": dict(zip(axes, mins, strict=True)),
        "max_values": dict(zip(axes, maxs, strict=True)),
        "weights": weights,
    }


def _status(result: dict[str, object]) -> str:
    support_count = int(result["support_count"])
    if support_count <= 0:
        return "missing_conditioned_support_pool"
    if bool(result["interior"]):
        return "finite_conditional_tilt_feasible_diagnostic_only"
    if bool(result["feasible"]):
        return "boundary_hull_match_requires_infinite_or_degenerate_tilt"
    return "target_outside_current_support_hull_fail_closed"


def build_root_selected_conditional_tilt_feasibility_rows(
    *,
    external_law_target_rows: pd.DataFrame,
    joined_feasibility_rows: pd.DataFrame,
    deformed_support_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return finite-support conditional tilt feasibility rows."""
    _require_columns(
        external_law_target_rows,
        {
            "target_case_id",
            "target_status",
            "target_root_tail_stratum_key",
            "required_tilt_axes",
            "target_moment_vector_json",
        },
        "external law target rows",
    )
    _require_columns(
        joined_feasibility_rows,
        {
            "case_id",
            "data_role",
            "calibration_role",
            "root_tie_rank_median_fraction",
            "root_sibling_selected_ratio",
            "root_edge_path_statistic_margin",
            "root_bandwidth_reopen_band",
        },
        "joined feasibility rows",
    )
    records: list[dict[str, object]] = []
    for _, target_row in external_law_target_rows.sort_values("target_case_id").iterrows():
        target = _parse_moment_json(target_row["target_moment_vector_json"])
        target_bandwidth = _target_bandwidth(
            _string_value(target_row, "target_root_tail_stratum_key")
        )
        axes_to_check: list[tuple[str, tuple[str, ...]]] = [("same_B_Hu", ALL_AXES)]
        if _string_value(target_row, "required_tilt_axes") != "none":
            axes_to_check.append(
                ("same_B_Hu", _axis_tuple(target_row.get("required_tilt_axes", "")))
            )
        axes_to_check.append(("all_B_Hu", ALL_AXES))
        for scope, axes in axes_to_check:
            support = _support_pool(
                joined_feasibility_rows=joined_feasibility_rows,
                deformed_support_rows=deformed_support_rows,
                target_bandwidth=target_bandwidth,
                scope=scope,  # type: ignore[arg-type]
            )
            result = _project_to_hull(support, target, tuple(axes))
            weights = result["weights"]
            assert isinstance(weights, np.ndarray)
            positive = weights[weights > INTERIOR_WEIGHT_TOLERANCE]
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "target_case_id": _string_value(target_row, "target_case_id"),
                    "target_status": _string_value(target_row, "target_status"),
                    "target_root_tail_stratum_key": _string_value(
                        target_row,
                        "target_root_tail_stratum_key",
                    ),
                    "support_pool_scope": scope,
                    "required_tilt_axes": _string_value(
                        target_row,
                        "required_tilt_axes",
                    ),
                    "checked_moment_axes": ",".join(axes),
                    "support_pool_count": int(result["support_count"]),
                    "finite_support_hull_residual_l2": float(result["residual_l2"]),
                    "moment_range_contains_target": bool(result["range_contains"]),
                    "convex_hull_moment_feasible": bool(result["feasible"]),
                    "relative_interior_witness_available": bool(result["interior"]),
                    "tilt_feasibility_status": _status(result),
                    "target_moment_vector_json": _finite_json(
                        {axis: target[axis] for axis in axes}
                    ),
                    "support_min_moment_vector_json": _finite_json(
                        result["min_values"]  # type: ignore[arg-type]
                    ),
                    "support_max_moment_vector_json": _finite_json(
                        result["max_values"]  # type: ignore[arg-type]
                    ),
                    "projected_moment_vector_json": _finite_json(
                        result["projected"]  # type: ignore[arg-type]
                    ),
                    "moment_residual_vector_json": _finite_json(
                        result["residual"]  # type: ignore[arg-type]
                    ),
                    "max_support_weight": float(np.max(weights)) if weights.size else math.nan,
                    "min_positive_support_weight": (
                        float(np.min(positive)) if positive.size else math.nan
                    ),
                    "effective_support_size": (
                        _effective_support_size(weights) if weights.size else 0.0
                    ),
                    "next_mathematical_step": (
                        "current_support_can_fit_diagnostic_tilt_not_production"
                        if bool(result["feasible"])
                        else "generate_new_selected_root_support_for_missing_moments"
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_conditional_tilt_feasibility_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize finite-support conditional tilt feasibility."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    full = rows.loc[
        rows["support_pool_scope"].astype(str).eq("same_B_Hu")
        & rows["checked_moment_axes"].astype(str).eq("T,A,E,S_Hu")
    ]
    required = rows.loc[
        rows["support_pool_scope"].astype(str).eq("same_B_Hu")
        & ~rows["checked_moment_axes"].astype(str).eq("T,A,E,S_Hu")
    ]
    full_feasible = full["convex_hull_moment_feasible"].astype(bool)
    required_feasible = required["convex_hull_moment_feasible"].astype(bool)
    full_missing = (
        full["tilt_feasibility_status"].astype(str).eq("missing_conditioned_support_pool")
    )
    required_missing = (
        required["tilt_feasibility_status"].astype(str).eq("missing_conditioned_support_pool")
    )
    full_outside = (
        full["tilt_feasibility_status"]
        .astype(str)
        .eq("target_outside_current_support_hull_fail_closed")
    )
    required_outside = (
        required["tilt_feasibility_status"]
        .astype(str)
        .eq("target_outside_current_support_hull_fail_closed")
    )
    residuals = pd.to_numeric(
        full["finite_support_hull_residual_l2"],
        errors="coerce",
    )
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "full_phi_feasible_count": int(full_feasible.sum()),
                "required_axes_feasible_count": int(required_feasible.sum()),
                "full_phi_missing_support_count": int(full_missing.sum()),
                "required_axes_missing_support_count": int(required_missing.sum()),
                "full_phi_outside_hull_count": int(full_outside.sum()),
                "required_axes_outside_hull_count": int(required_outside.sum()),
                "max_full_phi_residual_l2": float(residuals.max())
                if residuals.notna().any()
                else math.nan,
                "summary_status": (
                    "all_required_axes_feasible_diagnostic_only"
                    if int(required_feasible.sum()) == int(required.shape[0])
                    else "conditional_tilt_support_hull_incomplete"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_conditional_tilt_feasibility_panel(
    config: RootSelectedConditionalTiltFeasibilityConfig,
) -> dict[str, pd.DataFrame]:
    targets = pd.read_csv(config.external_law_target_rows_path, low_memory=False)
    joined = pd.read_csv(config.joined_feasibility_rows_path, low_memory=False)
    support = pd.read_csv(config.deformed_support_rows_path, low_memory=False)
    rows = build_root_selected_conditional_tilt_feasibility_rows(
        external_law_target_rows=targets,
        joined_feasibility_rows=joined,
        deformed_support_rows=support,
    )
    summary = summarize_root_selected_conditional_tilt_feasibility_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_conditional_tilt_feasibility_panel(
    config: RootSelectedConditionalTiltFeasibilityConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_conditional_tilt_feasibility_panel(config)
    paths = {
        "rows": config.output_dir / ROWS_OUTPUT,
        "summary": config.output_dir / SUMMARY_OUTPUT,
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
    manifest_path = config.output_dir / MANIFEST_OUTPUT
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "row_counts": {key: int(table.shape[0]) for key, table in tables.items()},
        "outputs": paths,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    args = parse_args()
    outputs = run_root_selected_conditional_tilt_feasibility_panel(
        RootSelectedConditionalTiltFeasibilityConfig(
            output_dir=args.output_dir,
            external_law_target_rows_path=args.external_law_target_rows_path,
            joined_feasibility_rows_path=args.joined_feasibility_rows_path,
            deformed_support_rows_path=args.deformed_support_rows_path,
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
