"""Support-gap diagnostic for deformed selected-root spectral tails.

This diagnostic does not create p-values. It localizes the missing selected-null
or external-null law after the selected-root tail variable has been switched
from identity MP excess to deformed-MP excess:

    S_Hu = log(lambda_root / b(H_u, gamma))_+.

Production calibration still requires same-stratum selected-null/external
support in T, A, E, B, and H_u. The panel reports which fail-closed roots lack
that support and what deformed spectral excess the nearest available support
rows currently reach.
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

from benchmarks.diagnostics.calibration.root.root_tail_values import (
    action_band,
    finite_float,
    is_calibration_support,
    is_observed_target,
    root_tail_stratum_key,
    safe_log1p,
    string_value,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
)

SCHEMA_VERSION = "root_selected_deformed_tail_support_gap_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_deformed_tail_support_gap_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.selected.root_selected_deformed_tail_support_gap_panel"
)

DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_importance_external_null_topology_join_mild_hu_replay_v3_smoke"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)
DEFAULT_DEFORMED_TAIL_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_spectral_tail_law_deformed_hu_mild_replay_v3_smoke"
    / "root_selected_spectral_tail_law_rows.csv"
)
DEFAULT_DEFORMED_SUPPORT_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_deformed_mp_edge_mild_hu_replay_v3_smoke"
    / "root_selected_deformed_mp_edge_support_rows.csv"
)

ROWS_OUTPUT = "root_selected_deformed_tail_support_gap_rows.csv"
SUMMARY_OUTPUT = "root_selected_deformed_tail_support_gap_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_root_tail_stratum_key",
    "target_s_h_u_excess_log",
    "target_s_identity_excess_log",
    "target_tie_fraction",
    "target_action_log1p",
    "target_edge_log1p",
    "target_bandwidth_topology_status",
    "tail_panel_support_count",
    "tail_panel_inference_status",
    "nearest_support_case_id",
    "nearest_support_data_role",
    "nearest_support_calibration_role",
    "nearest_support_proposal_family",
    "nearest_support_s_h_u_excess_log",
    "nearest_support_identity_excess_log",
    "nearest_support_distance",
    "nearest_support_tie_gap",
    "nearest_support_action_gap",
    "nearest_support_edge_gap",
    "nearest_support_bandwidth_mismatch",
    "same_stratum_support_count",
    "same_stratum_max_s_h_u_excess_log",
    "same_stratum_s_h_u_exceedance_count",
    "required_s_h_u_gap_to_nearest_support",
    "required_s_h_u_lift_multiplier",
    "dominant_conditioning_gap",
    "support_gap_status",
    "production_inference_status",
    "required_external_law",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "exact_support_target_count",
    "fail_closed_target_count",
    "nearest_deformed_support_available_target_count",
    "same_stratum_nonexceeding_target_count",
    "missing_same_stratum_target_count",
    "max_required_s_h_u_gap",
    "max_required_s_h_u_lift_multiplier",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedDeformedTailSupportGapConfig:
    """Input/output contract for deformed selected-root support gaps."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    deformed_tail_rows_path: Path = DEFAULT_DEFORMED_TAIL_ROWS
    deformed_support_rows_path: Path = DEFAULT_DEFORMED_SUPPORT_ROWS
    h_u_population_law_status: str = "deformed_mp_edge_measured_support_side"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joined-feasibility-rows-path",
        type=Path,
        default=DEFAULT_JOINED_FEASIBILITY_ROWS,
    )
    parser.add_argument(
        "--deformed-tail-rows-path",
        type=Path,
        default=DEFAULT_DEFORMED_TAIL_ROWS,
    )
    parser.add_argument(
        "--deformed-support-rows-path",
        type=Path,
        default=DEFAULT_DEFORMED_SUPPORT_ROWS,
    )
    parser.add_argument(
        "--h-u-population-law-status",
        default="deformed_mp_edge_measured_support_side",
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedDeformedTailSupportGapConfig):
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


def _tail_lookup(rows: pd.DataFrame) -> dict[str, pd.Series]:
    if rows.empty:
        return {}
    _require_columns(rows, {"target_case_id"}, "deformed tail rows")
    return {str(row["target_case_id"]): row for _, row in rows.iterrows()}


def _support_deformed_lookup(rows: pd.DataFrame) -> dict[str, pd.Series]:
    if rows.empty:
        return {}
    _require_columns(rows, {"case_id"}, "deformed support rows")
    return {str(row["case_id"]): row for _, row in rows.iterrows()}


def _tie_band(value: float) -> str:
    if not math.isfinite(value):
        return "tie_missing"
    if value < 0.70:
        return "tie_low_lt_0_70"
    if value < 0.85:
        return "tie_mid_0_70_0_85"
    return "tie_high_ge_0_85"


def _coordinates(
    row: pd.Series,
    *,
    h_u_population_law_status: str,
) -> dict[str, object]:
    tie = finite_float(row.get("root_tie_rank_median_fraction", math.nan))
    action = safe_log1p(row.get("root_sibling_selected_ratio", math.nan))
    edge = safe_log1p(row.get("root_edge_path_statistic_margin", math.nan))
    return {
        "tie": tie,
        "action": action,
        "edge": edge,
        "tie_band": _tie_band(tie),
        "action_band": action_band(action),
        "edge_band": action_band(edge),
        "bandwidth": string_value(row, "root_bandwidth_reopen_band", ""),
        "stratum": root_tail_stratum_key(
            target=row,
            h_u_population_law_status=h_u_population_law_status,
        ),
    }


def _identity_excess(row: pd.Series) -> float:
    ratio = finite_float(row.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan))
    return float(max(math.log(max(ratio, 1e-12)), 0.0)) if math.isfinite(ratio) else math.nan


def _support_rows(joined_rows: pd.DataFrame) -> pd.DataFrame:
    support_mask = joined_rows.apply(is_calibration_support, axis=1)
    target_mask = joined_rows.apply(is_observed_target, axis=1)
    return joined_rows.loc[support_mask & ~target_mask].copy()


def _distance_components(
    target: dict[str, object],
    support: dict[str, object],
) -> dict[str, float]:
    tie_gap = abs(float(support["tie"]) - float(target["tie"]))
    action_gap = abs(float(support["action"]) - float(target["action"]))
    edge_gap = abs(float(support["edge"]) - float(target["edge"]))
    bandwidth_mismatch = 0.0 if support["bandwidth"] == target["bandwidth"] else 1.0
    distance = math.sqrt(
        (tie_gap / 0.15) ** 2
        + (action_gap / 2.0) ** 2
        + (edge_gap / 2.0) ** 2
        + (2.0 * bandwidth_mismatch) ** 2
    )
    return {
        "nearest_support_distance": float(distance),
        "nearest_support_tie_gap": float(tie_gap),
        "nearest_support_action_gap": float(action_gap),
        "nearest_support_edge_gap": float(edge_gap),
        "nearest_support_bandwidth_mismatch": float(bandwidth_mismatch),
    }


def _dominant_gap(components: dict[str, float]) -> str:
    scaled = {
        "tie_rank": components["nearest_support_tie_gap"] / 0.15,
        "selected_ratio_action": components["nearest_support_action_gap"] / 2.0,
        "edge_action": components["nearest_support_edge_gap"] / 2.0,
        "bandwidth_topology": 2.0 * components["nearest_support_bandwidth_mismatch"],
    }
    return max(scaled, key=scaled.get)


def _empty_components() -> dict[str, float]:
    return {
        "nearest_support_distance": math.nan,
        "nearest_support_tie_gap": math.nan,
        "nearest_support_action_gap": math.nan,
        "nearest_support_edge_gap": math.nan,
        "nearest_support_bandwidth_mismatch": math.nan,
    }


def build_root_selected_deformed_tail_support_gap_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    deformed_tail_rows: pd.DataFrame,
    deformed_support_rows: pd.DataFrame,
    h_u_population_law_status: str = "deformed_mp_edge_measured_support_side",
) -> pd.DataFrame:
    """Return row-level selected-root deformed-tail support gaps."""
    _require_columns(
        joined_feasibility_rows,
        {
            "case_id",
            "data_role",
            "calibration_role",
            "proposal_family",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        "joined feasibility rows",
    )
    rows = joined_feasibility_rows.copy()
    if "root_bandwidth_reopen_band" not in rows.columns:
        rows["root_bandwidth_reopen_band"] = ""
    if "root_mixed_region_component" not in rows.columns:
        rows["root_mixed_region_component"] = "root_component_missing"
    tail_by_case = _tail_lookup(deformed_tail_rows)
    support_deformed_by_case = _support_deformed_lookup(deformed_support_rows)
    targets = rows.loc[rows.apply(is_observed_target, axis=1)].copy()
    supports = _support_rows(rows)
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        target_id = string_value(target, "case_id")
        target_tail = tail_by_case.get(target_id)
        target_coords = _coordinates(
            target,
            h_u_population_law_status=h_u_population_law_status,
        )
        target_s_h_u = finite_float(
            target_tail.get("s_root_deformed_excess_log", math.nan)
            if target_tail is not None
            else math.nan
        )
        target_identity = finite_float(
            target_tail.get("s_root_identity_excess_log", math.nan)
            if target_tail is not None
            else _identity_excess(target)
        )
        tail_support_count = int(
            finite_float(
                target_tail.get("selected_null_support_count", 0) if target_tail is not None else 0
            )
        )
        nearest: pd.Series | None = None
        nearest_components = _empty_components()
        nearest_s_h_u = math.nan
        nearest_identity = math.nan
        same_stratum_s: list[float] = []
        for _, support in supports.iterrows():
            support_id = string_value(support, "case_id")
            support_deformed = support_deformed_by_case.get(support_id)
            support_s_h_u = finite_float(
                support_deformed.get("s_root_deformed_excess_log", math.nan)
                if support_deformed is not None
                else math.nan
            )
            if not math.isfinite(support_s_h_u):
                continue
            support_coords = _coordinates(
                support,
                h_u_population_law_status=h_u_population_law_status,
            )
            if support_coords["stratum"] == target_coords["stratum"]:
                same_stratum_s.append(support_s_h_u)
            components = _distance_components(target_coords, support_coords)
            if (
                nearest is None
                or components["nearest_support_distance"]
                < nearest_components["nearest_support_distance"]
            ):
                nearest = support
                nearest_components = components
                nearest_s_h_u = support_s_h_u
                nearest_identity = finite_float(
                    support_deformed.get("s_root_identity_excess_log", math.nan)
                    if support_deformed is not None
                    else _identity_excess(support)
                )
        same_stratum_count = len(same_stratum_s)
        same_stratum_max = max(same_stratum_s) if same_stratum_s else math.nan
        same_stratum_exceedances = (
            int(np.sum(np.asarray(same_stratum_s) >= target_s_h_u))
            if same_stratum_s and math.isfinite(target_s_h_u)
            else 0
        )
        required_gap = (
            max(float(target_s_h_u - nearest_s_h_u), 0.0)
            if math.isfinite(target_s_h_u) and math.isfinite(nearest_s_h_u)
            else math.nan
        )
        required_lift = math.exp(required_gap) if math.isfinite(required_gap) else math.nan
        if tail_support_count > 0:
            gap_status = "exact_tail_support_available"
            production_status = "defer_to_deformed_tail_panel"
        elif same_stratum_count > 0:
            gap_status = "same_stratum_support_nonexceeding"
            production_status = "fail_closed_same_stratum_tail_nonexceeding"
        else:
            gap_status = "same_stratum_support_missing"
            production_status = "fail_closed_same_stratum_support_missing"
        dominant = (
            _dominant_gap(nearest_components)
            if nearest is not None
            else "no_support_rows_with_deformed_excess"
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": target_id,
                "target_root_tail_stratum_key": target_coords["stratum"],
                "target_s_h_u_excess_log": target_s_h_u,
                "target_s_identity_excess_log": target_identity,
                "target_tie_fraction": target_coords["tie"],
                "target_action_log1p": target_coords["action"],
                "target_edge_log1p": target_coords["edge"],
                "target_bandwidth_topology_status": target_coords["bandwidth"],
                "tail_panel_support_count": tail_support_count,
                "tail_panel_inference_status": string_value(
                    target_tail if target_tail is not None else {},
                    "root_tail_inference_status",
                ),
                "nearest_support_case_id": string_value(
                    nearest if nearest is not None else {},
                    "case_id",
                ),
                "nearest_support_data_role": string_value(
                    nearest if nearest is not None else {},
                    "data_role",
                ),
                "nearest_support_calibration_role": string_value(
                    nearest if nearest is not None else {},
                    "calibration_role",
                ),
                "nearest_support_proposal_family": string_value(
                    nearest if nearest is not None else {},
                    "proposal_family",
                ),
                "nearest_support_s_h_u_excess_log": nearest_s_h_u,
                "nearest_support_identity_excess_log": nearest_identity,
                **nearest_components,
                "same_stratum_support_count": same_stratum_count,
                "same_stratum_max_s_h_u_excess_log": same_stratum_max,
                "same_stratum_s_h_u_exceedance_count": same_stratum_exceedances,
                "required_s_h_u_gap_to_nearest_support": required_gap,
                "required_s_h_u_lift_multiplier": required_lift,
                "dominant_conditioning_gap": dominant,
                "support_gap_status": gap_status,
                "production_inference_status": production_status,
                "required_external_law": (
                    "generate_same_T_A_E_B_Hu_roots_with_nonzero_deformed_spectral_excess"
                    if tail_support_count <= 0
                    else "use_existing_conservative_deformed_tail_panel"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_deformed_tail_support_gap_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected-root deformed-tail support gaps."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    exact = int(rows["tail_panel_support_count"].gt(0).sum())
    fail_closed = int(
        rows["production_inference_status"].astype(str).str.startswith("fail_closed").sum()
    )
    same_nonexceeding = int(
        rows["support_gap_status"].astype(str).eq("same_stratum_support_nonexceeding").sum()
    )
    missing_same = int(
        rows["support_gap_status"].astype(str).eq("same_stratum_support_missing").sum()
    )
    gaps = pd.to_numeric(rows["required_s_h_u_gap_to_nearest_support"], errors="coerce")
    lifts = pd.to_numeric(rows["required_s_h_u_lift_multiplier"], errors="coerce")
    support_rows = int(
        rows["nearest_support_case_id"].astype(str).replace("", np.nan).notna().sum()
    )
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "exact_support_target_count": exact,
                "fail_closed_target_count": fail_closed,
                "nearest_deformed_support_available_target_count": support_rows,
                "same_stratum_nonexceeding_target_count": same_nonexceeding,
                "missing_same_stratum_target_count": missing_same,
                "max_required_s_h_u_gap": float(gaps.max()) if gaps.notna().any() else math.nan,
                "max_required_s_h_u_lift_multiplier": float(lifts.max())
                if lifts.notna().any()
                else math.nan,
                "summary_status": (
                    "deformed_tail_same_stratum_law_missing"
                    if fail_closed
                    else "all_targets_have_deformed_tail_support"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_deformed_tail_support_gap_panel(
    config: RootSelectedDeformedTailSupportGapConfig,
) -> dict[str, pd.DataFrame]:
    joined = pd.read_csv(config.joined_feasibility_rows_path, low_memory=False)
    tail = pd.read_csv(config.deformed_tail_rows_path, low_memory=False)
    support = pd.read_csv(config.deformed_support_rows_path, low_memory=False)
    rows = build_root_selected_deformed_tail_support_gap_rows(
        joined_feasibility_rows=joined,
        deformed_tail_rows=tail,
        deformed_support_rows=support,
        h_u_population_law_status=config.h_u_population_law_status,
    )
    summary = summarize_root_selected_deformed_tail_support_gap_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_deformed_tail_support_gap_panel(
    config: RootSelectedDeformedTailSupportGapConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_deformed_tail_support_gap_panel(config)
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
    outputs = run_root_selected_deformed_tail_support_gap_panel(
        RootSelectedDeformedTailSupportGapConfig(
            output_dir=args.output_dir,
            joined_feasibility_rows_path=args.joined_feasibility_rows_path,
            deformed_tail_rows_path=args.deformed_tail_rows_path,
            deformed_support_rows_path=args.deformed_support_rows_path,
            h_u_population_law_status=str(args.h_u_population_law_status),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
