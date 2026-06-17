"""Nearest-support diagnostic for selected-root spectral tails.

This panel does not calibrate a new p-value. It explains where selected-null
or external-null support sits relative to each observed root in the conditioning
coordinates used by the root spectral-tail law:

* S_root = log(lambda / lambda_MP)
* T = selected tie-rank fraction
* A = log selected-ratio action
* E = log edge-margin action
* B = measured bandwidth/topology status
* H_u = local null-whitened spectral population-law status

Exact same-stratum support remains the only route to a conservative empirical
tail p-value. Nearest support is diagnostic localization only.
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

from benchmarks.diagnostics.calibration.root_selected_spectral_tail_law_panel import (
    DEFAULT_LEGACY_FULL_PAIRWISE_ROWS,
    DEFAULT_LEGACY_INTERNAL_PAIRWISE_ROWS,
    DEFAULT_RESULT_ROOT,
    _action_band,
    _finite_float,
    _is_calibration_support,
    _is_observed_target,
    _root_tail_stratum_key,
    _safe_log1p,
    _spectral_excess_log,
)

SCHEMA_VERSION = "root_selected_spectral_tail_nearest_support_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_spectral_tail_nearest_support_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "root_selected_spectral_tail_nearest_support_panel"
)

DEFAULT_JOINED_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_importance_external_null_topology_join_mild_accumulated"
    / "conditioned_coherent_joined_feasibility_rows.csv"
)
DEFAULT_ROOT_TAIL_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_spectral_tail_law_importance_external_mild_accumulated"
    / "root_selected_spectral_tail_law_rows.csv"
)

ROWS_OUTPUT = "root_selected_spectral_tail_nearest_support_rows.csv"
SUMMARY_OUTPUT = "root_selected_spectral_tail_nearest_support_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_root_tail_stratum_key",
    "target_s_root_spectral_excess_log",
    "target_t_selected_tie_rank_fraction",
    "target_a_selected_ratio_action_log1p",
    "target_e_edge_margin_action_log1p",
    "target_b_bandwidth_topology_status",
    "exact_selected_null_support_count",
    "tail_panel_inference_status",
    "tail_panel_p_value_status",
    "tail_panel_conservative_p_value",
    "nearest_support_case_id",
    "nearest_support_data_role",
    "nearest_support_calibration_role",
    "nearest_support_proposal_family",
    "nearest_support_distance",
    "nearest_support_tie_gap",
    "nearest_support_action_gap",
    "nearest_support_edge_gap",
    "nearest_support_bandwidth_mismatch",
    "nearest_support_h_u_mismatch",
    "nearest_support_s_root_spectral_excess_log",
    "nearest_support_spectral_tail_gap",
    "nearest_support_pre_topology_match",
    "nearest_support_root_tail_stratum_match",
    "dominant_conditioning_gap",
    "nearest_support_status",
    "production_inference_status",
    "legacy_full_selected_null_legacy_false_split",
    "legacy_comparison_interpretation",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "exact_support_target_count",
    "nearest_support_available_count",
    "fail_closed_nearest_only_count",
    "legacy_full_selected_null_false_split_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedSpectralTailNearestSupportConfig:
    """Input/output contract for nearest selected-root support localization."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    root_tail_rows_path: Path = DEFAULT_ROOT_TAIL_ROWS
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated"
    legacy_full_pairwise_rows_path: Path | None = DEFAULT_LEGACY_FULL_PAIRWISE_ROWS
    legacy_internal_pairwise_rows_path: Path | None = DEFAULT_LEGACY_INTERNAL_PAIRWISE_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joined-feasibility-rows-path",
        type=Path,
        default=DEFAULT_JOINED_FEASIBILITY_ROWS,
    )
    parser.add_argument("--root-tail-rows-path", type=Path, default=DEFAULT_ROOT_TAIL_ROWS)
    parser.add_argument(
        "--h-u-population-law-status",
        default="identity_mp_assumed_deformed_mp_unestimated",
    )
    parser.add_argument(
        "--legacy-full-pairwise-rows-path",
        type=Path,
        default=DEFAULT_LEGACY_FULL_PAIRWISE_ROWS,
    )
    parser.add_argument(
        "--legacy-internal-pairwise-rows-path",
        type=Path,
        default=DEFAULT_LEGACY_INTERNAL_PAIRWISE_ROWS,
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedSpectralTailNearestSupportConfig):
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


def _string_value(row: pd.Series | dict[str, object], column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _target_tail_lookup(root_tail_rows: pd.DataFrame) -> dict[str, pd.Series]:
    if root_tail_rows.empty:
        return {}
    _require_columns(root_tail_rows, {"target_case_id"}, "root tail rows")
    return {
        str(row["target_case_id"]): row
        for _, row in root_tail_rows.set_index("target_case_id", drop=False).iterrows()
    }


def _support_rows(joined_feasibility_rows: pd.DataFrame) -> pd.DataFrame:
    if joined_feasibility_rows.empty:
        return joined_feasibility_rows.copy()
    support_mask = joined_feasibility_rows.apply(_is_calibration_support, axis=1)
    target_mask = joined_feasibility_rows.apply(_is_observed_target, axis=1)
    return joined_feasibility_rows.loc[support_mask & ~target_mask].copy()


def _conditioning_coordinates(
    row: pd.Series,
    *,
    h_u_population_law_status: str,
) -> dict[str, object]:
    return {
        "tie": _finite_float(row.get("root_tie_rank_median_fraction", math.nan)),
        "action": _safe_log1p(row.get("root_sibling_selected_ratio", math.nan)),
        "edge": _safe_log1p(row.get("root_edge_path_statistic_margin", math.nan)),
        "bandwidth": _string_value(row, "root_bandwidth_reopen_band", ""),
        "h_u": str(h_u_population_law_status),
        "s_root": _spectral_excess_log(
            row.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
        ),
        "stratum": _root_tail_stratum_key(
            target=row,
            h_u_population_law_status=h_u_population_law_status,
        ),
    }


def _pre_topology_match(target: dict[str, object], support: dict[str, object]) -> bool:
    return (
        _action_band(float(target["action"])) == _action_band(float(support["action"]))
        and _action_band(float(target["edge"])) == _action_band(float(support["edge"]))
        and _tie_pre_band(float(target["tie"])) == _tie_pre_band(float(support["tie"]))
    )


def _tie_pre_band(value: float) -> str:
    if not math.isfinite(value):
        return "tie_missing"
    if value < 0.70:
        return "tie_low_lt_0_70"
    if value < 0.85:
        return "tie_mid_0_70_0_85"
    return "tie_high_ge_0_85"


def _distance_components(
    target: dict[str, object],
    support: dict[str, object],
) -> dict[str, float]:
    tie_gap = abs(float(support["tie"]) - float(target["tie"]))
    action_gap = abs(float(support["action"]) - float(target["action"]))
    edge_gap = abs(float(support["edge"]) - float(target["edge"]))
    bandwidth_mismatch = 0.0 if support["bandwidth"] == target["bandwidth"] else 1.0
    h_u_mismatch = 0.0 if support["h_u"] == target["h_u"] else 1.0
    distance = math.sqrt(
        (tie_gap / 0.15) ** 2
        + (action_gap / 2.0) ** 2
        + (edge_gap / 2.0) ** 2
        + (2.0 * bandwidth_mismatch) ** 2
        + h_u_mismatch**2
    )
    return {
        "nearest_support_distance": float(distance),
        "nearest_support_tie_gap": float(tie_gap),
        "nearest_support_action_gap": float(action_gap),
        "nearest_support_edge_gap": float(edge_gap),
        "nearest_support_bandwidth_mismatch": float(bandwidth_mismatch),
        "nearest_support_h_u_mismatch": float(h_u_mismatch),
    }


def _dominant_gap(components: dict[str, float]) -> str:
    scaled = {
        "tie_rank": components["nearest_support_tie_gap"] / 0.15,
        "selected_ratio_action": components["nearest_support_action_gap"] / 2.0,
        "edge_action": components["nearest_support_edge_gap"] / 2.0,
        "bandwidth_topology": 2.0 * components["nearest_support_bandwidth_mismatch"],
        "h_u_population_law": components["nearest_support_h_u_mismatch"],
    }
    return max(scaled, key=scaled.get)


def _nearest_support(
    *,
    target: pd.Series,
    support: pd.DataFrame,
    h_u_population_law_status: str,
) -> tuple[pd.Series | None, dict[str, float], dict[str, object], dict[str, object]]:
    target_coords = _conditioning_coordinates(
        target,
        h_u_population_law_status=h_u_population_law_status,
    )
    if support.empty:
        return None, {}, target_coords, {}
    best_row: pd.Series | None = None
    best_components: dict[str, float] = {}
    best_support_coords: dict[str, object] = {}
    for _, candidate in support.iterrows():
        candidate_coords = _conditioning_coordinates(
            candidate,
            h_u_population_law_status=h_u_population_law_status,
        )
        components = _distance_components(target_coords, candidate_coords)
        if (
            best_row is None
            or components["nearest_support_distance"]
            < best_components["nearest_support_distance"]
        ):
            best_row = candidate
            best_components = components
            best_support_coords = candidate_coords
    return best_row, best_components, target_coords, best_support_coords


def build_root_selected_spectral_tail_nearest_support_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    root_tail_rows: pd.DataFrame,
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated",
) -> pd.DataFrame:
    """Return nearest-support localization rows for observed roots."""
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
    targets = rows.loc[rows.apply(_is_observed_target, axis=1)].copy()
    support = _support_rows(rows)
    tail_lookup = _target_tail_lookup(root_tail_rows)
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        case_id = _string_value(target, "case_id")
        tail = tail_lookup.get(case_id)
        nearest, components, target_coords, support_coords = _nearest_support(
            target=target,
            support=support,
            h_u_population_law_status=h_u_population_law_status,
        )
        has_nearest = nearest is not None
        exact_support_count = int(
            _finite_float(tail.get("selected_null_support_count", 0))
            if tail is not None
            else 0
        )
        if has_nearest:
            spectral_gap = float(support_coords["s_root"]) - float(target_coords["s_root"])
            root_tail_match = bool(support_coords["stratum"] == target_coords["stratum"])
            pre_topology_match = _pre_topology_match(target_coords, support_coords)
            dominant_gap = _dominant_gap(components)
        else:
            spectral_gap = math.nan
            root_tail_match = False
            pre_topology_match = False
            dominant_gap = "no_support_rows"
            components = {
                "nearest_support_distance": math.nan,
                "nearest_support_tie_gap": math.nan,
                "nearest_support_action_gap": math.nan,
                "nearest_support_edge_gap": math.nan,
                "nearest_support_bandwidth_mismatch": math.nan,
                "nearest_support_h_u_mismatch": math.nan,
            }
            support_coords = {"s_root": math.nan}
        production_status = (
            "exact_support_available_defer_to_tail_panel"
            if exact_support_count > 0
            else "fail_closed_nearest_support_only"
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": case_id,
                "target_root_tail_stratum_key": target_coords["stratum"],
                "target_s_root_spectral_excess_log": target_coords["s_root"],
                "target_t_selected_tie_rank_fraction": target_coords["tie"],
                "target_a_selected_ratio_action_log1p": target_coords["action"],
                "target_e_edge_margin_action_log1p": target_coords["edge"],
                "target_b_bandwidth_topology_status": target_coords["bandwidth"],
                "exact_selected_null_support_count": exact_support_count,
                "tail_panel_inference_status": _string_value(
                    tail if tail is not None else {},
                    "root_tail_inference_status",
                ),
                "tail_panel_p_value_status": _string_value(
                    tail if tail is not None else {},
                    "spectral_tail_p_value_status",
                ),
                "tail_panel_conservative_p_value": _finite_float(
                    tail.get("conservative_spectral_tail_p_value", math.nan)
                    if tail is not None
                    else math.nan
                ),
                "nearest_support_case_id": _string_value(
                    nearest if nearest is not None else {},
                    "case_id",
                ),
                "nearest_support_data_role": _string_value(
                    nearest if nearest is not None else {},
                    "data_role",
                ),
                "nearest_support_calibration_role": _string_value(
                    nearest if nearest is not None else {},
                    "calibration_role",
                ),
                "nearest_support_proposal_family": _string_value(
                    nearest if nearest is not None else {},
                    "proposal_family",
                ),
                **components,
                "nearest_support_s_root_spectral_excess_log": support_coords["s_root"],
                "nearest_support_spectral_tail_gap": spectral_gap,
                "nearest_support_pre_topology_match": pre_topology_match,
                "nearest_support_root_tail_stratum_match": root_tail_match,
                "dominant_conditioning_gap": dominant_gap,
                "nearest_support_status": (
                    "nearest_support_diagnostic_only"
                    if has_nearest
                    else "no_calibration_support_rows_available"
                ),
                "production_inference_status": production_status,
                "legacy_full_selected_null_legacy_false_split": bool(
                    tail.get("legacy_full_selected_null_legacy_false_split", False)
                    if tail is not None
                    else False
                ),
                "legacy_comparison_interpretation": _string_value(
                    tail if tail is not None else {},
                    "legacy_comparison_interpretation",
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_spectral_tail_nearest_support_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize nearest-support localization."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    exact = int(rows["exact_selected_null_support_count"].gt(0).sum())
    nearest = int(
        rows["nearest_support_status"].astype(str).eq("nearest_support_diagnostic_only").sum()
    )
    fail_closed = int(
        rows["production_inference_status"]
        .astype(str)
        .eq("fail_closed_nearest_support_only")
        .sum()
    )
    legacy_false = int(rows["legacy_full_selected_null_legacy_false_split"].sum())
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "exact_support_target_count": exact,
                "nearest_support_available_count": nearest,
                "fail_closed_nearest_only_count": fail_closed,
                "legacy_full_selected_null_false_split_count": legacy_false,
                "summary_status": (
                    "nearest_support_localized_but_tail_support_missing"
                    if fail_closed
                    else "all_targets_have_exact_tail_support"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_spectral_tail_nearest_support_panel(
    config: RootSelectedSpectralTailNearestSupportConfig,
) -> dict[str, pd.DataFrame]:
    joined = pd.read_csv(config.joined_feasibility_rows_path)
    tail = pd.read_csv(config.root_tail_rows_path)
    rows = build_root_selected_spectral_tail_nearest_support_rows(
        joined_feasibility_rows=joined,
        root_tail_rows=tail,
        h_u_population_law_status=config.h_u_population_law_status,
    )
    summary = summarize_root_selected_spectral_tail_nearest_support_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_spectral_tail_nearest_support_panel(
    config: RootSelectedSpectralTailNearestSupportConfig,
) -> dict[str, Path]:
    start = datetime.now(UTC)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_spectral_tail_nearest_support_panel(config)
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
        "generated_at": start.isoformat(),
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
    outputs = run_root_selected_spectral_tail_nearest_support_panel(
        RootSelectedSpectralTailNearestSupportConfig(
            output_dir=args.output_dir,
            joined_feasibility_rows_path=args.joined_feasibility_rows_path,
            root_tail_rows_path=args.root_tail_rows_path,
            h_u_population_law_status=str(args.h_u_population_law_status),
            legacy_full_pairwise_rows_path=args.legacy_full_pairwise_rows_path,
            legacy_internal_pairwise_rows_path=args.legacy_internal_pairwise_rows_path,
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
