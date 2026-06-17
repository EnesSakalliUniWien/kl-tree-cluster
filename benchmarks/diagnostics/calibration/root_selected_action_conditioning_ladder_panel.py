"""Conditioning ladder for selected-root action gaps.

The root spectral-tail law is exact-stratum only: conservative p-values are
valid only when selected-null or external-null support matches
T, A, E, B, and H_u. This diagnostic asks a narrower question after nearest
support localized the hard roots to selected-ratio action:

How much support appears if we relax A while keeping the other root-tail
conditioning coordinates fixed?

Relaxed rows are diagnostic only and never produce production p-values.
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
    DEFAULT_RESULT_ROOT,
    _action_band,
    _finite_float,
    _is_calibration_support,
    _is_observed_target,
    _root_tail_stratum_key,
    _safe_log1p,
    _spectral_excess_log,
    _tie_band,
)

SCHEMA_VERSION = "root_selected_action_conditioning_ladder_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_action_conditioning_ladder_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "root_selected_action_conditioning_ladder_panel"
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

ROWS_OUTPUT = "root_selected_action_conditioning_ladder_rows.csv"
SUMMARY_OUTPUT = "root_selected_action_conditioning_ladder_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

LADDER_LEVELS = (
    "exact_T_A_E_B_H",
    "relax_A_keep_T_E_B_H",
    "relax_A_B_keep_T_E_H",
    "relax_A_E_keep_T_B_H",
)

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "ladder_level",
    "target_root_tail_stratum_key",
    "target_tie_band",
    "target_action_band",
    "target_edge_band",
    "target_bandwidth_topology_status",
    "target_s_root_spectral_excess_log",
    "target_a_selected_ratio_action_log1p",
    "support_count",
    "spectral_exceedance_count",
    "action_ge_target_count",
    "min_abs_action_gap",
    "median_abs_action_gap",
    "best_support_case_id",
    "best_support_proposal_family",
    "best_support_action_log1p",
    "best_support_s_root_spectral_excess_log",
    "conservative_spectral_tail_p_value",
    "p_value_status",
    "production_inference_status",
    "conditioning_gap_interpretation",
    "legacy_full_selected_null_legacy_false_split",
    "legacy_comparison_interpretation",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "exact_supported_target_count",
    "action_relaxed_supported_target_count",
    "action_only_gap_target_count",
    "fail_closed_target_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedActionConditioningLadderConfig:
    """Input/output contract for selected-root action conditioning ladder."""

    output_dir: Path
    joined_feasibility_rows_path: Path = DEFAULT_JOINED_FEASIBILITY_ROWS
    root_tail_rows_path: Path = DEFAULT_ROOT_TAIL_ROWS
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated"


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
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedActionConditioningLadderConfig):
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


def _tail_lookup(root_tail_rows: pd.DataFrame) -> dict[str, pd.Series]:
    if root_tail_rows.empty:
        return {}
    _require_columns(root_tail_rows, {"target_case_id"}, "root tail rows")
    return {
        str(row["target_case_id"]): row
        for _, row in root_tail_rows.set_index("target_case_id", drop=False).iterrows()
    }


def _support_rows(rows: pd.DataFrame) -> pd.DataFrame:
    support_mask = rows.apply(_is_calibration_support, axis=1)
    target_mask = rows.apply(_is_observed_target, axis=1)
    return rows.loc[support_mask & ~target_mask].copy()


def _target_bands(
    row: pd.Series,
    *,
    h_u_population_law_status: str,
) -> dict[str, object]:
    tie = _finite_float(row.get("root_tie_rank_median_fraction", math.nan))
    action = _safe_log1p(row.get("root_sibling_selected_ratio", math.nan))
    edge = _safe_log1p(row.get("root_edge_path_statistic_margin", math.nan))
    bandwidth = _string_value(row, "root_bandwidth_reopen_band", "")
    return {
        "tie": tie,
        "action": action,
        "edge": edge,
        "tie_band": _tie_band(tie),
        "action_band": _action_band(action),
        "edge_band": _action_band(edge),
        "bandwidth": bandwidth,
        "h_u": str(h_u_population_law_status),
        "s_root": _spectral_excess_log(
            row.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
        ),
        "stratum": _root_tail_stratum_key(
            target=row,
            h_u_population_law_status=h_u_population_law_status,
        ),
    }


def _annotated_support(
    support: pd.DataFrame,
    *,
    h_u_population_law_status: str,
) -> pd.DataFrame:
    if support.empty:
        return support.copy()
    rows = support.copy()
    rows["_tie_band"] = rows["root_tie_rank_median_fraction"].map(
        lambda value: _tie_band(_finite_float(value))
    )
    rows["_action_log1p"] = rows["root_sibling_selected_ratio"].map(_safe_log1p)
    rows["_action_band"] = rows["_action_log1p"].map(_action_band)
    rows["_edge_log1p"] = rows["root_edge_path_statistic_margin"].map(_safe_log1p)
    rows["_edge_band"] = rows["_edge_log1p"].map(_action_band)
    rows["_bandwidth"] = rows.get("root_bandwidth_reopen_band", "").astype(str)
    rows["_h_u"] = str(h_u_population_law_status)
    rows["_s_root"] = rows["root_selected_eigenvalue_over_mp_upper_bound"].map(
        _spectral_excess_log
    )
    rows["_root_tail_stratum_key"] = rows.apply(
        lambda row: _root_tail_stratum_key(
            target=row,
            h_u_population_law_status=h_u_population_law_status,
        ),
        axis=1,
    )
    return rows


def _level_mask(
    support: pd.DataFrame,
    *,
    target: dict[str, object],
    level: str,
) -> pd.Series:
    base_tie = support["_tie_band"].eq(str(target["tie_band"]))
    base_edge = support["_edge_band"].eq(str(target["edge_band"]))
    base_bandwidth = support["_bandwidth"].eq(str(target["bandwidth"]))
    base_h = support["_h_u"].eq(str(target["h_u"]))
    base_action = support["_action_band"].eq(str(target["action_band"]))
    if level == "exact_T_A_E_B_H":
        return base_tie & base_action & base_edge & base_bandwidth & base_h
    if level == "relax_A_keep_T_E_B_H":
        return base_tie & base_edge & base_bandwidth & base_h
    if level == "relax_A_B_keep_T_E_H":
        return base_tie & base_edge & base_h
    if level == "relax_A_E_keep_T_B_H":
        return base_tie & base_bandwidth & base_h
    raise ValueError(f"Unsupported ladder level: {level!r}.")


def _conservative_p(exceedance_count: int, support_count: int) -> float:
    if support_count <= 0:
        return math.nan
    return float((int(exceedance_count) + 1) / (int(support_count) + 1))


def _interpret_level(level: str, support_count: int, exact_count: int) -> str:
    if level == "exact_T_A_E_B_H":
        return "exact_tail_support_available" if support_count else "exact_tail_support_missing"
    if level == "relax_A_keep_T_E_B_H" and support_count and not exact_count:
        return "selected_ratio_action_is_minimal_missing_coordinate"
    if support_count:
        return "relaxed_support_diagnostic_only"
    return "relaxed_support_missing"


def _build_level_row(
    *,
    target_case_id: str,
    target: dict[str, object],
    target_tail: pd.Series | None,
    level: str,
    level_support: pd.DataFrame,
    exact_count: int,
) -> dict[str, object]:
    support_count = int(level_support.shape[0])
    if support_count:
        spectral_exceed = int(level_support["_s_root"].ge(float(target["s_root"])).sum())
        action_ge = int(level_support["_action_log1p"].ge(float(target["action"])).sum())
        action_gaps = (level_support["_action_log1p"] - float(target["action"])).abs()
        ranked = level_support.assign(_action_gap=action_gaps).sort_values(
            ["_action_gap", "case_id"],
            ascending=[True, True],
        )
        best = ranked.iloc[0]
        min_gap = float(action_gaps.min())
        median_gap = float(action_gaps.median())
    else:
        spectral_exceed = 0
        action_ge = 0
        best = {}
        min_gap = math.nan
        median_gap = math.nan
    exact_level = level == "exact_T_A_E_B_H"
    p_value = _conservative_p(spectral_exceed, support_count) if exact_level else math.nan
    p_status = (
        "exact_conservative_empirical_tail"
        if exact_level and support_count
        else "exact_tail_support_missing"
        if exact_level
        else "relaxed_conditioning_not_calibration"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "target_case_id": target_case_id,
        "ladder_level": level,
        "target_root_tail_stratum_key": target["stratum"],
        "target_tie_band": target["tie_band"],
        "target_action_band": target["action_band"],
        "target_edge_band": target["edge_band"],
        "target_bandwidth_topology_status": target["bandwidth"],
        "target_s_root_spectral_excess_log": target["s_root"],
        "target_a_selected_ratio_action_log1p": target["action"],
        "support_count": support_count,
        "spectral_exceedance_count": spectral_exceed,
        "action_ge_target_count": action_ge,
        "min_abs_action_gap": min_gap,
        "median_abs_action_gap": median_gap,
        "best_support_case_id": _string_value(best, "case_id"),
        "best_support_proposal_family": _string_value(best, "proposal_family"),
        "best_support_action_log1p": _finite_float(
            best.get("_action_log1p", math.nan) if support_count else math.nan
        ),
        "best_support_s_root_spectral_excess_log": _finite_float(
            best.get("_s_root", math.nan) if support_count else math.nan
        ),
        "conservative_spectral_tail_p_value": p_value,
        "p_value_status": p_status,
        "production_inference_status": (
            "exact_support_available_defer_to_root_tail_panel"
            if exact_count
            else "fail_closed_action_ladder_diagnostic_only"
        ),
        "conditioning_gap_interpretation": _interpret_level(
            level,
            support_count,
            exact_count,
        ),
        "legacy_full_selected_null_legacy_false_split": bool(
            target_tail.get("legacy_full_selected_null_legacy_false_split", False)
            if target_tail is not None
            else False
        ),
        "legacy_comparison_interpretation": _string_value(
            target_tail if target_tail is not None else {},
            "legacy_comparison_interpretation",
        ),
    }


def build_root_selected_action_conditioning_ladder_rows(
    *,
    joined_feasibility_rows: pd.DataFrame,
    root_tail_rows: pd.DataFrame,
    h_u_population_law_status: str = "identity_mp_assumed_deformed_mp_unestimated",
) -> pd.DataFrame:
    """Build exact and relaxed action-conditioning support rows."""
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
    support = _annotated_support(
        _support_rows(rows),
        h_u_population_law_status=h_u_population_law_status,
    )
    tail_lookup = _tail_lookup(root_tail_rows)
    records: list[dict[str, object]] = []
    for _, target_row in targets.sort_values("case_id").iterrows():
        target_case_id = _string_value(target_row, "case_id")
        target = _target_bands(
            target_row,
            h_u_population_law_status=h_u_population_law_status,
        )
        target_tail = tail_lookup.get(target_case_id)
        exact_support = support.loc[
            _level_mask(support, target=target, level="exact_T_A_E_B_H")
        ]
        exact_count = int(exact_support.shape[0])
        for level in LADDER_LEVELS:
            level_support = support.loc[_level_mask(support, target=target, level=level)]
            records.append(
                _build_level_row(
                    target_case_id=target_case_id,
                    target=target,
                    target_tail=target_tail,
                    level=level,
                    level_support=level_support,
                    exact_count=exact_count,
                )
            )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_action_conditioning_ladder_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize target-level exact versus action-relaxed support."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    exact = rows.loc[rows["ladder_level"].eq("exact_T_A_E_B_H")]
    relaxed_a = rows.loc[rows["ladder_level"].eq("relax_A_keep_T_E_B_H")]
    exact_supported = int(exact["support_count"].gt(0).sum())
    relaxed_supported = int(relaxed_a["support_count"].gt(0).sum())
    exact_by_target = exact.set_index("target_case_id")["support_count"].gt(0)
    relaxed_by_target = relaxed_a.set_index("target_case_id")["support_count"].gt(0)
    action_only = int((~exact_by_target & relaxed_by_target).sum())
    fail_closed = int(exact["production_inference_status"].str.contains("fail_closed").sum())
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(exact.shape[0]),
                "exact_supported_target_count": exact_supported,
                "action_relaxed_supported_target_count": relaxed_supported,
                "action_only_gap_target_count": action_only,
                "fail_closed_target_count": fail_closed,
                "summary_status": (
                    "selected_ratio_action_conditioning_gap_localized"
                    if action_only
                    else "action_relaxation_does_not_restore_support"
                    if fail_closed
                    else "all_targets_exact_supported"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_action_conditioning_ladder_panel(
    config: RootSelectedActionConditioningLadderConfig,
) -> dict[str, pd.DataFrame]:
    joined = pd.read_csv(config.joined_feasibility_rows_path)
    tail = pd.read_csv(config.root_tail_rows_path)
    rows = build_root_selected_action_conditioning_ladder_rows(
        joined_feasibility_rows=joined,
        root_tail_rows=tail,
        h_u_population_law_status=config.h_u_population_law_status,
    )
    summary = summarize_root_selected_action_conditioning_ladder_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_action_conditioning_ladder_panel(
    config: RootSelectedActionConditioningLadderConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_action_conditioning_ladder_panel(config)
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
    outputs = run_root_selected_action_conditioning_ladder_panel(
        RootSelectedActionConditioningLadderConfig(
            output_dir=args.output_dir,
            joined_feasibility_rows_path=args.joined_feasibility_rows_path,
            root_tail_rows_path=args.root_tail_rows_path,
            h_u_population_law_status=str(args.h_u_population_law_status),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
