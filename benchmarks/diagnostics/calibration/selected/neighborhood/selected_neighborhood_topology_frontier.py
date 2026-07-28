"""Selected-neighborhood topology frontier diagnostics.

This panel compares three overlapping evidence surfaces on the same candidate
rows:

1. the current direct selected-family sibling decision;
2. the older bandwidth interpolation proxy at a reference tau_s;
3. root and non-root topology frontier proxies for a possible hybrid law.

The output is diagnostic-only. Structural root balance is reported as a proxy,
not as an admissible root-selected topology law.
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

SCHEMA_VERSION = "selected_neighborhood_topology_frontier/v1"
STUDY_ROLE = "diagnostic_selected_neighborhood_topology_frontier_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_topology_frontier"

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_MEASURABILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "selected_neighborhood_measurability_law_overlap_expanded_candidates"
    / "selected_neighborhood_measurability_law_rows.csv"
)

DEFAULT_REFERENCE_TAU_S = 20.0
DEFAULT_ALPHA = 0.01
DEFAULT_EFFECTIVE_SUPPORT_FLOOR = 2.0
DEFAULT_ROOT_OUTGOING_BALANCE_FLOOR = 0.30
DEFAULT_NONROOT_BALANCE_PRODUCT_FLOOR = 0.22
DEFAULT_NEAR_FRONTIER_WIDTH = 0.05
DEFAULT_REQUIRE_SPECTRAL_FLOW = True
DEFAULT_ROOT_GRID = (0.10, 0.15, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45)
DEFAULT_NONROOT_GRID = (0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22)

ROWS_OUTPUT = "selected_neighborhood_topology_frontier_rows.csv"
SUMMARY_OUTPUT = "selected_neighborhood_topology_frontier_summary.csv"
SWEEP_OUTPUT = "selected_neighborhood_topology_threshold_sweep.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "parent_id",
    "depth",
    "candidate_scope",
    "current_measurability_action",
    "current_measurability_bottleneck",
    "direct_sibling_measurable",
    "direct_sibling_open",
    "direct_sibling_p_value",
    "direct_sibling_significant",
    "interpolated_pair_null_prior",
    "interpolation_default_significant",
    "interpolation_effective_support",
    "interpolation_effective_support_pass",
    "interpolation_best_case_required_tau_s_for_alpha",
    "bandwidth_reference_tau_s",
    "bandwidth_reference_reopens",
    "bandwidth_reference_direct_positive_reopens",
    "root_outgoing_balance",
    "root_structural_proxy_pass",
    "root_structural_proxy_status",
    "root_selected_law_status",
    "root_margin_evidence_status",
    "root_child_min_merge_margin",
    "root_child_tied_minimum_merge_count",
    "root_child_discrete_tie_cell_count",
    "root_sibling_selected_ratio",
    "nonroot_balance_product",
    "nonroot_current_floor_margin",
    "nonroot_near_frontier",
    "nonroot_frontier_status",
    "spectral_bottleneck_status",
    "spectral_strict_pass",
    "hybrid_strict_support",
    "hybrid_strict_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "data_role",
    "method_id",
    "row_count",
    "current_split_count",
    "direct_measurable_count",
    "non_direct_count",
    "bandwidth_reference_reopen_count",
    "bandwidth_reference_direct_positive_reopen_count",
    "bandwidth_reference_reopen_non_direct_count",
    "root_non_direct_count",
    "root_structural_proxy_pass_count",
    "root_selected_law_missing_count",
    "root_discrete_tie_cell_count",
    "nonroot_non_direct_count",
    "nonroot_near_frontier_count",
    "nonroot_current_floor_pass_count",
    "spectral_strict_pass_non_direct_count",
    "hybrid_strict_support_count",
    "median_root_outgoing_balance",
    "median_nonroot_balance_product",
    "median_effective_support",
    "median_best_case_tau_s_non_direct",
    "frontier_status",
)

SWEEP_COLUMNS = (
    "schema_version",
    "study_role",
    "data_role",
    "method_id",
    "candidate_scope",
    "threshold_type",
    "threshold",
    "row_count",
    "pass_count",
    "pass_fraction",
    "bandwidth_reference_reopen_count",
    "bandwidth_reference_direct_positive_reopen_count",
    "hybrid_strict_support_count",
    "median_best_case_tau_s_for_pass",
    "sweep_status",
)


@dataclass(frozen=True)
class SelectedNeighborhoodTopologyFrontierConfig:
    """Input paths and threshold knobs for the topology frontier panel."""

    output_dir: Path
    measurability_rows_path: Path = DEFAULT_MEASURABILITY_ROWS
    root_selected_region_summary_path: Path | None = None
    reference_tau_s: float = DEFAULT_REFERENCE_TAU_S
    alpha: float = DEFAULT_ALPHA
    effective_support_floor: float = DEFAULT_EFFECTIVE_SUPPORT_FLOOR
    root_outgoing_balance_floor: float = DEFAULT_ROOT_OUTGOING_BALANCE_FLOOR
    nonroot_balance_product_floor: float = DEFAULT_NONROOT_BALANCE_PRODUCT_FLOOR
    near_frontier_width: float = DEFAULT_NEAR_FRONTIER_WIDTH
    require_spectral_flow: bool = DEFAULT_REQUIRE_SPECTRAL_FLOW
    root_threshold_grid: tuple[float, ...] = DEFAULT_ROOT_GRID
    nonroot_threshold_grid: tuple[float, ...] = DEFAULT_NONROOT_GRID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--measurability-rows-path",
        type=Path,
        default=DEFAULT_MEASURABILITY_ROWS,
    )
    parser.add_argument(
        "--root-selected-region-summary-path",
        type=Path,
        default=None,
        help="Optional root_selected_region_summary.csv to join by case_id.",
    )
    parser.add_argument("--reference-tau-s", type=float, default=DEFAULT_REFERENCE_TAU_S)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--effective-support-floor",
        type=float,
        default=DEFAULT_EFFECTIVE_SUPPORT_FLOOR,
    )
    parser.add_argument(
        "--root-outgoing-balance-floor",
        type=float,
        default=DEFAULT_ROOT_OUTGOING_BALANCE_FLOOR,
    )
    parser.add_argument(
        "--nonroot-balance-product-floor",
        type=float,
        default=DEFAULT_NONROOT_BALANCE_PRODUCT_FLOOR,
    )
    parser.add_argument(
        "--near-frontier-width",
        type=float,
        default=DEFAULT_NEAR_FRONTIER_WIDTH,
    )
    parser.add_argument(
        "--allow-spectral-unmeasured",
        action="store_true",
        help="Treat missing/floor-only spectral flow as neutral diagnostic evidence.",
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, SelectedNeighborhoodTopologyFrontierConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
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


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if not numeric.empty else math.nan


def _bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _bool_sum(values: pd.Series) -> int:
    if values.empty:
        return 0
    return int(values.map(_bool_value).sum())


def _string_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _fraction(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return math.nan
    return float(numerator) / float(denominator)


def _candidate_scope(row: pd.Series) -> str:
    if _bool_value(row.get("direct_sibling_measurable", False)):
        return "direct_measurable"
    parent_id = _string_value(row, "parent_id")
    bottleneck = _string_value(row, "measurability_bottleneck")
    topology_status = _string_value(row, "topology_coherence_status")
    if (
        not parent_id
        or bottleneck == "root_selected_topology_requires_root_law"
        or topology_status == "root_selected_topology_requires_root_law"
    ):
        return "root_non_direct"
    return "nonroot_non_direct"


def _root_structural_proxy_status(
    *,
    scope: str,
    outgoing_balance: float,
    effective_support_pass: bool,
    root_outgoing_balance_floor: float,
) -> tuple[bool, str]:
    if scope != "root_non_direct":
        return False, "not_root_candidate"
    if not effective_support_pass:
        return False, "effective_support_below_floor"
    if not math.isfinite(outgoing_balance):
        return False, "root_outgoing_balance_missing"
    if outgoing_balance < float(root_outgoing_balance_floor):
        return False, "root_outgoing_balance_below_floor"
    return True, "root_structural_proxy_passes_diagnostic_only"


def _root_margin_lookup(
    root_selected_region_summary: pd.DataFrame | None,
) -> dict[str, dict[str, object]]:
    if root_selected_region_summary is None or root_selected_region_summary.empty:
        return {}
    if "case_id" not in root_selected_region_summary.columns:
        raise ValueError("root selected-region summary missing required case_id column.")
    records: dict[str, dict[str, object]] = {}
    for _, row in root_selected_region_summary.iterrows():
        case_id = str(row["case_id"])
        records[case_id] = dict(row)
    return records


def _root_margin_status(
    *,
    scope: str,
    case_id: str,
    margin_by_case: dict[str, dict[str, object]],
) -> tuple[str, str, float, int, int, float]:
    if scope != "root_non_direct":
        return "not_root_candidate", "not_root_candidate", math.nan, 0, 0, math.nan
    margin = margin_by_case.get(str(case_id))
    if not margin:
        return (
            "root_selected_region_margin_missing",
            "root_margin_not_joined",
            math.nan,
            0,
            0,
            math.nan,
        )
    law_status = str(margin.get("root_selected_region_law_status", "root_margin_status_missing"))
    return (
        law_status,
        "root_margin_joined_case_family_diagnostic_only",
        _finite_float(margin.get("root_child_min_merge_margin", math.nan)),
        int(_finite_float(margin.get("root_child_tied_minimum_merge_count", 0))),
        int(_finite_float(margin.get("root_child_discrete_tie_cell_count", 0))),
        _finite_float(margin.get("root_sibling_selected_ratio", math.nan)),
    )


def _nonroot_frontier_status(
    *,
    scope: str,
    balance_product: float,
    nonroot_balance_product_floor: float,
    near_frontier_width: float,
) -> tuple[bool, str]:
    if scope != "nonroot_non_direct":
        return False, "not_nonroot_candidate"
    if not math.isfinite(balance_product):
        return False, "nonroot_balance_product_missing"
    margin = balance_product - float(nonroot_balance_product_floor)
    if margin >= 0.0:
        return False, "nonroot_current_floor_passes"
    if margin >= -float(near_frontier_width):
        return True, "nonroot_near_current_floor_diagnostic_only"
    return False, "nonroot_far_below_current_floor"


def _spectral_strict_pass(row: pd.Series, *, require_spectral_flow: bool) -> bool:
    if not bool(require_spectral_flow):
        return True
    return (
        _string_value(row, "spectral_bottleneck_status") == "spectral_flow_observed_diagnostic_only"
    )


def _hybrid_strict_status(
    *,
    scope: str,
    current_action: str,
    effective_support_pass: bool,
    root_law_status: str,
    nonroot_balance_product: float,
    nonroot_floor: float,
    spectral_pass: bool,
    require_spectral_flow: bool,
) -> tuple[bool, str]:
    if scope == "direct_measurable":
        if current_action == "split":
            return False, "current_direct_split"
        return False, "current_direct_fail_closed"
    if not effective_support_pass:
        return False, "effective_support_below_floor"
    if scope == "root_non_direct":
        return False, root_law_status
    if not math.isfinite(nonroot_balance_product):
        return False, "nonroot_balance_product_missing"
    if nonroot_balance_product < float(nonroot_floor):
        return False, "nonroot_topology_below_current_floor"
    if bool(require_spectral_flow) and not spectral_pass:
        return False, "spectral_transport_unmeasured_or_blocked"
    return True, "hybrid_support_observed_diagnostic_only"


def build_topology_frontier_rows(
    measurability_rows: pd.DataFrame,
    *,
    root_selected_region_summary: pd.DataFrame | None = None,
    reference_tau_s: float = DEFAULT_REFERENCE_TAU_S,
    alpha: float = DEFAULT_ALPHA,
    effective_support_floor: float = DEFAULT_EFFECTIVE_SUPPORT_FLOOR,
    root_outgoing_balance_floor: float = DEFAULT_ROOT_OUTGOING_BALANCE_FLOOR,
    nonroot_balance_product_floor: float = DEFAULT_NONROOT_BALANCE_PRODUCT_FLOOR,
    near_frontier_width: float = DEFAULT_NEAR_FRONTIER_WIDTH,
    require_spectral_flow: bool = DEFAULT_REQUIRE_SPECTRAL_FLOW,
) -> pd.DataFrame:
    """Annotate candidate rows with topology-frontier support statuses."""
    required = {
        "case_id",
        "data_role",
        "method_id",
        "node_id",
        "direct_sibling_measurable",
        "measurability_action",
        "measurability_bottleneck",
        "interpolation_effective_support",
        "interpolation_best_case_required_tau_s_for_alpha",
        "structural_outgoing_balance",
        "topology_balance_product_value",
        "spectral_bottleneck_status",
    }
    missing = required - set(measurability_rows.columns)
    if missing:
        raise ValueError(f"measurability rows missing required columns: {sorted(missing)!r}.")

    records: list[dict[str, object]] = []
    root_margin_by_case = _root_margin_lookup(root_selected_region_summary)
    for _, row in measurability_rows.iterrows():
        case_id = _string_value(row, "case_id")
        scope = _candidate_scope(row)
        current_action = _string_value(row, "measurability_action")
        direct_p_value = _finite_float(row.get("direct_sibling_p_value", math.nan))
        direct_significant = _bool_value(row.get("direct_sibling_open", False)) or (
            math.isfinite(direct_p_value) and direct_p_value <= float(alpha)
        )
        pair_prior = _finite_float(row.get("interpolated_pair_null_prior", math.nan))
        effective_support = _finite_float(row.get("interpolation_effective_support", math.nan))
        effective_support_pass = math.isfinite(effective_support) and effective_support >= float(
            effective_support_floor
        )
        best_case_tau = _finite_float(
            row.get("interpolation_best_case_required_tau_s_for_alpha", math.nan)
        )
        bandwidth_reference_reopens = math.isfinite(best_case_tau) and best_case_tau <= float(
            reference_tau_s
        )
        bandwidth_reference_direct_positive_reopens = (
            bandwidth_reference_reopens and direct_significant
        )
        root_outgoing = _finite_float(row.get("structural_outgoing_balance", math.nan))
        root_proxy_pass, root_proxy_status = _root_structural_proxy_status(
            scope=scope,
            outgoing_balance=root_outgoing,
            effective_support_pass=effective_support_pass,
            root_outgoing_balance_floor=float(root_outgoing_balance_floor),
        )
        (
            root_law_status,
            root_margin_evidence_status,
            root_child_min_merge_margin,
            root_child_tied_count,
            root_child_discrete_tie_count,
            root_sibling_selected_ratio,
        ) = _root_margin_status(
            scope=scope,
            case_id=case_id,
            margin_by_case=root_margin_by_case,
        )
        nonroot_balance = _finite_float(row.get("topology_balance_product_value", math.nan))
        near_frontier, nonroot_status = _nonroot_frontier_status(
            scope=scope,
            balance_product=nonroot_balance,
            nonroot_balance_product_floor=float(nonroot_balance_product_floor),
            near_frontier_width=float(near_frontier_width),
        )
        nonroot_margin = (
            nonroot_balance - float(nonroot_balance_product_floor)
            if math.isfinite(nonroot_balance)
            else math.nan
        )
        spectral_pass = _spectral_strict_pass(
            row,
            require_spectral_flow=bool(require_spectral_flow),
        )
        hybrid_support, hybrid_status = _hybrid_strict_status(
            scope=scope,
            current_action=current_action,
            effective_support_pass=effective_support_pass,
            root_law_status=root_law_status,
            nonroot_balance_product=nonroot_balance,
            nonroot_floor=float(nonroot_balance_product_floor),
            spectral_pass=spectral_pass,
            require_spectral_flow=bool(require_spectral_flow),
        )

        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": case_id,
                "data_role": _string_value(row, "data_role"),
                "method_id": _string_value(row, "method_id"),
                "replicate": row.get("replicate", math.nan),
                "node_id": _string_value(row, "node_id"),
                "parent_id": _string_value(row, "parent_id"),
                "depth": _finite_float(row.get("depth", math.nan)),
                "candidate_scope": scope,
                "current_measurability_action": current_action,
                "current_measurability_bottleneck": _string_value(
                    row,
                    "measurability_bottleneck",
                ),
                "direct_sibling_measurable": _bool_value(
                    row.get("direct_sibling_measurable", False)
                ),
                "direct_sibling_open": _bool_value(row.get("direct_sibling_open", False)),
                "direct_sibling_p_value": direct_p_value,
                "direct_sibling_significant": direct_significant,
                "interpolated_pair_null_prior": pair_prior,
                "interpolation_default_significant": (
                    math.isfinite(pair_prior) and pair_prior <= float(alpha)
                ),
                "interpolation_effective_support": effective_support,
                "interpolation_effective_support_pass": effective_support_pass,
                "interpolation_best_case_required_tau_s_for_alpha": best_case_tau,
                "bandwidth_reference_tau_s": float(reference_tau_s),
                "bandwidth_reference_reopens": bandwidth_reference_reopens,
                "bandwidth_reference_direct_positive_reopens": (
                    bandwidth_reference_direct_positive_reopens
                ),
                "root_outgoing_balance": root_outgoing,
                "root_structural_proxy_pass": root_proxy_pass,
                "root_structural_proxy_status": root_proxy_status,
                "root_selected_law_status": root_law_status,
                "root_margin_evidence_status": root_margin_evidence_status,
                "root_child_min_merge_margin": root_child_min_merge_margin,
                "root_child_tied_minimum_merge_count": root_child_tied_count,
                "root_child_discrete_tie_cell_count": root_child_discrete_tie_count,
                "root_sibling_selected_ratio": root_sibling_selected_ratio,
                "nonroot_balance_product": nonroot_balance,
                "nonroot_current_floor_margin": nonroot_margin,
                "nonroot_near_frontier": near_frontier,
                "nonroot_frontier_status": nonroot_status,
                "spectral_bottleneck_status": _string_value(
                    row,
                    "spectral_bottleneck_status",
                ),
                "spectral_strict_pass": spectral_pass,
                "hybrid_strict_support": hybrid_support,
                "hybrid_strict_status": hybrid_status,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_topology_frontier_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize frontier statuses by data role and method."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    records: list[dict[str, object]] = []
    for (data_role, method_id), group in rows.groupby(
        ["data_role", "method_id"],
        dropna=False,
        sort=True,
    ):
        non_direct = group[~group["direct_sibling_measurable"].map(_bool_value)]
        root = non_direct[non_direct["candidate_scope"].eq("root_non_direct")]
        nonroot = non_direct[non_direct["candidate_scope"].eq("nonroot_non_direct")]
        frontier_status = "diagnostic_only_not_promoted"
        if _bool_sum(non_direct["hybrid_strict_support"]) > 0:
            frontier_status = "strict_hybrid_support_observed_diagnostic_only"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "data_role": str(data_role),
                "method_id": str(method_id),
                "row_count": int(len(group)),
                "current_split_count": int(
                    group["current_measurability_action"].astype(str).eq("split").sum()
                ),
                "direct_measurable_count": _bool_sum(group["direct_sibling_measurable"]),
                "non_direct_count": int(len(non_direct)),
                "bandwidth_reference_reopen_count": _bool_sum(group["bandwidth_reference_reopens"]),
                "bandwidth_reference_direct_positive_reopen_count": _bool_sum(
                    group["bandwidth_reference_direct_positive_reopens"]
                ),
                "bandwidth_reference_reopen_non_direct_count": _bool_sum(
                    non_direct["bandwidth_reference_reopens"]
                ),
                "root_non_direct_count": int(len(root)),
                "root_structural_proxy_pass_count": _bool_sum(root["root_structural_proxy_pass"]),
                "root_selected_law_missing_count": int(
                    root["root_selected_law_status"]
                    .astype(str)
                    .eq("root_selected_region_margin_missing")
                    .sum()
                ),
                "root_discrete_tie_cell_count": int(
                    root["root_selected_law_status"]
                    .astype(str)
                    .eq("discrete_tie_cell_geometry_required")
                    .sum()
                ),
                "nonroot_non_direct_count": int(len(nonroot)),
                "nonroot_near_frontier_count": _bool_sum(nonroot["nonroot_near_frontier"]),
                "nonroot_current_floor_pass_count": int(
                    (
                        pd.to_numeric(nonroot["nonroot_current_floor_margin"], errors="coerce")
                        >= 0.0
                    ).sum()
                ),
                "spectral_strict_pass_non_direct_count": _bool_sum(
                    non_direct["spectral_strict_pass"]
                ),
                "hybrid_strict_support_count": _bool_sum(non_direct["hybrid_strict_support"]),
                "median_root_outgoing_balance": _finite_median(root["root_outgoing_balance"]),
                "median_nonroot_balance_product": _finite_median(
                    nonroot["nonroot_balance_product"]
                ),
                "median_effective_support": _finite_median(
                    group["interpolation_effective_support"]
                ),
                "median_best_case_tau_s_non_direct": _finite_median(
                    non_direct["interpolation_best_case_required_tau_s_for_alpha"]
                ),
                "frontier_status": frontier_status,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def build_threshold_sweep(
    rows: pd.DataFrame,
    *,
    root_threshold_grid: tuple[float, ...] = DEFAULT_ROOT_GRID,
    nonroot_threshold_grid: tuple[float, ...] = DEFAULT_NONROOT_GRID,
) -> pd.DataFrame:
    """Sweep structural root/non-root thresholds on non-direct rows."""
    records: list[dict[str, object]] = []
    non_direct = rows[~rows["direct_sibling_measurable"].map(_bool_value)].copy()
    specs = (
        (
            "root_non_direct",
            "root_outgoing_balance",
            "root_outgoing_balance_floor",
            root_threshold_grid,
        ),
        (
            "nonroot_non_direct",
            "nonroot_balance_product",
            "nonroot_balance_product_floor",
            nonroot_threshold_grid,
        ),
    )
    for scope, value_column, threshold_type, thresholds in specs:
        subset = non_direct[non_direct["candidate_scope"].eq(scope)].copy()
        for (data_role, method_id), group in subset.groupby(
            ["data_role", "method_id"],
            dropna=False,
            sort=True,
        ):
            values = pd.to_numeric(group[value_column], errors="coerce")
            for threshold in thresholds:
                passed = values >= float(threshold)
                passed_group = group[passed.fillna(False)]
                pass_count = int(passed.fillna(False).sum())
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "data_role": str(data_role),
                        "method_id": str(method_id),
                        "candidate_scope": scope,
                        "threshold_type": threshold_type,
                        "threshold": float(threshold),
                        "row_count": int(len(group)),
                        "pass_count": pass_count,
                        "pass_fraction": _fraction(pass_count, len(group)),
                        "bandwidth_reference_reopen_count": _bool_sum(
                            passed_group["bandwidth_reference_reopens"]
                        ),
                        "bandwidth_reference_direct_positive_reopen_count": _bool_sum(
                            passed_group["bandwidth_reference_direct_positive_reopens"]
                        ),
                        "hybrid_strict_support_count": _bool_sum(
                            passed_group["hybrid_strict_support"]
                        ),
                        "median_best_case_tau_s_for_pass": _finite_median(
                            passed_group["interpolation_best_case_required_tau_s_for_alpha"]
                        ),
                        "sweep_status": ("evaluated" if len(group) > 0 else "no_candidate_rows"),
                    }
                )
    return pd.DataFrame.from_records(records, columns=SWEEP_COLUMNS)


def evaluate_selected_neighborhood_topology_frontier(
    config: SelectedNeighborhoodTopologyFrontierConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read inputs and return row, summary, and threshold-sweep tables."""
    measurability_rows = pd.read_csv(config.measurability_rows_path, low_memory=False)
    root_selected_region_summary = (
        pd.read_csv(config.root_selected_region_summary_path, low_memory=False)
        if config.root_selected_region_summary_path is not None
        else None
    )
    rows = build_topology_frontier_rows(
        measurability_rows,
        root_selected_region_summary=root_selected_region_summary,
        reference_tau_s=float(config.reference_tau_s),
        alpha=float(config.alpha),
        effective_support_floor=float(config.effective_support_floor),
        root_outgoing_balance_floor=float(config.root_outgoing_balance_floor),
        nonroot_balance_product_floor=float(config.nonroot_balance_product_floor),
        near_frontier_width=float(config.near_frontier_width),
        require_spectral_flow=bool(config.require_spectral_flow),
    )
    summary = summarize_topology_frontier_rows(rows)
    sweep = build_threshold_sweep(
        rows,
        root_threshold_grid=tuple(config.root_threshold_grid),
        nonroot_threshold_grid=tuple(config.nonroot_threshold_grid),
    )
    return rows, summary, sweep


def run_selected_neighborhood_topology_frontier(
    config: SelectedNeighborhoodTopologyFrontierConfig,
) -> dict[str, Path]:
    """Write selected-neighborhood topology frontier outputs."""
    rows, summary, sweep = evaluate_selected_neighborhood_topology_frontier(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out = output_dir / ROWS_OUTPUT
    summary_out = output_dir / SUMMARY_OUTPUT
    sweep_out = output_dir / SWEEP_OUTPUT
    manifest_out = output_dir / MANIFEST_OUTPUT
    rows.to_csv(rows_out, index=False)
    summary.to_csv(summary_out, index=False)
    sweep.to_csv(sweep_out, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "outputs": {
            "rows": rows_out,
            "summary": summary_out,
            "threshold_sweep": sweep_out,
        },
        "row_count": int(len(rows)),
        "summary_row_count": int(len(summary)),
        "threshold_sweep_row_count": int(len(sweep)),
    }
    manifest_out.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": rows_out,
        "summary": summary_out,
        "threshold_sweep": sweep_out,
        "manifest": manifest_out,
    }


def main() -> None:
    args = parse_args()
    run_selected_neighborhood_topology_frontier(
        SelectedNeighborhoodTopologyFrontierConfig(
            output_dir=args.output_dir,
            measurability_rows_path=args.measurability_rows_path,
            root_selected_region_summary_path=args.root_selected_region_summary_path,
            reference_tau_s=float(args.reference_tau_s),
            alpha=float(args.alpha),
            effective_support_floor=float(args.effective_support_floor),
            root_outgoing_balance_floor=float(args.root_outgoing_balance_floor),
            nonroot_balance_product_floor=float(args.nonroot_balance_product_floor),
            near_frontier_width=float(args.near_frontier_width),
            require_spectral_flow=not bool(args.allow_spectral_unmeasured),
        )
    )


if __name__ == "__main__":
    main()
