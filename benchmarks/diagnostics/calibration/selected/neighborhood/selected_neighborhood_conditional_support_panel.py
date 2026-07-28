"""Selected-neighborhood conditional support diagnostics.

This panel joins the selected-neighborhood measurability rows with selected-root
validity/tail evidence. It asks whether a candidate row has admissible local
support after the selected-root context is known. The output is diagnostic-only:
it does not change traversal, gates, or production p-values.
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

SCHEMA_VERSION = "selected_neighborhood_conditional_support_panel/v1"
STUDY_ROLE = "diagnostic_selected_neighborhood_conditional_support_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_conditional_support_panel"

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_MEASURABILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "selected_neighborhood_measurability_law_overlap_expanded_candidates"
    / "selected_neighborhood_measurability_law_rows.csv"
)
DEFAULT_ROOT_VALIDITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_validity_replay_overlap_seven_signal_v1"
    / "root_selected_validity_replay_rows.csv"
)
DEFAULT_ROOT_TAIL_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_spectral_tail_law_with_legacy_overlay"
    / "root_selected_spectral_tail_law_rows.csv"
)

ROWS_OUTPUT = "selected_neighborhood_conditional_support_rows.csv"
CASE_SUMMARY_OUTPUT = "selected_neighborhood_conditional_support_case_summary.csv"
SUMMARY_OUTPUT = "selected_neighborhood_conditional_support_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

DEFAULT_HARD_NEGATIVE_CASES = ("overlap_extreme_4c",)
DEFAULT_MIN_SUPPORT_COUNT = 2
DEFAULT_MIN_EFFECTIVE_SUPPORT = 2.0
DEFAULT_MAX_WEIGHT_SHARE = 0.75
DEFAULT_REQUIRE_SPECTRAL_FLOW = True

JOIN_COLUMNS = ("case_id", "data_role", "method_id", "replicate", "node_id")

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
    "hard_negative_control",
    "direct_sibling_measurable",
    "direct_sibling_open",
    "direct_sibling_p_value",
    "measurability_action",
    "measurability_bottleneck",
    "root_validity_status",
    "root_tail_inference_status",
    "selected_root_usability_status",
    "root_context_status",
    "root_validity_pass",
    "root_tail_pass",
    "root_gate_allows_neighborhood",
    "local_support_count",
    "local_signal_anchor_count",
    "local_selected_nonnull_excluded_count",
    "local_effective_support",
    "local_support_weight",
    "local_max_weight_share",
    "local_max_weight_share_lower_bound",
    "local_max_weight_share_upper_bound",
    "local_max_weight_share_status",
    "local_support_pass",
    "nearest_support_distance",
    "nearest_signal_distance",
    "tau_t",
    "tau_s",
    "h_k",
    "interpolated_pair_null_prior",
    "interpolation_best_case_required_tau_s_for_alpha",
    "interpolation_behavior_label",
    "topology_coherent",
    "topology_coherence_status",
    "topology_balance_product_value",
    "structural_outgoing_balance",
    "spectral_bottleneck_status",
    "spectral_flow_pass",
    "conditional_support_pass",
    "promotion_eligible",
    "selected_null_leak",
    "hard_negative_leak",
    "dominant_blocker",
    "method_action",
    "mathematical_interpretation",
)

CASE_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "row_count",
    "hard_negative_control",
    "direct_split_count",
    "root_candidate_count",
    "nonroot_candidate_count",
    "root_validity_status",
    "root_tail_inference_status",
    "selected_root_usability_status",
    "root_gate_allows_neighborhood",
    "local_support_pass_count",
    "topology_pass_count",
    "spectral_flow_pass_count",
    "conditional_support_pass_count",
    "promotion_eligible_count",
    "selected_null_leak_count",
    "hard_negative_leak_count",
    "median_effective_support",
    "median_max_weight_share_upper_bound",
    "median_nearest_support_distance",
    "dominant_blocker",
    "case_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "case_count",
    "hard_negative_case_count",
    "hard_negative_leak_count",
    "selected_null_leak_count",
    "conditional_support_pass_count",
    "promotion_eligible_count",
    "root_validity_failed_case_count",
    "root_validity_unmeasured_case_count",
    "spectral_blocked_row_count",
    "topology_blocked_row_count",
    "support_blocked_row_count",
    "dominant_blocker",
    "summary_status",
)


@dataclass(frozen=True)
class SelectedNeighborhoodConditionalSupportConfig:
    """Input paths and thresholds for the conditional support panel."""

    output_dir: Path
    measurability_rows_path: Path = DEFAULT_MEASURABILITY_ROWS
    root_validity_rows_path: Path | None = DEFAULT_ROOT_VALIDITY_ROWS
    root_tail_rows_path: Path | None = DEFAULT_ROOT_TAIL_ROWS
    min_support_count: int = DEFAULT_MIN_SUPPORT_COUNT
    min_effective_support: float = DEFAULT_MIN_EFFECTIVE_SUPPORT
    max_weight_share: float = DEFAULT_MAX_WEIGHT_SHARE
    require_spectral_flow: bool = DEFAULT_REQUIRE_SPECTRAL_FLOW
    hard_negative_cases: tuple[str, ...] = DEFAULT_HARD_NEGATIVE_CASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--measurability-rows-path",
        type=Path,
        default=DEFAULT_MEASURABILITY_ROWS,
    )
    parser.add_argument(
        "--root-validity-rows-path",
        type=Path,
        default=DEFAULT_ROOT_VALIDITY_ROWS,
    )
    parser.add_argument(
        "--root-tail-rows-path",
        type=Path,
        default=DEFAULT_ROOT_TAIL_ROWS,
    )
    parser.add_argument(
        "--min-support-count",
        type=int,
        default=DEFAULT_MIN_SUPPORT_COUNT,
    )
    parser.add_argument(
        "--min-effective-support",
        type=float,
        default=DEFAULT_MIN_EFFECTIVE_SUPPORT,
    )
    parser.add_argument(
        "--max-weight-share",
        type=float,
        default=DEFAULT_MAX_WEIGHT_SHARE,
    )
    parser.add_argument(
        "--allow-spectral-unmeasured",
        action="store_true",
        help="Treat missing/floor-only spectral evidence as neutral diagnostic evidence.",
    )
    parser.add_argument(
        "--hard-negative-case",
        action="append",
        default=None,
        help=(
            "Case id that must not be rescued. May be supplied multiple times; "
            "defaults to overlap_extreme_4c."
        ),
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, SelectedNeighborhoodConditionalSupportConfig):
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


def _bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _string_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if not numeric.empty else math.nan


def _bool_sum(values: pd.Series) -> int:
    if values.empty:
        return 0
    return int(values.map(_bool_value).sum())


def _candidate_scope(row: pd.Series) -> str:
    parent_id = _string_value(row, "parent_id")
    bottleneck = _string_value(row, "measurability_bottleneck")
    topology_status = _string_value(row, "topology_coherence_status")
    if (
        parent_id == ""
        or bottleneck == "root_selected_topology_requires_root_law"
        or topology_status == "root_selected_topology_requires_root_law"
    ):
        return "root_non_direct"
    if _bool_value(row.get("direct_sibling_measurable", False)):
        return "direct_measurable"
    return "nonroot_non_direct"


def _root_validity_lookup(root_validity_rows: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if root_validity_rows is None or root_validity_rows.empty:
        return {}
    if "target_case_id" not in root_validity_rows.columns:
        raise ValueError("root validity rows must contain target_case_id.")
    records: dict[str, dict[str, object]] = {}
    for _, row in root_validity_rows.iterrows():
        records[str(row["target_case_id"])] = row.to_dict()
    return records


def _root_tail_lookup(root_tail_rows: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if root_tail_rows is None or root_tail_rows.empty:
        return {}
    if "target_case_id" not in root_tail_rows.columns:
        raise ValueError("root tail rows must contain target_case_id.")
    records: dict[str, dict[str, object]] = {}
    for _, row in root_tail_rows.iterrows():
        records[str(row["target_case_id"])] = row.to_dict()
    return records


def _root_context(
    *,
    row: pd.Series,
    validity_by_case: dict[str, dict[str, object]],
    tail_by_case: dict[str, dict[str, object]],
) -> tuple[str, str, str, str, bool, bool, bool]:
    case_id = _string_value(row, "case_id")
    data_role = _string_value(row, "data_role")
    validity_record = validity_by_case.get(case_id, {})
    tail_record = tail_by_case.get(case_id, {})
    validity_status = str(validity_record.get("root_validity_status", "root_validity_unmeasured"))
    validity_tail_status = str(validity_record.get("root_tail_inference_status", ""))
    tail_status = str(tail_record.get("root_tail_inference_status", ""))
    root_tail_status = validity_tail_status or tail_status or "root_tail_unmeasured"
    usability_status = str(
        validity_record.get(
            "selected_root_usability_status",
            "selected_root_usability_unmeasured",
        )
    )
    validity_pass = validity_status.startswith("root_validity_supported")
    tail_pass = root_tail_status == "calibrated_selected_root_spectral_tail_available"
    if data_role == "selected_null":
        return (
            validity_status,
            root_tail_status,
            usability_status,
            "root_context_not_applicable_for_selected_null_leak_diagnostic",
            False,
            tail_pass,
            True,
        )
    if not validity_record:
        return (
            validity_status,
            root_tail_status,
            usability_status,
            "root_validity_unmeasured",
            False,
            tail_pass,
            False,
        )
    if not validity_pass:
        return (
            validity_status,
            root_tail_status,
            usability_status,
            "root_validity_failed",
            False,
            tail_pass,
            False,
        )
    if not tail_pass:
        return (
            validity_status,
            root_tail_status,
            usability_status,
            "root_validity_supported_tail_missing",
            True,
            False,
            True,
        )
    return (
        validity_status,
        root_tail_status,
        usability_status,
        "root_validity_and_tail_supported",
        True,
        True,
        True,
    )


def _support_weight_share_fields(row: pd.Series) -> tuple[float, float, float, str]:
    effective_support = _finite_float(row.get("interpolation_effective_support", math.nan))
    observed = _finite_float(row.get("interpolation_max_weight_share", math.nan))
    if math.isfinite(observed):
        lower = observed
        upper = observed
        return observed, lower, upper, "max_weight_share_observed"
    if not math.isfinite(effective_support) or effective_support <= 0.0:
        return math.nan, math.nan, math.nan, "max_weight_share_unmeasured"
    lower = 1.0 / effective_support
    upper = 1.0 / math.sqrt(effective_support)
    return (
        math.nan,
        float(lower),
        float(upper),
        "max_weight_share_bounded_from_effective_support",
    )


def _local_support_pass(
    *,
    support_count: int,
    effective_support: float,
    max_weight_share: float,
    max_weight_share_upper_bound: float,
    min_support_count: int,
    min_effective_support: float,
    max_weight_share_threshold: float,
) -> tuple[bool, str]:
    if support_count < int(min_support_count):
        return False, "support_count_below_floor"
    if not math.isfinite(effective_support):
        return False, "effective_support_missing"
    if effective_support < float(min_effective_support):
        return False, "effective_support_below_floor"
    if math.isfinite(max_weight_share):
        if max_weight_share > float(max_weight_share_threshold):
            return False, "max_weight_share_above_floor"
        return True, "support_pass_with_observed_weight_share"
    if math.isfinite(max_weight_share_upper_bound):
        if max_weight_share_upper_bound > float(max_weight_share_threshold):
            return True, "support_pass_weight_share_bound_wide_diagnostic_only"
        return True, "support_pass_weight_share_bound_below_floor"
    return True, "support_pass_weight_share_unmeasured_diagnostic_only"


def _spectral_flow_pass(row: pd.Series, *, require_spectral_flow: bool) -> bool:
    if not bool(require_spectral_flow):
        return True
    return (
        _string_value(row, "spectral_bottleneck_status") == "spectral_flow_observed_diagnostic_only"
    )


def _classify_row(
    *,
    row: pd.Series,
    candidate_scope: str,
    hard_negative_control: bool,
    root_context_status: str,
    root_gate_allows_neighborhood: bool,
    root_tail_pass: bool,
    local_support_pass: bool,
    support_blocker: str,
    topology_pass: bool,
    spectral_pass: bool,
) -> tuple[bool, bool, bool, bool, str, str, str]:
    data_role = _string_value(row, "data_role")
    direct_measurable = _bool_value(row.get("direct_sibling_measurable", False))
    direct_open = _bool_value(row.get("direct_sibling_open", False))
    if hard_negative_control and not root_gate_allows_neighborhood:
        return (
            False,
            False,
            False,
            False,
            "root_validity_failed_hard_negative_control",
            "fail_closed_hard_negative_root_invalid",
            "The hard-negative case remains blocked because root replay rejects the selected root.",
        )
    if data_role == "signal" and not root_gate_allows_neighborhood:
        return (
            False,
            False,
            False,
            False,
            root_context_status,
            "fail_closed_root_context_blocks_neighborhood",
            "Signal neighborhood evidence cannot speak until root validity is supported.",
        )
    if candidate_scope == "root_non_direct":
        return (
            False,
            False,
            False,
            False,
            "root_candidate_requires_root_law",
            "fail_closed_root_neighborhood_rescue_disallowed",
            "Root rows require the selected-root law; neighborhood smoothing cannot rescue them.",
        )
    if direct_measurable:
        if direct_open:
            return (
                False,
                False,
                False,
                False,
                "direct_sibling_split_measured",
                "direct_split_already_measured_not_neighborhood_rescue",
                "The direct sibling test already opens this row; the neighborhood layer is not the deciding evidence.",
            )
        return (
            False,
            False,
            False,
            False,
            "direct_sibling_measurable_not_significant",
            "fail_closed_direct_sibling_test_closed",
            "Direct measurable sibling evidence closes the row.",
        )
    if not local_support_pass:
        return (
            False,
            False,
            False,
            False,
            support_blocker,
            "fail_closed_local_support_missing",
            "The selected neighborhood lacks admissible local support.",
        )
    if not topology_pass:
        return (
            False,
            False,
            False,
            False,
            _string_value(row, "topology_coherence_status", "topology_not_coherent"),
            "fail_closed_topology_not_coherent",
            "The local incoming/outgoing topology is not coherent enough for support borrowing.",
        )
    if not spectral_pass:
        return (
            False,
            False,
            False,
            False,
            _string_value(row, "spectral_bottleneck_status", "spectral_flow_missing"),
            "fail_closed_spectral_flow_missing",
            "The MP-supported spectral-flow check is missing or blocks the row.",
        )
    conditional_support = True
    promotion_eligible = bool(data_role == "signal" and root_tail_pass)
    selected_null_leak = bool(data_role == "selected_null")
    hard_negative_leak = bool(hard_negative_control)
    if selected_null_leak:
        action = "selected_null_conditional_support_leak_diagnostic"
        interpretation = (
            "This selected-null row would receive local conditional support; "
            "it is a leakage warning, not a promoted split."
        )
    elif promotion_eligible:
        action = "conditional_neighborhood_support_with_valid_root_tail_diagnostic"
        interpretation = (
            "Local support is observed under a valid root and calibrated root tail; "
            "this is a diagnostic candidate for future promotion."
        )
    else:
        action = "conditional_neighborhood_support_root_tail_missing_diagnostic"
        interpretation = (
            "Local support is observed, but promotion remains blocked until "
            "selected-root tail support exists."
        )
    return (
        conditional_support,
        promotion_eligible,
        selected_null_leak,
        hard_negative_leak,
        "conditional_support_observed",
        action,
        interpretation,
    )


def build_conditional_support_rows(
    measurability_rows: pd.DataFrame,
    *,
    root_validity_rows: pd.DataFrame | None = None,
    root_tail_rows: pd.DataFrame | None = None,
    min_support_count: int = DEFAULT_MIN_SUPPORT_COUNT,
    min_effective_support: float = DEFAULT_MIN_EFFECTIVE_SUPPORT,
    max_weight_share: float = DEFAULT_MAX_WEIGHT_SHARE,
    require_spectral_flow: bool = DEFAULT_REQUIRE_SPECTRAL_FLOW,
    hard_negative_cases: tuple[str, ...] = DEFAULT_HARD_NEGATIVE_CASES,
) -> pd.DataFrame:
    """Join root context and classify selected-neighborhood support rows."""
    required = {
        "case_id",
        "data_role",
        "method_id",
        "replicate",
        "node_id",
        "parent_id",
        "direct_sibling_measurable",
        "direct_sibling_open",
        "measurability_action",
        "measurability_bottleneck",
        "interpolation_support_count",
        "interpolation_signal_count",
        "interpolation_selected_nonnull_excluded_count",
        "interpolation_effective_support",
        "topology_coherent",
        "topology_coherence_status",
        "spectral_bottleneck_status",
    }
    missing = required - set(measurability_rows.columns)
    if missing:
        raise ValueError(f"measurability rows missing required columns: {sorted(missing)!r}.")

    validity_by_case = _root_validity_lookup(root_validity_rows)
    tail_by_case = _root_tail_lookup(root_tail_rows)
    hard_negative_set = set(hard_negative_cases)
    records: list[dict[str, object]] = []

    for _, row in measurability_rows.iterrows():
        case_id = _string_value(row, "case_id")
        candidate_scope = _candidate_scope(row)
        hard_negative_control = (
            case_id in hard_negative_set and _string_value(row, "data_role") == "signal"
        )
        (
            root_validity_status,
            root_tail_status,
            selected_root_usability_status,
            root_context_status,
            root_validity_pass,
            root_tail_pass,
            root_gate_allows_neighborhood,
        ) = _root_context(
            row=row,
            validity_by_case=validity_by_case,
            tail_by_case=tail_by_case,
        )
        (
            local_max_weight_share,
            local_max_weight_share_lower_bound,
            local_max_weight_share_upper_bound,
            local_max_weight_share_status,
        ) = _support_weight_share_fields(row)
        support_count = _finite_int(row.get("interpolation_support_count", 0))
        signal_anchor_count = _finite_int(row.get("interpolation_signal_count", 0))
        selected_nonnull_excluded_count = _finite_int(
            row.get("interpolation_selected_nonnull_excluded_count", 0)
        )
        effective_support = _finite_float(row.get("interpolation_effective_support", math.nan))
        support_pass, support_status = _local_support_pass(
            support_count=support_count,
            effective_support=effective_support,
            max_weight_share=local_max_weight_share,
            max_weight_share_upper_bound=local_max_weight_share_upper_bound,
            min_support_count=int(min_support_count),
            min_effective_support=float(min_effective_support),
            max_weight_share_threshold=float(max_weight_share),
        )
        topology_pass = _bool_value(row.get("topology_coherent", False))
        spectral_pass = _spectral_flow_pass(
            row,
            require_spectral_flow=bool(require_spectral_flow),
        )
        (
            conditional_support_pass,
            promotion_eligible,
            selected_null_leak,
            hard_negative_leak,
            dominant_blocker,
            method_action,
            mathematical_interpretation,
        ) = _classify_row(
            row=row,
            candidate_scope=candidate_scope,
            hard_negative_control=hard_negative_control,
            root_context_status=root_context_status,
            root_gate_allows_neighborhood=root_gate_allows_neighborhood,
            root_tail_pass=root_tail_pass,
            local_support_pass=support_pass,
            support_blocker=support_status,
            topology_pass=topology_pass,
            spectral_pass=spectral_pass,
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
                "candidate_scope": candidate_scope,
                "hard_negative_control": hard_negative_control,
                "direct_sibling_measurable": _bool_value(
                    row.get("direct_sibling_measurable", False)
                ),
                "direct_sibling_open": _bool_value(row.get("direct_sibling_open", False)),
                "direct_sibling_p_value": _finite_float(
                    row.get("direct_sibling_p_value", math.nan)
                ),
                "measurability_action": _string_value(row, "measurability_action"),
                "measurability_bottleneck": _string_value(
                    row,
                    "measurability_bottleneck",
                ),
                "root_validity_status": root_validity_status,
                "root_tail_inference_status": root_tail_status,
                "selected_root_usability_status": selected_root_usability_status,
                "root_context_status": root_context_status,
                "root_validity_pass": root_validity_pass,
                "root_tail_pass": root_tail_pass,
                "root_gate_allows_neighborhood": root_gate_allows_neighborhood,
                "local_support_count": support_count,
                "local_signal_anchor_count": signal_anchor_count,
                "local_selected_nonnull_excluded_count": selected_nonnull_excluded_count,
                "local_effective_support": effective_support,
                "local_support_weight": _finite_float(
                    row.get("interpolation_support_weight", math.nan)
                ),
                "local_max_weight_share": local_max_weight_share,
                "local_max_weight_share_lower_bound": (local_max_weight_share_lower_bound),
                "local_max_weight_share_upper_bound": (local_max_weight_share_upper_bound),
                "local_max_weight_share_status": local_max_weight_share_status,
                "local_support_pass": support_pass,
                "nearest_support_distance": _finite_float(
                    row.get("interpolation_nearest_support_distance", math.nan)
                ),
                "nearest_signal_distance": _finite_float(
                    row.get("interpolation_nearest_signal_distance", math.nan)
                ),
                "tau_t": _finite_float(row.get("interpolation_tau_t", math.nan)),
                "tau_s": _finite_float(row.get("interpolation_tau_s", math.nan)),
                "h_k": _finite_float(row.get("interpolation_h_k", math.nan)),
                "interpolated_pair_null_prior": _finite_float(
                    row.get("interpolated_pair_null_prior", math.nan)
                ),
                "interpolation_best_case_required_tau_s_for_alpha": _finite_float(
                    row.get(
                        "interpolation_best_case_required_tau_s_for_alpha",
                        math.nan,
                    )
                ),
                "interpolation_behavior_label": _string_value(
                    row,
                    "interpolation_behavior_label",
                ),
                "topology_coherent": topology_pass,
                "topology_coherence_status": _string_value(
                    row,
                    "topology_coherence_status",
                ),
                "topology_balance_product_value": _finite_float(
                    row.get("topology_balance_product_value", math.nan)
                ),
                "structural_outgoing_balance": _finite_float(
                    row.get("structural_outgoing_balance", math.nan)
                ),
                "spectral_bottleneck_status": _string_value(
                    row,
                    "spectral_bottleneck_status",
                ),
                "spectral_flow_pass": spectral_pass,
                "conditional_support_pass": conditional_support_pass,
                "promotion_eligible": promotion_eligible,
                "selected_null_leak": selected_null_leak,
                "hard_negative_leak": hard_negative_leak,
                "dominant_blocker": dominant_blocker,
                "method_action": method_action,
                "mathematical_interpretation": mathematical_interpretation,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _dominant(values: pd.Series) -> str:
    if values.empty:
        return ""
    counts = values.astype(str).value_counts(dropna=False)
    return str(counts.index[0]) if not counts.empty else ""


def summarize_conditional_support_cases(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize conditional support decisions by case, role, and method."""
    if rows.empty:
        return pd.DataFrame(columns=CASE_SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for (case_id, data_role, method_id), group in rows.groupby(
        ["case_id", "data_role", "method_id"],
        sort=True,
        dropna=False,
    ):
        hard_negative = _bool_sum(group["hard_negative_control"]) > 0
        selected_null_leaks = _bool_sum(group["selected_null_leak"])
        hard_negative_leaks = _bool_sum(group["hard_negative_leak"])
        conditional_support_count = _bool_sum(group["conditional_support_pass"])
        promotion_count = _bool_sum(group["promotion_eligible"])
        if hard_negative:
            case_status = (
                "hard_negative_control_leaked"
                if hard_negative_leaks
                else "hard_negative_control_supported"
            )
        elif selected_null_leaks:
            case_status = "selected_null_leak_observed_diagnostic"
        elif promotion_count:
            case_status = "promotion_candidate_observed_diagnostic_only"
        elif conditional_support_count:
            case_status = "conditional_support_observed_but_not_promotable"
        else:
            case_status = "conditional_support_fail_closed"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "data_role": str(data_role),
                "method_id": str(method_id),
                "row_count": int(len(group)),
                "hard_negative_control": hard_negative,
                "direct_split_count": int(
                    group["method_action"]
                    .astype(str)
                    .eq("direct_split_already_measured_not_neighborhood_rescue")
                    .sum()
                ),
                "root_candidate_count": int(
                    group["candidate_scope"].astype(str).eq("root_non_direct").sum()
                ),
                "nonroot_candidate_count": int(
                    group["candidate_scope"].astype(str).eq("nonroot_non_direct").sum()
                ),
                "root_validity_status": _dominant(group["root_validity_status"]),
                "root_tail_inference_status": _dominant(group["root_tail_inference_status"]),
                "selected_root_usability_status": _dominant(
                    group["selected_root_usability_status"]
                ),
                "root_gate_allows_neighborhood": _bool_sum(group["root_gate_allows_neighborhood"])
                > 0,
                "local_support_pass_count": _bool_sum(group["local_support_pass"]),
                "topology_pass_count": _bool_sum(group["topology_coherent"]),
                "spectral_flow_pass_count": _bool_sum(group["spectral_flow_pass"]),
                "conditional_support_pass_count": conditional_support_count,
                "promotion_eligible_count": promotion_count,
                "selected_null_leak_count": selected_null_leaks,
                "hard_negative_leak_count": hard_negative_leaks,
                "median_effective_support": _finite_median(group["local_effective_support"]),
                "median_max_weight_share_upper_bound": _finite_median(
                    group["local_max_weight_share_upper_bound"]
                ),
                "median_nearest_support_distance": _finite_median(
                    group["nearest_support_distance"]
                ),
                "dominant_blocker": _dominant(group["dominant_blocker"]),
                "case_status": case_status,
            }
        )
    return pd.DataFrame.from_records(records, columns=CASE_SUMMARY_COLUMNS)


def summarize_conditional_support(rows: pd.DataFrame) -> pd.DataFrame:
    """Build one overall summary row."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    cases = rows[["case_id", "data_role", "method_id"]].drop_duplicates()
    hard_negative_cases = rows.loc[
        rows["hard_negative_control"].map(_bool_value),
        ["case_id", "data_role", "method_id"],
    ].drop_duplicates()
    root_context_by_case = rows[
        ["case_id", "data_role", "method_id", "root_context_status"]
    ].drop_duplicates()
    root_failed_cases = root_context_by_case[
        root_context_by_case["root_context_status"].astype(str).eq("root_validity_failed")
    ]
    root_unmeasured_cases = root_context_by_case[
        root_context_by_case["root_context_status"].astype(str).eq("root_validity_unmeasured")
    ]
    hard_negative_leaks = _bool_sum(rows["hard_negative_leak"])
    selected_null_leaks = _bool_sum(rows["selected_null_leak"])
    if hard_negative_leaks:
        summary_status = "hard_negative_control_failed"
    elif selected_null_leaks:
        summary_status = "selected_null_leak_observed_diagnostic"
    elif _bool_sum(rows["promotion_eligible"]):
        summary_status = "promotion_candidates_observed_diagnostic_only"
    else:
        summary_status = "conditional_support_fail_closed_or_not_promotable"
    record = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "row_count": int(len(rows)),
        "case_count": int(len(cases)),
        "hard_negative_case_count": int(len(hard_negative_cases)),
        "hard_negative_leak_count": hard_negative_leaks,
        "selected_null_leak_count": selected_null_leaks,
        "conditional_support_pass_count": _bool_sum(rows["conditional_support_pass"]),
        "promotion_eligible_count": _bool_sum(rows["promotion_eligible"]),
        "root_validity_failed_case_count": int(len(root_failed_cases)),
        "root_validity_unmeasured_case_count": int(len(root_unmeasured_cases)),
        "spectral_blocked_row_count": int(
            rows["dominant_blocker"].astype(str).str.contains("spectral").sum()
        ),
        "topology_blocked_row_count": int(
            rows["dominant_blocker"].astype(str).str.contains("topology").sum()
        ),
        "support_blocked_row_count": int(
            rows["dominant_blocker"].astype(str).str.contains("support").sum()
        ),
        "dominant_blocker": _dominant(rows["dominant_blocker"]),
        "summary_status": summary_status,
    }
    return pd.DataFrame.from_records([record], columns=SUMMARY_COLUMNS)


def evaluate_selected_neighborhood_conditional_support(
    config: SelectedNeighborhoodConditionalSupportConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read inputs and return row, case-summary, and overall-summary tables."""
    measurability_rows = pd.read_csv(config.measurability_rows_path, low_memory=False)
    root_validity_rows = (
        pd.read_csv(config.root_validity_rows_path, low_memory=False)
        if config.root_validity_rows_path is not None
        and Path(config.root_validity_rows_path).exists()
        else None
    )
    root_tail_rows = (
        pd.read_csv(config.root_tail_rows_path, low_memory=False)
        if config.root_tail_rows_path is not None and Path(config.root_tail_rows_path).exists()
        else None
    )
    rows = build_conditional_support_rows(
        measurability_rows,
        root_validity_rows=root_validity_rows,
        root_tail_rows=root_tail_rows,
        min_support_count=int(config.min_support_count),
        min_effective_support=float(config.min_effective_support),
        max_weight_share=float(config.max_weight_share),
        require_spectral_flow=bool(config.require_spectral_flow),
        hard_negative_cases=tuple(config.hard_negative_cases),
    )
    case_summary = summarize_conditional_support_cases(rows)
    summary = summarize_conditional_support(rows)
    return rows, case_summary, summary


def run_selected_neighborhood_conditional_support(
    config: SelectedNeighborhoodConditionalSupportConfig,
) -> dict[str, Path]:
    """Write selected-neighborhood conditional support outputs."""
    rows, case_summary, summary = evaluate_selected_neighborhood_conditional_support(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out = output_dir / ROWS_OUTPUT
    case_summary_out = output_dir / CASE_SUMMARY_OUTPUT
    summary_out = output_dir / SUMMARY_OUTPUT
    manifest_out = output_dir / MANIFEST_OUTPUT
    rows.to_csv(rows_out, index=False)
    case_summary.to_csv(case_summary_out, index=False)
    summary.to_csv(summary_out, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "outputs": {
            "rows": rows_out,
            "case_summary": case_summary_out,
            "summary": summary_out,
        },
        "row_count": int(len(rows)),
        "case_summary_row_count": int(len(case_summary)),
        "summary_row_count": int(len(summary)),
    }
    manifest_out.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": rows_out,
        "case_summary": case_summary_out,
        "summary": summary_out,
        "manifest": manifest_out,
    }


def main() -> None:
    args = parse_args()
    hard_negative_cases = (
        tuple(args.hard_negative_case) if args.hard_negative_case else DEFAULT_HARD_NEGATIVE_CASES
    )
    run_selected_neighborhood_conditional_support(
        SelectedNeighborhoodConditionalSupportConfig(
            output_dir=args.output_dir,
            measurability_rows_path=args.measurability_rows_path,
            root_validity_rows_path=args.root_validity_rows_path,
            root_tail_rows_path=args.root_tail_rows_path,
            min_support_count=int(args.min_support_count),
            min_effective_support=float(args.min_effective_support),
            max_weight_share=float(args.max_weight_share),
            require_spectral_flow=not bool(args.allow_spectral_unmeasured),
            hard_negative_cases=hard_negative_cases,
        )
    )


if __name__ == "__main__":
    main()
