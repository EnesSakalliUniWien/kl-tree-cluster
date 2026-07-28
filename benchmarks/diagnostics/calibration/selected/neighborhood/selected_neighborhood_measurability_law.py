"""Selected-neighborhood measurability-law diagnostics.

This module implements the formal law documented in the wiki:

1. use a direct sibling p-value first when it is measurable under the selected
   family contract;
2. use bandwidth-interpolated null evidence only when direct measurement is not
   valid and the local support contract is satisfied;
3. otherwise fail closed with an explicit bottleneck label.

The output is diagnostic-only. It does not change production traversal.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "selected_neighborhood_measurability_law/v1"
STUDY_ROLE = "diagnostic_selected_neighborhood_measurability_law_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_measurability_law"

DEFAULT_DIRECT_ALPHA = 0.01
DEFAULT_INTERPOLATED_ALPHA = 0.01
DEFAULT_MIN_INTERPOLATION_SUPPORT = 2
DEFAULT_BALANCE_PRODUCT_FLOOR = 0.22
DEFAULT_OUTGOING_EDGE_NORM_FLOOR = 0.95
PROBABILITY_TOLERANCE = 1e-10

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
    "traversal_decision",
    "decision_class",
    "traversal_state",
    "traversal_stop_reason",
    "n_descendant_leaves",
    "structural_n_parent_context",
    "structural_n_node",
    "structural_n_incoming_sibling",
    "structural_n_left",
    "structural_n_right",
    "structural_incoming_branch_balance",
    "structural_outgoing_balance",
    "structural_balance_product",
    "structural_topology_feature_count",
    "structural_topology_context_status",
    "sibling_projection_dimension",
    "topology_incidence_role",
    "topology_pass_through_candidate",
    "guard_truth_role",
    "topology_support_role",
    "topology_signal_role",
    "neighborhood_evidence_family",
    "direct_sibling_measurable",
    "direct_sibling_p_value",
    "direct_sibling_open",
    "direct_contract_status",
    "interpolation_measurable",
    "interpolated_left_null_prior",
    "interpolated_right_null_prior",
    "interpolated_pair_null_prior",
    "interpolation_support_count",
    "interpolation_signal_count",
    "interpolation_selected_nonnull_excluded_count",
    "interpolation_support_status",
    "interpolation_effective_support",
    "interpolation_support_weight",
    "interpolation_stable_weighted_p_mean",
    "interpolation_signal_attenuation",
    "interpolation_nearest_support_distance",
    "interpolation_nearest_signal_distance",
    "interpolation_tau_t",
    "interpolation_tau_s",
    "interpolation_h_k",
    "interpolation_best_case_required_tau_s_for_alpha",
    "interpolation_comparison_class",
    "interpolation_behavior_label",
    "topology_neighborhood_tau_b",
    "topology_neighborhood_tau_t",
    "topology_neighborhood_tau_s",
    "topology_neighborhood_h_k",
    "topology_neighborhood_nearest_stable_distance",
    "topology_neighborhood_nearest_signal_distance",
    "incoming_branch_balance",
    "outgoing_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "balance_product",
    "topology_balance_product_value",
    "topology_balance_product_source",
    "outgoing_balance_edge_product",
    "topology_coherent",
    "topology_coherence_status",
    "spectral_parent_id",
    "spectral_mp_common_dimension",
    "spectral_mp_pair_supported",
    "spectral_mp_subspace_chordal_distance",
    "spectral_mp_log_eigenvalue_delta",
    "spectral_barrier",
    "spectral_flow_affinity",
    "spectral_flow_status",
    "spectral_bottleneck_status",
    "measurability_action",
    "measurability_bottleneck",
    "measurability_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "data_role",
    "method_id",
    "measurability_action",
    "measurability_bottleneck",
    "row_count",
    "direct_measurable_count",
    "interpolation_measurable_count",
    "topology_coherent_count",
)

JOIN_COLUMNS = ("case_id", "data_role", "method_id", "replicate", "node_id")


@dataclass(frozen=True)
class ChildNullPriorResult:
    """Diagnostic child-level interpolated null prior."""

    prior: float
    support_weight: float
    signal_attenuation: float
    status: str


@dataclass(frozen=True)
class MeasurabilityDecision:
    """Selected-neighborhood action and bottleneck for one candidate row."""

    action: str
    bottleneck: str
    status: str
    direct_measurable: bool
    interpolation_measurable: bool
    topology_coherent: bool
    pair_prior: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate selected-neighborhood rows with measurability-law actions."
    )
    parser.add_argument(
        "--rows",
        type=Path,
        required=True,
        help="Selected-neighborhood distribution rows CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for measurability rows, summary, and manifest.",
    )
    parser.add_argument(
        "--direct-alpha",
        type=float,
        default=DEFAULT_DIRECT_ALPHA,
        help="Direct sibling-test alpha used when sibling_open is unavailable.",
    )
    parser.add_argument(
        "--interpolated-alpha",
        type=float,
        default=DEFAULT_INTERPOLATED_ALPHA,
        help="Diagnostic threshold for supported interpolated pair priors.",
    )
    parser.add_argument(
        "--min-interpolation-support",
        type=int,
        default=DEFAULT_MIN_INTERPOLATION_SUPPORT,
        help="Minimum strict support count for interpolation measurability.",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Restrict output to rows with direct/interpolated/guard traversal evidence.",
    )
    parser.add_argument(
        "--pvalue-comparison-rows",
        type=Path,
        default=None,
        help=(
            "Optional selected_neighborhood_pvalue_interpolation_rows.csv to join "
            "hold-out interpolation diagnostics into the candidate audit table."
        ),
    )
    parser.add_argument(
        "--spectral-flow-edges",
        type=Path,
        default=None,
        help=(
            "Optional selected_neighborhood_spectral_flow_edges.csv to join "
            "parent-child MP eigenspace bottleneck diagnostics."
        ),
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
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


def _probability_or_nan(value: object, *, tolerance: float = PROBABILITY_TOLERANCE) -> float:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    if numeric < -tolerance or numeric > 1.0 + tolerance:
        return math.nan
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _finite_int(value: object, default: int = 0) -> int:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return int(default)
    return int(numeric)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return math.nan
    return float(numerator) / float(denominator)


def _first_finite(row: pd.Series, columns: tuple[str, ...], default: float = math.nan) -> float:
    for column in columns:
        if column in row:
            value = _finite_float(row[column])
            if math.isfinite(value):
                return value
    return float(default)


def _first_string(row: pd.Series, columns: tuple[str, ...], default: str = "") -> str:
    for column in columns:
        value = _string_value(row, column, default="")
        if value:
            return value
    return default


def _first_probability(row: pd.Series, columns: tuple[str, ...]) -> float:
    for column in columns:
        if column in row:
            value = _probability_or_nan(row[column])
            if math.isfinite(value):
                return value
    return math.nan


def validate_probability(value: object, *, label: str) -> float:
    """Return a probability or raise for values outside [0, 1]."""
    numeric = _finite_float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{label} must be a finite probability in [0, 1].")
    return numeric


def compute_child_interpolated_null_prior(
    *,
    ancestor_p_values: list[float] | tuple[float, ...] = (),
    ancestor_weights: list[float] | tuple[float, ...] = (),
    stable_p_values: list[float] | tuple[float, ...] = (),
    stable_weights: list[float] | tuple[float, ...] = (),
    signal_p_values: list[float] | tuple[float, ...] = (),
    signal_distances: list[float] | tuple[float, ...] = (),
    tau_s: float = 1.0,
) -> ChildNullPriorResult:
    """Compute the unclipped diagnostic child null prior.

    The result is invalid rather than clipped if the mathematical assumptions
    fail. Tiny floating-point drift is not expected because the formula is a
    convex average multiplied by a unit attenuation.
    """
    ancestor_p = [
        validate_probability(value, label="ancestor_p_value") for value in ancestor_p_values
    ]
    stable_p = [validate_probability(value, label="stable_p_value") for value in stable_p_values]
    signal_p = [validate_probability(value, label="signal_p_value") for value in signal_p_values]
    ancestor_w = [float(value) for value in ancestor_weights]
    stable_w = [float(value) for value in stable_weights]
    signal_d = [float(value) for value in signal_distances]

    if len(ancestor_p) != len(ancestor_w):
        raise ValueError("ancestor_p_values and ancestor_weights must have equal length.")
    if len(stable_p) != len(stable_w):
        raise ValueError("stable_p_values and stable_weights must have equal length.")
    if len(signal_p) != len(signal_d):
        raise ValueError("signal_p_values and signal_distances must have equal length.")
    if any((not math.isfinite(weight)) or weight < 0.0 for weight in (*ancestor_w, *stable_w)):
        raise ValueError("Interpolation weights must be finite and nonnegative.")
    tau_s_value = float(tau_s)
    if not math.isfinite(tau_s_value) or tau_s_value <= 0.0:
        raise ValueError("tau_s must be finite and positive.")

    support_weight = float(sum(ancestor_w) + sum(stable_w))
    if support_weight <= 0.0:
        return ChildNullPriorResult(
            prior=math.nan,
            support_weight=support_weight,
            signal_attenuation=math.nan,
            status="support_bottleneck",
        )

    numerator = float(
        sum(weight * value for weight, value in zip(ancestor_w, ancestor_p))
        + sum(weight * value for weight, value in zip(stable_w, stable_p))
    )
    interpolated = numerator / support_weight
    attenuation = 0.0
    if signal_p:
        attenuation = max(
            (1.0 - p_value) * math.exp(-distance / tau_s_value)
            for p_value, distance in zip(signal_p, signal_d)
        )
    prior = interpolated * (1.0 - attenuation)

    if prior < -PROBABILITY_TOLERANCE or prior > 1.0 + PROBABILITY_TOLERANCE:
        return ChildNullPriorResult(
            prior=math.nan,
            support_weight=support_weight,
            signal_attenuation=attenuation,
            status="invalid_probability_domain",
        )
    return ChildNullPriorResult(
        prior=float(min(max(prior, 0.0), 1.0)),
        support_weight=support_weight,
        signal_attenuation=float(attenuation),
        status="interpolated_prior_observed_diagnostic_only",
    )


def _guard_blocked(row: pd.Series) -> bool:
    return any(
        _bool_value(row[column]) if column in row else False
        for column in (
            "explicit_guard_blocked",
            "root_stability_guard_blocked",
            "root_selective_guard_blocked",
            "selected_family_guard_blocked",
        )
    )


def _direct_contract_status(row: pd.Series) -> tuple[bool, str, float, bool]:
    p_value = _probability_or_nan(row.get("sibling_p_value", math.nan))
    sibling_open = _bool_value(row.get("sibling_open", False))
    if _guard_blocked(row):
        return False, "selected_family_or_root_guard_blocked", p_value, sibling_open
    if not math.isfinite(p_value):
        return False, "direct_sibling_p_value_unavailable", p_value, sibling_open
    return True, "direct_sibling_p_value_measurable", p_value, sibling_open


def _pair_prior_from_row(row: pd.Series) -> tuple[float, str]:
    explicit = _first_probability(
        row,
        (
            "interpolated_pair_null_prior",
            "interpolated_sibling_null_p_like",
        ),
    )
    if math.isfinite(explicit):
        return explicit, "pair_prior_observed"
    left = _probability_or_nan(row.get("interpolated_left_null_prior", math.nan))
    right = _probability_or_nan(row.get("interpolated_right_null_prior", math.nan))
    if math.isfinite(left) and math.isfinite(right):
        return min(left, right), "pair_prior_from_child_minimum"

    for column in (
        "interpolated_pair_null_prior",
        "interpolated_sibling_null_p_like",
        "interpolated_left_null_prior",
        "interpolated_right_null_prior",
    ):
        raw = _finite_float(row.get(column, math.nan))
        if math.isfinite(raw) and (raw < 0.0 or raw > 1.0):
            return math.nan, "invalid_interpolated_prior_probability_domain"
    return math.nan, "interpolated_prior_unavailable"


def _interpolation_contract_status(
    row: pd.Series,
    *,
    min_support: int,
) -> tuple[bool, str, float, int, int, int]:
    pair_prior, prior_status = _pair_prior_from_row(row)
    topology_support_count = _finite_int(row.get("topology_neighborhood_support_count", 0.0))
    support_anchor_count = _finite_int(row.get("support_anchor_count", 0.0))
    support_count = topology_support_count if topology_support_count > 0 else support_anchor_count
    topology_signal_count = _finite_int(row.get("topology_neighborhood_signal_count", 0.0))
    signal_anchor_count = _finite_int(row.get("signal_anchor_count", 0.0))
    signal_count = topology_signal_count if topology_signal_count > 0 else signal_anchor_count
    topology_excluded_count = _finite_int(
        row.get("topology_neighborhood_selected_nonnull_excluded_count", 0.0)
    )
    anchor_excluded_count = _finite_int(row.get("selected_nonnull_excluded_count", 0.0))
    excluded_count = (
        topology_excluded_count if topology_excluded_count > 0 else anchor_excluded_count
    )
    support_status = _first_string(
        row,
        ("topology_neighborhood_support_status", "interpolation_status"),
    )

    if prior_status == "invalid_interpolated_prior_probability_domain":
        return False, prior_status, pair_prior, support_count, signal_count, excluded_count
    if not math.isfinite(pair_prior):
        return False, prior_status, pair_prior, support_count, signal_count, excluded_count
    if support_count <= 0 and excluded_count > 0:
        return (
            False,
            "selection_bottleneck_selected_nonnull_only",
            pair_prior,
            support_count,
            signal_count,
            excluded_count,
        )
    if support_count < int(min_support):
        return (
            False,
            "support_bottleneck",
            pair_prior,
            support_count,
            signal_count,
            excluded_count,
        )
    if support_status and support_status.endswith("_fail_closed"):
        return (
            False,
            support_status,
            pair_prior,
            support_count,
            signal_count,
            excluded_count,
        )
    tau_b = _finite_float(row.get("topology_neighborhood_tau_b", math.nan))
    if math.isfinite(tau_b) and tau_b <= 0.0:
        return (
            False,
            "bandwidth_bottleneck",
            pair_prior,
            support_count,
            signal_count,
            excluded_count,
        )
    for columns in (
        ("topology_neighborhood_tau_t", "tau_t"),
        ("topology_neighborhood_tau_s", "tau_s"),
        ("topology_neighborhood_h_k", "h_k"),
    ):
        value = _first_finite(row, columns)
        if not math.isfinite(value) or value <= 0.0:
            return (
                False,
                "bandwidth_bottleneck",
                pair_prior,
                support_count,
                signal_count,
                excluded_count,
            )
    return (
        True,
        "interpolation_support_observed_diagnostic_only",
        pair_prior,
        support_count,
        signal_count,
        excluded_count,
    )


def _topology_coherence_status(
    row: pd.Series,
    *,
    balance_product_floor: float,
    outgoing_edge_norm_floor: float,
) -> tuple[bool, str]:
    balance_product, _source = _topology_balance_product(row)
    outgoing_edge = _finite_float(row.get("outgoing_edge_norm_balance", math.nan))

    if not math.isfinite(balance_product):
        structural_incoming = _finite_float(row.get("structural_incoming_branch_balance", math.nan))
        structural_outgoing = _finite_float(row.get("structural_outgoing_balance", math.nan))
        if math.isfinite(structural_outgoing) and not math.isfinite(structural_incoming):
            if not _string_value(row, "parent_id"):
                return False, "root_selected_topology_requires_root_law"
            return False, "structural_incoming_topology_missing"
        if math.isfinite(structural_incoming) and not math.isfinite(structural_outgoing):
            return False, "structural_outgoing_topology_missing"
        return False, "topology_balance_product_missing"
    if balance_product < float(balance_product_floor):
        return False, "topology_balance_product_below_floor"
    if not math.isfinite(outgoing_edge):
        return False, "topology_outgoing_edge_norm_missing"
    if outgoing_edge < float(outgoing_edge_norm_floor):
        return False, "topology_outgoing_edge_norm_below_floor"
    return True, "topology_coherent"


def _topology_balance_product(row: pd.Series) -> tuple[float, str]:
    balance_product = _finite_float(row.get("balance_product", math.nan))
    if math.isfinite(balance_product):
        return balance_product, "observed_balance_product"
    incoming = _finite_float(row.get("incoming_branch_balance", math.nan))
    outgoing = _finite_float(row.get("outgoing_balance", math.nan))
    if math.isfinite(incoming) and math.isfinite(outgoing):
        return float(incoming * outgoing), "observed_incoming_outgoing"
    structural_balance_product = _finite_float(row.get("structural_balance_product", math.nan))
    if math.isfinite(structural_balance_product):
        return structural_balance_product, "structural_selected_tree_fallback"
    return math.nan, "unavailable"


def _spectral_bottleneck_status(row: pd.Series) -> str:
    flow_status = _string_value(row, "spectral_flow_status")
    if not flow_status:
        flow_status = _string_value(row, "flow_status")
    if not flow_status:
        return "spectral_not_joined"

    common_dimension = _first_finite(
        row,
        ("spectral_mp_common_dimension", "mp_common_dimension"),
    )
    pair_supported = (
        _bool_value(row["spectral_mp_pair_supported"])
        if "spectral_mp_pair_supported" in row
        else _bool_value(row.get("mp_pair_supported", False))
    )
    chordal = _first_finite(
        row,
        ("spectral_mp_subspace_chordal_distance", "mp_subspace_chordal_distance"),
    )
    eigen_delta = _first_finite(
        row,
        ("spectral_mp_log_eigenvalue_delta", "mp_log_eigenvalue_delta"),
    )

    if math.isfinite(common_dimension) and common_dimension <= 0:
        return "spectral_floor_only"
    if not pair_supported:
        return "spectral_mp_pair_unsupported"
    if math.isfinite(chordal) and chordal >= 0.75:
        return "spectral_subspace_rotation_bottleneck"
    if math.isfinite(eigen_delta) and eigen_delta >= 1.0:
        return "spectral_eigenvalue_drift_bottleneck"
    return "spectral_flow_observed_diagnostic_only"


def evaluate_measurability_decision(
    row: pd.Series,
    *,
    direct_alpha: float = DEFAULT_DIRECT_ALPHA,
    interpolated_alpha: float = DEFAULT_INTERPOLATED_ALPHA,
    min_interpolation_support: int = DEFAULT_MIN_INTERPOLATION_SUPPORT,
    balance_product_floor: float = DEFAULT_BALANCE_PRODUCT_FLOOR,
    outgoing_edge_norm_floor: float = DEFAULT_OUTGOING_EDGE_NORM_FLOOR,
) -> MeasurabilityDecision:
    """Evaluate the selected-neighborhood measurability law for one row."""
    direct_measurable, direct_status, p_value, sibling_open = _direct_contract_status(row)
    interpolation_measurable, interpolation_status, pair_prior, *_ = _interpolation_contract_status(
        row, min_support=int(min_interpolation_support)
    )
    topology_coherent, topology_status = _topology_coherence_status(
        row,
        balance_product_floor=float(balance_product_floor),
        outgoing_edge_norm_floor=float(outgoing_edge_norm_floor),
    )

    if direct_measurable:
        direct_open = sibling_open or (math.isfinite(p_value) and p_value <= float(direct_alpha))
        if direct_open:
            return MeasurabilityDecision(
                action="split",
                bottleneck="none",
                status="direct_sibling_test_split",
                direct_measurable=True,
                interpolation_measurable=interpolation_measurable,
                topology_coherent=topology_coherent,
                pair_prior=pair_prior,
            )
        return MeasurabilityDecision(
            action="fail_closed",
            bottleneck="direct_measurable_not_significant",
            status="direct_sibling_test_fail_closed",
            direct_measurable=True,
            interpolation_measurable=interpolation_measurable,
            topology_coherent=topology_coherent,
            pair_prior=pair_prior,
        )

    if interpolation_status == "invalid_interpolated_prior_probability_domain":
        bottleneck = "invalid_interpolated_prior_probability_domain"
    elif not interpolation_measurable:
        bottleneck = interpolation_status
    elif not topology_coherent:
        bottleneck = topology_status
    elif math.isfinite(pair_prior) and pair_prior <= float(interpolated_alpha):
        return MeasurabilityDecision(
            action="diagnostic_rescue",
            bottleneck="none",
            status="supported_interpolated_topology_rescue_diagnostic_only",
            direct_measurable=False,
            interpolation_measurable=True,
            topology_coherent=True,
            pair_prior=pair_prior,
        )
    else:
        bottleneck = "interpolated_null_not_strong"

    return MeasurabilityDecision(
        action="fail_closed",
        bottleneck=bottleneck,
        status="measurability_fail_closed",
        direct_measurable=False,
        interpolation_measurable=interpolation_measurable,
        topology_coherent=topology_coherent,
        pair_prior=pair_prior,
    )


def build_measurability_law_rows(
    rows: pd.DataFrame,
    *,
    direct_alpha: float = DEFAULT_DIRECT_ALPHA,
    interpolated_alpha: float = DEFAULT_INTERPOLATED_ALPHA,
    min_interpolation_support: int = DEFAULT_MIN_INTERPOLATION_SUPPORT,
    balance_product_floor: float = DEFAULT_BALANCE_PRODUCT_FLOOR,
    outgoing_edge_norm_floor: float = DEFAULT_OUTGOING_EDGE_NORM_FLOOR,
) -> pd.DataFrame:
    """Return row-level selected-neighborhood measurability diagnostics."""
    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        direct_measurable, direct_status, p_value, sibling_open = _direct_contract_status(row)
        (
            interpolation_measurable,
            interpolation_status,
            pair_prior,
            support_count,
            signal_count,
            excluded_count,
        ) = _interpolation_contract_status(
            row,
            min_support=int(min_interpolation_support),
        )
        topology_coherent, topology_status = _topology_coherence_status(
            row,
            balance_product_floor=float(balance_product_floor),
            outgoing_edge_norm_floor=float(outgoing_edge_norm_floor),
        )
        decision = evaluate_measurability_decision(
            row,
            direct_alpha=float(direct_alpha),
            interpolated_alpha=float(interpolated_alpha),
            min_interpolation_support=int(min_interpolation_support),
            balance_product_floor=float(balance_product_floor),
            outgoing_edge_norm_floor=float(outgoing_edge_norm_floor),
        )
        topology_balance_product, topology_balance_source = _topology_balance_product(row)
        record = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": _string_value(row, "case_id"),
            "data_role": _string_value(row, "data_role"),
            "method_id": _string_value(row, "method_id"),
            "replicate": row.get("replicate", math.nan),
            "node_id": _string_value(row, "node_id"),
            "parent_id": _string_value(row, "parent_id"),
            "depth": _finite_float(row.get("depth", math.nan)),
            "traversal_decision": _string_value(row, "traversal_decision"),
            "decision_class": _string_value(row, "decision_class"),
            "traversal_state": _string_value(row, "traversal_state"),
            "traversal_stop_reason": _string_value(row, "traversal_stop_reason"),
            "n_descendant_leaves": _finite_float(row.get("n_descendant_leaves", math.nan)),
            "structural_n_parent_context": _finite_float(
                row.get("structural_n_parent_context", math.nan)
            ),
            "structural_n_node": _finite_float(row.get("structural_n_node", math.nan)),
            "structural_n_incoming_sibling": _finite_float(
                row.get("structural_n_incoming_sibling", math.nan)
            ),
            "structural_n_left": _finite_float(row.get("structural_n_left", math.nan)),
            "structural_n_right": _finite_float(row.get("structural_n_right", math.nan)),
            "structural_incoming_branch_balance": _finite_float(
                row.get("structural_incoming_branch_balance", math.nan)
            ),
            "structural_outgoing_balance": _finite_float(
                row.get("structural_outgoing_balance", math.nan)
            ),
            "structural_balance_product": _finite_float(
                row.get("structural_balance_product", math.nan)
            ),
            "structural_topology_feature_count": _finite_int(
                row.get("structural_topology_feature_count", 0)
            ),
            "structural_topology_context_status": _string_value(
                row,
                "structural_topology_context_status",
            ),
            "sibling_projection_dimension": _finite_float(
                row.get("sibling_projection_dimension", math.nan)
            ),
            "topology_incidence_role": _string_value(row, "topology_incidence_role"),
            "topology_pass_through_candidate": _bool_value(
                row.get("topology_pass_through_candidate", False)
            ),
            "guard_truth_role": _string_value(row, "guard_truth_role"),
            "topology_support_role": _string_value(row, "topology_support_role"),
            "topology_signal_role": _string_value(row, "topology_signal_role"),
            "neighborhood_evidence_family": _string_value(
                row,
                "neighborhood_evidence_family",
            ),
            "direct_sibling_measurable": direct_measurable,
            "direct_sibling_p_value": p_value,
            "direct_sibling_open": sibling_open,
            "direct_contract_status": direct_status,
            "interpolation_measurable": interpolation_measurable,
            "interpolated_left_null_prior": _probability_or_nan(
                row.get("interpolated_left_null_prior", math.nan)
            ),
            "interpolated_right_null_prior": _probability_or_nan(
                row.get("interpolated_right_null_prior", math.nan)
            ),
            "interpolated_pair_null_prior": pair_prior,
            "interpolation_support_count": support_count,
            "interpolation_signal_count": signal_count,
            "interpolation_selected_nonnull_excluded_count": excluded_count,
            "interpolation_support_status": interpolation_status,
            "interpolation_effective_support": _first_finite(
                row,
                ("effective_support", "topology_neighborhood_support_count"),
                default=float(support_count),
            ),
            "interpolation_support_weight": _finite_float(row.get("support_weight", math.nan)),
            "interpolation_stable_weighted_p_mean": _finite_float(
                row.get("stable_weighted_p_mean", math.nan)
            ),
            "interpolation_signal_attenuation": _finite_float(
                row.get("signal_attenuation", math.nan)
            ),
            "interpolation_nearest_support_distance": _first_finite(
                row,
                (
                    "nearest_support_distance",
                    "topology_neighborhood_nearest_stable_distance",
                ),
            ),
            "interpolation_nearest_signal_distance": _first_finite(
                row,
                (
                    "nearest_signal_distance",
                    "topology_neighborhood_nearest_signal_distance",
                ),
            ),
            "interpolation_tau_t": _finite_float(row.get("tau_t", math.nan)),
            "interpolation_tau_s": _finite_float(row.get("tau_s", math.nan)),
            "interpolation_h_k": _finite_float(row.get("h_k", math.nan)),
            "interpolation_best_case_required_tau_s_for_alpha": _finite_float(
                row.get("best_case_required_tau_s_for_alpha", math.nan)
            ),
            "interpolation_comparison_class": _string_value(row, "comparison_class"),
            "interpolation_behavior_label": _string_value(row, "behavior_label"),
            "topology_neighborhood_tau_b": _finite_float(
                row.get("topology_neighborhood_tau_b", math.nan)
            ),
            "topology_neighborhood_tau_t": _first_finite(
                row,
                ("topology_neighborhood_tau_t", "tau_t"),
            ),
            "topology_neighborhood_tau_s": _first_finite(
                row,
                ("topology_neighborhood_tau_s", "tau_s"),
            ),
            "topology_neighborhood_h_k": _first_finite(
                row,
                ("topology_neighborhood_h_k", "h_k"),
            ),
            "topology_neighborhood_nearest_stable_distance": _finite_float(
                row.get("topology_neighborhood_nearest_stable_distance", math.nan)
            ),
            "topology_neighborhood_nearest_signal_distance": _finite_float(
                row.get("topology_neighborhood_nearest_signal_distance", math.nan)
            ),
            "incoming_branch_balance": _finite_float(row.get("incoming_branch_balance", math.nan)),
            "outgoing_balance": _finite_float(row.get("outgoing_balance", math.nan)),
            "outgoing_edge_norm_balance": _finite_float(
                row.get("outgoing_edge_norm_balance", math.nan)
            ),
            "outgoing_fragment_risk_proxy_score": _finite_float(
                row.get("outgoing_fragment_risk_proxy_score", math.nan)
            ),
            "balance_product": _finite_float(row.get("balance_product", math.nan)),
            "topology_balance_product_value": topology_balance_product,
            "topology_balance_product_source": topology_balance_source,
            "outgoing_balance_edge_product": _finite_float(
                row.get("outgoing_balance_edge_product", math.nan)
            ),
            "topology_coherent": topology_coherent,
            "topology_coherence_status": topology_status,
            "spectral_parent_id": _string_value(row, "spectral_parent_id"),
            "spectral_mp_common_dimension": _first_finite(
                row,
                ("spectral_mp_common_dimension", "mp_common_dimension"),
            ),
            "spectral_mp_pair_supported": _bool_value(row.get("spectral_mp_pair_supported", False)),
            "spectral_mp_subspace_chordal_distance": _first_finite(
                row,
                ("spectral_mp_subspace_chordal_distance", "mp_subspace_chordal_distance"),
            ),
            "spectral_mp_log_eigenvalue_delta": _first_finite(
                row,
                ("spectral_mp_log_eigenvalue_delta", "mp_log_eigenvalue_delta"),
            ),
            "spectral_barrier": _finite_float(row.get("spectral_barrier", math.nan)),
            "spectral_flow_affinity": _finite_float(row.get("spectral_flow_affinity", math.nan)),
            "spectral_flow_status": _first_string(
                row,
                ("spectral_flow_status", "flow_status"),
            ),
            "spectral_bottleneck_status": _spectral_bottleneck_status(row),
            "measurability_action": decision.action,
            "measurability_bottleneck": decision.bottleneck,
            "measurability_status": decision.status,
        }
        records.append(record)
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def candidate_evidence_mask(rows: pd.DataFrame) -> pd.Series:
    """Return rows that are meaningful candidates for measurability auditing."""
    mask = pd.Series(False, index=rows.index)
    if "sibling_p_value" in rows:
        mask |= pd.to_numeric(rows["sibling_p_value"], errors="coerce").notna()
    if "traversal_state" in rows:
        mask |= rows["traversal_state"].astype(str).isin({"split", "pass_through"})
    if "traversal_decision" in rows:
        mask |= rows["traversal_decision"].astype(str).isin({"split", "pass_through"})
    for column in (
        "explicit_guard_blocked",
        "root_stability_guard_blocked",
        "root_selective_guard_blocked",
        "selected_family_guard_blocked",
    ):
        if column in rows:
            mask |= rows[column].fillna(False).astype(bool)
    for column in (
        "topology_neighborhood_support_count",
        "topology_neighborhood_signal_count",
        "topology_neighborhood_selected_nonnull_excluded_count",
    ):
        if column in rows:
            mask |= pd.to_numeric(rows[column], errors="coerce").fillna(0.0).gt(0.0)
    for column in (
        "interpolated_pair_null_prior",
        "interpolated_left_null_prior",
        "interpolated_right_null_prior",
    ):
        if column in rows:
            mask |= pd.to_numeric(rows[column], errors="coerce").notna()
    return mask


def _deduplicated_join_frame(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    keep = [column for column in columns if column in frame.columns]
    join_columns = [column for column in JOIN_COLUMNS if column in keep]
    if len(join_columns) != len(JOIN_COLUMNS):
        return pd.DataFrame(columns=list(JOIN_COLUMNS))
    joined = frame.loc[:, keep].copy()
    return joined.drop_duplicates(subset=list(JOIN_COLUMNS), keep="first")


def _add_selected_tree_structural_topology(rows: pd.DataFrame) -> pd.DataFrame:
    """Add structural balance fields from parent links and descendant sizes."""
    if "node_id" not in rows or "parent_id" not in rows or "n_descendant_leaves" not in rows:
        enriched = rows.copy()
        for column in (
            "structural_n_parent_context",
            "structural_n_node",
            "structural_n_incoming_sibling",
            "structural_n_left",
            "structural_n_right",
            "structural_incoming_branch_balance",
            "structural_outgoing_balance",
            "structural_balance_product",
        ):
            enriched[column] = math.nan
        enriched["structural_topology_feature_count"] = 0
        enriched["structural_topology_context_status"] = "structural_topology_unavailable"
        return enriched

    enriched = rows.copy()
    records: dict[object, dict[str, object]] = {}
    group_columns = [
        column
        for column in ("case_id", "data_role", "method_id", "replicate")
        if column in enriched.columns
    ]
    grouped = (
        enriched.groupby(group_columns, dropna=False, sort=False)
        if group_columns
        else [((), enriched)]
    )

    for _keys, group in grouped:
        node_size = {
            str(row.node_id): _finite_float(row.n_descendant_leaves) for row in group.itertuples()
        }
        parent_by_node = {
            str(row.node_id): ("" if pd.isna(row.parent_id) else str(row.parent_id))
            for row in group.itertuples()
        }
        children_by_node: dict[str, list[str]] = {}
        for node_id, parent_id in parent_by_node.items():
            if parent_id:
                children_by_node.setdefault(parent_id, []).append(node_id)

        for index, row in group.iterrows():
            node_id = str(row["node_id"])
            n_node = node_size.get(node_id, math.nan)
            parent_id = parent_by_node.get(node_id, "")
            n_parent = node_size.get(parent_id, math.nan) if parent_id else math.nan
            incoming_siblings = [
                sibling for sibling in children_by_node.get(parent_id, []) if sibling != node_id
            ]
            n_incoming_sibling = (
                float(
                    sum(
                        node_size[sibling]
                        for sibling in incoming_siblings
                        if math.isfinite(node_size.get(sibling, math.nan))
                    )
                )
                if incoming_siblings
                else math.nan
            )
            incoming_balance = _safe_ratio(
                min(n_node, n_incoming_sibling),
                n_parent,
            )

            child_sizes = [
                node_size[child]
                for child in children_by_node.get(node_id, [])
                if math.isfinite(node_size.get(child, math.nan))
            ]
            if len(child_sizes) >= 2:
                child_sizes = sorted(child_sizes, reverse=True)
                n_left = float(child_sizes[0])
                n_right = float(child_sizes[1])
                outgoing_balance = _safe_ratio(min(n_left, n_right), n_node)
            else:
                n_left = math.nan
                n_right = math.nan
                outgoing_balance = math.nan

            structural_balance_product = (
                float(incoming_balance * outgoing_balance)
                if math.isfinite(incoming_balance) and math.isfinite(outgoing_balance)
                else math.nan
            )
            feature_count = (
                int(math.isfinite(incoming_balance))
                + int(math.isfinite(outgoing_balance))
                + int(math.isfinite(structural_balance_product))
            )
            if feature_count == 3:
                status = "structural_topology_observed"
            elif feature_count > 0:
                status = "structural_topology_partial"
            else:
                status = "structural_topology_unavailable"
            records[index] = {
                "structural_n_parent_context": n_parent,
                "structural_n_node": n_node,
                "structural_n_incoming_sibling": n_incoming_sibling,
                "structural_n_left": n_left,
                "structural_n_right": n_right,
                "structural_incoming_branch_balance": incoming_balance,
                "structural_outgoing_balance": outgoing_balance,
                "structural_balance_product": structural_balance_product,
                "structural_topology_feature_count": feature_count,
                "structural_topology_context_status": status,
            }

    structural = pd.DataFrame.from_dict(records, orient="index")
    for column in structural.columns:
        enriched[column] = structural[column].reindex(enriched.index)
    return enriched


def enrich_measurability_input_rows(
    rows: pd.DataFrame,
    *,
    pvalue_rows: pd.DataFrame | None = None,
    spectral_flow_edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join optional old-neighborhood and spectral bottleneck diagnostics.

    The joins are diagnostic only. Missing joins leave fields empty rather than
    changing a traversal decision.
    """
    enriched = _add_selected_tree_structural_topology(rows)

    if pvalue_rows is not None and not pvalue_rows.empty:
        pvalue_columns = (
            *JOIN_COLUMNS,
            "interpolated_sibling_null_p_like",
            "direct_significant",
            "interpolated_significant",
            "support_anchor_count",
            "signal_anchor_count",
            "selected_nonnull_excluded_count",
            "support_weight",
            "effective_support",
            "stable_weighted_p_mean",
            "signal_attenuation",
            "nearest_support_distance",
            "nearest_signal_distance",
            "tau_t",
            "tau_s",
            "h_k",
            "best_case_required_tau_s_for_alpha",
            "tree_distance_status",
            "support_anchor_rule",
            "signal_anchor_rule",
            "interpolation_status",
            "comparison_class",
            "behavior_label",
        )
        pvalue_join = _deduplicated_join_frame(
            pvalue_rows,
            columns=pvalue_columns,
        )
        if not pvalue_join.empty:
            enriched = enriched.merge(
                pvalue_join,
                on=list(JOIN_COLUMNS),
                how="left",
            )

    if spectral_flow_edges is not None and not spectral_flow_edges.empty:
        spectral_columns = (
            "case_id",
            "data_role",
            "method_id",
            "replicate",
            "parent_id",
            "child_id",
            "mp_common_dimension",
            "mp_pair_supported",
            "mp_subspace_chordal_distance",
            "mp_log_eigenvalue_delta",
            "spectral_barrier",
            "spectral_flow_affinity",
            "flow_status",
        )
        spectral_keep = [
            column for column in spectral_columns if column in spectral_flow_edges.columns
        ]
        if "child_id" in spectral_keep:
            spectral_join = spectral_flow_edges.loc[:, spectral_keep].copy()
            spectral_join = spectral_join.rename(
                columns={
                    "child_id": "node_id",
                    "parent_id": "spectral_parent_id",
                    "mp_common_dimension": "spectral_mp_common_dimension",
                    "mp_pair_supported": "spectral_mp_pair_supported",
                    "mp_subspace_chordal_distance": ("spectral_mp_subspace_chordal_distance"),
                    "mp_log_eigenvalue_delta": "spectral_mp_log_eigenvalue_delta",
                    "flow_status": "spectral_flow_status",
                }
            )
            spectral_join = spectral_join.drop_duplicates(
                subset=list(JOIN_COLUMNS),
                keep="first",
            )
            enriched = enriched.merge(
                spectral_join,
                on=list(JOIN_COLUMNS),
                how="left",
            )

    return enriched


def summarize_measurability_law_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-neighborhood measurability diagnostics."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    group_columns = [
        "data_role",
        "method_id",
        "measurability_action",
        "measurability_bottleneck",
    ]
    summary_rows = []
    for keys, group in rows.groupby(group_columns, dropna=False):
        summary_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "data_role": keys[0],
                "method_id": keys[1],
                "measurability_action": keys[2],
                "measurability_bottleneck": keys[3],
                "row_count": int(len(group)),
                "direct_measurable_count": int(group["direct_sibling_measurable"].sum()),
                "interpolation_measurable_count": int(group["interpolation_measurable"].sum()),
                "topology_coherent_count": int(group["topology_coherent"].sum()),
            }
        )
    return pd.DataFrame.from_records(summary_rows, columns=SUMMARY_COLUMNS)


def run_selected_neighborhood_measurability_law(
    *,
    rows_path: Path,
    output_dir: Path,
    direct_alpha: float = DEFAULT_DIRECT_ALPHA,
    interpolated_alpha: float = DEFAULT_INTERPOLATED_ALPHA,
    min_interpolation_support: int = DEFAULT_MIN_INTERPOLATION_SUPPORT,
    candidate_only: bool = False,
    pvalue_comparison_rows_path: Path | None = None,
    spectral_flow_edges_path: Path | None = None,
) -> dict[str, Path]:
    input_rows = pd.read_csv(rows_path)
    pvalue_rows = (
        pd.read_csv(pvalue_comparison_rows_path)
        if pvalue_comparison_rows_path is not None and pvalue_comparison_rows_path.exists()
        else None
    )
    spectral_flow_edges = (
        pd.read_csv(spectral_flow_edges_path)
        if spectral_flow_edges_path is not None and spectral_flow_edges_path.exists()
        else None
    )
    input_rows = enrich_measurability_input_rows(
        input_rows,
        pvalue_rows=pvalue_rows,
        spectral_flow_edges=spectral_flow_edges,
    )
    if candidate_only:
        input_rows = input_rows.loc[candidate_evidence_mask(input_rows)].copy()
    law_rows = build_measurability_law_rows(
        input_rows,
        direct_alpha=float(direct_alpha),
        interpolated_alpha=float(interpolated_alpha),
        min_interpolation_support=int(min_interpolation_support),
    )
    summary = summarize_measurability_law_rows(law_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out = output_dir / "selected_neighborhood_measurability_law_rows.csv"
    summary_out = output_dir / "selected_neighborhood_measurability_law_summary.csv"
    manifest_out = output_dir / "manifest.json"
    law_rows.to_csv(rows_out, index=False)
    summary.to_csv(summary_out, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": rows_path,
        "direct_alpha": float(direct_alpha),
        "interpolated_alpha": float(interpolated_alpha),
        "min_interpolation_support": int(min_interpolation_support),
        "candidate_only": bool(candidate_only),
        "pvalue_comparison_rows": pvalue_comparison_rows_path,
        "spectral_flow_edges": spectral_flow_edges_path,
        "input_row_count": int(len(input_rows)),
        "output_row_count": int(len(law_rows)),
        "outputs": {
            "rows": rows_out,
            "summary": summary_out,
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return {"rows": rows_out, "summary": summary_out, "manifest": manifest_out}


def main() -> None:
    args = parse_args()
    run_selected_neighborhood_measurability_law(
        rows_path=args.rows,
        output_dir=args.output_dir,
        direct_alpha=args.direct_alpha,
        interpolated_alpha=args.interpolated_alpha,
        min_interpolation_support=args.min_interpolation_support,
        candidate_only=bool(args.candidate_only),
        pvalue_comparison_rows_path=args.pvalue_comparison_rows,
        spectral_flow_edges_path=args.spectral_flow_edges,
    )


if __name__ == "__main__":
    main()
