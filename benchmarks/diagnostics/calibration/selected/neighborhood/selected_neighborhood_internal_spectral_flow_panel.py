"""Compare leaf-only and internal-barycenter selected-neighborhood spectral flow.

This panel is diagnostic-only. It runs the same selected tree/profile on the
same generated data while toggling only the opt-in internal-barycenter spectral
context. Internal barycenters are deterministic tree-filtered leaf averages, so
they are measured here as spectral support evidence, not as independent MP
samples or production p-value calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
    INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE,
    INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
)

from benchmarks.diagnostics.calibration.selected.family.selected_family_traversal_panel import (
    _output_data_role,
)
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_spectral_flow import (
    DEFAULT_EIGENVALUE_BLOCK_LOG_TOLERANCE,
    DEFAULT_METHOD_ID,
    DEFAULT_UNMATCHED_MODE_PENALTY,
    build_multiplicity_spectral_panels,
    build_spectral_flow_panels,
)
from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_panel import (
    validate_data_roles,
)
from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.shared.runners.tbs_runner import _run_tbs_method
from benchmarks.validation.statistics.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)

SCHEMA_VERSION = "selected_neighborhood_internal_spectral_flow_panel/v1"
STUDY_ROLE = "diagnostic_internal_barycenter_spectral_flow_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_internal_spectral_flow_panel"

LEAF_VARIANT = "leaf_only_spectral_flow"
INTERNAL_VARIANT = "internal_barycenter_spectral_flow"
BRANCH_LENGTH_INTERNAL_VARIANT = "branch_length_internal_state_spectral_flow"

EMPIRICAL_INTERNAL_COMPARISON = "leaf_vs_internal_barycenter"
BRANCH_LENGTH_INTERNAL_COMPARISON = "leaf_vs_branch_length_internal_state"

DEFAULT_CASE_NAMES = (
    "overlap_part_4c_small",
    "overlap_mod_4c_small",
    "overlap_heavy_4c_small_feat",
)

NODES_OUTPUT = "selected_neighborhood_internal_spectral_flow_nodes.csv"
EDGES_OUTPUT = "selected_neighborhood_internal_spectral_flow_edges.csv"
BLOCKS_OUTPUT = "selected_neighborhood_internal_spectral_flow_blocks.csv"
MODE_EDGES_OUTPUT = "selected_neighborhood_internal_spectral_flow_mode_edges.csv"
NODE_PAIRWISE_OUTPUT = "selected_neighborhood_internal_spectral_flow_node_pairwise.csv"
EDGE_PAIRWISE_OUTPUT = "selected_neighborhood_internal_spectral_flow_edge_pairwise.csv"
MODE_PAIRWISE_OUTPUT = "selected_neighborhood_internal_spectral_flow_mode_pairwise.csv"
NEIGHBORHOOD_ENERGY_OUTPUT = "selected_neighborhood_internal_spectral_flow_neighborhood_energy.csv"
SUMMARY_OUTPUT = "selected_neighborhood_internal_spectral_flow_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

EDGE_KEY_COLUMNS = (
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "parent_id",
    "child_id",
)

NODE_KEY_COLUMNS = (
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
)


@dataclass(frozen=True)
class InternalSpectralFlowConfig:
    """Configuration for the internal-barycenter spectral-flow comparison."""

    output_dir: Path
    suite: str = "full"
    case_names: tuple[str, ...] = DEFAULT_CASE_NAMES
    data_roles: tuple[str, ...] = ("null", "signal")
    method_id: str = DEFAULT_METHOD_ID
    replicates: int = 1
    base_seed: int = 20260618
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA
    edge_alpha: float = DEFAULT_EDGE_ALPHA
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION
    eigenvalue_block_log_tolerance: float = DEFAULT_EIGENVALUE_BLOCK_LOG_TOLERANCE
    unmatched_mode_penalty: float = DEFAULT_UNMATCHED_MODE_PENALTY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--case-names", type=parse_names, default=DEFAULT_CASE_NAMES)
    parser.add_argument("--data-roles", type=parse_names, default=("null", "signal"))
    parser.add_argument("--method-id", default=DEFAULT_METHOD_ID)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260618)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument(
        "--spectral-minimum-dimension",
        type=int,
        default=EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    )
    parser.add_argument(
        "--eigenvalue-block-log-tolerance",
        type=float,
        default=DEFAULT_EIGENVALUE_BLOCK_LOG_TOLERANCE,
    )
    parser.add_argument(
        "--unmatched-mode-penalty",
        type=float,
        default=DEFAULT_UNMATCHED_MODE_PENALTY,
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


def _finite_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return math.nan if finite.empty else float(finite.median())


def _mean_square(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return math.nan
    return float(np.mean(np.square(finite.to_numpy(dtype=float))))


def _positive_eigenvalues(eigenvalues: object) -> np.ndarray:
    if eigenvalues is None:
        return np.zeros(0, dtype=float)
    values = np.asarray(eigenvalues, dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def _principal_angle_transport(
    left_projection: object,
    right_projection: object,
    *,
    dimension: int,
) -> dict[str, object]:
    if int(dimension) <= 0:
        return {
            "angle_common_dimension": 0,
            "principal_angles_rad": "[]",
            "mean_sin_angle": math.nan,
            "max_sin_angle": math.nan,
            "effective_rotating_rank": math.nan,
            "rotation_concentration": math.nan,
            "angle_status": "no_common_dimension",
        }
    if left_projection is None or right_projection is None:
        return {
            "angle_common_dimension": 0,
            "principal_angles_rad": "[]",
            "mean_sin_angle": math.nan,
            "max_sin_angle": math.nan,
            "effective_rotating_rank": math.nan,
            "rotation_concentration": math.nan,
            "angle_status": "projection_missing",
        }
    left = np.asarray(left_projection, dtype=float)
    right = np.asarray(right_projection, dtype=float)
    common_dimension = min(int(dimension), left.shape[0], right.shape[0])
    if left.ndim != 2 or right.ndim != 2 or common_dimension <= 0:
        return {
            "angle_common_dimension": 0,
            "principal_angles_rad": "[]",
            "mean_sin_angle": math.nan,
            "max_sin_angle": math.nan,
            "effective_rotating_rank": math.nan,
            "rotation_concentration": math.nan,
            "angle_status": "projection_empty",
        }
    singular_values = np.linalg.svd(
        left[:common_dimension] @ right[:common_dimension].T,
        compute_uv=False,
    )
    clipped = np.clip(singular_values[:common_dimension], 0.0, 1.0)
    angles = np.arccos(clipped)
    sin_sq = np.maximum(1.0 - clipped**2, 0.0)
    sin_sum = float(np.sum(sin_sq))
    sin_four_sum = float(np.sum(sin_sq**2))
    effective_rank = (
        float((sin_sum**2) / sin_four_sum) if sin_four_sum > 0.0 and sin_sum > 0.0 else 0.0
    )
    concentration = float(np.max(sin_sq) / sin_sum) if sin_sum > 0.0 and sin_sq.size > 0 else 0.0
    return {
        "angle_common_dimension": int(common_dimension),
        "principal_angles_rad": json.dumps(
            [float(value) for value in angles],
            separators=(",", ":"),
        ),
        "mean_sin_angle": float(math.sqrt(np.mean(sin_sq))),
        "max_sin_angle": float(math.sqrt(np.max(sin_sq))) if sin_sq.size else math.nan,
        "effective_rotating_rank": effective_rank,
        "rotation_concentration": concentration,
        "angle_status": "angles_compared",
    }


def _radial_transport(
    left_eigenvalues: object,
    right_eigenvalues: object,
    *,
    dimension: int,
) -> dict[str, object]:
    if int(dimension) <= 0:
        return {
            "radius_common_dimension": 0,
            "log_radius_deltas": "[]",
            "mean_log_radius_delta": math.nan,
            "rms_log_radius_delta": math.nan,
            "max_abs_log_radius_delta": math.nan,
            "contracting_mode_count": 0,
            "expanding_mode_count": 0,
            "total_log_power_delta": math.nan,
            "radius_status": "no_common_dimension",
        }
    left = _positive_eigenvalues(left_eigenvalues)
    right = _positive_eigenvalues(right_eigenvalues)
    common_dimension = min(int(dimension), left.size, right.size)
    if common_dimension <= 0:
        return {
            "radius_common_dimension": 0,
            "log_radius_deltas": "[]",
            "mean_log_radius_delta": math.nan,
            "rms_log_radius_delta": math.nan,
            "max_abs_log_radius_delta": math.nan,
            "contracting_mode_count": 0,
            "expanding_mode_count": 0,
            "total_log_power_delta": math.nan,
            "radius_status": "eigenvalues_missing",
        }
    deltas = np.log(right[:common_dimension]) - np.log(left[:common_dimension])
    return {
        "radius_common_dimension": int(common_dimension),
        "log_radius_deltas": json.dumps(
            [float(value) for value in deltas],
            separators=(",", ":"),
        ),
        "mean_log_radius_delta": float(np.mean(deltas)),
        "rms_log_radius_delta": float(math.sqrt(np.mean(deltas**2))),
        "max_abs_log_radius_delta": float(np.max(np.abs(deltas))),
        "contracting_mode_count": int(np.sum(deltas < -1e-12)),
        "expanding_mode_count": int(np.sum(deltas > 1e-12)),
        "total_log_power_delta": float(
            math.log(float(np.sum(right[:common_dimension])))
            - math.log(float(np.sum(left[:common_dimension])))
        ),
        "radius_status": "radii_compared",
    }


def _object_transport_status(row: pd.Series) -> str:
    leaf_q = int(_finite_float(row.get("leaf_raw_mp_signal_count", 0.0)))
    internal_q = int(_finite_float(row.get("internal_raw_mp_signal_count", 0.0)))
    common = int(_finite_float(row.get("mp_common_dimension", 0.0)))
    if leaf_q <= 0 and internal_q > 0:
        return "internal_only_spike_created"
    if leaf_q > 0 and internal_q <= 0:
        return "leaf_spike_vanished"
    if common <= 0:
        return "no_shared_mp_object"

    mean_sin = _finite_float(row.get("mp_mean_sin_angle", row.get("mean_sin_angle")))
    max_sin = _finite_float(row.get("mp_max_sin_angle", row.get("max_sin_angle")))
    rotating_rank = _finite_float(
        row.get("mp_effective_rotating_rank", row.get("effective_rotating_rank"))
    )
    rms_radius = _finite_float(row.get("mp_rms_log_radius_delta", row.get("rms_log_radius_delta")))
    mean_radius = _finite_float(
        row.get("mp_mean_log_radius_delta", row.get("mean_log_radius_delta"))
    )

    angle_small = math.isfinite(mean_sin) and mean_sin <= 0.25
    radius_small = math.isfinite(rms_radius) and rms_radius <= 0.25
    angle_large = math.isfinite(mean_sin) and mean_sin >= 0.50
    single_mode = (
        math.isfinite(max_sin)
        and max_sin >= 0.65
        and math.isfinite(rotating_rank)
        and rotating_rank <= 1.5
    )

    if angle_small and radius_small:
        return "stable_angle_stable_radius"
    if angle_small and math.isfinite(mean_radius) and mean_radius < -0.25:
        return "stable_angle_power_loss"
    if angle_small and math.isfinite(mean_radius) and mean_radius > 0.25:
        return "stable_angle_power_gain"
    if single_mode:
        return "single_mode_rotation"
    if angle_large and math.isfinite(rotating_rank) and rotating_rank > 1.5:
        return "whole_object_rotation"
    if angle_large and math.isfinite(mean_radius) and mean_radius < -0.25:
        return "rotating_and_fading"
    if angle_large and math.isfinite(mean_radius) and mean_radius > 0.25:
        return "rotating_and_expanding"
    return "mixed_angular_radial_shift"


def _stamp_variant(
    frame: pd.DataFrame,
    *,
    variant_id: str,
    include_internal_barycenters: bool,
    internal_distribution_mode: str,
) -> pd.DataFrame:
    stamped = frame.copy()
    stamped["schema_version"] = SCHEMA_VERSION
    stamped["study_role"] = STUDY_ROLE
    stamped["spectral_variant_id"] = str(variant_id)
    stamped["spectral_include_internal_barycenters"] = bool(include_internal_barycenters)
    stamped["spectral_internal_distribution_mode"] = str(internal_distribution_mode)
    return stamped


def _variant_spec() -> tuple[tuple[str, bool, str], ...]:
    return (
        (LEAF_VARIANT, False, INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER),
        (INTERNAL_VARIANT, True, INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER),
        (
            BRANCH_LENGTH_INTERNAL_VARIANT,
            True,
            INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE,
        ),
    )


def _comparison_spec() -> tuple[tuple[str, str], ...]:
    return (
        (EMPIRICAL_INTERNAL_COMPARISON, INTERNAL_VARIANT),
        (BRANCH_LENGTH_INTERNAL_COMPARISON, BRANCH_LENGTH_INTERNAL_VARIANT),
    )


def _run_one_variant(
    *,
    data: pd.DataFrame,
    feature_space: object,
    distance_condensed: np.ndarray,
    case_id: str,
    data_role: str,
    method_id: str,
    replicate: int,
    sibling_alpha: float,
    edge_alpha: float,
    spectral_minimum_dimension: int,
    eigenvalue_block_log_tolerance: float,
    unmatched_mode_penalty: float,
    variant_id: str,
    include_internal_barycenters: bool,
    internal_distribution_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    profile_id = None if method_id == "baseline_projected_wald" else str(method_id)
    result = _run_tbs_method(
        data,
        distance_condensed,
        sibling_significance_level=float(sibling_alpha),
        tree_linkage_method="average",
        edge_alpha=float(edge_alpha),
        feature_space=feature_space,  # type: ignore[arg-type]
        spectral_minimum_dimension=int(spectral_minimum_dimension),
        spectral_include_internal_barycenters=bool(include_internal_barycenters),
        spectral_internal_distribution_mode=str(internal_distribution_mode),
        sibling_gate_profile=profile_id,
    )
    if result.status != "ok":
        raise RuntimeError(
            f"TBS run failed for {case_id}/{data_role}/{variant_id}: {result.skip_reason}"
        )

    gate_bundle = result.extra["gate_bundle"]
    spectral_context = gate_bundle.edge_gate_result.spectral_context
    node_rows, edge_rows = build_spectral_flow_panels(
        case_id=case_id,
        data_role=_output_data_role(data_role),
        method_id=method_id,
        replicate=int(replicate),
        tree=result.extra["tree"],
        annotations=result.extra["annotations"],
        spectral_context=spectral_context,
    )
    block_rows, mode_edge_rows = build_multiplicity_spectral_panels(
        case_id=case_id,
        data_role=_output_data_role(data_role),
        method_id=method_id,
        replicate=int(replicate),
        tree=result.extra["tree"],
        annotations=result.extra["annotations"],
        spectral_context=spectral_context,
        eigenvalue_block_log_tolerance=float(eigenvalue_block_log_tolerance),
        unmatched_mode_penalty=float(unmatched_mode_penalty),
    )
    return (
        _stamp_variant(
            node_rows,
            variant_id=variant_id,
            include_internal_barycenters=include_internal_barycenters,
            internal_distribution_mode=internal_distribution_mode,
        ),
        _stamp_variant(
            edge_rows,
            variant_id=variant_id,
            include_internal_barycenters=include_internal_barycenters,
            internal_distribution_mode=internal_distribution_mode,
        ),
        _stamp_variant(
            block_rows,
            variant_id=variant_id,
            include_internal_barycenters=include_internal_barycenters,
            internal_distribution_mode=internal_distribution_mode,
        ),
        _stamp_variant(
            mode_edge_rows,
            variant_id=variant_id,
            include_internal_barycenters=include_internal_barycenters,
            internal_distribution_mode=internal_distribution_mode,
        ),
        spectral_context,
    )


def build_node_pairwise_rows(
    *,
    case_id: str,
    data_role: str,
    method_id: str,
    replicate: int,
    leaf_context: Any,
    internal_context: Any,
    comparison_id: str = EMPIRICAL_INTERNAL_COMPARISON,
    internal_variant_id: str = INTERNAL_VARIANT,
) -> pd.DataFrame:
    """Compare leaf-only and internal-barycenter spectral objects by node."""
    leaf_projections = leaf_context.principal_component_projections_by_node
    internal_projections = internal_context.principal_component_projections_by_node
    leaf_eigenvalues = leaf_context.principal_component_eigenvalues_by_node
    internal_eigenvalues = internal_context.principal_component_eigenvalues_by_node
    leaf_raw_mp = leaf_context.raw_mp_signal_counts_by_node
    internal_raw_mp = internal_context.raw_mp_signal_counts_by_node
    leaf_test_dimensions = leaf_context.test_projection_dimensions_by_node
    internal_test_dimensions = internal_context.test_projection_dimensions_by_node

    node_ids = sorted(set(leaf_raw_mp).union(internal_raw_mp))
    records: list[dict[str, object]] = []
    for node_id in node_ids:
        leaf_q = int(leaf_raw_mp.get(node_id, 0))
        internal_q = int(internal_raw_mp.get(node_id, 0))
        mp_common_dimension = min(leaf_q, internal_q)
        floor_common_dimension = min(
            int(leaf_test_dimensions.get(node_id, 0)),
            int(internal_test_dimensions.get(node_id, 0)),
        )

        mp_angles = _principal_angle_transport(
            leaf_projections.get(node_id),
            internal_projections.get(node_id),
            dimension=mp_common_dimension,
        )
        mp_radii = _radial_transport(
            leaf_eigenvalues.get(node_id),
            internal_eigenvalues.get(node_id),
            dimension=mp_common_dimension,
        )
        floor_angles = _principal_angle_transport(
            leaf_projections.get(node_id),
            internal_projections.get(node_id),
            dimension=floor_common_dimension,
        )
        floor_radii = _radial_transport(
            leaf_eigenvalues.get(node_id),
            internal_eigenvalues.get(node_id),
            dimension=floor_common_dimension,
        )

        record = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": str(case_id),
            "data_role": str(data_role),
            "method_id": str(method_id),
            "replicate": int(replicate),
            "node_id": str(node_id),
            "spectral_variant_comparison_id": str(comparison_id),
            "leaf_spectral_variant_id": LEAF_VARIANT,
            "internal_spectral_variant_id": str(internal_variant_id),
            "leaf_raw_mp_signal_count": leaf_q,
            "internal_raw_mp_signal_count": internal_q,
            "delta_raw_mp_signal_count": int(internal_q - leaf_q),
            "mp_common_dimension": int(mp_common_dimension),
            "floor_common_dimension": int(floor_common_dimension),
            "mp_angle_common_dimension": int(mp_angles["angle_common_dimension"]),
            "mp_principal_angles_rad": mp_angles["principal_angles_rad"],
            "mp_mean_sin_angle": mp_angles["mean_sin_angle"],
            "mp_max_sin_angle": mp_angles["max_sin_angle"],
            "mp_effective_rotating_rank": mp_angles["effective_rotating_rank"],
            "mp_rotation_concentration": mp_angles["rotation_concentration"],
            "mp_angle_status": mp_angles["angle_status"],
            "mp_radius_common_dimension": int(mp_radii["radius_common_dimension"]),
            "mp_log_radius_deltas": mp_radii["log_radius_deltas"],
            "mp_mean_log_radius_delta": mp_radii["mean_log_radius_delta"],
            "mp_rms_log_radius_delta": mp_radii["rms_log_radius_delta"],
            "mp_max_abs_log_radius_delta": mp_radii["max_abs_log_radius_delta"],
            "mp_contracting_mode_count": int(mp_radii["contracting_mode_count"]),
            "mp_expanding_mode_count": int(mp_radii["expanding_mode_count"]),
            "mp_total_log_power_delta": mp_radii["total_log_power_delta"],
            "mp_radius_status": mp_radii["radius_status"],
            "floor_angle_common_dimension": int(floor_angles["angle_common_dimension"]),
            "floor_mean_sin_angle": floor_angles["mean_sin_angle"],
            "floor_max_sin_angle": floor_angles["max_sin_angle"],
            "floor_effective_rotating_rank": floor_angles["effective_rotating_rank"],
            "floor_rotation_concentration": floor_angles["rotation_concentration"],
            "floor_mean_log_radius_delta": floor_radii["mean_log_radius_delta"],
            "floor_rms_log_radius_delta": floor_radii["rms_log_radius_delta"],
            "floor_total_log_power_delta": floor_radii["total_log_power_delta"],
        }
        record["object_transport_status"] = _object_transport_status(pd.Series(record))
        records.append(record)

    return pd.DataFrame.from_records(records)


def _edge_gain_status(row: pd.Series) -> str:
    leaf_supported = _finite_bool(
        row.get("leaf_mp_pair_supported", row.get("mp_pair_supported_leaf"))
    )
    internal_supported = _finite_bool(
        row.get("internal_mp_pair_supported", row.get("mp_pair_supported_internal"))
    )
    if not leaf_supported and internal_supported:
        return "mp_support_created_by_internal_barycenter"
    if leaf_supported and not internal_supported:
        return "mp_support_lost_by_internal_barycenter"
    if not leaf_supported and not internal_supported:
        return "floor_only_both_variants"

    delta_barrier = _finite_float(row.get("delta_spectral_barrier"))
    delta_affinity = _finite_float(row.get("delta_spectral_flow_affinity"))
    if math.isfinite(delta_barrier) and delta_barrier < -1e-12:
        return "both_supported_barrier_improved"
    if math.isfinite(delta_affinity) and delta_affinity > 1e-12:
        return "both_supported_affinity_improved"
    if math.isfinite(delta_barrier) and delta_barrier > 1e-12:
        return "both_supported_barrier_degraded"
    return "both_supported_no_material_change"


def _mode_gain_status(row: pd.Series) -> str:
    leaf_matched = _finite_float(
        row.get("leaf_matched_mp_block_count", row.get("matched_mp_block_count_leaf"))
    )
    internal_matched = _finite_float(
        row.get(
            "internal_matched_mp_block_count",
            row.get("matched_mp_block_count_internal"),
        )
    )
    leaf_supported = math.isfinite(leaf_matched) and int(leaf_matched) > 0
    internal_supported = math.isfinite(internal_matched) and int(internal_matched) > 0
    if not leaf_supported and internal_supported:
        return "mp_block_support_created_by_internal_barycenter"
    if leaf_supported and not internal_supported:
        return "mp_block_support_lost_by_internal_barycenter"
    if not leaf_supported and not internal_supported:
        return "floor_only_or_unmatched_both_variants"

    delta_cost = _finite_float(row.get("delta_mode_transport_cost"))
    delta_affinity = _finite_float(row.get("delta_mode_transport_affinity"))
    if math.isfinite(delta_cost) and delta_cost < -1e-12:
        return "both_supported_transport_improved"
    if math.isfinite(delta_affinity) and delta_affinity > 1e-12:
        return "both_supported_transport_affinity_improved"
    if math.isfinite(delta_cost) and delta_cost > 1e-12:
        return "both_supported_transport_degraded"
    return "both_supported_transport_no_material_change"


def build_edge_pairwise_rows(edge_rows: pd.DataFrame) -> pd.DataFrame:
    """Join leaf-only spectral-flow edge rows against each internal-state variant."""
    if edge_rows.empty:
        return pd.DataFrame()
    leaf = edge_rows[edge_rows["spectral_variant_id"].astype(str).eq(LEAF_VARIANT)].copy()
    records: list[dict[str, object]] = []
    for comparison_id, internal_variant_id in _comparison_spec():
        internal = edge_rows[
            edge_rows["spectral_variant_id"].astype(str).eq(internal_variant_id)
        ].copy()
        if internal.empty:
            continue
        merged = leaf.merge(
            internal,
            on=list(EDGE_KEY_COLUMNS),
            how="inner",
            suffixes=("_leaf", "_internal"),
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            record = {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                **{column: row[column] for column in EDGE_KEY_COLUMNS},
                "spectral_variant_comparison_id": str(comparison_id),
                "leaf_spectral_variant_id": LEAF_VARIANT,
                "internal_spectral_variant_id": str(internal_variant_id),
                "leaf_flow_status": row["flow_status_leaf"],
                "internal_flow_status": row["flow_status_internal"],
                "leaf_mp_pair_supported": bool(row["mp_pair_supported_leaf"]),
                "internal_mp_pair_supported": bool(row["mp_pair_supported_internal"]),
                "leaf_parent_raw_mp_signal_count": int(row["parent_raw_mp_signal_count_leaf"]),
                "internal_parent_raw_mp_signal_count": int(
                    row["parent_raw_mp_signal_count_internal"]
                ),
                "leaf_child_raw_mp_signal_count": int(row["child_raw_mp_signal_count_leaf"]),
                "internal_child_raw_mp_signal_count": int(
                    row["child_raw_mp_signal_count_internal"]
                ),
                "leaf_mp_common_dimension": int(row["mp_common_dimension_leaf"]),
                "internal_mp_common_dimension": int(row["mp_common_dimension_internal"]),
                "leaf_mp_subspace_chordal_distance": _finite_float(
                    row["mp_subspace_chordal_distance_leaf"]
                ),
                "internal_mp_subspace_chordal_distance": _finite_float(
                    row["mp_subspace_chordal_distance_internal"]
                ),
                "leaf_mp_log_eigenvalue_delta": _finite_float(row["mp_log_eigenvalue_delta_leaf"]),
                "internal_mp_log_eigenvalue_delta": _finite_float(
                    row["mp_log_eigenvalue_delta_internal"]
                ),
                "leaf_spectral_barrier": _finite_float(row["spectral_barrier_leaf"]),
                "internal_spectral_barrier": _finite_float(row["spectral_barrier_internal"]),
                "leaf_spectral_flow_affinity": _finite_float(row["spectral_flow_affinity_leaf"]),
                "internal_spectral_flow_affinity": _finite_float(
                    row["spectral_flow_affinity_internal"]
                ),
            }
            record.update(
                {
                    "delta_parent_raw_mp_signal_count": (
                        record["internal_parent_raw_mp_signal_count"]
                        - record["leaf_parent_raw_mp_signal_count"]
                    ),
                    "delta_child_raw_mp_signal_count": (
                        record["internal_child_raw_mp_signal_count"]
                        - record["leaf_child_raw_mp_signal_count"]
                    ),
                    "delta_mp_common_dimension": (
                        record["internal_mp_common_dimension"] - record["leaf_mp_common_dimension"]
                    ),
                    "delta_mp_subspace_chordal_distance": (
                        record["internal_mp_subspace_chordal_distance"]
                        - record["leaf_mp_subspace_chordal_distance"]
                    ),
                    "delta_mp_log_eigenvalue_delta": (
                        record["internal_mp_log_eigenvalue_delta"]
                        - record["leaf_mp_log_eigenvalue_delta"]
                    ),
                    "delta_spectral_barrier": (
                        record["internal_spectral_barrier"] - record["leaf_spectral_barrier"]
                    ),
                    "delta_spectral_flow_affinity": (
                        record["internal_spectral_flow_affinity"]
                        - record["leaf_spectral_flow_affinity"]
                    ),
                }
            )
            record["edge_gain_status"] = _edge_gain_status(pd.Series(record))
            records.append(record)
    return pd.DataFrame.from_records(records)


def build_mode_pairwise_rows(mode_edge_rows: pd.DataFrame) -> pd.DataFrame:
    """Join leaf-only multiplicity-mode rows against each internal-state variant."""
    if mode_edge_rows.empty:
        return pd.DataFrame()
    leaf = mode_edge_rows[mode_edge_rows["spectral_variant_id"].astype(str).eq(LEAF_VARIANT)].copy()
    records: list[dict[str, object]] = []
    for comparison_id, internal_variant_id in _comparison_spec():
        internal = mode_edge_rows[
            mode_edge_rows["spectral_variant_id"].astype(str).eq(internal_variant_id)
        ].copy()
        if internal.empty:
            continue
        merged = leaf.merge(
            internal,
            on=list(EDGE_KEY_COLUMNS),
            how="inner",
            suffixes=("_leaf", "_internal"),
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            record = {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                **{column: row[column] for column in EDGE_KEY_COLUMNS},
                "spectral_variant_comparison_id": str(comparison_id),
                "leaf_spectral_variant_id": LEAF_VARIANT,
                "internal_spectral_variant_id": str(internal_variant_id),
                "leaf_mode_flow_status": row["mode_flow_status_leaf"],
                "internal_mode_flow_status": row["mode_flow_status_internal"],
                "leaf_matched_mp_block_count": int(row["matched_mp_block_count_leaf"]),
                "internal_matched_mp_block_count": int(row["matched_mp_block_count_internal"]),
                "leaf_unmatched_mp_block_count": int(row["unmatched_mp_block_count_leaf"]),
                "internal_unmatched_mp_block_count": int(row["unmatched_mp_block_count_internal"]),
                "leaf_mode_transport_cost": _finite_float(row["mode_transport_cost_leaf"]),
                "internal_mode_transport_cost": _finite_float(row["mode_transport_cost_internal"]),
                "leaf_mode_transport_affinity": _finite_float(row["mode_transport_affinity_leaf"]),
                "internal_mode_transport_affinity": _finite_float(
                    row["mode_transport_affinity_internal"]
                ),
                "leaf_connection_laplacian_residual": _finite_float(
                    row["connection_laplacian_residual_leaf"]
                ),
                "internal_connection_laplacian_residual": _finite_float(
                    row["connection_laplacian_residual_internal"]
                ),
                "leaf_mean_block_projector_chordal_distance": _finite_float(
                    row["mean_block_projector_chordal_distance_leaf"]
                ),
                "internal_mean_block_projector_chordal_distance": _finite_float(
                    row["mean_block_projector_chordal_distance_internal"]
                ),
            }
            record.update(
                {
                    "delta_matched_mp_block_count": (
                        record["internal_matched_mp_block_count"]
                        - record["leaf_matched_mp_block_count"]
                    ),
                    "delta_unmatched_mp_block_count": (
                        record["internal_unmatched_mp_block_count"]
                        - record["leaf_unmatched_mp_block_count"]
                    ),
                    "delta_mode_transport_cost": (
                        record["internal_mode_transport_cost"] - record["leaf_mode_transport_cost"]
                    ),
                    "delta_mode_transport_affinity": (
                        record["internal_mode_transport_affinity"]
                        - record["leaf_mode_transport_affinity"]
                    ),
                    "delta_connection_laplacian_residual": (
                        record["internal_connection_laplacian_residual"]
                        - record["leaf_connection_laplacian_residual"]
                    ),
                    "delta_mean_block_projector_chordal_distance": (
                        record["internal_mean_block_projector_chordal_distance"]
                        - record["leaf_mean_block_projector_chordal_distance"]
                    ),
                }
            )
            record["mode_gain_status"] = _mode_gain_status(pd.Series(record))
            records.append(record)
    return pd.DataFrame.from_records(records)


def _energy_terms(
    frame: pd.DataFrame,
    *,
    angle_column: str,
    radius_column: str,
) -> dict[str, object]:
    """Return Dirichlet-style transport energies over a selected edge set."""
    if frame.empty:
        return {
            "edge_count": 0,
            "angle_dirichlet_energy": math.nan,
            "radius_dirichlet_energy": math.nan,
            "joint_transport_energy": math.nan,
        }
    angle_energy = _mean_square(frame[angle_column])
    radius_energy = _mean_square(frame[radius_column])
    joint_terms = []
    for _, row in frame.iterrows():
        angle = _finite_float(row.get(angle_column))
        radius = _finite_float(row.get(radius_column))
        terms = []
        if math.isfinite(angle):
            terms.append(angle**2)
        if math.isfinite(radius):
            terms.append(radius**2)
        if terms:
            joint_terms.append(float(sum(terms)))
    return {
        "edge_count": int(len(frame)),
        "angle_dirichlet_energy": angle_energy,
        "radius_dirichlet_energy": radius_energy,
        "joint_transport_energy": (float(np.mean(joint_terms)) if joint_terms else math.nan),
    }


def _energy_status(row: pd.Series) -> str:
    internal_only_value = _finite_float(row.get("internal_only_mp_supported_edge_count", 0))
    strict_count_value = _finite_float(row.get("strict_shared_mp_supported_edge_count", 0))
    internal_only = int(internal_only_value) if math.isfinite(internal_only_value) else 0
    strict_count = int(strict_count_value) if math.isfinite(strict_count_value) else 0
    data_role = str(row.get("data_role", ""))
    if data_role == "selected_null" and internal_only > 0:
        return "diagnostic_warn_selected_null_internal_only_transport_energy"
    if strict_count <= 0:
        return "diagnostic_no_strict_shared_mp_transport_energy"
    delta = _finite_float(row.get("delta_strict_shared_mp_joint_transport_energy"))
    if math.isfinite(delta) and delta < -1e-12:
        return "diagnostic_internal_smooths_strict_shared_transport"
    if math.isfinite(delta) and delta > 1e-12:
        return "diagnostic_internal_degrades_strict_shared_transport"
    return "diagnostic_strict_shared_transport_energy_unchanged"


def build_neighborhood_energy_rows(edge_pairwise: pd.DataFrame) -> pd.DataFrame:
    """Aggregate spectral-flow edges into graph Dirichlet-style energies.

    The selected tree topology supplies the neighborhood graph and each
    parent-child edge has unit weight. Energies are diagnostic evidence only:
    they summarize local angle/radius coherence and do not create p-values.
    """
    if edge_pairwise.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    group_columns = ["case_id", "data_role", "method_id", "replicate"]
    if "spectral_variant_comparison_id" in edge_pairwise.columns:
        group_columns.append("spectral_variant_comparison_id")
    for key, group in edge_pairwise.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        group_metadata = dict(zip(group_columns, key_values, strict=True))
        case_id = group_metadata["case_id"]
        data_role = group_metadata["data_role"]
        method_id = group_metadata["method_id"]
        replicate = group_metadata["replicate"]
        leaf_supported = group["leaf_mp_pair_supported"].astype(bool)
        internal_supported = group["internal_mp_pair_supported"].astype(bool)
        strict_shared = leaf_supported & internal_supported
        internal_only = ~leaf_supported & internal_supported
        leaf_only = leaf_supported & ~internal_supported

        leaf_own = _energy_terms(
            group.loc[leaf_supported],
            angle_column="leaf_mp_subspace_chordal_distance",
            radius_column="leaf_mp_log_eigenvalue_delta",
        )
        internal_own = _energy_terms(
            group.loc[internal_supported],
            angle_column="internal_mp_subspace_chordal_distance",
            radius_column="internal_mp_log_eigenvalue_delta",
        )
        strict_leaf = _energy_terms(
            group.loc[strict_shared],
            angle_column="leaf_mp_subspace_chordal_distance",
            radius_column="leaf_mp_log_eigenvalue_delta",
        )
        strict_internal = _energy_terms(
            group.loc[strict_shared],
            angle_column="internal_mp_subspace_chordal_distance",
            radius_column="internal_mp_log_eigenvalue_delta",
        )
        internal_only_energy = _energy_terms(
            group.loc[internal_only],
            angle_column="internal_mp_subspace_chordal_distance",
            radius_column="internal_mp_log_eigenvalue_delta",
        )

        record = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": str(case_id),
            "data_role": str(data_role),
            "method_id": str(method_id),
            "replicate": int(replicate),
            "edge_count": int(len(group)),
            "leaf_mp_supported_edge_count": int(leaf_supported.sum()),
            "internal_mp_supported_edge_count": int(internal_supported.sum()),
            "strict_shared_mp_supported_edge_count": int(strict_shared.sum()),
            "internal_only_mp_supported_edge_count": int(internal_only.sum()),
            "leaf_only_mp_supported_edge_count": int(leaf_only.sum()),
            "leaf_mp_angle_dirichlet_energy": leaf_own["angle_dirichlet_energy"],
            "leaf_mp_radius_dirichlet_energy": leaf_own["radius_dirichlet_energy"],
            "leaf_mp_joint_transport_energy": leaf_own["joint_transport_energy"],
            "internal_mp_angle_dirichlet_energy": internal_own["angle_dirichlet_energy"],
            "internal_mp_radius_dirichlet_energy": internal_own["radius_dirichlet_energy"],
            "internal_mp_joint_transport_energy": internal_own["joint_transport_energy"],
            "strict_shared_leaf_mp_angle_dirichlet_energy": strict_leaf["angle_dirichlet_energy"],
            "strict_shared_leaf_mp_radius_dirichlet_energy": strict_leaf["radius_dirichlet_energy"],
            "strict_shared_leaf_mp_joint_transport_energy": strict_leaf["joint_transport_energy"],
            "strict_shared_internal_mp_angle_dirichlet_energy": strict_internal[
                "angle_dirichlet_energy"
            ],
            "strict_shared_internal_mp_radius_dirichlet_energy": strict_internal[
                "radius_dirichlet_energy"
            ],
            "strict_shared_internal_mp_joint_transport_energy": strict_internal[
                "joint_transport_energy"
            ],
            "internal_only_mp_angle_dirichlet_energy": internal_only_energy[
                "angle_dirichlet_energy"
            ],
            "internal_only_mp_radius_dirichlet_energy": internal_only_energy[
                "radius_dirichlet_energy"
            ],
            "internal_only_mp_joint_transport_energy": internal_only_energy[
                "joint_transport_energy"
            ],
        }
        if "spectral_variant_comparison_id" in group_metadata:
            record["spectral_variant_comparison_id"] = str(
                group_metadata["spectral_variant_comparison_id"]
            )
        record.update(
            {
                "delta_mp_angle_dirichlet_energy": (
                    record["internal_mp_angle_dirichlet_energy"]
                    - record["leaf_mp_angle_dirichlet_energy"]
                ),
                "delta_mp_radius_dirichlet_energy": (
                    record["internal_mp_radius_dirichlet_energy"]
                    - record["leaf_mp_radius_dirichlet_energy"]
                ),
                "delta_mp_joint_transport_energy": (
                    record["internal_mp_joint_transport_energy"]
                    - record["leaf_mp_joint_transport_energy"]
                ),
                "delta_strict_shared_mp_angle_dirichlet_energy": (
                    record["strict_shared_internal_mp_angle_dirichlet_energy"]
                    - record["strict_shared_leaf_mp_angle_dirichlet_energy"]
                ),
                "delta_strict_shared_mp_radius_dirichlet_energy": (
                    record["strict_shared_internal_mp_radius_dirichlet_energy"]
                    - record["strict_shared_leaf_mp_radius_dirichlet_energy"]
                ),
                "delta_strict_shared_mp_joint_transport_energy": (
                    record["strict_shared_internal_mp_joint_transport_energy"]
                    - record["strict_shared_leaf_mp_joint_transport_energy"]
                ),
            }
        )
        record["neighborhood_energy_status"] = _energy_status(pd.Series(record))
        records.append(record)

    return pd.DataFrame.from_records(records)


def _summary_status(data_role: str, group: pd.DataFrame) -> str:
    created = int(group["edge_gain_status"].eq("mp_support_created_by_internal_barycenter").sum())
    lost = int(group["edge_gain_status"].eq("mp_support_lost_by_internal_barycenter").sum())
    median_barrier_delta = _median(group["delta_spectral_barrier"])
    median_affinity_delta = _median(group["delta_spectral_flow_affinity"])
    if data_role == "selected_null" and created > 0:
        return "diagnostic_warn_selected_null_internal_support_gain"
    if data_role == "signal" and created > 0:
        return "diagnostic_signal_internal_mp_support_gain_observed"
    if data_role == "signal" and math.isfinite(median_barrier_delta):
        if median_barrier_delta < 0.0 or median_affinity_delta > 0.0:
            return "diagnostic_signal_internal_flow_stability_gain_observed"
    if lost > created:
        return "diagnostic_internal_flow_support_loss_observed"
    return "diagnostic_no_internal_flow_gain_observed"


def summarize_internal_spectral_flow(
    edge_pairwise: pd.DataFrame,
    mode_pairwise: pd.DataFrame,
    node_pairwise: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize internal-barycenter spectral-flow effects by data role."""
    if edge_pairwise.empty:
        return pd.DataFrame(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "data_role": "all",
                    "method_id": "",
                    "edge_count": 0,
                    "summary_status": "no_pairwise_edges",
                }
            ]
        )
    summary_group_columns = ["data_role", "method_id"]
    comparison_column = "spectral_variant_comparison_id"
    if comparison_column in edge_pairwise.columns:
        summary_group_columns.append(comparison_column)
    mode_lookup = (
        mode_pairwise.groupby(summary_group_columns, dropna=False)
        if set(summary_group_columns).issubset(mode_pairwise.columns)
        else None
    )
    node_pairwise = pd.DataFrame() if node_pairwise is None else node_pairwise
    node_lookup = (
        node_pairwise.groupby(summary_group_columns, dropna=False)
        if set(summary_group_columns).issubset(node_pairwise.columns)
        else None
    )
    records: list[dict[str, object]] = []
    for key, group in edge_pairwise.groupby(
        summary_group_columns,
        dropna=False,
    ):
        key_values = key if isinstance(key, tuple) else (key,)
        group_metadata = dict(zip(summary_group_columns, key_values, strict=True))
        data_role = group_metadata["data_role"]
        method_id = group_metadata["method_id"]
        mode_group = pd.DataFrame()
        if mode_lookup is not None and key_values in mode_lookup.groups:
            mode_group = mode_lookup.get_group(key_values)
        node_group = pd.DataFrame()
        if node_lookup is not None and key_values in node_lookup.groups:
            node_group = node_lookup.get_group(key_values)
        node_status = (
            node_group.get("object_transport_status", pd.Series(dtype=str))
            if not node_group.empty
            else pd.Series(dtype=str)
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "data_role": str(data_role),
            "method_id": str(method_id),
            "edge_count": int(len(group)),
            "leaf_mp_supported_edge_count": int(group["leaf_mp_pair_supported"].sum()),
            "internal_mp_supported_edge_count": int(group["internal_mp_pair_supported"].sum()),
            "mp_support_created_count": int(
                group["edge_gain_status"].eq("mp_support_created_by_internal_barycenter").sum()
            ),
            "mp_support_lost_count": int(
                group["edge_gain_status"].eq("mp_support_lost_by_internal_barycenter").sum()
            ),
            "both_supported_improved_count": int(
                group["edge_gain_status"]
                .isin(
                    [
                        "both_supported_barrier_improved",
                        "both_supported_affinity_improved",
                    ]
                )
                .sum()
            ),
            "both_supported_degraded_count": int(
                group["edge_gain_status"].eq("both_supported_barrier_degraded").sum()
            ),
            "median_delta_parent_raw_mp_signal_count": _median(
                group["delta_parent_raw_mp_signal_count"]
            ),
            "median_delta_child_raw_mp_signal_count": _median(
                group["delta_child_raw_mp_signal_count"]
            ),
            "median_delta_mp_common_dimension": _median(group["delta_mp_common_dimension"]),
            "median_delta_mp_subspace_chordal_distance": _median(
                group["delta_mp_subspace_chordal_distance"]
            ),
            "median_delta_mp_log_eigenvalue_delta": _median(group["delta_mp_log_eigenvalue_delta"]),
            "median_delta_spectral_barrier": _median(group["delta_spectral_barrier"]),
            "median_delta_spectral_flow_affinity": _median(group["delta_spectral_flow_affinity"]),
            "mode_edge_count": int(len(mode_group)),
            "mode_support_created_count": int(
                mode_group.get("mode_gain_status", pd.Series(dtype=str))
                .eq("mp_block_support_created_by_internal_barycenter")
                .sum()
            ),
            "mode_support_lost_count": int(
                mode_group.get("mode_gain_status", pd.Series(dtype=str))
                .eq("mp_block_support_lost_by_internal_barycenter")
                .sum()
            ),
            "median_delta_mode_transport_cost": (
                _median(mode_group["delta_mode_transport_cost"])
                if "delta_mode_transport_cost" in mode_group
                else math.nan
            ),
            "median_delta_mode_transport_affinity": (
                _median(mode_group["delta_mode_transport_affinity"])
                if "delta_mode_transport_affinity" in mode_group
                else math.nan
            ),
            "node_pairwise_count": int(len(node_group)),
            "node_internal_only_spike_created_count": int(
                node_status.eq("internal_only_spike_created").sum()
            ),
            "node_leaf_spike_vanished_count": int(node_status.eq("leaf_spike_vanished").sum()),
            "node_stable_angle_radius_count": int(
                node_status.eq("stable_angle_stable_radius").sum()
            ),
            "node_stable_angle_power_gain_count": int(
                node_status.eq("stable_angle_power_gain").sum()
            ),
            "node_stable_angle_power_loss_count": int(
                node_status.eq("stable_angle_power_loss").sum()
            ),
            "node_single_mode_rotation_count": int(node_status.eq("single_mode_rotation").sum()),
            "node_whole_object_rotation_count": int(node_status.eq("whole_object_rotation").sum()),
            "median_node_mp_mean_sin_angle": (
                _median(node_group["mp_mean_sin_angle"])
                if "mp_mean_sin_angle" in node_group
                else math.nan
            ),
            "median_node_mp_rms_log_radius_delta": (
                _median(node_group["mp_rms_log_radius_delta"])
                if "mp_rms_log_radius_delta" in node_group
                else math.nan
            ),
            "median_node_mp_total_log_power_delta": (
                _median(node_group["mp_total_log_power_delta"])
                if "mp_total_log_power_delta" in node_group
                else math.nan
            ),
            "summary_status": _summary_status(str(data_role), group),
        }
        if comparison_column in group_metadata:
            record[comparison_column] = str(group_metadata[comparison_column])
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_internal_spectral_flow_rows(
    config: InternalSpectralFlowConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both spectral variants and return node, edge, block, and mode rows."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    roles = validate_data_roles(config.data_roles)
    cases = _select_cases(suite=config.suite, case_names=config.case_names)

    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    block_frames: list[pd.DataFrame] = []
    mode_edge_frames: list[pd.DataFrame] = []
    node_pairwise_frames: list[pd.DataFrame] = []
    for case in cases:
        (
            case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        for replicate in range(int(config.replicates)):
            data_seed = int(config.base_seed) + replicate * 1009
            for raw_role in roles:
                data, feature_space, _truth_labels, _true_clusters = _generate_data_with_truth(
                    case=case,
                    case_id=case_id,
                    source_family=source_family,
                    feature_representation=feature_representation,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_categories=n_categories,
                    data_role=raw_role,
                    seed=data_seed,
                )
                distance_condensed = pdist(data.to_numpy(dtype=float), metric="hamming")
                contexts_by_variant: dict[str, Any] = {}
                for (
                    variant_id,
                    include_internal,
                    internal_distribution_mode,
                ) in _variant_spec():
                    (
                        node_rows,
                        edge_rows,
                        block_rows,
                        mode_edge_rows,
                        spectral_context,
                    ) = _run_one_variant(
                        data=data,
                        feature_space=feature_space,
                        distance_condensed=distance_condensed,
                        case_id=case_id,
                        data_role=str(raw_role),
                        method_id=str(config.method_id),
                        replicate=replicate,
                        sibling_alpha=float(config.sibling_alpha),
                        edge_alpha=float(config.edge_alpha),
                        spectral_minimum_dimension=int(config.spectral_minimum_dimension),
                        eigenvalue_block_log_tolerance=float(config.eigenvalue_block_log_tolerance),
                        unmatched_mode_penalty=float(config.unmatched_mode_penalty),
                        variant_id=variant_id,
                        include_internal_barycenters=include_internal,
                        internal_distribution_mode=internal_distribution_mode,
                    )
                    node_frames.append(node_rows)
                    edge_frames.append(edge_rows)
                    block_frames.append(block_rows)
                    mode_edge_frames.append(mode_edge_rows)
                    contexts_by_variant[variant_id] = spectral_context
                if LEAF_VARIANT in contexts_by_variant:
                    for comparison_id, internal_variant_id in _comparison_spec():
                        if internal_variant_id not in contexts_by_variant:
                            continue
                        node_pairwise_frames.append(
                            build_node_pairwise_rows(
                                case_id=case_id,
                                data_role=_output_data_role(str(raw_role)),
                                method_id=str(config.method_id),
                                replicate=replicate,
                                leaf_context=contexts_by_variant[LEAF_VARIANT],
                                internal_context=contexts_by_variant[internal_variant_id],
                                comparison_id=comparison_id,
                                internal_variant_id=internal_variant_id,
                            )
                        )

    return (
        pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame(),
        pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame(),
        pd.concat(block_frames, ignore_index=True) if block_frames else pd.DataFrame(),
        pd.concat(mode_edge_frames, ignore_index=True) if mode_edge_frames else pd.DataFrame(),
        pd.concat(node_pairwise_frames, ignore_index=True)
        if node_pairwise_frames
        else pd.DataFrame(),
    )


def run_selected_neighborhood_internal_spectral_flow_panel(
    config: InternalSpectralFlowConfig,
) -> dict[str, Path]:
    """Run the diagnostic and write CSV outputs plus a manifest."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    nodes, edges, blocks, mode_edges, node_pairwise = build_internal_spectral_flow_rows(config)
    edge_pairwise = build_edge_pairwise_rows(edges)
    mode_pairwise = build_mode_pairwise_rows(mode_edges)
    neighborhood_energy = build_neighborhood_energy_rows(edge_pairwise)
    summary = summarize_internal_spectral_flow(
        edge_pairwise,
        mode_pairwise,
        node_pairwise,
    )

    paths = {
        "nodes": config.output_dir / NODES_OUTPUT,
        "edges": config.output_dir / EDGES_OUTPUT,
        "blocks": config.output_dir / BLOCKS_OUTPUT,
        "mode_edges": config.output_dir / MODE_EDGES_OUTPUT,
        "node_pairwise": config.output_dir / NODE_PAIRWISE_OUTPUT,
        "edge_pairwise": config.output_dir / EDGE_PAIRWISE_OUTPUT,
        "mode_pairwise": config.output_dir / MODE_PAIRWISE_OUTPUT,
        "neighborhood_energy": config.output_dir / NEIGHBORHOOD_ENERGY_OUTPUT,
        "summary": config.output_dir / SUMMARY_OUTPUT,
        "manifest": config.output_dir / MANIFEST_OUTPUT,
    }

    nodes.to_csv(paths["nodes"], index=False)
    edges.to_csv(paths["edges"], index=False)
    blocks.to_csv(paths["blocks"], index=False)
    mode_edges.to_csv(paths["mode_edges"], index=False)
    node_pairwise.to_csv(paths["node_pairwise"], index=False)
    edge_pairwise.to_csv(paths["edge_pairwise"], index=False)
    mode_pairwise.to_csv(paths["mode_pairwise"], index=False)
    neighborhood_energy.to_csv(paths["neighborhood_energy"], index=False)
    summary.to_csv(paths["summary"], index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "suite": config.suite,
            "case_names": list(config.case_names),
            "data_roles": list(config.data_roles),
            "method_id": str(config.method_id),
            "replicates": int(config.replicates),
            "base_seed": int(config.base_seed),
            "sibling_alpha": float(config.sibling_alpha),
            "edge_alpha": float(config.edge_alpha),
            "spectral_minimum_dimension": int(config.spectral_minimum_dimension),
            "eigenvalue_block_log_tolerance": float(config.eigenvalue_block_log_tolerance),
            "unmatched_mode_penalty": float(config.unmatched_mode_penalty),
        },
        "variants": {
            variant_id: {
                "spectral_include_internal_barycenters": bool(include_internal),
                "spectral_internal_distribution_mode": str(internal_distribution_mode),
            }
            for variant_id, include_internal, internal_distribution_mode in _variant_spec()
        },
        "comparisons": [
            {
                "spectral_variant_comparison_id": comparison_id,
                "leaf_spectral_variant_id": LEAF_VARIANT,
                "internal_spectral_variant_id": internal_variant_id,
            }
            for comparison_id, internal_variant_id in _comparison_spec()
        ],
        "outputs": paths,
        "n_node_rows": int(len(nodes)),
        "n_edge_rows": int(len(edges)),
        "n_block_rows": int(len(blocks)),
        "n_mode_edge_rows": int(len(mode_edges)),
        "n_node_pairwise_rows": int(len(node_pairwise)),
        "n_edge_pairwise_rows": int(len(edge_pairwise)),
        "n_mode_pairwise_rows": int(len(mode_pairwise)),
        "n_neighborhood_energy_rows": int(len(neighborhood_energy)),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    args = parse_args()
    run_selected_neighborhood_internal_spectral_flow_panel(
        InternalSpectralFlowConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=tuple(args.case_names),
            data_roles=tuple(args.data_roles),
            method_id=str(args.method_id),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            sibling_alpha=float(args.sibling_alpha),
            edge_alpha=float(args.edge_alpha),
            spectral_minimum_dimension=int(args.spectral_minimum_dimension),
            eigenvalue_block_log_tolerance=float(args.eigenvalue_block_log_tolerance),
            unmatched_mode_penalty=float(args.unmatched_mode_penalty),
        )
    )


if __name__ == "__main__":
    main()
