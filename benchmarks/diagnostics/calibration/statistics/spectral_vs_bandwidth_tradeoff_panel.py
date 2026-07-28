"""Compare strict spectral transport with the older bandwidth interpolation layer.

The old bandwidth object is not a runnable clustering profile in the current
tree. It is a selected-neighborhood p-like interpolation diagnostic. This panel
therefore compares the overlapping evidence surfaces: selected-null false-split
control, signal retention/catch behavior, and full-Julia fragmentation summaries.
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

SCHEMA_VERSION = "spectral_vs_bandwidth_tradeoff_panel/v1"
STUDY_ROLE = "diagnostic_spectral_vs_bandwidth_tradeoff_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.statistics.spectral_vs_bandwidth_tradeoff_panel"

BASELINE_PROFILE = "fixed_coordinate_global_passthrough_refined_v1"
CANDIDATE_PROFILE = "fixed_coordinate_spectral_transport_passthrough_v1"
BANDWIDTH_METHOD = "fixed_coordinate_global_passthrough_refined_v1"
REFERENCE_TAU_S = 20.0

DEFAULT_SMALL_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_OLD_STACK_ROOT = Path(
    "raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison"
)
DEFAULT_SELECTED_FAMILY_ROWS = (
    DEFAULT_SMALL_ROOT
    / "selected_family_traversal_spectral_transport_promoted_replicates"
    / "selected_family_traversal_rows.csv"
)
DEFAULT_PROMOTION_COMPONENTS = (
    DEFAULT_SMALL_ROOT
    / "spectral_transport_promotion_gate_promoted_replicates"
    / "spectral_transport_promotion_components.csv"
)
DEFAULT_PROMOTION_SUMMARY = (
    DEFAULT_SMALL_ROOT
    / "spectral_transport_promotion_gate_promoted_replicates"
    / "spectral_transport_promotion_summary.csv"
)
DEFAULT_BANDWIDTH_SUMMARY = (
    DEFAULT_SMALL_ROOT
    / "selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates"
    / "selected_neighborhood_pvalue_interpolation_summary.csv"
)
DEFAULT_BANDWIDTH_TAU_SENSITIVITY = (
    DEFAULT_SMALL_ROOT
    / "selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates"
    / "selected_neighborhood_pvalue_interpolation_tau_s_sensitivity.csv"
)
DEFAULT_OLD_STACK_BEHAVIOR = DEFAULT_OLD_STACK_ROOT / "method_stack_behavior_summary.csv"
DEFAULT_OLD_STACK_OVERLAP = DEFAULT_OLD_STACK_ROOT / "method_stack_pairwise_overlap.csv"

ROWS_OUTPUT = "spectral_vs_bandwidth_tradeoff_rows.csv"
SUMMARY_OUTPUT = "spectral_vs_bandwidth_tradeoff_summary.csv"
MANIFEST_OUTPUT = "manifest.json"


@dataclass(frozen=True)
class SpectralVsBandwidthTradeoffConfig:
    """Input paths and comparison knobs for the tradeoff panel."""

    output_dir: Path
    selected_family_rows_path: Path = DEFAULT_SELECTED_FAMILY_ROWS
    promotion_components_path: Path = DEFAULT_PROMOTION_COMPONENTS
    promotion_summary_path: Path = DEFAULT_PROMOTION_SUMMARY
    bandwidth_summary_path: Path = DEFAULT_BANDWIDTH_SUMMARY
    bandwidth_tau_sensitivity_path: Path = DEFAULT_BANDWIDTH_TAU_SENSITIVITY
    old_stack_behavior_path: Path = DEFAULT_OLD_STACK_BEHAVIOR
    old_stack_overlap_path: Path = DEFAULT_OLD_STACK_OVERLAP
    baseline_profile: str = BASELINE_PROFILE
    candidate_profile: str = CANDIDATE_PROFILE
    bandwidth_method_id: str = BANDWIDTH_METHOD
    reference_tau_s: float = REFERENCE_TAU_S


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-family-rows-path",
        type=Path,
        default=DEFAULT_SELECTED_FAMILY_ROWS,
    )
    parser.add_argument(
        "--promotion-components-path",
        type=Path,
        default=DEFAULT_PROMOTION_COMPONENTS,
    )
    parser.add_argument(
        "--promotion-summary-path",
        type=Path,
        default=DEFAULT_PROMOTION_SUMMARY,
    )
    parser.add_argument(
        "--bandwidth-summary-path",
        type=Path,
        default=DEFAULT_BANDWIDTH_SUMMARY,
    )
    parser.add_argument(
        "--bandwidth-tau-sensitivity-path",
        type=Path,
        default=DEFAULT_BANDWIDTH_TAU_SENSITIVITY,
    )
    parser.add_argument(
        "--old-stack-behavior-path",
        type=Path,
        default=DEFAULT_OLD_STACK_BEHAVIOR,
    )
    parser.add_argument(
        "--old-stack-overlap-path",
        type=Path,
        default=DEFAULT_OLD_STACK_OVERLAP,
    )
    parser.add_argument("--baseline-profile", default=BASELINE_PROFILE)
    parser.add_argument("--candidate-profile", default=CANDIDATE_PROFILE)
    parser.add_argument("--bandwidth-method-id", default=BANDWIDTH_METHOD)
    parser.add_argument("--reference-tau-s", type=float, default=REFERENCE_TAU_S)
    return parser.parse_args()


def _json_default(value: object) -> object:
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


def _to_bool_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)
    normalized = values.fillna(False)
    if normalized.dtype == bool:
        return normalized.astype(bool)
    return normalized.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _finite_sum(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.sum()) if not numeric.empty else math.nan


def _finite_value(values: pd.Series, reducer: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return math.nan
    if reducer == "min":
        return float(numeric.min())
    if reducer == "mean":
        return float(numeric.mean())
    if reducer == "median":
        return float(numeric.median())
    raise ValueError(f"unknown reducer {reducer!r}.")


def _fraction(numerator: float, denominator: float) -> float:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return math.nan
    if float(denominator) == 0.0:
        return math.nan
    return float(numerator) / float(denominator)


def _metric_row(
    *,
    method_family: str,
    evidence_level: str,
    metric_id: str,
    metric_value: float = math.nan,
    metric_denominator: float = math.nan,
    metric_fraction: float = math.nan,
    data_role: str = "",
    threshold: float = math.nan,
    status: str = "",
    interpretation: str = "",
    source_path: Path | str = "",
    metric_text: str = "",
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "method_family": method_family,
        "evidence_level": evidence_level,
        "data_role": data_role,
        "metric_id": metric_id,
        "metric_value": metric_value,
        "metric_denominator": metric_denominator,
        "metric_fraction": metric_fraction,
        "threshold": threshold,
        "status": status,
        "interpretation": interpretation,
        "source_path": str(source_path),
        "metric_text": metric_text,
    }


def _paired_selected_family_rows(
    rows: pd.DataFrame,
    *,
    baseline_profile: str,
    candidate_profile: str,
    data_role: str,
) -> pd.DataFrame:
    required = {"case_id", "data_role", "method_id", "replicate", "ari", "false_split"}
    _require_columns(rows, required, "selected-family rows")
    subset = rows[rows["data_role"].astype(str).eq(str(data_role))].copy()
    baseline = subset[subset["method_id"].astype(str).eq(str(baseline_profile))].copy()
    candidate = subset[subset["method_id"].astype(str).eq(str(candidate_profile))].copy()
    return baseline.merge(
        candidate,
        on=["case_id", "data_role", "replicate"],
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )


def build_spectral_rows(
    *,
    selected_family_rows: pd.DataFrame,
    promotion_components: pd.DataFrame,
    promotion_summary: pd.DataFrame,
    config: SpectralVsBandwidthTradeoffConfig,
) -> list[dict[str, object]]:
    """Build metric rows for strict MP spectral transport."""
    null_paired = _paired_selected_family_rows(
        selected_family_rows,
        baseline_profile=config.baseline_profile,
        candidate_profile=config.candidate_profile,
        data_role="selected_null",
    )
    signal_paired = _paired_selected_family_rows(
        selected_family_rows,
        baseline_profile=config.baseline_profile,
        candidate_profile=config.candidate_profile,
        data_role="signal",
    )

    baseline_false = _to_bool_series(null_paired["false_split_baseline"])
    candidate_false = _to_bool_series(null_paired["false_split_candidate"])
    baseline_false_count = int(baseline_false.sum())
    candidate_false_count = int(candidate_false.sum())
    false_split_reduction = baseline_false_count - candidate_false_count

    delta_ari = pd.to_numeric(signal_paired["ari_candidate"], errors="coerce") - pd.to_numeric(
        signal_paired["ari_baseline"], errors="coerce"
    )
    signal_regression_count = int((delta_ari < -1e-12).sum())
    min_delta_ari = _finite_value(delta_ari, "min")
    mean_delta_ari = _finite_value(delta_ari, "mean")

    promotion_decision = ""
    blocking_components = ""
    if not promotion_summary.empty:
        first_summary = promotion_summary.iloc[0]
        promotion_decision = str(first_summary.get("promotion_decision", ""))
        blocking_components = str(first_summary.get("blocking_component_ids", ""))

    rows = [
        _metric_row(
            method_family="spectral_transport_strict_mp",
            evidence_level="selected_family_replicate_traversal",
            data_role="selected_null",
            metric_id="baseline_false_split_count",
            metric_value=float(baseline_false_count),
            metric_denominator=float(len(null_paired)),
            metric_fraction=_fraction(baseline_false_count, len(null_paired)),
            status="baseline",
            interpretation="refined baseline selected-null oversplitting",
            source_path=config.selected_family_rows_path,
        ),
        _metric_row(
            method_family="spectral_transport_strict_mp",
            evidence_level="selected_family_replicate_traversal",
            data_role="selected_null",
            metric_id="candidate_false_split_count",
            metric_value=float(candidate_false_count),
            metric_denominator=float(len(null_paired)),
            metric_fraction=_fraction(candidate_false_count, len(null_paired)),
            status="candidate",
            interpretation="strict MP spectral support remaining false splits",
            source_path=config.selected_family_rows_path,
        ),
        _metric_row(
            method_family="spectral_transport_strict_mp",
            evidence_level="selected_family_replicate_traversal",
            data_role="selected_null",
            metric_id="false_split_reduction",
            metric_value=float(false_split_reduction),
            metric_denominator=float(baseline_false_count),
            metric_fraction=_fraction(false_split_reduction, baseline_false_count),
            status="candidate_vs_baseline",
            interpretation="false splits removed by strict MP support",
            source_path=config.selected_family_rows_path,
        ),
        _metric_row(
            method_family="spectral_transport_strict_mp",
            evidence_level="selected_family_replicate_traversal",
            data_role="signal",
            metric_id="signal_regression_count",
            metric_value=float(signal_regression_count),
            metric_denominator=float(len(signal_paired)),
            metric_fraction=_fraction(signal_regression_count, len(signal_paired)),
            status="promotion_blocker" if signal_regression_count else "passes",
            interpretation="paired signal rows with lower ARI under strict MP support",
            source_path=config.selected_family_rows_path,
        ),
        _metric_row(
            method_family="spectral_transport_strict_mp",
            evidence_level="selected_family_replicate_traversal",
            data_role="signal",
            metric_id="min_delta_ari_candidate_minus_baseline",
            metric_value=min_delta_ari,
            status="promotion_blocker" if min_delta_ari < 0.0 else "passes",
            interpretation="worst paired signal ARI delta under strict MP support",
            source_path=config.selected_family_rows_path,
        ),
        _metric_row(
            method_family="spectral_transport_strict_mp",
            evidence_level="selected_family_replicate_traversal",
            data_role="signal",
            metric_id="mean_delta_ari_candidate_minus_baseline",
            metric_value=mean_delta_ari,
            status="diagnostic",
            interpretation="mean paired signal ARI delta under strict MP support",
            source_path=config.selected_family_rows_path,
        ),
        _metric_row(
            method_family="spectral_transport_strict_mp",
            evidence_level="promotion_gate",
            metric_id="promotion_decision",
            status=promotion_decision,
            interpretation="promotion gate decision over traversal components",
            source_path=config.promotion_summary_path,
            metric_text=promotion_decision,
        ),
        _metric_row(
            method_family="spectral_transport_strict_mp",
            evidence_level="promotion_gate",
            metric_id="blocking_component_ids",
            status="blocked" if blocking_components else "passes",
            interpretation="components blocking default promotion",
            source_path=config.promotion_summary_path,
            metric_text=blocking_components,
        ),
    ]

    if not promotion_components.empty:
        _require_columns(
            promotion_components,
            {"component_id", "component_status"},
            "promotion components",
        )
        for _, component in promotion_components.iterrows():
            rows.append(
                _metric_row(
                    method_family="spectral_transport_strict_mp",
                    evidence_level="promotion_gate_component",
                    metric_id=str(component["component_id"]),
                    status=str(component["component_status"]),
                    interpretation=str(component.get("failure_reason", "")),
                    source_path=config.promotion_components_path,
                    metric_text=str(component["component_status"]),
                )
            )
    return rows


def build_bandwidth_rows(
    *,
    bandwidth_summary: pd.DataFrame,
    tau_s_sensitivity: pd.DataFrame,
    config: SpectralVsBandwidthTradeoffConfig,
) -> list[dict[str, object]]:
    """Build metric rows for the older bandwidth interpolation diagnostic."""
    _require_columns(
        bandwidth_summary,
        {
            "data_role",
            "method_id",
            "direct_significant_count",
            "interpolated_significant_count",
            "signal_miss_count",
            "signal_catch_count",
            "median_best_case_required_tau_s_for_alpha",
            "behavior_label",
        },
        "bandwidth summary",
    )
    summary = bandwidth_summary[
        bandwidth_summary["method_id"].astype(str).eq(str(config.bandwidth_method_id))
    ].copy()
    if summary.empty:
        raise ValueError(
            f"bandwidth summary has no rows for method_id={config.bandwidth_method_id!r}."
        )

    selected_null = summary[summary["data_role"].astype(str).eq("selected_null")]
    signal = summary[summary["data_role"].astype(str).eq("signal")]

    null_direct = _finite_sum(selected_null["direct_significant_count"])
    null_interpolated = _finite_sum(selected_null["interpolated_significant_count"])
    signal_direct = _finite_sum(signal["direct_significant_count"])
    signal_interpolated = _finite_sum(signal["interpolated_significant_count"])
    signal_miss = _finite_sum(signal["signal_miss_count"])
    signal_catch = _finite_sum(signal["signal_catch_count"])

    false_suppressed = selected_null[
        selected_null["behavior_label"].astype(str).eq("selected_null_false_signal_suppressed")
    ]
    signal_missed = signal[
        signal["behavior_label"].astype(str).eq("signal_not_caught_by_interpolation")
    ]

    tau_rows = _tau_s_reference_rows(
        tau_s_sensitivity=tau_s_sensitivity,
        method_id=config.bandwidth_method_id,
        reference_tau_s=config.reference_tau_s,
    )
    null_tau = tau_rows.get("selected_null", {})
    signal_tau = tau_rows.get("signal", {})

    return [
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="candidate_pvalue_interpolation_default",
            data_role="selected_null",
            metric_id="direct_significant_selected_null_count",
            metric_value=null_direct,
            status="measured_direct_pvalues",
            interpretation="selected-null rows directly significant before interpolation",
            source_path=config.bandwidth_summary_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="candidate_pvalue_interpolation_default",
            data_role="selected_null",
            metric_id="interpolated_significant_selected_null_count",
            metric_value=null_interpolated,
            metric_denominator=null_direct,
            metric_fraction=_fraction(null_interpolated, null_direct),
            status="conservative_smoother",
            interpretation="default bandwidth does not reopen selected-null rows",
            source_path=config.bandwidth_summary_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="candidate_pvalue_interpolation_default",
            data_role="selected_null",
            metric_id="suppressed_direct_selected_null_positive_count",
            metric_value=_finite_sum(false_suppressed["direct_significant_count"]),
            status="conservative_smoother",
            interpretation="direct selected-null positives made nonsignificant",
            source_path=config.bandwidth_summary_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="candidate_pvalue_interpolation_default",
            data_role="signal",
            metric_id="direct_significant_signal_count",
            metric_value=signal_direct,
            status="measured_direct_pvalues",
            interpretation="signal rows directly significant before interpolation",
            source_path=config.bandwidth_summary_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="candidate_pvalue_interpolation_default",
            data_role="signal",
            metric_id="interpolated_significant_signal_count",
            metric_value=signal_interpolated,
            metric_denominator=signal_direct,
            metric_fraction=_fraction(signal_interpolated, signal_direct),
            status="misses_signal",
            interpretation="default bandwidth does not catch direct signal positives",
            source_path=config.bandwidth_summary_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="candidate_pvalue_interpolation_default",
            data_role="signal",
            metric_id="signal_miss_count",
            metric_value=signal_miss,
            metric_denominator=signal_direct,
            metric_fraction=_fraction(signal_miss, signal_direct),
            status="misses_signal",
            interpretation="direct signal positives made nonsignificant",
            source_path=config.bandwidth_summary_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="candidate_pvalue_interpolation_default",
            data_role="signal",
            metric_id="signal_catch_count",
            metric_value=signal_catch,
            metric_denominator=signal_direct,
            metric_fraction=_fraction(signal_catch, signal_direct),
            status="no_rescue_at_default_tau",
            interpretation="direct signal positives caught by default interpolation",
            source_path=config.bandwidth_summary_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="tau_s_sensitivity",
            data_role="selected_null",
            metric_id="reference_tau_s_reopened_fraction",
            metric_value=float(null_tau.get("best_case_significant_count", math.nan)),
            metric_denominator=float(null_tau.get("direct_significant_count", math.nan)),
            metric_fraction=float(null_tau.get("best_case_significant_fraction", math.nan)),
            threshold=float(config.reference_tau_s),
            status="reopens_selected_null",
            interpretation="best-case selected-null reopening when tau_s is widened",
            source_path=config.bandwidth_tau_sensitivity_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="tau_s_sensitivity",
            data_role="signal",
            metric_id="reference_tau_s_signal_recovered_fraction",
            metric_value=float(signal_tau.get("best_case_significant_count", math.nan)),
            metric_denominator=float(signal_tau.get("direct_significant_count", math.nan)),
            metric_fraction=float(signal_tau.get("best_case_significant_fraction", math.nan)),
            threshold=float(config.reference_tau_s),
            status="partial_signal_recovery",
            interpretation="best-case signal recovery when tau_s is widened",
            source_path=config.bandwidth_tau_sensitivity_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="candidate_pvalue_interpolation_default",
            data_role="selected_null",
            metric_id="median_required_tau_s_selected_null_direct_positive",
            metric_value=_finite_value(
                false_suppressed["median_best_case_required_tau_s_for_alpha"],
                "median",
            ),
            status="locality_bottleneck",
            interpretation="optimistic tau_s needed for selected-null positives",
            source_path=config.bandwidth_summary_path,
        ),
        _metric_row(
            method_family="old_bandwidth_interpolation",
            evidence_level="candidate_pvalue_interpolation_default",
            data_role="signal",
            metric_id="median_required_tau_s_signal_direct_positive",
            metric_value=_finite_value(
                signal_missed["median_best_case_required_tau_s_for_alpha"],
                "median",
            ),
            status="locality_bottleneck",
            interpretation="optimistic tau_s needed for signal positives",
            source_path=config.bandwidth_summary_path,
        ),
    ]


def _tau_s_reference_rows(
    *,
    tau_s_sensitivity: pd.DataFrame,
    method_id: str,
    reference_tau_s: float,
) -> dict[str, dict[str, object]]:
    _require_columns(
        tau_s_sensitivity,
        {
            "data_role",
            "method_id",
            "tau_s_threshold",
            "direct_significant_count",
            "best_case_significant_count",
            "best_case_significant_fraction",
        },
        "tau_s sensitivity",
    )
    subset = tau_s_sensitivity[
        tau_s_sensitivity["method_id"].astype(str).eq(str(method_id))
        & np.isclose(
            pd.to_numeric(tau_s_sensitivity["tau_s_threshold"], errors="coerce"),
            float(reference_tau_s),
        )
    ]
    out: dict[str, dict[str, object]] = {}
    for _, row in subset.iterrows():
        out[str(row["data_role"])] = row.to_dict()
    return out


def build_old_stack_rows(
    *,
    old_stack_behavior: pd.DataFrame,
    old_stack_overlap: pd.DataFrame,
    config: SpectralVsBandwidthTradeoffConfig,
) -> list[dict[str, object]]:
    """Build full-Julia fragmentation rows from old/current stack summaries."""
    _require_columns(
        old_stack_behavior,
        {"method", "n_samples", "n_clusters", "n_singleton_clusters"},
        "old stack behavior",
    )
    rows: list[dict[str, object]] = []
    for _, behavior in old_stack_behavior.iterrows():
        method = str(behavior["method"])
        family = (
            "prior_full_julia_old_stack"
            if method.startswith("prior_full_julia")
            else "current_conditional_topology"
        )
        n_samples = float(behavior["n_samples"])
        n_clusters = float(behavior["n_clusters"])
        n_singletons = float(behavior["n_singleton_clusters"])
        rows.extend(
            [
                _metric_row(
                    method_family=family,
                    evidence_level="full_julia_clustering_summary",
                    metric_id="n_clusters",
                    metric_value=n_clusters,
                    metric_denominator=n_samples,
                    metric_fraction=_fraction(n_clusters, n_samples),
                    status="fragmentation_audit",
                    interpretation=method,
                    source_path=config.old_stack_behavior_path,
                ),
                _metric_row(
                    method_family=family,
                    evidence_level="full_julia_clustering_summary",
                    metric_id="n_singleton_clusters",
                    metric_value=n_singletons,
                    metric_denominator=n_clusters,
                    metric_fraction=_fraction(n_singletons, n_clusters),
                    status="fragmentation_audit",
                    interpretation=method,
                    source_path=config.old_stack_behavior_path,
                ),
            ]
        )

    if not old_stack_overlap.empty:
        _require_columns(
            old_stack_overlap,
            {"adjusted_rand_index", "normalized_mutual_information"},
            "old stack pairwise overlap",
        )
        first = old_stack_overlap.iloc[0]
        rows.extend(
            [
                _metric_row(
                    method_family="old_current_full_julia_overlap",
                    evidence_level="full_julia_pairwise_overlap",
                    metric_id="adjusted_rand_index",
                    metric_value=float(first["adjusted_rand_index"]),
                    status="partition_overlap",
                    interpretation="prior full TBS stack versus current topology diagnostic",
                    source_path=config.old_stack_overlap_path,
                ),
                _metric_row(
                    method_family="old_current_full_julia_overlap",
                    evidence_level="full_julia_pairwise_overlap",
                    metric_id="normalized_mutual_information",
                    metric_value=float(first["normalized_mutual_information"]),
                    status="partition_overlap",
                    interpretation="prior full TBS stack versus current topology diagnostic",
                    source_path=config.old_stack_overlap_path,
                ),
            ]
        )
    return rows


def build_tradeoff_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize the row metrics into one comparison decision."""

    def value(method_family: str, metric_id: str) -> float:
        found = rows[
            rows["method_family"].astype(str).eq(method_family)
            & rows["metric_id"].astype(str).eq(metric_id)
        ]
        if found.empty:
            return math.nan
        return float(found.iloc[0]["metric_value"])

    def fraction(method_family: str, metric_id: str) -> float:
        found = rows[
            rows["method_family"].astype(str).eq(method_family)
            & rows["metric_id"].astype(str).eq(metric_id)
        ]
        if found.empty:
            return math.nan
        return float(found.iloc[0]["metric_fraction"])

    def text(method_family: str, metric_id: str) -> str:
        found = rows[
            rows["method_family"].astype(str).eq(method_family)
            & rows["metric_id"].astype(str).eq(metric_id)
        ]
        if found.empty:
            return ""
        return str(found.iloc[0]["metric_text"])

    spectral_regressions = value(
        "spectral_transport_strict_mp",
        "signal_regression_count",
    )
    tau_null_fraction = fraction(
        "old_bandwidth_interpolation",
        "reference_tau_s_reopened_fraction",
    )
    tau_signal_fraction = fraction(
        "old_bandwidth_interpolation",
        "reference_tau_s_signal_recovered_fraction",
    )
    comparison_status = "hybrid_needed_diagnostic_only"
    if spectral_regressions == 0 and tau_null_fraction <= tau_signal_fraction:
        comparison_status = "spectral_candidate_ready_bandwidth_secondary"

    record = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "comparison_id": "spectral_transport_vs_old_bandwidth_v1",
        "comparison_status": comparison_status,
        "spectral_promotion_decision": text(
            "spectral_transport_strict_mp",
            "promotion_decision",
        ),
        "spectral_blocking_component_ids": text(
            "spectral_transport_strict_mp",
            "blocking_component_ids",
        ),
        "spectral_null_baseline_false_split_count": value(
            "spectral_transport_strict_mp",
            "baseline_false_split_count",
        ),
        "spectral_null_candidate_false_split_count": value(
            "spectral_transport_strict_mp",
            "candidate_false_split_count",
        ),
        "spectral_null_false_split_reduction": value(
            "spectral_transport_strict_mp",
            "false_split_reduction",
        ),
        "spectral_signal_regression_count": spectral_regressions,
        "spectral_signal_regression_fraction": fraction(
            "spectral_transport_strict_mp",
            "signal_regression_count",
        ),
        "spectral_min_delta_ari": value(
            "spectral_transport_strict_mp",
            "min_delta_ari_candidate_minus_baseline",
        ),
        "bandwidth_default_selected_null_direct_significant_count": value(
            "old_bandwidth_interpolation",
            "direct_significant_selected_null_count",
        ),
        "bandwidth_default_selected_null_interpolated_significant_count": value(
            "old_bandwidth_interpolation",
            "interpolated_significant_selected_null_count",
        ),
        "bandwidth_default_signal_direct_significant_count": value(
            "old_bandwidth_interpolation",
            "direct_significant_signal_count",
        ),
        "bandwidth_default_signal_interpolated_significant_count": value(
            "old_bandwidth_interpolation",
            "interpolated_significant_signal_count",
        ),
        "bandwidth_default_signal_miss_count": value(
            "old_bandwidth_interpolation",
            "signal_miss_count",
        ),
        "bandwidth_reference_tau_s_selected_null_reopened_fraction": tau_null_fraction,
        "bandwidth_reference_tau_s_signal_recovered_fraction": tau_signal_fraction,
        "old_full_julia_prior_cluster_count": value(
            "prior_full_julia_old_stack",
            "n_clusters",
        ),
        "current_full_julia_cluster_count": value(
            "current_conditional_topology",
            "n_clusters",
        ),
        "old_current_full_julia_adjusted_rand_index": value(
            "old_current_full_julia_overlap",
            "adjusted_rand_index",
        ),
        "old_current_full_julia_normalized_mutual_information": value(
            "old_current_full_julia_overlap",
            "normalized_mutual_information",
        ),
        "recommended_next_step": (
            "use bandwidth interpolation as support-gated locality evidence and "
            "spectral transport as a fail-closed bottleneck guard; do not promote "
            "either as a standalone rescue rule"
        ),
    }
    return pd.DataFrame.from_records([record])


def evaluate_spectral_vs_bandwidth_tradeoff(
    config: SpectralVsBandwidthTradeoffConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all inputs and return comparison rows plus one-row summary."""
    selected_family_rows = pd.read_csv(config.selected_family_rows_path)
    promotion_components = pd.read_csv(config.promotion_components_path)
    promotion_summary = pd.read_csv(config.promotion_summary_path)
    bandwidth_summary = pd.read_csv(config.bandwidth_summary_path)
    tau_s_sensitivity = pd.read_csv(config.bandwidth_tau_sensitivity_path)
    old_stack_behavior = pd.read_csv(config.old_stack_behavior_path)
    old_stack_overlap = pd.read_csv(config.old_stack_overlap_path)

    records = []
    records.extend(
        build_spectral_rows(
            selected_family_rows=selected_family_rows,
            promotion_components=promotion_components,
            promotion_summary=promotion_summary,
            config=config,
        )
    )
    records.extend(
        build_bandwidth_rows(
            bandwidth_summary=bandwidth_summary,
            tau_s_sensitivity=tau_s_sensitivity,
            config=config,
        )
    )
    records.extend(
        build_old_stack_rows(
            old_stack_behavior=old_stack_behavior,
            old_stack_overlap=old_stack_overlap,
            config=config,
        )
    )
    rows = pd.DataFrame.from_records(records)
    summary = build_tradeoff_summary(rows)
    return rows, summary


def run_spectral_vs_bandwidth_tradeoff_panel(
    config: SpectralVsBandwidthTradeoffConfig,
) -> dict[str, Path]:
    """Run the panel and write rows, summary, and manifest."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate_spectral_vs_bandwidth_tradeoff(config)

    rows_path = config.output_dir / ROWS_OUTPUT
    summary_path = config.output_dir / SUMMARY_OUTPUT
    manifest_path = config.output_dir / MANIFEST_OUTPUT
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "selected_family_rows_path": config.selected_family_rows_path,
            "promotion_components_path": config.promotion_components_path,
            "promotion_summary_path": config.promotion_summary_path,
            "bandwidth_summary_path": config.bandwidth_summary_path,
            "bandwidth_tau_sensitivity_path": config.bandwidth_tau_sensitivity_path,
            "old_stack_behavior_path": config.old_stack_behavior_path,
            "old_stack_overlap_path": config.old_stack_overlap_path,
        },
        "parameters": {
            "baseline_profile": config.baseline_profile,
            "candidate_profile": config.candidate_profile,
            "bandwidth_method_id": config.bandwidth_method_id,
            "reference_tau_s": config.reference_tau_s,
        },
        "outputs": {
            "rows": rows_path,
            "summary": summary_path,
            "manifest": manifest_path,
        },
        "n_rows": int(len(rows)),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return {"rows": rows_path, "summary": summary_path, "manifest": manifest_path}


def main() -> None:
    args = parse_args()
    run_spectral_vs_bandwidth_tradeoff_panel(
        SpectralVsBandwidthTradeoffConfig(
            output_dir=args.output_dir,
            selected_family_rows_path=args.selected_family_rows_path,
            promotion_components_path=args.promotion_components_path,
            promotion_summary_path=args.promotion_summary_path,
            bandwidth_summary_path=args.bandwidth_summary_path,
            bandwidth_tau_sensitivity_path=args.bandwidth_tau_sensitivity_path,
            old_stack_behavior_path=args.old_stack_behavior_path,
            old_stack_overlap_path=args.old_stack_overlap_path,
            baseline_profile=args.baseline_profile,
            candidate_profile=args.candidate_profile,
            bandwidth_method_id=args.bandwidth_method_id,
            reference_tau_s=args.reference_tau_s,
        )
    )


if __name__ == "__main__":
    main()
