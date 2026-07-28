"""Run standard-dispatch overlap comparisons for spectral transport traversal.

This panel compares the refined global pass-through profile against the same
profile plus the fail-closed MP spectral transport pass-through guard. It uses
the benchmark registry and shared method dispatcher, so the result answers
whether the registered traversal method behaves as expected outside the
selected-family checkpoint-only diagnostics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.result_records.dataframe import benchmark_rows_to_dataframe
from benchmarks.shared.result_records.models import BenchmarkResultRow
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import run_single_method_once

SCHEMA_VERSION = "spectral_transport_overlap_dispatch_panel/v1"
STUDY_ROLE = "diagnostic_spectral_transport_overlap_dispatch_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.spectral_transport.spectral_transport_overlap_dispatch_panel"

BASELINE_METHOD = "tbs_global_passthrough_refined_diagnostic"
CANDIDATE_METHOD = "tbs_spectral_transport_passthrough"
DEFAULT_METHODS = (BASELINE_METHOD, CANDIDATE_METHOD)
DEFAULT_CASE_NAMES = (
    "overlap_part_4c_small",
    "overlap_mod_4c_small",
    "overlap_heavy_4c_small_feat",
)

ROW_OUTPUT = "spectral_transport_overlap_dispatch_rows.csv"
PAIRWISE_OUTPUT = "spectral_transport_overlap_dispatch_pairwise.csv"
SUMMARY_OUTPUT = "spectral_transport_overlap_dispatch_summary.csv"
MANIFEST_OUTPUT = "manifest.json"


@dataclass(frozen=True)
class SpectralTransportOverlapDispatchConfig:
    """Configuration for the registered-method overlap comparison panel."""

    output_dir: Path
    case_names: tuple[str, ...] = DEFAULT_CASE_NAMES
    methods: tuple[str, ...] = DEFAULT_METHODS
    baseline_method: str = BASELINE_METHOD
    candidate_method: str = CANDIDATE_METHOD
    significance_level: float = 0.01
    edge_alpha: float = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the registered spectral transport TBS method against the "
            "refined global pass-through TBS baseline on overlap cases."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        dest="case_names",
        action="append",
        default=None,
        help="Case name to include. Repeat to run more than one case.",
    )
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=None,
        help="Registered method id to run. Repeat to run more than one method.",
    )
    parser.add_argument("--baseline-method", default=BASELINE_METHOD)
    parser.add_argument("--candidate-method", default=CANDIDATE_METHOD)
    parser.add_argument("--significance-level", type=float, default=0.01)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def select_cases(case_names: tuple[str, ...]) -> list[dict[str, object]]:
    """Select benchmark cases in the requested order."""
    available = {str(case["name"]): case for case in get_default_test_cases()}
    missing = [case_name for case_name in case_names if case_name not in available]
    if missing:
        raise ValueError(
            f"Unknown case name(s): {missing!r}. Available examples: {sorted(available)[:10]!r}."
        )
    return [dict(available[case_name]) for case_name in case_names]


def _validate_methods(methods: tuple[str, ...]) -> None:
    missing = [method for method in methods if method not in METHOD_SPECS]
    if missing:
        raise ValueError(f"Unknown registered method id(s): {missing!r}.")
    multi_param_methods = [
        method for method in methods if len(METHOD_SPECS[method].param_grid) != 1
    ]
    if multi_param_methods:
        raise ValueError(
            "This diagnostic requires exactly one parameter set per method; "
            f"got multi-param methods {multi_param_methods!r}."
        )


def _spectral_annotation_counts(annotations: object) -> dict[str, float]:
    if not isinstance(annotations, pd.DataFrame):
        return {
            "spectral_transport_passthrough_supported_count": np.nan,
            "spectral_transport_passthrough_blocked_count": np.nan,
            "spectral_transport_bottleneck_count": np.nan,
            "spectral_transport_unmeasured_no_matched_mp_path_count": np.nan,
            "spectral_transport_supported_mp_mode_path_count": np.nan,
            "spectral_transport_mp_evidence_path_count": np.nan,
            "spectral_transport_max_path_cost": np.nan,
        }

    def _bool_sum(column: str) -> float:
        if column not in annotations.columns:
            return np.nan
        return float(annotations[column].fillna(False).astype(bool).sum())

    bottleneck_count = np.nan
    unmeasured_no_matched_mp_count = np.nan
    supported_mp_path_count = np.nan
    if "Spectral_Transport_Bottleneck" in annotations.columns:
        values = annotations["Spectral_Transport_Bottleneck"].fillna("").astype(str)
        bottleneck_count = float(values.eq("spectral_transport_bottleneck").sum())
        unmeasured_no_matched_mp_count = float(values.eq("unmeasured_no_matched_mp_path").sum())
        supported_mp_path_count = float(values.eq("supported_mp_mode_path").sum())

    mp_evidence_path_count = np.nan
    mp_evidence_column = "Spectral_Transport_Best_Descendant_Split_Path_Has_MP_Evidence"
    if mp_evidence_column in annotations.columns:
        mp_evidence_path_count = float(
            annotations[mp_evidence_column].fillna(False).astype(bool).sum()
        )

    max_path_cost = np.nan
    if "Spectral_Transport_Best_Descendant_Split_Path_Cost" in annotations.columns:
        costs = pd.to_numeric(
            annotations["Spectral_Transport_Best_Descendant_Split_Path_Cost"],
            errors="coerce",
        )
        finite = costs[np.isfinite(costs)]
        if not finite.empty:
            max_path_cost = float(finite.max())

    return {
        "spectral_transport_passthrough_supported_count": _bool_sum(
            "Spectral_Transport_Pass_Through_Supported"
        ),
        "spectral_transport_passthrough_blocked_count": _bool_sum(
            "Spectral_Transport_Pass_Through_Blocked"
        ),
        "spectral_transport_bottleneck_count": bottleneck_count,
        "spectral_transport_unmeasured_no_matched_mp_path_count": (unmeasured_no_matched_mp_count),
        "spectral_transport_supported_mp_mode_path_count": supported_mp_path_count,
        "spectral_transport_mp_evidence_path_count": mp_evidence_path_count,
        "spectral_transport_max_path_cost": max_path_cost,
    }


def _sibling_gate_profile_for_method(method_id: str) -> str:
    params = METHOD_SPECS[method_id].param_grid[0]
    return str(params.get("sibling_gate_profile", ""))


def run_dispatch_rows(
    config: SpectralTransportOverlapDispatchConfig,
) -> tuple[pd.DataFrame, dict[tuple[str, str], Any]]:
    """Run all configured case/method pairs and return row metrics."""
    _validate_methods(config.methods)
    cases = select_cases(config.case_names)
    benchmark_rows: list[BenchmarkResultRow] = []
    computed_by_case_method: dict[tuple[str, str], Any] = {}
    spectral_rows: list[dict[str, object]] = []

    for case_index, test_case in enumerate(cases, start=1):
        case_name = str(test_case["name"])
        inputs = prepare_case_inputs(test_case, list(config.methods))
        for method_id in config.methods:
            spec = METHOD_SPECS[method_id]
            params = dict(spec.param_grid[0])
            row, computed, _method_audit = run_single_method_once(
                method_id=method_id,
                spec=spec,
                params=params,
                case_idx=case_index,
                case_name=case_name,
                tc_seed=test_case["seed"],
                significance_level=config.significance_level,
                edge_alpha=config.edge_alpha,
                data_t=inputs.data,
                y_t=inputs.labels,
                x_original=inputs.original_features,
                meta=inputs.metadata,
                distance_matrix=inputs.distance_matrix,
                distance_condensed=inputs.distance_condensed,
                matrix_audit=False,
            )
            benchmark_rows.append(row)
            if computed is not None:
                computed_by_case_method[(case_name, method_id)] = computed
            spectral_counts = _spectral_annotation_counts(
                computed.annotations if computed is not None else None
            )
            spectral_rows.append(
                {
                    "case_id": case_name,
                    "method": method_id,
                    "sibling_gate_profile": _sibling_gate_profile_for_method(method_id),
                    **spectral_counts,
                }
            )

    rows = benchmark_rows_to_dataframe(benchmark_rows)
    if rows.empty:
        return rows, computed_by_case_method
    rows.insert(0, "schema_version", SCHEMA_VERSION)
    rows.insert(1, "study_role", STUDY_ROLE)
    spectral_df = pd.DataFrame(spectral_rows)
    rows = rows.merge(
        spectral_df,
        on=["case_id", "method"],
        how="left",
        validate="one_to_one",
    )
    return rows, computed_by_case_method


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _partition_metrics(left_labels: object, right_labels: object) -> dict[str, float]:
    if left_labels is None or right_labels is None:
        return {
            "partition_ari_between_methods": np.nan,
            "partition_nmi_between_methods": np.nan,
            "partition_ami_between_methods": np.nan,
            "partition_fowlkes_mallows_between_methods": np.nan,
        }
    left = np.asarray(left_labels)
    right = np.asarray(right_labels)
    if left.shape[0] != right.shape[0] or left.shape[0] == 0:
        return {
            "partition_ari_between_methods": np.nan,
            "partition_nmi_between_methods": np.nan,
            "partition_ami_between_methods": np.nan,
            "partition_fowlkes_mallows_between_methods": np.nan,
        }
    return {
        "partition_ari_between_methods": float(adjusted_rand_score(left, right)),
        "partition_nmi_between_methods": float(normalized_mutual_info_score(left, right)),
        "partition_ami_between_methods": float(adjusted_mutual_info_score(left, right)),
        "partition_fowlkes_mallows_between_methods": float(fowlkes_mallows_score(left, right)),
    }


def _guard_effect_class(
    *,
    baseline_status: str,
    candidate_status: str,
    delta_found_clusters: float,
    delta_ari: float,
    candidate_blocked_count: float,
) -> str:
    if baseline_status != "ok" or candidate_status != "ok":
        return "skip_or_missing"
    if _to_float(candidate_blocked_count) > 0.0 and delta_found_clusters < 0:
        if delta_ari >= -1e-12:
            return "spectral_block_less_fragmented_not_worse"
        return "spectral_block_less_fragmented_lower_ari"
    if delta_found_clusters < 0:
        return "candidate_less_fragmented_without_recorded_block"
    if delta_found_clusters > 0:
        return "candidate_more_fragmented"
    if abs(delta_ari) > 1e-12:
        return "same_cluster_count_different_quality"
    return "unchanged"


def build_pairwise_rows(
    rows: pd.DataFrame,
    computed_by_case_method: dict[tuple[str, str], Any],
    *,
    baseline_method: str = BASELINE_METHOD,
    candidate_method: str = CANDIDATE_METHOD,
) -> pd.DataFrame:
    """Compare baseline and candidate method rows case by case."""
    if rows.empty:
        return pd.DataFrame()
    row_by_case_method = {
        (str(row.case_id), str(row.method)): row for row in rows.itertuples(index=False)
    }
    case_ids = sorted(
        {
            case_id
            for case_id, method in row_by_case_method
            if method == baseline_method and (case_id, candidate_method) in row_by_case_method
        }
    )

    pairwise_rows = []
    for case_id in case_ids:
        baseline = row_by_case_method[(case_id, baseline_method)]
        candidate = row_by_case_method[(case_id, candidate_method)]
        baseline_computed = computed_by_case_method.get((case_id, baseline_method))
        candidate_computed = computed_by_case_method.get((case_id, candidate_method))
        partition = _partition_metrics(
            getattr(baseline_computed, "labels", None),
            getattr(candidate_computed, "labels", None),
        )
        delta_found_clusters = _to_float(candidate.found_clusters) - _to_float(
            baseline.found_clusters
        )
        delta_ari = _to_float(candidate.ari) - _to_float(baseline.ari)
        candidate_blocked_count = _to_float(candidate.spectral_transport_passthrough_blocked_count)
        pairwise_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": case_id,
                "baseline_method": baseline_method,
                "candidate_method": candidate_method,
                "baseline_status": baseline.status,
                "candidate_status": candidate.status,
                "true_clusters": baseline.true_clusters,
                "baseline_found_clusters": baseline.found_clusters,
                "candidate_found_clusters": candidate.found_clusters,
                "delta_found_clusters_candidate_minus_baseline": delta_found_clusters,
                "baseline_ari": baseline.ari,
                "candidate_ari": candidate.ari,
                "delta_ari_candidate_minus_baseline": delta_ari,
                "baseline_nmi": baseline.nmi,
                "candidate_nmi": candidate.nmi,
                "delta_nmi_candidate_minus_baseline": _to_float(candidate.nmi)
                - _to_float(baseline.nmi),
                "baseline_ami": baseline.ami,
                "candidate_ami": candidate.ami,
                "delta_ami_candidate_minus_baseline": _to_float(candidate.ami)
                - _to_float(baseline.ami),
                "baseline_singleton_fraction": baseline.singleton_fraction,
                "candidate_singleton_fraction": candidate.singleton_fraction,
                "delta_singleton_fraction_candidate_minus_baseline": _to_float(
                    candidate.singleton_fraction
                )
                - _to_float(baseline.singleton_fraction),
                "baseline_effective_cluster_count": baseline.effective_cluster_count,
                "candidate_effective_cluster_count": candidate.effective_cluster_count,
                "delta_effective_cluster_count_candidate_minus_baseline": _to_float(
                    candidate.effective_cluster_count
                )
                - _to_float(baseline.effective_cluster_count),
                "baseline_silhouette_score": baseline.silhouette_score,
                "candidate_silhouette_score": candidate.silhouette_score,
                "delta_silhouette_candidate_minus_baseline": _to_float(candidate.silhouette_score)
                - _to_float(baseline.silhouette_score),
                "baseline_davies_bouldin_index": baseline.davies_bouldin_index,
                "candidate_davies_bouldin_index": candidate.davies_bouldin_index,
                "delta_davies_bouldin_candidate_minus_baseline": _to_float(
                    candidate.davies_bouldin_index
                )
                - _to_float(baseline.davies_bouldin_index),
                "baseline_calinski_harabasz_index": baseline.calinski_harabasz_index,
                "candidate_calinski_harabasz_index": candidate.calinski_harabasz_index,
                "delta_calinski_harabasz_candidate_minus_baseline": _to_float(
                    candidate.calinski_harabasz_index
                )
                - _to_float(baseline.calinski_harabasz_index),
                "candidate_spectral_transport_blocked_count": candidate_blocked_count,
                "candidate_spectral_transport_bottleneck_count": _to_float(
                    candidate.spectral_transport_bottleneck_count
                ),
                "guard_effect_class": _guard_effect_class(
                    baseline_status=str(baseline.status),
                    candidate_status=str(candidate.status),
                    delta_found_clusters=delta_found_clusters,
                    delta_ari=delta_ari,
                    candidate_blocked_count=candidate_blocked_count,
                ),
                **partition,
            }
        )
    return pd.DataFrame(pairwise_rows)


def build_summary_rows(rows: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    """Summarize method-level and pairwise overlap behavior."""
    summary_rows: list[dict[str, object]] = []
    if not rows.empty:
        metric_columns = [
            "ari",
            "nmi",
            "ami",
            "found_clusters",
            "cluster_count_abs_error",
            "singleton_fraction",
            "effective_cluster_count",
            "silhouette_score",
            "davies_bouldin_index",
            "calinski_harabasz_index",
            "spectral_transport_passthrough_blocked_count",
        ]
        for method, group in rows.groupby("method", dropna=False):
            row: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "summary_family": "per_method",
                "method": method,
                "case_count": int(group["case_id"].nunique()),
                "ok_count": int(group["status"].eq("ok").sum()),
                "skip_count": int(group["status"].ne("ok").sum()),
            }
            ok = group[group["status"].eq("ok")]
            for column in metric_columns:
                values = pd.to_numeric(ok[column], errors="coerce")
                row[f"mean_{column}"] = float(values.mean())
                row[f"median_{column}"] = float(values.median())
            summary_rows.append(row)

    if not pairwise.empty:
        pairwise_ok = pairwise["baseline_status"].eq("ok") & pairwise["candidate_status"].eq("ok")
        baseline_method = str(pairwise["baseline_method"].iloc[0])
        candidate_method = str(pairwise["candidate_method"].iloc[0])
        metric_columns = [
            "delta_found_clusters_candidate_minus_baseline",
            "delta_ari_candidate_minus_baseline",
            "delta_nmi_candidate_minus_baseline",
            "delta_ami_candidate_minus_baseline",
            "delta_singleton_fraction_candidate_minus_baseline",
            "delta_effective_cluster_count_candidate_minus_baseline",
            "delta_silhouette_candidate_minus_baseline",
            "delta_davies_bouldin_candidate_minus_baseline",
            "delta_calinski_harabasz_candidate_minus_baseline",
            "partition_ari_between_methods",
            "candidate_spectral_transport_blocked_count",
        ]
        row = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "summary_family": "pairwise_candidate_minus_baseline",
            "method": f"{candidate_method}_vs_{baseline_method}",
            "case_count": int(pairwise["case_id"].nunique()),
            "ok_count": int(pairwise_ok.sum()),
            "skip_count": int((~pairwise_ok).sum()),
            "n_candidate_less_fragmented": int(
                (
                    pd.to_numeric(
                        pairwise["delta_found_clusters_candidate_minus_baseline"],
                        errors="coerce",
                    )
                    < 0
                ).sum()
            ),
            "n_candidate_ari_not_worse": int(
                (
                    pd.to_numeric(
                        pairwise["delta_ari_candidate_minus_baseline"],
                        errors="coerce",
                    )
                    >= -1e-12
                ).sum()
            ),
            "dominant_guard_effect_class": str(
                pairwise["guard_effect_class"].value_counts().idxmax()
            ),
        }
        for column in metric_columns:
            values = pd.to_numeric(pairwise[column], errors="coerce")
            row[f"mean_{column}"] = float(values.mean())
            row[f"median_{column}"] = float(values.median())
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def run_spectral_transport_overlap_dispatch_panel(
    config: SpectralTransportOverlapDispatchConfig,
) -> dict[str, Path]:
    """Run the panel and write durable CSV/manifest outputs."""
    rows, computed = run_dispatch_rows(config)
    pairwise = build_pairwise_rows(
        rows,
        computed,
        baseline_method=config.baseline_method,
        candidate_method=config.candidate_method,
    )
    summary = build_summary_rows(rows, pairwise)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = config.output_dir / ROW_OUTPUT
    pairwise_path = config.output_dir / PAIRWISE_OUTPUT
    summary_path = config.output_dir / SUMMARY_OUTPUT
    manifest_path = config.output_dir / MANIFEST_OUTPUT

    rows.to_csv(rows_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "case_names": config.case_names,
        "methods": config.methods,
        "baseline_method": config.baseline_method,
        "candidate_method": config.candidate_method,
        "significance_level": config.significance_level,
        "edge_alpha": config.edge_alpha,
        "row_count": int(len(rows)),
        "pairwise_count": int(len(pairwise)),
        "summary_count": int(len(summary)),
        "outputs": {
            "rows": rows_path,
            "pairwise": pairwise_path,
            "summary": summary_path,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return {
        "rows": rows_path,
        "pairwise": pairwise_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def main() -> None:
    args = parse_args()
    methods = tuple(args.methods) if args.methods else DEFAULT_METHODS
    case_names = tuple(args.case_names) if args.case_names else DEFAULT_CASE_NAMES
    run_spectral_transport_overlap_dispatch_panel(
        SpectralTransportOverlapDispatchConfig(
            output_dir=args.output_dir,
            case_names=case_names,
            methods=methods,
            baseline_method=str(args.baseline_method),
            candidate_method=str(args.candidate_method),
            significance_level=float(args.significance_level),
            edge_alpha=float(args.edge_alpha),
        )
    )


if __name__ == "__main__":
    main()
