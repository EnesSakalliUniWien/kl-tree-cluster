"""Threshold calibration panel for spectral transport pass-through traversal.

The promotion gate says whether the current registered profile is promotable.
This panel asks the next calibration question: is there any spectral transport
max-cost threshold, on the same overlap evidence, that both preserves signal
rows and reduces selected-null false splitting?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel import (
    validate_data_roles,
)
from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)

SCHEMA_VERSION = "spectral_transport_threshold_calibration_panel/v1"
STUDY_ROLE = "diagnostic_spectral_transport_threshold_calibration_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "spectral_transport_threshold_calibration_panel"
)

NULL_OUTPUT_ROLE = "selected_null"
SIGNAL_OUTPUT_ROLE = "signal"
BASELINE_VARIANT = "refined_baseline"
SPECTRAL_VARIANT = "spectral_transport"

DEFAULT_CASE_NAMES = (
    "overlap_part_4c_small",
    "overlap_mod_4c_small",
    "overlap_heavy_4c_small_feat",
)
DEFAULT_THRESHOLDS = (0.75, 1.0, 1.2, 1.5)

ROWS_OUTPUT = "spectral_transport_threshold_rows.csv"
PAIRWISE_OUTPUT = "spectral_transport_threshold_pairwise.csv"
SUMMARY_OUTPUT = "spectral_transport_threshold_summary.csv"
MANIFEST_OUTPUT = "manifest.json"


@dataclass(frozen=True)
class SpectralTransportThresholdCalibrationConfig:
    """Configuration for the threshold calibration panel."""

    output_dir: Path
    suite: str = "binary"
    case_names: tuple[str, ...] = DEFAULT_CASE_NAMES
    data_roles: tuple[str, ...] = ("null", "signal")
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    base_seed: int = 20260613
    edge_alpha: float = 0.001
    sibling_alpha: float = 0.01
    max_allowed_signal_ari_drop: float = 0.0
    min_null_false_split_reduction: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names", type=parse_names, default=DEFAULT_CASE_NAMES)
    parser.add_argument("--data-roles", type=parse_names, default=("null", "signal"))
    parser.add_argument(
        "--thresholds",
        type=parse_names,
        default=tuple(str(value) for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--max-allowed-signal-ari-drop", type=float, default=0.0)
    parser.add_argument("--min-null-false-split-reduction", type=int, default=1)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _refined_passthrough_params(
    *,
    spectral_transport: bool,
    threshold: float | None = None,
) -> dict[str, object]:
    params: dict[str, object] = {
        "tree_distance_metric": "hamming",
        "tree_linkage_method": "average",
        "sibling_gate_method": "fixed_coordinate_bh",
        "sibling_gate_alpha_penalty": 50.0,
        "root_stability_guard_threshold": 0.24,
        "root_stability_subsample_replicates": 12,
        "root_stability_feature_fraction": 0.8,
        "root_stability_seed": 0,
        "root_selective_permutation_guard_replicates": 99,
        "root_selective_permutation_guard_seed": 0,
        "root_selective_permutation_guard_alpha": 0.01,
        "root_selective_permutation_guard_scope": (
            "global_sibling_min_passthrough_descendant_refined"
        ),
    }
    if spectral_transport:
        params["spectral_transport_passthrough_guard"] = True
        params["spectral_transport_require_mp_blocks"] = True
        if threshold is not None:
            params["spectral_transport_max_cost"] = float(threshold)
    return params


def _spectral_annotation_counts(annotations: object) -> dict[str, float]:
    if not isinstance(annotations, pd.DataFrame):
        return {
            "spectral_transport_blocked_count": np.nan,
            "spectral_transport_supported_count": np.nan,
            "spectral_transport_bottleneck_count": np.nan,
            "spectral_transport_supported_mp_mode_path_count": np.nan,
            "spectral_transport_unmeasured_no_matched_mp_path_count": np.nan,
        }

    def _bool_sum(column: str) -> float:
        if column not in annotations.columns:
            return np.nan
        return float(annotations[column].fillna(False).astype(bool).sum())

    bottleneck_count = np.nan
    supported_mp_count = np.nan
    unmeasured_count = np.nan
    if "Spectral_Transport_Bottleneck" in annotations.columns:
        values = annotations["Spectral_Transport_Bottleneck"].fillna("").astype(str)
        bottleneck_count = float(values.eq("spectral_transport_bottleneck").sum())
        supported_mp_count = float(values.eq("supported_mp_mode_path").sum())
        unmeasured_count = float(values.eq("unmeasured_no_matched_mp_path").sum())

    return {
        "spectral_transport_blocked_count": _bool_sum(
            "Spectral_Transport_Pass_Through_Blocked"
        ),
        "spectral_transport_supported_count": _bool_sum(
            "Spectral_Transport_Pass_Through_Supported"
        ),
        "spectral_transport_bottleneck_count": bottleneck_count,
        "spectral_transport_supported_mp_mode_path_count": supported_mp_count,
        "spectral_transport_unmeasured_no_matched_mp_path_count": unmeasured_count,
    }


def _data_role_label(raw_role: str) -> str:
    return NULL_OUTPUT_ROLE if raw_role == "null" else SIGNAL_OUTPUT_ROLE


def _run_one_variant(
    *,
    data: pd.DataFrame,
    feature_space: object,
    truth_labels: np.ndarray,
    true_clusters: int,
    distance: np.ndarray,
    case_id: str,
    data_role: str,
    variant_id: str,
    threshold: float | None,
    data_seed: int,
    edge_alpha: float,
    sibling_alpha: float,
) -> dict[str, object]:
    spectral = variant_id == SPECTRAL_VARIANT
    result = run_clustering_result(
        data,
        method_id="tbs",
        params=_refined_passthrough_params(
            spectral_transport=spectral,
            threshold=threshold,
        ),
        seed=data_seed,
        significance_level=float(sibling_alpha),
        edge_alpha=float(edge_alpha),
        distance_condensed=distance,
        feature_space=feature_space,  # type: ignore[arg-type]
    )
    labels = np.asarray(result.labels) if result.labels is not None else np.array([])
    ari = (
        float(adjusted_rand_score(np.asarray(truth_labels, dtype=int), labels))
        if result.status == "ok" and labels.size == len(truth_labels)
        else np.nan
    )
    found_clusters = int(result.found_clusters)
    counts = _spectral_annotation_counts(
        result.extra.get("annotations") if result.extra else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": case_id,
        "data_role": data_role,
        "variant_id": variant_id,
        "spectral_transport_max_cost": np.nan if threshold is None else float(threshold),
        "data_seed": int(data_seed),
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "true_clusters": int(true_clusters),
        "found_clusters": found_clusters,
        "ari": ari,
        "exact_cluster_count": bool(found_clusters == int(true_clusters)),
        "false_split": bool(
            result.status == "ok" and found_clusters > int(true_clusters)
        ),
        "status": result.status,
        "skip_reason": result.skip_reason or "",
        **counts,
    }


def run_threshold_rows(
    config: SpectralTransportThresholdCalibrationConfig,
) -> pd.DataFrame:
    """Run baseline and thresholded spectral variants."""
    roles = validate_data_roles(config.data_roles)
    cases = _select_cases(suite=config.suite, case_names=config.case_names)
    rows: list[dict[str, object]] = []
    for case in cases:
        (
            case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        for raw_role in roles:
            data_role = _data_role_label(raw_role)
            data, feature_space, truth_labels, true_clusters = _generate_data_with_truth(
                case=case,
                case_id=case_id,
                source_family=source_family,
                feature_representation=feature_representation,
                n_samples=n_samples,
                n_features=n_features,
                n_categories=n_categories,
                data_role=raw_role,
                seed=int(config.base_seed),
            )
            distance = pdist(data.to_numpy(dtype=float), metric="hamming")
            rows.append(
                _run_one_variant(
                    data=data,
                    feature_space=feature_space,
                    truth_labels=np.asarray(truth_labels, dtype=int),
                    true_clusters=int(true_clusters),
                    distance=distance,
                    case_id=case_id,
                    data_role=data_role,
                    variant_id=BASELINE_VARIANT,
                    threshold=None,
                    data_seed=int(config.base_seed),
                    edge_alpha=float(config.edge_alpha),
                    sibling_alpha=float(config.sibling_alpha),
                )
            )
            for threshold in config.thresholds:
                rows.append(
                    _run_one_variant(
                        data=data,
                        feature_space=feature_space,
                        truth_labels=np.asarray(truth_labels, dtype=int),
                        true_clusters=int(true_clusters),
                        distance=distance,
                        case_id=case_id,
                        data_role=data_role,
                        variant_id=SPECTRAL_VARIANT,
                        threshold=float(threshold),
                        data_seed=int(config.base_seed),
                        edge_alpha=float(config.edge_alpha),
                        sibling_alpha=float(config.sibling_alpha),
                    )
                )
    return pd.DataFrame.from_records(rows)


def _to_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def build_threshold_pairwise_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Pair each spectral threshold with its per-case baseline."""
    required = {
        "case_id",
        "data_role",
        "variant_id",
        "spectral_transport_max_cost",
        "found_clusters",
        "ari",
        "false_split",
        "status",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"threshold rows missing columns: {sorted(missing)!r}.")
    baseline = rows[rows["variant_id"].astype(str).eq(BASELINE_VARIANT)].copy()
    spectral = rows[rows["variant_id"].astype(str).eq(SPECTRAL_VARIANT)].copy()
    paired = baseline.merge(
        spectral,
        on=["case_id", "data_role"],
        suffixes=("_baseline", "_candidate"),
        how="inner",
        validate="one_to_many",
    )
    out = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": paired["case_id"].astype(str),
            "data_role": paired["data_role"].astype(str),
            "spectral_transport_max_cost": pd.to_numeric(
                paired["spectral_transport_max_cost_candidate"],
                errors="coerce",
            ),
            "baseline_status": paired["status_baseline"].astype(str),
            "candidate_status": paired["status_candidate"].astype(str),
            "baseline_found_clusters": pd.to_numeric(
                paired["found_clusters_baseline"],
                errors="coerce",
            ),
            "candidate_found_clusters": pd.to_numeric(
                paired["found_clusters_candidate"],
                errors="coerce",
            ),
            "delta_found_clusters_candidate_minus_baseline": pd.to_numeric(
                paired["found_clusters_candidate"],
                errors="coerce",
            )
            - pd.to_numeric(paired["found_clusters_baseline"], errors="coerce"),
            "baseline_ari": pd.to_numeric(paired["ari_baseline"], errors="coerce"),
            "candidate_ari": pd.to_numeric(paired["ari_candidate"], errors="coerce"),
            "delta_ari_candidate_minus_baseline": pd.to_numeric(
                paired["ari_candidate"],
                errors="coerce",
            )
            - pd.to_numeric(paired["ari_baseline"], errors="coerce"),
            "baseline_false_split": paired["false_split_baseline"].map(_to_bool),
            "candidate_false_split": paired["false_split_candidate"].map(_to_bool),
            "candidate_spectral_transport_blocked_count": pd.to_numeric(
                paired["spectral_transport_blocked_count_candidate"],
                errors="coerce",
            ),
        }
    )
    out["false_split_reduced"] = (
        out["baseline_false_split"].astype(bool)
        & ~out["candidate_false_split"].astype(bool)
    )
    return out.sort_values(["spectral_transport_max_cost", "case_id", "data_role"])


def summarize_thresholds(
    pairwise: pd.DataFrame,
    *,
    max_allowed_signal_ari_drop: float = 0.0,
    min_null_false_split_reduction: int = 1,
) -> pd.DataFrame:
    """Summarize promotion-readiness by threshold."""
    if pairwise.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []
    for threshold, group in pairwise.groupby(
        "spectral_transport_max_cost",
        sort=True,
        dropna=True,
    ):
        null_group = group[group["data_role"].astype(str).eq(NULL_OUTPUT_ROLE)]
        signal_group = group[group["data_role"].astype(str).eq(SIGNAL_OUTPUT_ROLE)]
        baseline_false = int(null_group["baseline_false_split"].astype(bool).sum())
        candidate_false = int(null_group["candidate_false_split"].astype(bool).sum())
        false_split_reduction = int(baseline_false - candidate_false)
        signal_delta = pd.to_numeric(
            signal_group["delta_ari_candidate_minus_baseline"],
            errors="coerce",
        )
        signal_delta = signal_delta[np.isfinite(signal_delta)]
        min_signal_delta = (
            float(signal_delta.min()) if not signal_delta.empty else np.nan
        )
        mean_signal_delta = (
            float(signal_delta.mean()) if not signal_delta.empty else np.nan
        )
        signal_regression_count = int(
            (signal_delta < -float(max_allowed_signal_ari_drop)).sum()
        )
        null_pass = bool(false_split_reduction >= int(min_null_false_split_reduction))
        signal_pass = bool(
            not signal_delta.empty
            and min_signal_delta >= -float(max_allowed_signal_ari_drop)
        )
        if null_pass and signal_pass:
            status = "threshold_candidate"
        elif not null_pass and signal_pass:
            status = "null_false_split_not_reduced"
        elif null_pass and not signal_pass:
            status = "signal_regression"
        else:
            status = "null_not_reduced_and_signal_regression"
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "spectral_transport_max_cost": float(threshold),
                "selected_null_case_count": int(null_group["case_id"].nunique()),
                "signal_case_count": int(signal_group["case_id"].nunique()),
                "baseline_false_split_count": baseline_false,
                "candidate_false_split_count": candidate_false,
                "false_split_reduction": false_split_reduction,
                "min_null_false_split_reduction": int(min_null_false_split_reduction),
                "min_signal_delta_ari": min_signal_delta,
                "mean_signal_delta_ari": mean_signal_delta,
                "max_allowed_signal_ari_drop": float(max_allowed_signal_ari_drop),
                "signal_regression_count": signal_regression_count,
                "operating_point_status": status,
            }
        )
    return pd.DataFrame.from_records(summaries)


def run_spectral_transport_threshold_calibration_panel(
    config: SpectralTransportThresholdCalibrationConfig,
) -> dict[str, Path]:
    """Run threshold calibration and write CSV outputs."""
    rows = run_threshold_rows(config)
    pairwise = build_threshold_pairwise_rows(rows)
    summary = summarize_thresholds(
        pairwise,
        max_allowed_signal_ari_drop=float(config.max_allowed_signal_ari_drop),
        min_null_false_split_reduction=int(config.min_null_false_split_reduction),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = config.output_dir / ROWS_OUTPUT
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
        "suite": config.suite,
        "case_names": config.case_names,
        "data_roles": config.data_roles,
        "thresholds": config.thresholds,
        "base_seed": int(config.base_seed),
        "edge_alpha": float(config.edge_alpha),
        "sibling_alpha": float(config.sibling_alpha),
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
    thresholds = tuple(float(value) for value in args.thresholds)
    run_spectral_transport_threshold_calibration_panel(
        SpectralTransportThresholdCalibrationConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=tuple(args.case_names),
            data_roles=tuple(args.data_roles),
            thresholds=thresholds,
            base_seed=int(args.base_seed),
            edge_alpha=float(args.edge_alpha),
            sibling_alpha=float(args.sibling_alpha),
            max_allowed_signal_ari_drop=float(args.max_allowed_signal_ari_drop),
            min_null_false_split_reduction=int(args.min_null_false_split_reduction),
        )
    )


if __name__ == "__main__":
    main()
