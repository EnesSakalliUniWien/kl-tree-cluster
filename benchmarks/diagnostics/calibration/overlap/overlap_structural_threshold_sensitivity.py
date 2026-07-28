"""Threshold sensitivity for overlap structural sibling diagnostics.

This post-run analyzer reads ``overlap_structural_sibling_rows.csv`` and
evaluates diagnostic traversal gates over homogeneity gain, heterogeneity gain,
subspace consensus, and sibling p-value thresholds. It does not change
production traversal behavior.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_structural_threshold_sensitivity_not_calibration"
SCHEMA_VERSION = "overlap_structural_threshold_sensitivity/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap.overlap_structural_threshold_sensitivity"
DEFAULT_HOMOGENEITY_THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03)
DEFAULT_HETEROGENEITY_THRESHOLDS = (0.0, 0.005, 0.01, 0.02)
DEFAULT_SUBSPACE_THRESHOLDS = (0.15, 0.25, 0.35, 0.50)
DEFAULT_SIBLING_P_THRESHOLDS = (0.001, 0.01, 0.05)

SENSITIVITY_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "homogeneity_gain_threshold",
    "heterogeneity_gain_threshold",
    "subspace_consensus_threshold",
    "sibling_p_threshold",
    "accepted_split_count",
    "structural_homogeneous_accept_count",
    "same_subspace_heterogeneous_warning_count",
    "unrelated_subspace_signal_count",
    "weak_or_mixed_rejected_count",
    "null_false_accept_count",
    "signal_truth_aligned_accept_count",
    "signal_truth_misaligned_accept_count",
    "signal_truth_unavailable_accept_count",
    "accepted_truth_alignment_rate",
    "accepted_median_truth_split_ari",
    "blocked_accepted_split_count",
    "blocked_signal_truth_aligned_count",
    "blocked_null_false_split_count",
    "threshold_status",
)

RECOMMENDATION_COLUMNS = (
    "schema_version",
    "study_role",
    "homogeneity_gain_threshold",
    "heterogeneity_gain_threshold",
    "subspace_consensus_threshold",
    "sibling_p_threshold",
    "null_false_accept_count",
    "null_false_accept_rate",
    "signal_truth_aligned_accept_count",
    "signal_truth_aligned_total",
    "signal_truth_aligned_retention",
    "signal_truth_misaligned_accept_count",
    "same_subspace_heterogeneous_warning_count",
    "unrelated_subspace_signal_count",
    "null_case_replicate_count",
    "null_case_replicates_with_false_accept",
    "signal_truth_aligned_case_replicate_total",
    "signal_truth_aligned_case_replicates_retained",
    "signal_truth_misaligned_case_replicates_accepted",
    "threshold_stability_status",
    "threshold_tradeoff_score",
    "threshold_recommendation_status",
)


@dataclass(frozen=True)
class OverlapStructuralThresholdSensitivityConfig:
    """Runtime contract for overlap structural threshold sensitivity."""

    rows_path: Path
    output_dir: Path
    homogeneity_thresholds: tuple[float, ...] = DEFAULT_HOMOGENEITY_THRESHOLDS
    heterogeneity_thresholds: tuple[float, ...] = DEFAULT_HETEROGENEITY_THRESHOLDS
    subspace_thresholds: tuple[float, ...] = DEFAULT_SUBSPACE_THRESHOLDS
    sibling_p_thresholds: tuple[float, ...] = DEFAULT_SIBLING_P_THRESHOLDS
    truth_alignment_threshold: float = 0.50

    @property
    def sensitivity_path(self) -> Path:
        return self.output_dir / "overlap_structural_threshold_sensitivity.csv"

    @property
    def recommendation_path(self) -> Path:
        return self.output_dir / "overlap_structural_threshold_recommendations.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(token) for token in str(value).split(",") if token.strip())
    if not values:
        raise ValueError("Threshold grid must contain at least one value.")
    return values


def _is_null_role(series: pd.Series) -> pd.Series:
    return series.astype(str).isin({"null", "selected_null"})


def _is_signal_role(series: pd.Series) -> pd.Series:
    return series.astype(str).eq("signal")


def _required_columns() -> set[str]:
    return {
        "case_id",
        "data_role",
        "replicate",
        "decision_class",
        "sibling_p_value",
        "homogeneity_gain_min",
        "heterogeneity_gain_max",
        "subspace_consensus_jaccard_topk",
        "heterogeneity_subspace_consensus_jaccard_topk",
        "truth_split_ari",
    }


def _validate_rows(rows: pd.DataFrame) -> None:
    missing = sorted(_required_columns() - set(rows.columns))
    if missing:
        raise ValueError(f"Overlap structural rows are missing columns: {missing!r}")


def _threshold_status(
    *,
    data_role: str,
    null_false_accept_count: int,
    signal_truth_aligned_accept_count: int,
    blocked_signal_truth_aligned_count: int,
    same_subspace_heterogeneous_warning_count: int,
    unrelated_subspace_signal_count: int,
) -> str:
    if str(data_role) in {"null", "selected_null"}:
        if null_false_accept_count == 0:
            return "null_false_splits_blocked"
        return "null_false_splits_remain"
    if signal_truth_aligned_accept_count > 0 and blocked_signal_truth_aligned_count == 0:
        if same_subspace_heterogeneous_warning_count > 0:
            return "signal_retained_with_heterogeneity_warnings"
        return "signal_truth_aligned_retained"
    if signal_truth_aligned_accept_count > 0:
        return "signal_partially_retained"
    if unrelated_subspace_signal_count > 0:
        return "signal_unrelated_subspace_only"
    return "signal_truth_aligned_blocked"


def _evaluate_group(
    group: pd.DataFrame,
    *,
    homogeneity_threshold: float,
    heterogeneity_threshold: float,
    subspace_threshold: float,
    sibling_p_threshold: float,
    truth_alignment_threshold: float,
) -> dict[str, object]:
    role = str(group["data_role"].iloc[0])
    accepted_split = group["decision_class"].astype(str).eq("accepted_internal_split")
    sibling_pass = pd.to_numeric(group["sibling_p_value"], errors="coerce").le(
        float(sibling_p_threshold)
    )
    homogeneous_same_subspace = pd.to_numeric(group["homogeneity_gain_min"], errors="coerce").ge(
        float(homogeneity_threshold)
    ) & pd.to_numeric(group["subspace_consensus_jaccard_topk"], errors="coerce").ge(
        float(subspace_threshold)
    )
    heterogeneous_same_subspace = pd.to_numeric(
        group["heterogeneity_gain_max"], errors="coerce"
    ).ge(float(heterogeneity_threshold)) & pd.to_numeric(
        group["heterogeneity_subspace_consensus_jaccard_topk"],
        errors="coerce",
    ).ge(float(subspace_threshold))
    unrelated_signal = (
        sibling_pass
        & ~homogeneous_same_subspace
        & ~heterogeneous_same_subspace
        & (
            pd.to_numeric(group["homogeneity_gain_min"], errors="coerce").ge(
                float(homogeneity_threshold)
            )
            | pd.to_numeric(group["heterogeneity_gain_max"], errors="coerce").ge(
                float(heterogeneity_threshold)
            )
        )
    )
    structural_accept = accepted_split & sibling_pass & homogeneous_same_subspace
    structural_warning = accepted_split & sibling_pass & heterogeneous_same_subspace
    unrelated_accept = accepted_split & unrelated_signal
    blocked_accepted = accepted_split & ~(structural_accept | structural_warning)
    truth_ari = pd.to_numeric(group["truth_split_ari"], errors="coerce")
    truth_aligned = truth_ari.ge(float(truth_alignment_threshold))
    truth_available = truth_ari.notna()
    truth_misaligned = truth_available & ~truth_aligned
    null_mask = _is_null_role(group["data_role"])
    signal_mask = _is_signal_role(group["data_role"])
    accepted_truth = truth_ari[structural_accept & truth_available]
    signal_truth_aligned_accept = int((structural_accept & signal_mask & truth_aligned).sum())
    blocked_signal_truth_aligned = int((blocked_accepted & signal_mask & truth_aligned).sum())
    null_false_accept = int((structural_accept & null_mask).sum())
    same_subspace_heterogeneous_warning = int(structural_warning.sum())
    unrelated_subspace_signal = int(unrelated_accept.sum())
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": str(group["case_id"].iloc[0]),
        "data_role": role,
        "replicate": int(group["replicate"].iloc[0]),
        "homogeneity_gain_threshold": float(homogeneity_threshold),
        "heterogeneity_gain_threshold": float(heterogeneity_threshold),
        "subspace_consensus_threshold": float(subspace_threshold),
        "sibling_p_threshold": float(sibling_p_threshold),
        "accepted_split_count": int(accepted_split.sum()),
        "structural_homogeneous_accept_count": int(structural_accept.sum()),
        "same_subspace_heterogeneous_warning_count": same_subspace_heterogeneous_warning,
        "unrelated_subspace_signal_count": unrelated_subspace_signal,
        "weak_or_mixed_rejected_count": int(blocked_accepted.sum()),
        "null_false_accept_count": null_false_accept,
        "signal_truth_aligned_accept_count": signal_truth_aligned_accept,
        "signal_truth_misaligned_accept_count": int(
            (structural_accept & signal_mask & truth_misaligned).sum()
        ),
        "signal_truth_unavailable_accept_count": int(
            (structural_accept & signal_mask & ~truth_available).sum()
        ),
        "accepted_truth_alignment_rate": (
            float((accepted_truth >= float(truth_alignment_threshold)).mean())
            if not accepted_truth.empty
            else np.nan
        ),
        "accepted_median_truth_split_ari": (
            float(accepted_truth.median()) if not accepted_truth.empty else np.nan
        ),
        "blocked_accepted_split_count": int(blocked_accepted.sum()),
        "blocked_signal_truth_aligned_count": blocked_signal_truth_aligned,
        "blocked_null_false_split_count": int((blocked_accepted & null_mask).sum()),
        "threshold_status": _threshold_status(
            data_role=role,
            null_false_accept_count=null_false_accept,
            signal_truth_aligned_accept_count=signal_truth_aligned_accept,
            blocked_signal_truth_aligned_count=blocked_signal_truth_aligned,
            same_subspace_heterogeneous_warning_count=(same_subspace_heterogeneous_warning),
            unrelated_subspace_signal_count=unrelated_subspace_signal,
        ),
    }


def build_threshold_sensitivity(
    rows: pd.DataFrame,
    *,
    homogeneity_thresholds: Sequence[float] = DEFAULT_HOMOGENEITY_THRESHOLDS,
    heterogeneity_thresholds: Sequence[float] = DEFAULT_HETEROGENEITY_THRESHOLDS,
    subspace_thresholds: Sequence[float] = DEFAULT_SUBSPACE_THRESHOLDS,
    sibling_p_thresholds: Sequence[float] = DEFAULT_SIBLING_P_THRESHOLDS,
    truth_alignment_threshold: float = 0.50,
) -> pd.DataFrame:
    """Build case-level threshold sensitivity rows."""
    _validate_rows(rows)
    records: list[dict[str, object]] = []
    for homogeneity_threshold in homogeneity_thresholds:
        for heterogeneity_threshold in heterogeneity_thresholds:
            for subspace_threshold in subspace_thresholds:
                for sibling_p_threshold in sibling_p_thresholds:
                    for _, group in rows.groupby(
                        ["case_id", "data_role", "replicate"],
                        sort=True,
                    ):
                        records.append(
                            _evaluate_group(
                                group,
                                homogeneity_threshold=float(homogeneity_threshold),
                                heterogeneity_threshold=float(heterogeneity_threshold),
                                subspace_threshold=float(subspace_threshold),
                                sibling_p_threshold=float(sibling_p_threshold),
                                truth_alignment_threshold=float(truth_alignment_threshold),
                            )
                        )
    return pd.DataFrame.from_records(records, columns=SENSITIVITY_COLUMNS)


def summarize_threshold_recommendations(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Aggregate threshold sensitivity into grid-level tradeoff rows."""
    if sensitivity.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
    records: list[dict[str, object]] = []
    keys = [
        "homogeneity_gain_threshold",
        "heterogeneity_gain_threshold",
        "subspace_consensus_threshold",
        "sibling_p_threshold",
    ]
    for key_values, group in sensitivity.groupby(keys, sort=True):
        null_rows = group[_is_null_role(group["data_role"])]
        signal_rows = group[_is_signal_role(group["data_role"])]
        null_false_accept = int(null_rows["null_false_accept_count"].sum())
        null_accepted_total = int(null_rows["accepted_split_count"].sum())
        signal_truth_aligned_accept = int(signal_rows["signal_truth_aligned_accept_count"].sum())
        signal_truth_aligned_total = int(
            signal_rows[
                [
                    "signal_truth_aligned_accept_count",
                    "blocked_signal_truth_aligned_count",
                ]
            ]
            .sum(axis=1)
            .sum()
        )
        signal_truth_misaligned_accept = int(
            signal_rows["signal_truth_misaligned_accept_count"].sum()
        )
        heterogeneity_warnings = int(group["same_subspace_heterogeneous_warning_count"].sum())
        unrelated_signal = int(group["unrelated_subspace_signal_count"].sum())
        null_case_replicate_count = int(null_rows.shape[0])
        null_case_replicates_with_false_accept = int(
            null_rows["null_false_accept_count"].gt(0).sum()
        )
        signal_truth_aligned_by_row = signal_rows[
            [
                "signal_truth_aligned_accept_count",
                "blocked_signal_truth_aligned_count",
            ]
        ].sum(axis=1)
        signal_truth_aligned_case_replicate_total = int(signal_truth_aligned_by_row.gt(0).sum())
        signal_truth_aligned_case_replicates_retained = int(
            (
                signal_truth_aligned_by_row.gt(0)
                & signal_rows["signal_truth_aligned_accept_count"].gt(0)
            ).sum()
        )
        signal_truth_misaligned_case_replicates_accepted = int(
            signal_rows["signal_truth_misaligned_accept_count"].gt(0).sum()
        )
        null_rate = null_false_accept / null_accepted_total if null_accepted_total > 0 else 0.0
        retention = (
            signal_truth_aligned_accept / signal_truth_aligned_total
            if signal_truth_aligned_total > 0
            else np.nan
        )
        score = (
            (0.0 if np.isnan(retention) else retention)
            - null_rate
            - 0.25 * signal_truth_misaligned_accept
            - 0.10 * unrelated_signal
        )
        if null_false_accept > 0:
            status = "threshold_leaves_null_false_splits"
        elif signal_truth_aligned_total > 0 and signal_truth_aligned_accept == 0:
            status = "threshold_blocks_truth_aligned_signal"
        elif signal_truth_misaligned_accept > 0:
            status = "threshold_keeps_truth_misaligned_signal"
        elif heterogeneity_warnings > 0:
            status = "threshold_warns_same_subspace_heterogeneity"
        else:
            status = "threshold_candidate_diagnostic"
        if null_case_replicates_with_false_accept > 0:
            stability_status = "threshold_unstable_null_false_accepts"
        elif (
            signal_truth_aligned_case_replicate_total > 0
            and signal_truth_aligned_case_replicates_retained
            < signal_truth_aligned_case_replicate_total
        ):
            stability_status = "threshold_unstable_signal_loss"
        elif signal_truth_misaligned_case_replicates_accepted > 0:
            stability_status = "threshold_unstable_misaligned_signal"
        elif heterogeneity_warnings > 0:
            stability_status = "threshold_stable_with_heterogeneity_warnings"
        else:
            stability_status = "threshold_stable_candidate"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "homogeneity_gain_threshold": float(key_values[0]),
                "heterogeneity_gain_threshold": float(key_values[1]),
                "subspace_consensus_threshold": float(key_values[2]),
                "sibling_p_threshold": float(key_values[3]),
                "null_false_accept_count": null_false_accept,
                "null_false_accept_rate": float(null_rate),
                "signal_truth_aligned_accept_count": signal_truth_aligned_accept,
                "signal_truth_aligned_total": signal_truth_aligned_total,
                "signal_truth_aligned_retention": (
                    float(retention) if not np.isnan(retention) else np.nan
                ),
                "signal_truth_misaligned_accept_count": (signal_truth_misaligned_accept),
                "same_subspace_heterogeneous_warning_count": heterogeneity_warnings,
                "unrelated_subspace_signal_count": unrelated_signal,
                "null_case_replicate_count": null_case_replicate_count,
                "null_case_replicates_with_false_accept": (null_case_replicates_with_false_accept),
                "signal_truth_aligned_case_replicate_total": (
                    signal_truth_aligned_case_replicate_total
                ),
                "signal_truth_aligned_case_replicates_retained": (
                    signal_truth_aligned_case_replicates_retained
                ),
                "signal_truth_misaligned_case_replicates_accepted": (
                    signal_truth_misaligned_case_replicates_accepted
                ),
                "threshold_stability_status": stability_status,
                "threshold_tradeoff_score": float(score),
                "threshold_recommendation_status": status,
            }
        )
    frame = pd.DataFrame.from_records(records, columns=RECOMMENDATION_COLUMNS)
    return frame.sort_values(
        [
            "threshold_recommendation_status",
            "threshold_tradeoff_score",
            "homogeneity_gain_threshold",
            "subspace_consensus_threshold",
        ],
        ascending=[True, False, True, True],
        ignore_index=True,
    )


def run_overlap_structural_threshold_sensitivity(
    config: OverlapStructuralThresholdSensitivityConfig,
) -> dict[str, Path]:
    """Run threshold sensitivity and write outputs."""
    rows = pd.read_csv(config.rows_path)
    sensitivity = build_threshold_sensitivity(
        rows,
        homogeneity_thresholds=config.homogeneity_thresholds,
        heterogeneity_thresholds=config.heterogeneity_thresholds,
        subspace_thresholds=config.subspace_thresholds,
        sibling_p_thresholds=config.sibling_p_thresholds,
        truth_alignment_threshold=float(config.truth_alignment_threshold),
    )
    recommendations = summarize_threshold_recommendations(sensitivity)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(config.sensitivity_path, index=False)
    recommendations.to_csv(config.recommendation_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "rows_path": str(config.rows_path),
        "homogeneity_thresholds": list(config.homogeneity_thresholds),
        "heterogeneity_thresholds": list(config.heterogeneity_thresholds),
        "subspace_thresholds": list(config.subspace_thresholds),
        "sibling_p_thresholds": list(config.sibling_p_thresholds),
        "truth_alignment_threshold": float(config.truth_alignment_threshold),
        "sensitivity_rows": int(sensitivity.shape[0]),
        "recommendation_rows": int(recommendations.shape[0]),
        "outputs": {
            "sensitivity": str(config.sensitivity_path),
            "recommendations": str(config.recommendation_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "sensitivity": config.sensitivity_path,
        "recommendations": config.recommendation_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--homogeneity-thresholds",
        default=",".join(str(value) for value in DEFAULT_HOMOGENEITY_THRESHOLDS),
    )
    parser.add_argument(
        "--heterogeneity-thresholds",
        default=",".join(str(value) for value in DEFAULT_HETEROGENEITY_THRESHOLDS),
    )
    parser.add_argument(
        "--subspace-thresholds",
        default=",".join(str(value) for value in DEFAULT_SUBSPACE_THRESHOLDS),
    )
    parser.add_argument(
        "--sibling-p-thresholds",
        default=",".join(str(value) for value in DEFAULT_SIBLING_P_THRESHOLDS),
    )
    parser.add_argument("--truth-alignment-threshold", default=0.50, type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_structural_threshold_sensitivity(
        OverlapStructuralThresholdSensitivityConfig(
            rows_path=args.rows_path,
            output_dir=args.output_dir,
            homogeneity_thresholds=_parse_float_grid(args.homogeneity_thresholds),
            heterogeneity_thresholds=_parse_float_grid(args.heterogeneity_thresholds),
            subspace_thresholds=_parse_float_grid(args.subspace_thresholds),
            sibling_p_thresholds=_parse_float_grid(args.sibling_p_thresholds),
            truth_alignment_threshold=float(args.truth_alignment_threshold),
        )
    )


if __name__ == "__main__":
    main()
