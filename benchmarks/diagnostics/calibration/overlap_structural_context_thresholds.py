"""Context-binned threshold analysis for overlap structural traversal rows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_structural_context_thresholds_not_calibration"
SCHEMA_VERSION = "overlap_structural_context_thresholds/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_structural_context_thresholds"
DEFAULT_HOMOGENEITY_THRESHOLDS = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
DEFAULT_SUBSPACE_THRESHOLDS = (0.15, 0.25, 0.35, 0.50)
DEFAULT_SIBLING_P_THRESHOLDS = (0.001, 0.01, 0.05)
CONTEXT_AXES = (
    "depth_bin",
    "parent_size_bin",
    "balance_bin",
    "subspace_consensus_bin",
)

SENSITIVITY_COLUMNS = (
    "schema_version",
    "study_role",
    "context_axis",
    "context_bin",
    "homogeneity_gain_threshold",
    "subspace_consensus_threshold",
    "sibling_p_threshold",
    "accepted_split_count",
    "null_accepted_split_count",
    "signal_accepted_split_count",
    "null_structural_accept_count",
    "signal_truth_aligned_total",
    "signal_truth_aligned_accept_count",
    "signal_truth_aligned_retention",
    "signal_truth_misaligned_accept_count",
    "median_parent_size",
    "median_depth",
    "median_balance",
    "median_homogeneity_gain",
    "median_subspace_consensus",
    "context_threshold_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "context_axis",
    "context_bin",
    "best_homogeneity_gain_threshold",
    "best_subspace_consensus_threshold",
    "best_sibling_p_threshold",
    "accepted_split_count",
    "null_structural_accept_count",
    "signal_truth_aligned_total",
    "signal_truth_aligned_accept_count",
    "signal_truth_aligned_retention",
    "signal_truth_misaligned_accept_count",
    "median_parent_size",
    "median_depth",
    "median_balance",
    "median_homogeneity_gain",
    "median_subspace_consensus",
    "context_threshold_status",
)


@dataclass(frozen=True)
class OverlapStructuralContextThresholdConfig:
    """Runtime contract for context-binned threshold diagnostics."""

    rows_path: Path
    output_dir: Path
    homogeneity_thresholds: tuple[float, ...] = DEFAULT_HOMOGENEITY_THRESHOLDS
    subspace_thresholds: tuple[float, ...] = DEFAULT_SUBSPACE_THRESHOLDS
    sibling_p_thresholds: tuple[float, ...] = DEFAULT_SIBLING_P_THRESHOLDS
    truth_alignment_threshold: float = 0.50

    @property
    def sensitivity_path(self) -> Path:
        return self.output_dir / "overlap_structural_context_threshold_sensitivity.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_structural_context_threshold_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(token) for token in str(value).split(",") if token.strip())
    if not values:
        raise ValueError("Threshold grid must contain at least one value.")
    return values


def _required_columns() -> set[str]:
    return {
        "data_role",
        "decision_class",
        "depth",
        "n_parent",
        "barycentric_balance",
        "sibling_p_value",
        "homogeneity_gain_min",
        "subspace_consensus_jaccard_topk",
        "truth_split_ari",
    }


def _validate_rows(rows: pd.DataFrame) -> None:
    missing = sorted(_required_columns() - set(rows.columns))
    if missing:
        raise ValueError(f"Overlap structural rows are missing columns: {missing!r}")


def _is_null_role(series: pd.Series) -> pd.Series:
    return series.astype(str).isin({"null", "selected_null"})


def _is_signal_role(series: pd.Series) -> pd.Series:
    return series.astype(str).eq("signal")


def add_context_bins(rows: pd.DataFrame) -> pd.DataFrame:
    """Add traversal-context bins used by the context-threshold diagnostic."""
    _validate_rows(rows)
    enriched = rows.copy()
    depth = pd.to_numeric(enriched["depth"], errors="coerce")
    parent_size = pd.to_numeric(enriched["n_parent"], errors="coerce")
    balance = pd.to_numeric(enriched["barycentric_balance"], errors="coerce")
    consensus = pd.to_numeric(
        enriched["subspace_consensus_jaccard_topk"],
        errors="coerce",
    )
    enriched["depth_bin"] = np.select(
        [depth <= 0, depth.between(1, 2), depth >= 3],
        ["root", "shallow_1_2", "deep_3_plus"],
        default="unknown_depth",
    )
    enriched["parent_size_bin"] = np.select(
        [parent_size < 150, parent_size.between(150, 299), parent_size >= 300],
        ["small_parent_lt150", "medium_parent_150_299", "large_parent_ge300"],
        default="unknown_parent_size",
    )
    enriched["balance_bin"] = np.select(
        [balance < 0.25, balance.between(0.25, 0.399999), balance >= 0.40],
        ["unbalanced_lt0.25", "moderate_balance_0.25_0.4", "balanced_ge0.4"],
        default="unknown_balance",
    )
    enriched["subspace_consensus_bin"] = np.select(
        [consensus < 0.25, consensus.between(0.25, 0.499999), consensus >= 0.50],
        ["low_consensus_lt0.25", "mid_consensus_0.25_0.5", "high_consensus_ge0.5"],
        default="unknown_consensus",
    )
    return enriched


def _status_for_context(
    *,
    null_structural_accept_count: int,
    signal_truth_aligned_total: int,
    signal_truth_aligned_accept_count: int,
    signal_truth_misaligned_accept_count: int,
) -> str:
    if null_structural_accept_count > 0:
        return "context_null_uncontrolled"
    if signal_truth_aligned_total > 0 and signal_truth_aligned_accept_count == 0:
        return "context_signal_blocked"
    if signal_truth_aligned_accept_count < signal_truth_aligned_total:
        return "context_signal_partially_lost"
    if signal_truth_misaligned_accept_count > 0:
        return "context_truth_misaligned_kept"
    if signal_truth_aligned_total == 0:
        return "context_no_truth_aligned_signal"
    return "context_threshold_candidate"


def _evaluate_context_threshold(
    group: pd.DataFrame,
    *,
    context_axis: str,
    context_bin: str,
    homogeneity_threshold: float,
    subspace_threshold: float,
    sibling_p_threshold: float,
    truth_alignment_threshold: float,
) -> dict[str, object]:
    accepted_split = group["decision_class"].astype(str).eq("accepted_internal_split")
    null_mask = _is_null_role(group["data_role"])
    signal_mask = _is_signal_role(group["data_role"])
    truth_ari = pd.to_numeric(group["truth_split_ari"], errors="coerce")
    truth_aligned = truth_ari.ge(float(truth_alignment_threshold))
    truth_misaligned = truth_ari.notna() & ~truth_aligned
    structural_accept = (
        accepted_split
        & pd.to_numeric(group["sibling_p_value"], errors="coerce").le(
            float(sibling_p_threshold)
        )
        & pd.to_numeric(group["homogeneity_gain_min"], errors="coerce").ge(
            float(homogeneity_threshold)
        )
        & pd.to_numeric(
            group["subspace_consensus_jaccard_topk"],
            errors="coerce",
        ).ge(float(subspace_threshold))
    )
    null_structural_accept_count = int((structural_accept & null_mask).sum())
    signal_truth_aligned_total = int((accepted_split & signal_mask & truth_aligned).sum())
    signal_truth_aligned_accept_count = int(
        (structural_accept & signal_mask & truth_aligned).sum()
    )
    signal_truth_misaligned_accept_count = int(
        (structural_accept & signal_mask & truth_misaligned).sum()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "context_axis": context_axis,
        "context_bin": context_bin,
        "homogeneity_gain_threshold": float(homogeneity_threshold),
        "subspace_consensus_threshold": float(subspace_threshold),
        "sibling_p_threshold": float(sibling_p_threshold),
        "accepted_split_count": int(accepted_split.sum()),
        "null_accepted_split_count": int((accepted_split & null_mask).sum()),
        "signal_accepted_split_count": int((accepted_split & signal_mask).sum()),
        "null_structural_accept_count": null_structural_accept_count,
        "signal_truth_aligned_total": signal_truth_aligned_total,
        "signal_truth_aligned_accept_count": signal_truth_aligned_accept_count,
        "signal_truth_aligned_retention": (
            signal_truth_aligned_accept_count / signal_truth_aligned_total
            if signal_truth_aligned_total > 0
            else np.nan
        ),
        "signal_truth_misaligned_accept_count": signal_truth_misaligned_accept_count,
        "median_parent_size": float(
            pd.to_numeric(group["n_parent"], errors="coerce").median()
        ),
        "median_depth": float(pd.to_numeric(group["depth"], errors="coerce").median()),
        "median_balance": float(
            pd.to_numeric(group["barycentric_balance"], errors="coerce").median()
        ),
        "median_homogeneity_gain": float(
            pd.to_numeric(group["homogeneity_gain_min"], errors="coerce").median()
        ),
        "median_subspace_consensus": float(
            pd.to_numeric(
                group["subspace_consensus_jaccard_topk"],
                errors="coerce",
            ).median()
        ),
        "context_threshold_status": _status_for_context(
            null_structural_accept_count=null_structural_accept_count,
            signal_truth_aligned_total=signal_truth_aligned_total,
            signal_truth_aligned_accept_count=signal_truth_aligned_accept_count,
            signal_truth_misaligned_accept_count=signal_truth_misaligned_accept_count,
        ),
    }


def build_context_threshold_sensitivity(
    rows: pd.DataFrame,
    *,
    homogeneity_thresholds: Sequence[float] = DEFAULT_HOMOGENEITY_THRESHOLDS,
    subspace_thresholds: Sequence[float] = DEFAULT_SUBSPACE_THRESHOLDS,
    sibling_p_thresholds: Sequence[float] = DEFAULT_SIBLING_P_THRESHOLDS,
    truth_alignment_threshold: float = 0.50,
) -> pd.DataFrame:
    """Build context-binned threshold sensitivity rows."""
    enriched = add_context_bins(rows)
    records: list[dict[str, object]] = []
    for context_axis in CONTEXT_AXES:
        for context_bin, context_group in enriched.groupby(context_axis, sort=True):
            for homogeneity_threshold in homogeneity_thresholds:
                for subspace_threshold in subspace_thresholds:
                    for sibling_p_threshold in sibling_p_thresholds:
                        records.append(
                            _evaluate_context_threshold(
                                context_group,
                                context_axis=context_axis,
                                context_bin=str(context_bin),
                                homogeneity_threshold=float(homogeneity_threshold),
                                subspace_threshold=float(subspace_threshold),
                                sibling_p_threshold=float(sibling_p_threshold),
                                truth_alignment_threshold=float(
                                    truth_alignment_threshold
                                ),
                            )
                        )
    return pd.DataFrame.from_records(records, columns=SENSITIVITY_COLUMNS)


def summarize_context_thresholds(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Pick the best diagnostic threshold row per context bin."""
    if sensitivity.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    status_rank = {
        "context_threshold_candidate": 0,
        "context_no_truth_aligned_signal": 1,
        "context_truth_misaligned_kept": 2,
        "context_signal_partially_lost": 3,
        "context_signal_blocked": 4,
        "context_null_uncontrolled": 5,
    }
    ranked = sensitivity.copy()
    ranked["_status_rank"] = ranked["context_threshold_status"].map(status_rank).fillna(9)
    retention = pd.to_numeric(
        ranked["signal_truth_aligned_retention"],
        errors="coerce",
    ).fillna(0.0)
    ranked["_score"] = (
        -ranked["_status_rank"].astype(float)
        + retention
        - 0.25
        * pd.to_numeric(
            ranked["signal_truth_misaligned_accept_count"],
            errors="coerce",
        ).fillna(0.0)
        - 0.50
        * pd.to_numeric(
            ranked["null_structural_accept_count"],
            errors="coerce",
        ).fillna(0.0)
    )
    records: list[dict[str, object]] = []
    for (context_axis, context_bin), group in ranked.groupby(
        ["context_axis", "context_bin"],
        sort=True,
    ):
        best = group.sort_values(
            [
                "_score",
                "_status_rank",
                "homogeneity_gain_threshold",
                "subspace_consensus_threshold",
            ],
            ascending=[False, True, True, True],
        ).iloc[0]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "context_axis": context_axis,
                "context_bin": context_bin,
                "best_homogeneity_gain_threshold": float(
                    best["homogeneity_gain_threshold"]
                ),
                "best_subspace_consensus_threshold": float(
                    best["subspace_consensus_threshold"]
                ),
                "best_sibling_p_threshold": float(best["sibling_p_threshold"]),
                "accepted_split_count": int(best["accepted_split_count"]),
                "null_structural_accept_count": int(best["null_structural_accept_count"]),
                "signal_truth_aligned_total": int(best["signal_truth_aligned_total"]),
                "signal_truth_aligned_accept_count": int(
                    best["signal_truth_aligned_accept_count"]
                ),
                "signal_truth_aligned_retention": float(
                    best["signal_truth_aligned_retention"]
                )
                if pd.notna(best["signal_truth_aligned_retention"])
                else np.nan,
                "signal_truth_misaligned_accept_count": int(
                    best["signal_truth_misaligned_accept_count"]
                ),
                "median_parent_size": float(best["median_parent_size"]),
                "median_depth": float(best["median_depth"]),
                "median_balance": float(best["median_balance"]),
                "median_homogeneity_gain": float(best["median_homogeneity_gain"]),
                "median_subspace_consensus": float(best["median_subspace_consensus"]),
                "context_threshold_status": str(best["context_threshold_status"]),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def run_overlap_structural_context_thresholds(
    config: OverlapStructuralContextThresholdConfig,
) -> dict[str, Path]:
    """Run context-binned threshold diagnostics and write outputs."""
    rows = pd.read_csv(config.rows_path)
    sensitivity = build_context_threshold_sensitivity(
        rows,
        homogeneity_thresholds=config.homogeneity_thresholds,
        subspace_thresholds=config.subspace_thresholds,
        sibling_p_thresholds=config.sibling_p_thresholds,
        truth_alignment_threshold=float(config.truth_alignment_threshold),
    )
    summary = summarize_context_thresholds(sensitivity)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(config.sensitivity_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "rows_path": str(config.rows_path),
        "homogeneity_thresholds": list(config.homogeneity_thresholds),
        "subspace_thresholds": list(config.subspace_thresholds),
        "sibling_p_thresholds": list(config.sibling_p_thresholds),
        "truth_alignment_threshold": float(config.truth_alignment_threshold),
        "sensitivity_rows": int(sensitivity.shape[0]),
        "summary_rows": int(summary.shape[0]),
        "outputs": {
            "sensitivity": str(config.sensitivity_path),
            "summary": str(config.summary_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "sensitivity": config.sensitivity_path,
        "summary": config.summary_path,
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
    run_overlap_structural_context_thresholds(
        OverlapStructuralContextThresholdConfig(
            rows_path=args.rows_path,
            output_dir=args.output_dir,
            homogeneity_thresholds=_parse_float_grid(args.homogeneity_thresholds),
            subspace_thresholds=_parse_float_grid(args.subspace_thresholds),
            sibling_p_thresholds=_parse_float_grid(args.sibling_p_thresholds),
            truth_alignment_threshold=float(args.truth_alignment_threshold),
        )
    )


if __name__ == "__main__":
    main()
