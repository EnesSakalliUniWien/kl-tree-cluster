"""Oracle truth-geometry diagnostics for weak overlap traversal splits.

This panel uses truth labels, so it is diagnostic-only. It explains why
statistically extreme weak-homogeneity selected families can still be wrong:
many are one-sided pure-fragment splits or wrong-granularity splits rather than
balanced recovery of the intended overlap structure.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_weak_truth_geometry_not_calibration"
SCHEMA_VERSION = "overlap_weak_truth_geometry/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap_weak_truth_geometry"
DEFAULT_TRUTH_ALIGNMENT_THRESHOLD = 0.50
DEFAULT_HIGH_CHILD_PURITY = 0.85
DEFAULT_LOW_CHILD_PURITY = 0.55
DEFAULT_BALANCED_CHILD_PURITY = 0.65
DEFAULT_PURITY_GAP = 0.25

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "depth",
    "n_parent",
    "n_left",
    "n_right",
    "barycentric_balance",
    "truth_split_ari",
    "parent_truth_purity",
    "left_truth_purity",
    "right_truth_purity",
    "min_child_truth_purity",
    "max_child_truth_purity",
    "child_truth_purity_gap",
    "best_child_purity_gain",
    "worst_child_purity_gain",
    "homogeneity_gain_min",
    "subspace_consensus_jaccard_topk",
    "truth_geometry_mode",
)

FAMILY_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "family_truth_geometry_mode",
    "family_size",
    "truth_aligned_row_count",
    "one_sided_fragment_count",
    "wrong_granularity_count",
    "diffuse_mismatch_count",
    "max_truth_split_ari",
    "median_truth_split_ari",
    "max_child_truth_purity",
    "median_min_child_truth_purity",
    "median_child_truth_purity_gap",
    "max_homogeneity_gain_min",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "truth_geometry_mode",
    "row_count",
    "family_count",
    "median_truth_split_ari",
    "median_parent_truth_purity",
    "median_min_child_truth_purity",
    "median_max_child_truth_purity",
    "median_child_truth_purity_gap",
    "median_homogeneity_gain_min",
)


@dataclass(frozen=True)
class OverlapWeakTruthGeometryConfig:
    """Runtime contract for oracle weak truth-geometry diagnostics."""

    rows_path: Path
    output_dir: Path
    truth_alignment_threshold: float = DEFAULT_TRUTH_ALIGNMENT_THRESHOLD
    high_child_purity: float = DEFAULT_HIGH_CHILD_PURITY
    low_child_purity: float = DEFAULT_LOW_CHILD_PURITY
    balanced_child_purity: float = DEFAULT_BALANCED_CHILD_PURITY
    purity_gap: float = DEFAULT_PURITY_GAP

    @property
    def row_geometry_path(self) -> Path:
        return self.output_dir / "overlap_weak_truth_geometry_rows.csv"

    @property
    def family_geometry_path(self) -> Path:
        return self.output_dir / "overlap_weak_truth_geometry_families.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_weak_truth_geometry_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _required_columns() -> set[str]:
    return {
        "case_id",
        "data_role",
        "replicate",
        "node_id",
        "depth",
        "decision_class",
        "structural_sibling_status",
        "n_parent",
        "n_left",
        "n_right",
        "barycentric_balance",
        "truth_split_ari",
        "parent_truth_purity",
        "left_truth_purity",
        "right_truth_purity",
        "homogeneity_gain_min",
        "subspace_consensus_jaccard_topk",
    }


def _validate_rows(rows: pd.DataFrame) -> None:
    missing = sorted(_required_columns() - set(rows.columns))
    if missing:
        raise ValueError(f"Overlap structural rows are missing columns: {missing!r}")


def classify_truth_geometry(
    *,
    truth_split_ari: float,
    min_child_truth_purity: float,
    max_child_truth_purity: float,
    child_truth_purity_gap: float,
    truth_alignment_threshold: float = DEFAULT_TRUTH_ALIGNMENT_THRESHOLD,
    high_child_purity: float = DEFAULT_HIGH_CHILD_PURITY,
    low_child_purity: float = DEFAULT_LOW_CHILD_PURITY,
    balanced_child_purity: float = DEFAULT_BALANCED_CHILD_PURITY,
    purity_gap: float = DEFAULT_PURITY_GAP,
) -> str:
    """Classify oracle truth geometry for one weak accepted split."""
    if pd.isna(truth_split_ari):
        return "truth_unavailable"
    if float(truth_split_ari) >= float(truth_alignment_threshold):
        if float(min_child_truth_purity) >= float(balanced_child_purity):
            return "balanced_truth_recovery"
        return "partial_truth_recovery"
    if (
        float(max_child_truth_purity) >= float(high_child_purity)
        and float(min_child_truth_purity) < float(low_child_purity)
    ):
        return "one_sided_pure_fragment"
    if (
        float(max_child_truth_purity) >= float(balanced_child_purity)
        and float(child_truth_purity_gap) >= float(purity_gap)
    ):
        return "one_sided_mixed_remainder"
    if float(min_child_truth_purity) >= float(balanced_child_purity):
        return "balanced_but_wrong_granularity"
    return "diffuse_truth_mismatch"


def build_weak_truth_geometry_rows(
    rows: pd.DataFrame,
    *,
    truth_alignment_threshold: float = DEFAULT_TRUTH_ALIGNMENT_THRESHOLD,
    high_child_purity: float = DEFAULT_HIGH_CHILD_PURITY,
    low_child_purity: float = DEFAULT_LOW_CHILD_PURITY,
    balanced_child_purity: float = DEFAULT_BALANCED_CHILD_PURITY,
    purity_gap: float = DEFAULT_PURITY_GAP,
) -> pd.DataFrame:
    """Build row-level oracle truth geometry for weak accepted signal splits."""
    _validate_rows(rows)
    signal = rows["data_role"].astype(str).eq("signal")
    weak_accept = (
        rows["decision_class"].astype(str).eq("accepted_internal_split")
        & rows["structural_sibling_status"].astype(str).eq("weak_homogeneity_gain")
        & signal
    )
    working = rows.loc[weak_accept].copy()
    if working.empty:
        return pd.DataFrame(columns=ROW_COLUMNS)
    left = pd.to_numeric(working["left_truth_purity"], errors="coerce")
    right = pd.to_numeric(working["right_truth_purity"], errors="coerce")
    parent = pd.to_numeric(working["parent_truth_purity"], errors="coerce")
    min_child = pd.concat([left, right], axis=1).min(axis=1)
    max_child = pd.concat([left, right], axis=1).max(axis=1)
    gap = max_child - min_child
    truth_ari = pd.to_numeric(working["truth_split_ari"], errors="coerce")
    modes = [
        classify_truth_geometry(
            truth_split_ari=float(truth_ari.loc[idx]),
            min_child_truth_purity=float(min_child.loc[idx]),
            max_child_truth_purity=float(max_child.loc[idx]),
            child_truth_purity_gap=float(gap.loc[idx]),
            truth_alignment_threshold=float(truth_alignment_threshold),
            high_child_purity=float(high_child_purity),
            low_child_purity=float(low_child_purity),
            balanced_child_purity=float(balanced_child_purity),
            purity_gap=float(purity_gap),
        )
        for idx in working.index
    ]
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": working["case_id"].astype(str),
            "data_role": working["data_role"].astype(str),
            "replicate": working["replicate"].astype(int),
            "node_id": working["node_id"].astype(str),
            "depth": pd.to_numeric(working["depth"], errors="coerce"),
            "n_parent": pd.to_numeric(working["n_parent"], errors="coerce"),
            "n_left": pd.to_numeric(working["n_left"], errors="coerce"),
            "n_right": pd.to_numeric(working["n_right"], errors="coerce"),
            "barycentric_balance": pd.to_numeric(
                working["barycentric_balance"],
                errors="coerce",
            ),
            "truth_split_ari": truth_ari,
            "parent_truth_purity": parent,
            "left_truth_purity": left,
            "right_truth_purity": right,
            "min_child_truth_purity": min_child,
            "max_child_truth_purity": max_child,
            "child_truth_purity_gap": gap,
            "best_child_purity_gain": max_child - parent,
            "worst_child_purity_gain": min_child - parent,
            "homogeneity_gain_min": pd.to_numeric(
                working["homogeneity_gain_min"],
                errors="coerce",
            ),
            "subspace_consensus_jaccard_topk": pd.to_numeric(
                working["subspace_consensus_jaccard_topk"],
                errors="coerce",
            ),
            "truth_geometry_mode": modes,
        },
        columns=ROW_COLUMNS,
    )


def _family_mode(group: pd.DataFrame) -> str:
    modes = set(group["truth_geometry_mode"].astype(str))
    if modes & {"balanced_truth_recovery", "partial_truth_recovery"}:
        return "family_contains_truth_recovery"
    if "one_sided_pure_fragment" in modes:
        return "family_one_sided_pure_fragment"
    if "one_sided_mixed_remainder" in modes:
        return "family_one_sided_mixed_remainder"
    if "balanced_but_wrong_granularity" in modes:
        return "family_balanced_wrong_granularity"
    return "family_diffuse_truth_mismatch"


def build_weak_truth_geometry_families(row_geometry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level truth geometry to selected weak signal families."""
    if row_geometry.empty:
        return pd.DataFrame(columns=FAMILY_COLUMNS)
    records: list[dict[str, object]] = []
    group_cols = ["case_id", "data_role", "replicate"]
    for (case_id, data_role, replicate), group in row_geometry.groupby(
        group_cols,
        sort=True,
    ):
        modes = group["truth_geometry_mode"].astype(str)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "data_role": str(data_role),
                "replicate": int(replicate),
                "family_truth_geometry_mode": _family_mode(group),
                "family_size": int(group.shape[0]),
                "truth_aligned_row_count": int(
                    modes.isin(
                        {"balanced_truth_recovery", "partial_truth_recovery"}
                    ).sum()
                ),
                "one_sided_fragment_count": int(
                    modes.isin(
                        {
                            "one_sided_pure_fragment",
                            "one_sided_mixed_remainder",
                        }
                    ).sum()
                ),
                "wrong_granularity_count": int(
                    modes.eq("balanced_but_wrong_granularity").sum()
                ),
                "diffuse_mismatch_count": int(
                    modes.eq("diffuse_truth_mismatch").sum()
                ),
                "max_truth_split_ari": float(group["truth_split_ari"].max()),
                "median_truth_split_ari": float(group["truth_split_ari"].median()),
                "max_child_truth_purity": float(
                    group["max_child_truth_purity"].max()
                ),
                "median_min_child_truth_purity": float(
                    group["min_child_truth_purity"].median()
                ),
                "median_child_truth_purity_gap": float(
                    group["child_truth_purity_gap"].median()
                ),
                "max_homogeneity_gain_min": float(
                    group["homogeneity_gain_min"].max()
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=FAMILY_COLUMNS)


def summarize_weak_truth_geometry(
    row_geometry: pd.DataFrame,
    family_geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize row and family truth-geometry modes."""
    if row_geometry.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    family_key = (
        family_geometry["case_id"].astype(str)
        + "::"
        + family_geometry["replicate"].astype(str)
    )
    records: list[dict[str, object]] = []
    for mode, group in row_geometry.groupby("truth_geometry_mode", sort=True):
        row_family_key = group["case_id"].astype(str) + "::" + group["replicate"].astype(str)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "truth_geometry_mode": str(mode),
                "row_count": int(group.shape[0]),
                "family_count": int(family_key.isin(row_family_key.unique()).sum()),
                "median_truth_split_ari": float(group["truth_split_ari"].median()),
                "median_parent_truth_purity": float(
                    group["parent_truth_purity"].median()
                ),
                "median_min_child_truth_purity": float(
                    group["min_child_truth_purity"].median()
                ),
                "median_max_child_truth_purity": float(
                    group["max_child_truth_purity"].median()
                ),
                "median_child_truth_purity_gap": float(
                    group["child_truth_purity_gap"].median()
                ),
                "median_homogeneity_gain_min": float(
                    group["homogeneity_gain_min"].median()
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def run_overlap_weak_truth_geometry(
    config: OverlapWeakTruthGeometryConfig,
) -> dict[str, Path]:
    """Run oracle truth-geometry diagnostics and write outputs."""
    rows = pd.read_csv(config.rows_path)
    row_geometry = build_weak_truth_geometry_rows(
        rows,
        truth_alignment_threshold=float(config.truth_alignment_threshold),
        high_child_purity=float(config.high_child_purity),
        low_child_purity=float(config.low_child_purity),
        balanced_child_purity=float(config.balanced_child_purity),
        purity_gap=float(config.purity_gap),
    )
    family_geometry = build_weak_truth_geometry_families(row_geometry)
    summary = summarize_weak_truth_geometry(row_geometry, family_geometry)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    row_geometry.to_csv(config.row_geometry_path, index=False)
    family_geometry.to_csv(config.family_geometry_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "rows_path": str(config.rows_path),
        "truth_alignment_threshold": float(config.truth_alignment_threshold),
        "high_child_purity": float(config.high_child_purity),
        "low_child_purity": float(config.low_child_purity),
        "balanced_child_purity": float(config.balanced_child_purity),
        "purity_gap": float(config.purity_gap),
        "outputs": {
            "row_geometry": str(config.row_geometry_path),
            "family_geometry": str(config.family_geometry_path),
            "summary": str(config.summary_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "row_geometry": config.row_geometry_path,
        "family_geometry": config.family_geometry_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--truth-alignment-threshold",
        default=DEFAULT_TRUTH_ALIGNMENT_THRESHOLD,
        type=float,
    )
    parser.add_argument(
        "--high-child-purity",
        default=DEFAULT_HIGH_CHILD_PURITY,
        type=float,
    )
    parser.add_argument(
        "--low-child-purity",
        default=DEFAULT_LOW_CHILD_PURITY,
        type=float,
    )
    parser.add_argument(
        "--balanced-child-purity",
        default=DEFAULT_BALANCED_CHILD_PURITY,
        type=float,
    )
    parser.add_argument("--purity-gap", default=DEFAULT_PURITY_GAP, type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_weak_truth_geometry(
        OverlapWeakTruthGeometryConfig(
            rows_path=args.rows_path,
            output_dir=args.output_dir,
            truth_alignment_threshold=float(args.truth_alignment_threshold),
            high_child_purity=float(args.high_child_purity),
            low_child_purity=float(args.low_child_purity),
            balanced_child_purity=float(args.balanced_child_purity),
            purity_gap=float(args.purity_gap),
        )
    )


if __name__ == "__main__":
    main()
