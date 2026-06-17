"""Localize signal runs where the refined traversal guard suppresses movement.

This diagnostic joins paired clustering outcomes with selected-neighborhood
candidate contrast rows. It answers a narrower question than the method-wide
comparison: among signal runs, where does the refined guarded profile suppress
conditional-profile splits/pass-throughs, and does the run-level clustering
evidence make that suppressed movement look useful?
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "overlap_signal_suppression_localizer/v1"
STUDY_ROLE = "diagnostic_overlap_signal_suppression_localizer_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "overlap_signal_suppression_localizer"
)

JOIN_KEYS = ["case_id", "data_role", "replicate"]
SIGNAL_ROLE = "signal"
SUPPRESSED_REASONS = {
    "left_pass_through",
    "left_split|right_guard_blocked",
    "left_split|right_pass_through|right_guard_blocked",
}

RUN_VALUE_COLUMNS = [
    "partition_ari_between_methods",
    "delta_n_clusters_left_minus_right",
    "delta_singletons_left_minus_right",
    "delta_singleton_fraction_left_minus_right",
    "delta_largest_cluster_fraction_left_minus_right",
    "delta_effective_cluster_count_left_minus_right",
    "delta_run_row_ari_left_minus_right",
]

NODE_VALUE_COLUMNS = [
    "left_depth",
    "left_n_descendant_leaves",
    "left_sibling_p_value",
    "left_balance_product",
    "left_outgoing_edge_norm_balance",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "left_descendant_pass_through_count",
    "right_descendant_pass_through_count",
    "left_descendant_guard_blocked_count",
    "right_descendant_guard_blocked_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join paired overlap clustering rows to candidate contrast rows and "
            "localize signal suppression cases."
        )
    )
    parser.add_argument(
        "--pairwise",
        type=Path,
        required=True,
        help="overlap_method_clustering_pairwise.csv from the paired method comparison.",
    )
    parser.add_argument(
        "--candidate-rows",
        type=Path,
        required=True,
        help="selected_neighborhood_candidate_method_contrast_rows.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for localization outputs.",
    )
    parser.add_argument(
        "--positive-ari-epsilon",
        type=float,
        default=1e-12,
        help="Minimum left-minus-right ARI delta counted as useful movement.",
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


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _count_true(series: pd.Series) -> int:
    if series.empty:
        return 0
    return int(series.fillna(False).astype(bool).sum())


def _suppression_status(row: pd.Series, positive_ari_epsilon: float) -> str:
    delta_clusters = float(row.get("delta_n_clusters_left_minus_right", np.nan))
    delta_ari = float(row.get("delta_run_row_ari_left_minus_right", np.nan))
    suppressed_nodes = int(row.get("suppressed_candidate_count", 0))
    if suppressed_nodes == 0:
        return "no_suppressed_signal_candidate"
    if not np.isfinite(delta_clusters) or delta_clusters <= 0:
        return "suppressed_candidate_no_extra_cluster_movement"
    if np.isfinite(delta_ari) and delta_ari > positive_ari_epsilon:
        return "useful_movement_suppressed"
    if np.isfinite(delta_ari) and delta_ari < -positive_ari_epsilon:
        return "extra_movement_hurts_or_overfragments"
    return "extra_movement_neutral"


def _node_examples(group: pd.DataFrame, limit: int = 5) -> str:
    if group.empty:
        return ""
    parts = []
    sort_columns = [
        "candidate_reason",
        "left_depth",
        "left_n_descendant_leaves",
        "node_id",
    ]
    for _, row in group.sort_values(sort_columns, na_position="last").head(limit).iterrows():
        parts.append(
            (
                f"{row['node_id']}:{row['candidate_reason']}"
                f":depth={row.get('left_depth', '')}"
                f":leaves={row.get('left_n_descendant_leaves', '')}"
            )
        )
    return ";".join(parts)


def _summarize_candidate_nodes(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=[*JOIN_KEYS, "candidate_count"])

    candidates = candidates.copy()
    candidates["is_suppressed_candidate"] = candidates["candidate_reason"].isin(
        SUPPRESSED_REASONS
    )
    candidates["is_split_suppressed_by_guard"] = (
        candidates["candidate_reason"] == "left_split|right_guard_blocked"
    )
    candidates["is_pass_through_suppressed"] = candidates["candidate_reason"] == "left_pass_through"
    candidates["is_split_to_pass_through_guard"] = (
        candidates["candidate_reason"] == "left_split|right_pass_through|right_guard_blocked"
    )
    candidates["has_old_and_current_evidence"] = (
        (candidates["left_neighborhood_evidence_family"] == "old_and_current")
        | (candidates["right_neighborhood_evidence_family"] == "old_and_current")
    )
    candidates["is_traversal_only_pair"] = (
        (candidates["left_neighborhood_evidence_family"] == "traversal_only")
        & (candidates["right_neighborhood_evidence_family"] == "traversal_only")
    )

    rows = []
    for keys, group in candidates.groupby(JOIN_KEYS, dropna=False):
        suppressed = group[group["is_suppressed_candidate"]]
        row: dict[str, object] = {
            "case_id": keys[0],
            "data_role": keys[1],
            "replicate": keys[2],
            "candidate_count": int(len(group)),
            "decision_divergence_count": int(
                (group["candidate_contrast_status"] == "candidate_decision_diverges").sum()
            ),
            "suppressed_candidate_count": int(len(suppressed)),
            "split_suppressed_by_guard_count": _count_true(
                group["is_split_suppressed_by_guard"]
            ),
            "pass_through_suppressed_count": _count_true(group["is_pass_through_suppressed"]),
            "split_to_pass_through_guard_count": _count_true(
                group["is_split_to_pass_through_guard"]
            ),
            "right_guard_blocked_count": _count_true(group["right_explicit_guard_blocked"]),
            "suppressed_right_guard_blocked_count": _count_true(
                suppressed["right_explicit_guard_blocked"]
            ),
            "old_and_current_evidence_count": _count_true(group["has_old_and_current_evidence"]),
            "suppressed_old_and_current_evidence_count": _count_true(
                suppressed["has_old_and_current_evidence"]
            ),
            "traversal_only_pair_count": _count_true(group["is_traversal_only_pair"]),
            "suppressed_traversal_only_pair_count": _count_true(
                suppressed["is_traversal_only_pair"]
            ),
            "suppressed_node_examples": _node_examples(suppressed),
        }
        for column in NODE_VALUE_COLUMNS:
            values = _to_numeric(suppressed[column]) if column in suppressed.columns else pd.Series(dtype=float)
            row[f"suppressed_median_{column}"] = float(values.median()) if not values.dropna().empty else np.nan
            row[f"suppressed_max_{column}"] = float(values.max()) if not values.dropna().empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _load_signal_pairwise(pairwise_path: Path) -> pd.DataFrame:
    pairwise = pd.read_csv(pairwise_path)
    missing = set(JOIN_KEYS) - set(pairwise.columns)
    if missing:
        msg = f"{pairwise_path} is missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    pairwise = pairwise[pairwise["data_role"] == SIGNAL_ROLE].copy()
    for column in RUN_VALUE_COLUMNS:
        if column in pairwise.columns:
            pairwise[column] = _to_numeric(pairwise[column])
    return pairwise


def _load_candidate_rows(candidate_rows_path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_rows_path)
    missing = set(JOIN_KEYS + ["candidate_reason"]) - set(candidates.columns)
    if missing:
        msg = f"{candidate_rows_path} is missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    return candidates[candidates["data_role"] == SIGNAL_ROLE].copy()


def build_localization_rows(
    pairwise: pd.DataFrame,
    candidates: pd.DataFrame,
    positive_ari_epsilon: float = 1e-12,
) -> pd.DataFrame:
    node_summary = _summarize_candidate_nodes(candidates)
    localized = pairwise.merge(node_summary, on=JOIN_KEYS, how="left")

    fill_zero_columns = [
        "candidate_count",
        "decision_divergence_count",
        "suppressed_candidate_count",
        "split_suppressed_by_guard_count",
        "pass_through_suppressed_count",
        "split_to_pass_through_guard_count",
        "right_guard_blocked_count",
        "suppressed_right_guard_blocked_count",
        "old_and_current_evidence_count",
        "suppressed_old_and_current_evidence_count",
        "traversal_only_pair_count",
        "suppressed_traversal_only_pair_count",
    ]
    for column in fill_zero_columns:
        if column not in localized.columns:
            localized[column] = 0
        localized[column] = localized[column].fillna(0).astype(int)
    if "suppressed_node_examples" not in localized.columns:
        localized["suppressed_node_examples"] = ""
    localized["suppressed_node_examples"] = localized["suppressed_node_examples"].fillna("")

    localized["suppression_localization_status"] = localized.apply(
        _suppression_status,
        axis=1,
        positive_ari_epsilon=positive_ari_epsilon,
    )
    localized["schema_version"] = SCHEMA_VERSION
    localized["study_role"] = STUDY_ROLE
    front = [
        "schema_version",
        "study_role",
        "case_id",
        "data_role",
        "replicate",
        "suppression_localization_status",
    ]
    remaining = [column for column in localized.columns if column not in front]
    return localized[front + remaining].sort_values(
        [
            "suppression_localization_status",
            "delta_run_row_ari_left_minus_right",
            "delta_n_clusters_left_minus_right",
            "case_id",
            "replicate",
        ],
        ascending=[True, False, False, True, True],
    )


def build_case_summary(localized: pd.DataFrame) -> pd.DataFrame:
    if localized.empty:
        return pd.DataFrame()
    rows = []
    for case_id, group in localized.groupby("case_id", dropna=False):
        useful = group[group["suppression_localization_status"] == "useful_movement_suppressed"]
        suppressed = group[group["suppressed_candidate_count"] > 0]
        row: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": case_id,
            "signal_replicate_count": int(group["replicate"].nunique()),
            "suppressed_replicate_count": int(suppressed["replicate"].nunique()),
            "useful_suppressed_replicate_count": int(useful["replicate"].nunique()),
            "harmful_or_overfragmented_replicate_count": int(
                (
                    group["suppression_localization_status"]
                    == "extra_movement_hurts_or_overfragments"
                ).sum()
            ),
            "neutral_extra_movement_replicate_count": int(
                (group["suppression_localization_status"] == "extra_movement_neutral").sum()
            ),
            "mean_delta_run_row_ari_left_minus_right": float(
                group["delta_run_row_ari_left_minus_right"].mean()
            ),
            "max_delta_run_row_ari_left_minus_right": float(
                group["delta_run_row_ari_left_minus_right"].max()
            ),
            "mean_delta_n_clusters_left_minus_right": float(
                group["delta_n_clusters_left_minus_right"].mean()
            ),
            "suppressed_candidate_count": int(group["suppressed_candidate_count"].sum()),
            "split_suppressed_by_guard_count": int(
                group["split_suppressed_by_guard_count"].sum()
            ),
            "pass_through_suppressed_count": int(
                group["pass_through_suppressed_count"].sum()
            ),
            "suppressed_old_and_current_evidence_count": int(
                group["suppressed_old_and_current_evidence_count"].sum()
            ),
            "suppressed_traversal_only_pair_count": int(
                group["suppressed_traversal_only_pair_count"].sum()
            ),
        }
        if not useful.empty:
            row["top_useful_replicates"] = ";".join(
                str(int(rep)) for rep in useful.sort_values(
                    "delta_run_row_ari_left_minus_right",
                    ascending=False,
                )["replicate"].head(5)
            )
        else:
            row["top_useful_replicates"] = ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "useful_suppressed_replicate_count",
            "max_delta_run_row_ari_left_minus_right",
            "suppressed_candidate_count",
        ],
        ascending=[False, False, False],
    )


def build_status_summary(localized: pd.DataFrame) -> pd.DataFrame:
    if localized.empty:
        return pd.DataFrame()
    rows = []
    for status, group in localized.groupby("suppression_localization_status", dropna=False):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "suppression_localization_status": status,
                "run_count": int(len(group)),
                "case_count": int(group["case_id"].nunique()),
                "mean_delta_run_row_ari_left_minus_right": float(
                    group["delta_run_row_ari_left_minus_right"].mean()
                ),
                "max_delta_run_row_ari_left_minus_right": float(
                    group["delta_run_row_ari_left_minus_right"].max()
                ),
                "mean_delta_n_clusters_left_minus_right": float(
                    group["delta_n_clusters_left_minus_right"].mean()
                ),
                "suppressed_candidate_count": int(group["suppressed_candidate_count"].sum()),
                "split_suppressed_by_guard_count": int(
                    group["split_suppressed_by_guard_count"].sum()
                ),
                "pass_through_suppressed_count": int(
                    group["pass_through_suppressed_count"].sum()
                ),
                "suppressed_old_and_current_evidence_count": int(
                    group["suppressed_old_and_current_evidence_count"].sum()
                ),
                "suppressed_traversal_only_pair_count": int(
                    group["suppressed_traversal_only_pair_count"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("run_count", ascending=False)


def run_analysis(
    pairwise_path: Path,
    candidate_rows_path: Path,
    output_dir: Path,
    positive_ari_epsilon: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairwise = _load_signal_pairwise(pairwise_path)
    candidates = _load_candidate_rows(candidate_rows_path)
    localized = build_localization_rows(
        pairwise,
        candidates,
        positive_ari_epsilon=positive_ari_epsilon,
    )
    case_summary = build_case_summary(localized)
    status_summary = build_status_summary(localized)

    output_dir.mkdir(parents=True, exist_ok=True)
    localized_path = output_dir / "overlap_signal_suppression_localization_rows.csv"
    case_summary_path = output_dir / "overlap_signal_suppression_case_summary.csv"
    status_summary_path = output_dir / "overlap_signal_suppression_status_summary.csv"
    manifest_path = output_dir / "manifest.json"

    localized.to_csv(localized_path, index=False)
    case_summary.to_csv(case_summary_path, index=False)
    status_summary.to_csv(status_summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "pairwise": pairwise_path,
        "candidate_rows": candidate_rows_path,
        "positive_ari_epsilon": positive_ari_epsilon,
        "signal_pair_count": int(len(pairwise)),
        "signal_candidate_row_count": int(len(candidates)),
        "localized_row_count": int(len(localized)),
        "outputs": {
            "localized_rows": localized_path,
            "case_summary": case_summary_path,
            "status_summary": status_summary_path,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return localized, case_summary, status_summary


def main() -> None:
    args = parse_args()
    run_analysis(
        pairwise_path=args.pairwise,
        candidate_rows_path=args.candidate_rows,
        output_dir=args.output_dir,
        positive_ari_epsilon=args.positive_ari_epsilon,
    )


if __name__ == "__main__":
    main()
