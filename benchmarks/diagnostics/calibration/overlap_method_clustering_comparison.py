"""Compare two clustering methods on shared overlap checkpoint assignments.

The selected-pass-through diagnostics write sample-level ``*.genes.csv`` files
for each case, data role, method, and replicate. This panel pairs two method
profiles on the same case/role/replicate keys and reports both per-method
fragmentation metrics and label-invariant partition agreement.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

from benchmarks.shared.metrics import _calculate_fragmentation_metrics

SCHEMA_VERSION = "overlap_method_clustering_comparison/v1"
STUDY_ROLE = "diagnostic_overlap_method_clustering_comparison_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "overlap_method_clustering_comparison"
)

DEFAULT_LEFT_METHOD = "fixed_coordinate_conditional_topology_diagnostic_v1"
DEFAULT_RIGHT_METHOD = "fixed_coordinate_global_passthrough_refined_v1"
DEFAULT_CHECKPOINT_ROOT = Path(
    "raw/assets/benchmark-results/specific_small_method_benchmark_20260615/"
    "overlap_selected_pass_through_expanded_traversal/checkpoint_rows"
)

FILENAME_RE = re.compile(
    r"^case=(?P<case_id>.+?)__role=(?P<filename_role>.+?)"
    r"__method=(?P<method_id>.+?)__replicate=(?P<replicate>\d+)\.genes\.csv$"
)

CLUSTER_COLUMN_CANDIDATES = (
    "final_fragment_cluster_id",
    "cluster_id",
    "predicted_cluster_id",
    "predicted_cluster",
    "assignment",
)
TRUTH_COLUMN_CANDIDATES = (
    "true_cluster_id",
    "truth_cluster_id",
    "truth_label",
    "true_label",
    "ground_truth",
    "reference_cluster_id",
)
ROW_METADATA_COLUMNS = (
    "true_clusters",
    "found_clusters",
    "ari",
    "exact_cluster_count",
    "false_split",
    "adaptive_projection_avoided",
    "stable_boundary_count",
    "selected_root_blocked_count",
    "selected_family_blocked_count",
    "unstable_passthrough_zone_count",
    "accepted_internal_split_count",
    "leaf_fragment_count",
    "selected_family_guard_tested_count",
    "selected_family_guard_block_count",
    "min_selected_family_p_value",
)


@dataclass(frozen=True)
class CheckpointKey:
    """Identity shared by method outputs that can be compared directly."""

    case_id: str
    data_role: str
    replicate: int


@dataclass(frozen=True)
class AssignmentRecord:
    """Loaded checkpoint assignment plus its computed row metrics."""

    key: CheckpointKey
    filename_role: str
    method_id: str
    path: Path
    assignments: pd.DataFrame
    row: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two method profiles on overlapping checkpoint assignments."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Directory containing checkpoint_rows/*.genes.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for comparison CSVs and manifest.json.",
    )
    parser.add_argument(
        "--left-method",
        default=DEFAULT_LEFT_METHOD,
        help="First method profile to compare.",
    )
    parser.add_argument(
        "--right-method",
        default=DEFAULT_RIGHT_METHOD,
        help="Second method profile to compare.",
    )
    parser.add_argument(
        "--case-prefix",
        default=None,
        help="Optional case_id prefix filter, e.g. overlap_.",
    )
    return parser.parse_args()


def _parse_filename(path: Path) -> dict[str, object]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        msg = f"Checkpoint filename does not match expected schema: {path.name}"
        raise ValueError(msg)
    parsed = match.groupdict()
    parsed["replicate"] = int(parsed["replicate"])
    return parsed


def _first_present_column(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _read_row_metadata(path: Path) -> dict[str, object]:
    row_path = path.with_name(path.name.removesuffix(".genes.csv") + ".row.csv")
    if not row_path.exists():
        return {}
    row_df = pd.read_csv(row_path)
    if row_df.empty:
        return {}
    return row_df.iloc[0].to_dict()


def _to_float(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(value: object) -> int | None:
    as_float = _to_float(value)
    if np.isnan(as_float):
        return None
    return int(as_float)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _infer_truth_labels(
    genes: pd.DataFrame,
    row_metadata: dict[str, object],
) -> tuple[np.ndarray | None, str]:
    truth_column = _first_present_column(genes.columns, TRUTH_COLUMN_CANDIDATES)
    if truth_column is not None:
        return genes[truth_column].to_numpy(), truth_column

    true_clusters = _to_int(row_metadata.get("true_clusters"))
    if true_clusters == 1:
        return np.zeros(len(genes), dtype=int), "single_true_cluster_from_row_metadata"

    return None, "unavailable"


def _fragmentation_row(cluster_labels: pd.Series) -> dict[str, float]:
    (
        n_singleton_clusters,
        singleton_fraction,
        median_cluster_size,
        largest_cluster_fraction,
        effective_cluster_count,
        cluster_size_entropy,
        cluster_size_gini,
        noise_label_fraction,
    ) = _calculate_fragmentation_metrics(cluster_labels)
    return {
        "n_singleton_clusters": n_singleton_clusters,
        "singleton_fraction": singleton_fraction,
        "median_cluster_size": median_cluster_size,
        "largest_cluster_fraction": largest_cluster_fraction,
        "effective_cluster_count": effective_cluster_count,
        "cluster_size_entropy": cluster_size_entropy,
        "cluster_size_gini": cluster_size_gini,
        "noise_label_fraction": noise_label_fraction,
    }


def _external_metric_row(
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
) -> dict[str, float | bool]:
    if y_true is None:
        return {
            "external_metrics_available": False,
            "ari": float("nan"),
            "nmi": float("nan"),
            "ami": float("nan"),
            "homogeneity": float("nan"),
            "completeness": float("nan"),
            "v_measure": float("nan"),
            "fowlkes_mallows": float("nan"),
        }
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        y_true,
        y_pred,
    )
    return {
        "external_metrics_available": True,
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "fowlkes_mallows": float(fowlkes_mallows_score(y_true, y_pred)),
    }


def load_assignment_record(path: Path) -> AssignmentRecord:
    parsed = _parse_filename(path)
    genes = pd.read_csv(path)
    if "sample_id" not in genes.columns:
        msg = f"{path} does not contain a sample_id column."
        raise ValueError(msg)
    cluster_column = _first_present_column(genes.columns, CLUSTER_COLUMN_CANDIDATES)
    if cluster_column is None:
        msg = f"{path} does not contain a supported cluster assignment column."
        raise ValueError(msg)

    row_metadata = _read_row_metadata(path)
    data_role = str(genes["data_role"].iloc[0]) if "data_role" in genes.columns else str(parsed["filename_role"])
    method_id = str(genes["method_id"].iloc[0]) if "method_id" in genes.columns else str(parsed["method_id"])
    replicate = int(genes["replicate"].iloc[0]) if "replicate" in genes.columns else int(parsed["replicate"])
    case_id = str(genes["case_id"].iloc[0]) if "case_id" in genes.columns else str(parsed["case_id"])

    assignments = genes[["sample_id", cluster_column]].rename(
        columns={cluster_column: "cluster_id"}
    )
    assignments["sample_id"] = assignments["sample_id"].astype(str)
    if assignments["sample_id"].duplicated().any():
        msg = f"{path} contains duplicate sample_id values."
        raise ValueError(msg)

    y_true, truth_source = _infer_truth_labels(genes, row_metadata)
    row = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": case_id,
        "data_role": data_role,
        "filename_role": str(parsed["filename_role"]),
        "method_id": method_id,
        "replicate": replicate,
        "n_samples": int(len(assignments)),
        "n_clusters": int(assignments["cluster_id"].nunique(dropna=False)),
        "cluster_column": cluster_column,
        "truth_label_source": truth_source,
        "source_path": str(path),
    }
    row.update(_fragmentation_row(assignments["cluster_id"]))
    row.update(_external_metric_row(y_true, assignments["cluster_id"].to_numpy()))
    for column in ROW_METADATA_COLUMNS:
        value = row_metadata.get(column, np.nan)
        row[f"run_row_{column}"] = value

    return AssignmentRecord(
        key=CheckpointKey(case_id=case_id, data_role=data_role, replicate=replicate),
        filename_role=str(parsed["filename_role"]),
        method_id=method_id,
        path=path,
        assignments=assignments,
        row=row,
    )


def discover_records(
    checkpoint_root: Path,
    methods: tuple[str, str],
    case_prefix: str | None = None,
) -> list[AssignmentRecord]:
    records = []
    for path in sorted(checkpoint_root.glob("*.genes.csv")):
        parsed = _parse_filename(path)
        if parsed["method_id"] not in methods:
            continue
        if case_prefix is not None and not str(parsed["case_id"]).startswith(case_prefix):
            continue
        records.append(load_assignment_record(path))
    if not records:
        msg = f"No checkpoint assignment files found for {methods} under {checkpoint_root}"
        raise FileNotFoundError(msg)
    return records


def _pair_pattern(delta_n_clusters: float, delta_singletons: float) -> str:
    if delta_n_clusters > 0:
        return "left_more_fragmented"
    if delta_n_clusters < 0:
        return "right_more_fragmented"
    if delta_singletons > 0:
        return "left_more_singletons"
    if delta_singletons < 0:
        return "right_more_singletons"
    return "same_fragment_count"


def build_pairwise_rows(
    records: list[AssignmentRecord],
    left_method: str,
    right_method: str,
) -> pd.DataFrame:
    by_key_method = {(record.key, record.method_id): record for record in records}
    common_keys = sorted(
        {
            key
            for key, method_id in by_key_method
            if method_id == left_method and (key, right_method) in by_key_method
        },
        key=lambda key: (key.case_id, key.data_role, key.replicate),
    )

    rows = []
    for key in common_keys:
        left = by_key_method[(key, left_method)]
        right = by_key_method[(key, right_method)]
        merged = left.assignments.merge(
            right.assignments,
            on="sample_id",
            how="inner",
            suffixes=("_left", "_right"),
        )
        left_only = len(left.assignments) - len(merged)
        right_only = len(right.assignments) - len(merged)
        y_left = merged["cluster_id_left"].to_numpy()
        y_right = merged["cluster_id_right"].to_numpy()

        left_row = left.row
        right_row = right.row
        delta_n_clusters = _to_float(left_row["n_clusters"]) - _to_float(right_row["n_clusters"])
        delta_singletons = _to_float(left_row["n_singleton_clusters"]) - _to_float(
            right_row["n_singleton_clusters"]
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": key.case_id,
                "data_role": key.data_role,
                "replicate": key.replicate,
                "left_method_id": left_method,
                "right_method_id": right_method,
                "paired_sample_count": int(len(merged)),
                "left_only_sample_count": int(left_only),
                "right_only_sample_count": int(right_only),
                "same_raw_cluster_id_fraction": float(np.mean(y_left == y_right))
                if len(merged)
                else float("nan"),
                "partition_ari_between_methods": float(adjusted_rand_score(y_left, y_right))
                if len(merged)
                else float("nan"),
                "partition_nmi_between_methods": float(normalized_mutual_info_score(y_left, y_right))
                if len(merged)
                else float("nan"),
                "partition_ami_between_methods": float(adjusted_mutual_info_score(y_left, y_right))
                if len(merged)
                else float("nan"),
                "partition_fowlkes_mallows_between_methods": float(
                    fowlkes_mallows_score(y_left, y_right)
                )
                if len(merged)
                else float("nan"),
                "left_n_clusters": left_row["n_clusters"],
                "right_n_clusters": right_row["n_clusters"],
                "delta_n_clusters_left_minus_right": delta_n_clusters,
                "left_n_singleton_clusters": left_row["n_singleton_clusters"],
                "right_n_singleton_clusters": right_row["n_singleton_clusters"],
                "delta_singletons_left_minus_right": delta_singletons,
                "left_singleton_fraction": left_row["singleton_fraction"],
                "right_singleton_fraction": right_row["singleton_fraction"],
                "delta_singleton_fraction_left_minus_right": _to_float(
                    left_row["singleton_fraction"]
                )
                - _to_float(right_row["singleton_fraction"]),
                "left_largest_cluster_fraction": left_row["largest_cluster_fraction"],
                "right_largest_cluster_fraction": right_row["largest_cluster_fraction"],
                "delta_largest_cluster_fraction_left_minus_right": _to_float(
                    left_row["largest_cluster_fraction"]
                )
                - _to_float(right_row["largest_cluster_fraction"]),
                "left_effective_cluster_count": left_row["effective_cluster_count"],
                "right_effective_cluster_count": right_row["effective_cluster_count"],
                "delta_effective_cluster_count_left_minus_right": _to_float(
                    left_row["effective_cluster_count"]
                )
                - _to_float(right_row["effective_cluster_count"]),
                "left_run_row_ari": left_row["run_row_ari"],
                "right_run_row_ari": right_row["run_row_ari"],
                "delta_run_row_ari_left_minus_right": _to_float(left_row["run_row_ari"])
                - _to_float(right_row["run_row_ari"]),
                "fragmentation_pattern": _pair_pattern(delta_n_clusters, delta_singletons),
            }
        )
    return pd.DataFrame(rows)


def build_summary_rows(rows: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    if not rows.empty:
        group_columns = ["case_id", "data_role", "method_id"]
        value_columns = [
            "n_samples",
            "n_clusters",
            "run_row_found_clusters",
            "run_row_ari",
            "n_singleton_clusters",
            "singleton_fraction",
            "median_cluster_size",
            "largest_cluster_fraction",
            "effective_cluster_count",
            "cluster_size_entropy",
            "cluster_size_gini",
        ]
        for keys, group in rows.groupby(group_columns, dropna=False):
            row = {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "summary_family": "per_method",
                "case_id": keys[0],
                "data_role": keys[1],
                "method_id": keys[2],
                "replicate_count": int(group["replicate"].nunique()),
            }
            for column in value_columns:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"mean_{column}"] = float(values.mean())
                row[f"std_{column}"] = float(values.std(ddof=0))
            summary_rows.append(row)

    if not pairwise.empty:
        group_columns = ["case_id", "data_role"]
        value_columns = [
            "partition_ari_between_methods",
            "partition_nmi_between_methods",
            "delta_n_clusters_left_minus_right",
            "delta_singletons_left_minus_right",
            "delta_singleton_fraction_left_minus_right",
            "delta_largest_cluster_fraction_left_minus_right",
            "delta_effective_cluster_count_left_minus_right",
            "delta_run_row_ari_left_minus_right",
        ]
        for keys, group in pairwise.groupby(group_columns, dropna=False):
            row = {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "summary_family": "pairwise",
                "case_id": keys[0],
                "data_role": keys[1],
                "method_id": "paired_methods",
                "replicate_count": int(group["replicate"].nunique()),
                "dominant_fragmentation_pattern": str(
                    group["fragmentation_pattern"].value_counts().idxmax()
                ),
            }
            for column in value_columns:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"mean_{column}"] = float(values.mean())
                row[f"std_{column}"] = float(values.std(ddof=0))
            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def run_analysis(
    checkpoint_root: Path,
    output_dir: Path,
    left_method: str = DEFAULT_LEFT_METHOD,
    right_method: str = DEFAULT_RIGHT_METHOD,
    case_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = discover_records(
        checkpoint_root,
        (left_method, right_method),
        case_prefix=case_prefix,
    )
    rows = pd.DataFrame([record.row for record in records]).sort_values(
        ["case_id", "data_role", "method_id", "replicate"]
    )
    pairwise = build_pairwise_rows(records, left_method, right_method)
    summary = build_summary_rows(rows, pairwise)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "overlap_method_clustering_rows.csv"
    pairwise_path = output_dir / "overlap_method_clustering_pairwise.csv"
    summary_path = output_dir / "overlap_method_clustering_summary.csv"
    manifest_path = output_dir / "manifest.json"

    rows.to_csv(rows_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "checkpoint_root": checkpoint_root,
        "left_method": left_method,
        "right_method": right_method,
        "case_prefix": case_prefix,
        "assignment_file_count": int(len(records)),
        "paired_run_count": int(len(pairwise)),
        "outputs": {
            "rows": rows_path,
            "pairwise": pairwise_path,
            "summary": summary_path,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return rows, pairwise, summary


def main() -> None:
    args = parse_args()
    run_analysis(
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        left_method=args.left_method,
        right_method=args.right_method,
        case_prefix=args.case_prefix,
    )


if __name__ == "__main__":
    main()
