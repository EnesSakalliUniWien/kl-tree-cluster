"""Consensus selector and artifacts for graphtools adaptive-K tree grids."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from tree_break_selection.tree.construction import (
    NEIGHBOR_JOINING_TREE_BUILDER,
    TREE_CONSENSUS_STRATEGIES,
    TREE_CONSENSUS_STRATEGY_PRIORITY,
)

TARGET_TREE_CONSENSUS_METHOD = "tbs_diffusion_graphtools_adaptive_nnls"
TARGET_TREE_CONSENSUS_GRID = "graphtools_adaptive_k_tree_strategy"

TREE_PRIORITY: dict[str, int] = TREE_CONSENSUS_STRATEGY_PRIORITY

LABEL_COLUMNS: list[str] = [
    "test_case",
    "case_id",
    "method",
    "run_id",
    "sample_id",
    "cluster_label",
]

EXTERNAL_AUDIT_COLUMNS: tuple[str, ...] = (
    "ari",
    "nmi",
    "macro_f1",
    "purity",
    "true_clusters",
)


@dataclass(frozen=True)
class TreeConsensusTables:
    """In-memory tables produced by the tree-consensus analysis."""

    label_assignments: pd.DataFrame
    pairwise_agreement: pd.DataFrame
    stability: pd.DataFrame
    rankings: pd.DataFrame
    selection: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class TreeConsensusArtifacts:
    """Filesystem artifacts emitted by the tree-consensus writer."""

    label_assignments_csv: Path
    pairwise_agreement_csv: Path
    stability_csv: Path
    rankings_csv: Path
    selection_csv: Path
    summary_csv: Path
    report_md: Path


def tree_inference_from_run_id(run_id: str) -> str:
    """Return the tree strategy encoded in a benchmark grid run id."""
    raw = str(run_id)
    if "tree_builder_neighbor_joining" in raw:
        return NEIGHBOR_JOINING_TREE_BUILDER
    for tree_inference in TREE_CONSENSUS_STRATEGIES:
        if f"tree_linkage_method_{tree_inference}" in raw:
            return tree_inference
    raise ValueError(f"Cannot infer tree strategy from run_id={run_id!r}.")


def _slugify(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, table: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def _target_results(results: pd.DataFrame) -> pd.DataFrame:
    required = {
        "test_case",
        "case_id",
        "case_category",
        "method",
        "run_id",
        "benchmark_grid",
        "status",
        "true_clusters",
        "found_clusters",
        "labels_length",
        "ari",
        "nmi",
        "macro_f1",
        "purity",
        "silhouette_score",
        "calinski_harabasz_index",
        "davies_bouldin_index",
        "largest_cluster_fraction",
    }
    _require_columns(results, required, table="Tree consensus results")

    target = results[
        results["method"].astype(str).eq(TARGET_TREE_CONSENSUS_METHOD)
        & results["benchmark_grid"].astype(str).eq(TARGET_TREE_CONSENSUS_GRID)
    ].copy()
    if target.empty:
        raise ValueError("Tree consensus requires graphtools adaptive-K NNLS tree-strategy rows.")

    duplicates = target.duplicated(["case_id", "run_id"], keep=False)
    if duplicates.any():
        examples = (
            target.loc[duplicates, ["case_id", "run_id"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Tree consensus requires unique case_id/run_id rows; duplicates include {examples}."
        )

    target["tree_inference"] = target["run_id"].map(tree_inference_from_run_id)
    target["tree_priority"] = target["tree_inference"].map(TREE_PRIORITY).astype(int)
    for column in [
        "test_case",
        "true_clusters",
        "found_clusters",
        "labels_length",
        "ari",
        "nmi",
        "macro_f1",
        "purity",
        "silhouette_score",
        "calinski_harabasz_index",
        "davies_bouldin_index",
        "largest_cluster_fraction",
    ]:
        target[column] = pd.to_numeric(target[column], errors="coerce")
    return target


def _read_label_assignments(labels_dir: Path | str) -> pd.DataFrame:
    root = Path(labels_dir)
    if not root.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*.csv")):
        frame = pd.read_csv(path)
        _require_columns(frame, LABEL_COLUMNS, table=f"Label assignment file {path}")
        frames.append(frame[LABEL_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    labels = pd.concat(frames, ignore_index=True)
    labels["cluster_label"] = pd.to_numeric(labels["cluster_label"], errors="raise").astype(int)
    return labels


def _relevant_labels(target: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    case_ids = set(target["case_id"].astype(str))
    run_ids = set(target["run_id"].astype(str))
    relevant = labels[
        labels["case_id"].astype(str).isin(case_ids) & labels["run_id"].astype(str).isin(run_ids)
    ].copy()
    return relevant[LABEL_COLUMNS].reset_index(drop=True)


def _validate_label_integrity(target: pd.DataFrame, labels: pd.DataFrame) -> bool:
    ok_rows = target[target["status"].astype(str).eq("ok") & target["labels_length"].gt(0)]
    expected_label_count = int(ok_rows["labels_length"].sum())
    if expected_label_count == 0:
        return labels.empty

    counts = labels.groupby(["case_id", "run_id"], dropna=False).size()
    missing: list[dict[str, object]] = []
    for row in ok_rows.itertuples(index=False):
        key = (str(row.case_id), str(row.run_id))
        observed = int(counts.get(key, 0))
        expected = int(row.labels_length)
        if observed != expected:
            missing.append(
                {
                    "case_id": row.case_id,
                    "run_id": row.run_id,
                    "expected": expected,
                    "observed": observed,
                }
            )
    if missing:
        raise ValueError(
            f"Missing label assignments for successful tree consensus rows; examples={missing[:5]}."
        )
    if len(labels) != expected_label_count:
        raise ValueError(
            "Tree consensus label row count mismatch: "
            f"expected {expected_label_count}, observed {len(labels)}."
        )
    return True


def _labels_for(labels: pd.DataFrame, *, case_id: str, run_id: str) -> pd.Series | None:
    subset = labels[
        labels["case_id"].astype(str).eq(str(case_id))
        & labels["run_id"].astype(str).eq(str(run_id))
    ]
    if subset.empty:
        return None
    series = subset.set_index("sample_id")["cluster_label"]
    return series.sort_index()


def _pairwise_agreement(target: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for case_id, group in target.groupby("case_id", sort=False):
        ordered = group.sort_values(["tree_priority", "run_id"])
        for left, right in combinations(ordered.itertuples(index=False), 2):
            left_labels = _labels_for(labels, case_id=str(case_id), run_id=str(left.run_id))
            right_labels = _labels_for(labels, case_id=str(case_id), run_id=str(right.run_id))
            if (
                left_labels is None
                or right_labels is None
                or set(left_labels.index) != set(right_labels.index)
            ):
                partition_ari = np.nan
            else:
                right_aligned = right_labels.loc[left_labels.index]
                partition_ari = float(adjusted_rand_score(left_labels, right_aligned))
            records.append(
                {
                    "case_id": case_id,
                    "left_run_id": left.run_id,
                    "right_run_id": right.run_id,
                    "left_method": left.method,
                    "right_method": right.method,
                    "left_tree_inference": left.tree_inference,
                    "right_tree_inference": right.tree_inference,
                    "partition_ari": partition_ari,
                }
            )
    return pd.DataFrame.from_records(records)


def _stability(target: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in target.itertuples(index=False):
        values = pairs[
            pairs["case_id"].astype(str).eq(str(row.case_id))
            & (
                pairs["left_run_id"].astype(str).eq(str(row.run_id))
                | pairs["right_run_id"].astype(str).eq(str(row.run_id))
            )
        ]["partition_ari"].dropna()
        records.append(
            {
                "case_id": row.case_id,
                "method": row.method,
                "run_id": row.run_id,
                "tree_inference": row.tree_inference,
                "mean_partition_agreement": float(values.mean()) if not values.empty else np.nan,
                "median_partition_agreement": (
                    float(values.median()) if not values.empty else np.nan
                ),
                "min_partition_agreement": float(values.min()) if not values.empty else np.nan,
                "max_partition_agreement": float(values.max()) if not values.empty else np.nan,
                "agreement_pair_count": int(values.size),
            }
        )
    return pd.DataFrame.from_records(records)


def _add_case_rank(
    frame: pd.DataFrame,
    *,
    source: str,
    rank_column: str,
    ascending: bool,
) -> None:
    frame[rank_column] = np.nan
    for _case_id, index in frame.groupby("case_id").groups.items():
        subset = frame.loc[index]
        valid = subset["valid_internal_metrics"] & subset[source].notna()
        ranks = subset.loc[valid, source].rank(method="average", ascending=ascending)
        frame.loc[ranks.index, rank_column] = ranks


def _rank_and_select(
    target: pd.DataFrame, stability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rankings = target.merge(
        stability[
            [
                "case_id",
                "run_id",
                "mean_partition_agreement",
                "median_partition_agreement",
                "agreement_pair_count",
            ]
        ],
        on=["case_id", "run_id"],
        how="left",
    )
    rankings["valid_internal_metrics"] = (
        rankings["status"].astype(str).eq("ok")
        & rankings["found_clusters"].ge(2)
        & rankings["silhouette_score"].notna()
        & rankings["calinski_harabasz_index"].notna()
        & rankings["davies_bouldin_index"].notna()
    )
    rankings["dominant_cluster_penalty"] = np.where(
        rankings["largest_cluster_fraction"].ge(0.45),
        3.0,
        0.0,
    )
    rankings.loc[~rankings["valid_internal_metrics"], "dominant_cluster_penalty"] = 100.0

    _add_case_rank(
        rankings,
        source="silhouette_score",
        rank_column="silhouette_rank",
        ascending=False,
    )
    _add_case_rank(
        rankings,
        source="calinski_harabasz_index",
        rank_column="calinski_harabasz_rank",
        ascending=False,
    )
    _add_case_rank(
        rankings,
        source="davies_bouldin_index",
        rank_column="davies_bouldin_rank",
        ascending=True,
    )
    _add_case_rank(
        rankings,
        source="found_clusters",
        rank_column="cluster_count_rank",
        ascending=True,
    )
    _add_case_rank(
        rankings,
        source="mean_partition_agreement",
        rank_column="method_stability_rank",
        ascending=False,
    )

    core_cols = [
        "silhouette_rank",
        "calinski_harabasz_rank",
        "davies_bouldin_rank",
    ]
    rankings["internal_core_rank_mean"] = rankings[core_cols].mean(axis=1, skipna=False)
    rankings["sharp_label_free_score"] = (
        rankings["internal_core_rank_mean"]
        + 0.25 * rankings["cluster_count_rank"]
        + rankings["dominant_cluster_penalty"]
    )
    rankings["sharp_label_free_score_rounded"] = rankings["sharp_label_free_score"].round(9)

    selections: list[dict[str, object]] = []
    for case_id, subset in rankings.groupby("case_id", sort=False):
        case_meta = subset.iloc[0]
        valid = subset[subset["valid_internal_metrics"]]
        if valid.empty:
            selections.append(
                {
                    "case_id": case_id,
                    "test_case": case_meta["test_case"],
                    "case_category": case_meta["case_category"],
                    "selector_status": "skip_no_valid_topology",
                    "selected_run_id": np.nan,
                    "selected_tree_inference": np.nan,
                    "selected_found_clusters": np.nan,
                    "selected_sharp_label_free_score": np.nan,
                    "selected_mean_partition_agreement": np.nan,
                    "selected_ari": np.nan,
                    "selected_nmi": np.nan,
                    "selected_macro_f1": np.nan,
                    "selected_purity": np.nan,
                    "average_ari": _metric_for_tree(subset, "average", "ari"),
                    "average_nmi": _metric_for_tree(subset, "average", "nmi"),
                    "average_macro_f1": _metric_for_tree(subset, "average", "macro_f1"),
                    "weighted_ari": _metric_for_tree(subset, "weighted", "ari"),
                    "weighted_nmi": _metric_for_tree(subset, "weighted", "nmi"),
                    "weighted_macro_f1": _metric_for_tree(subset, "weighted", "macro_f1"),
                }
            )
            continue
        selected = valid.sort_values(
            [
                "sharp_label_free_score_rounded",
                "method_stability_rank",
                "internal_core_rank_mean",
                "cluster_count_rank",
                "tree_priority",
                "run_id",
            ],
            ascending=[True, True, True, True, True, True],
            na_position="last",
        ).iloc[0]
        selections.append(
            {
                "case_id": case_id,
                "test_case": selected["test_case"],
                "case_category": selected["case_category"],
                "selector_status": "selected",
                "selected_run_id": selected["run_id"],
                "selected_tree_inference": selected["tree_inference"],
                "selected_found_clusters": selected["found_clusters"],
                "selected_sharp_label_free_score": selected["sharp_label_free_score"],
                "selected_mean_partition_agreement": selected["mean_partition_agreement"],
                "selected_ari": selected["ari"],
                "selected_nmi": selected["nmi"],
                "selected_macro_f1": selected["macro_f1"],
                "selected_purity": selected["purity"],
                "average_ari": _metric_for_tree(subset, "average", "ari"),
                "average_nmi": _metric_for_tree(subset, "average", "nmi"),
                "average_macro_f1": _metric_for_tree(subset, "average", "macro_f1"),
                "weighted_ari": _metric_for_tree(subset, "weighted", "ari"),
                "weighted_nmi": _metric_for_tree(subset, "weighted", "nmi"),
                "weighted_macro_f1": _metric_for_tree(subset, "weighted", "macro_f1"),
            }
        )
    selection = pd.DataFrame.from_records(selections).sort_values(["test_case", "case_id"])
    return rankings, selection.reset_index(drop=True)


def _metric_for_tree(subset: pd.DataFrame, tree_inference: str, metric: str) -> float:
    rows = subset[subset["tree_inference"].astype(str).eq(tree_inference)]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0][metric])


def _label_free_invariant(
    target: pd.DataFrame,
    stability: pd.DataFrame,
    selection: pd.DataFrame,
) -> bool:
    perturbed = target.copy()
    for column in EXTERNAL_AUDIT_COLUMNS:
        if column in perturbed.columns:
            perturbed[column] = np.nan
    _rankings, perturbed_selection = _rank_and_select(perturbed, stability)
    left = selection[["case_id", "selected_run_id"]].sort_values("case_id").reset_index(drop=True)
    right = (
        perturbed_selection[["case_id", "selected_run_id"]]
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    return left.equals(right)


def _paired_metric_summary(
    selection: pd.DataFrame,
    *,
    selected_col: str,
    baseline_col: str,
) -> tuple[int, float, float, float, float]:
    paired = selection[
        selection["selector_status"].eq("selected")
        & selection[selected_col].notna()
        & selection[baseline_col].notna()
    ]
    if paired.empty:
        return 0, np.nan, np.nan, np.nan, np.nan
    return (
        int(len(paired)),
        float(paired[selected_col].mean()),
        float(paired[baseline_col].mean()),
        float(paired[selected_col].median()),
        float(paired[baseline_col].median()),
    )


def _category_loss_count(selection: pd.DataFrame) -> int:
    valid = selection[
        selection["selector_status"].eq("selected")
        & selection["selected_ari"].notna()
        & selection["average_ari"].notna()
        & selection["weighted_ari"].notna()
    ]
    if valid.empty:
        return 0
    losses = 0
    for _category, group in valid.groupby("case_category", dropna=False):
        if len(group) < 5:
            continue
        selected_mean = float(group["selected_ari"].mean())
        average_mean = float(group["average_ari"].mean())
        weighted_mean = float(group["weighted_ari"].mean())
        if selected_mean < average_mean - 0.02 and selected_mean < weighted_mean - 0.02:
            losses += 1
    return losses


def _summary(
    target: pd.DataFrame,
    labels: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    label_integrity_pass: bool,
    label_free_integrity_pass: bool,
) -> pd.DataFrame:
    case_count = int(target["case_id"].nunique())
    expected_rows = case_count * len(TREE_CONSENSUS_STRATEGIES)
    result_rows = int(len(target))
    per_case_complete = target.groupby("case_id")["tree_inference"].agg(
        lambda values: set(values.astype(str)) == set(TREE_CONSENSUS_STRATEGIES)
    )
    completeness_pass = (
        result_rows == expected_rows
        and int(target["run_id"].nunique()) == len(TREE_CONSENSUS_STRATEGIES)
        and bool(per_case_complete.all())
    )

    metric_passes: dict[str, bool] = {}
    median_passes: dict[str, bool] = {}
    metric_records: dict[str, object] = {}
    for metric in ("ari", "nmi", "macro_f1"):
        selected_col = f"selected_{metric}"
        for baseline in ("average", "weighted"):
            baseline_col = f"{baseline}_{metric}"
            count, selected_mean, baseline_mean, selected_median, baseline_median = (
                _paired_metric_summary(
                    selection,
                    selected_col=selected_col,
                    baseline_col=baseline_col,
                )
            )
            key = f"{metric}_vs_{baseline}"
            metric_records[f"{key}_paired_cases"] = count
            metric_records[f"{key}_selected_mean"] = selected_mean
            metric_records[f"{key}_baseline_mean"] = baseline_mean
            metric_records[f"{key}_selected_median"] = selected_median
            metric_records[f"{key}_baseline_median"] = baseline_median
            metric_passes[key] = count > 0 and selected_mean > baseline_mean
            median_passes[key] = count > 0 and selected_median >= baseline_median - 0.005

    category_loss_count = _category_loss_count(selection)
    promotion_pass = (
        completeness_pass
        and label_integrity_pass
        and label_free_integrity_pass
        and all(metric_passes.values())
        and all(median_passes.values())
        and category_loss_count == 0
    )
    selected_rows = selection[selection["selector_status"].eq("selected")]
    summary = {
        "gate_status": "pass" if promotion_pass else "no_promotion",
        "cases": case_count,
        "result_rows": result_rows,
        "expected_result_rows": expected_rows,
        "run_cells": int(target["run_id"].nunique()),
        "expected_run_cells": len(TREE_CONSENSUS_STRATEGIES),
        "selected_cases": int(len(selected_rows)),
        "skipped_cases": int(selection["selector_status"].ne("selected").sum()),
        "label_rows": int(len(labels)),
        "expected_label_rows": int(
            target[target["status"].astype(str).eq("ok")]["labels_length"].sum()
        ),
        "completeness_pass": bool(completeness_pass),
        "label_integrity_pass": bool(label_integrity_pass),
        "label_free_integrity_pass": bool(label_free_integrity_pass),
        "metric_mean_pass": bool(all(metric_passes.values())),
        "metric_median_pass": bool(all(median_passes.values())),
        "category_loss_count": int(category_loss_count),
        "category_gate_pass": bool(category_loss_count == 0),
        "mean_selected_ari": float(selected_rows["selected_ari"].mean())
        if not selected_rows.empty
        else np.nan,
        "median_selected_ari": float(selected_rows["selected_ari"].median())
        if not selected_rows.empty
        else np.nan,
        "mean_selected_nmi": float(selected_rows["selected_nmi"].mean())
        if not selected_rows.empty
        else np.nan,
        "mean_selected_macro_f1": float(selected_rows["selected_macro_f1"].mean())
        if not selected_rows.empty
        else np.nan,
        **metric_records,
    }
    return pd.DataFrame([summary])


def build_tree_consensus_tables(
    results: pd.DataFrame,
    label_assignments: pd.DataFrame,
) -> TreeConsensusTables:
    """Build label-free tree-consensus tables from benchmark rows and labels."""
    target = _target_results(results)
    labels = _relevant_labels(target, label_assignments)
    label_integrity_pass = _validate_label_integrity(target, labels)
    pairs = _pairwise_agreement(target, labels)
    stability = _stability(target, pairs)
    rankings, selection = _rank_and_select(target, stability)
    label_free_integrity_pass = _label_free_invariant(target, stability, selection)
    summary = _summary(
        target,
        labels,
        selection,
        label_integrity_pass=label_integrity_pass,
        label_free_integrity_pass=label_free_integrity_pass,
    )
    return TreeConsensusTables(
        label_assignments=labels,
        pairwise_agreement=pairs,
        stability=stability,
        rankings=rankings,
        selection=selection,
        summary=summary,
    )


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    visible = frame.head(max_rows).copy()
    for column in visible.columns:
        if pd.api.types.is_float_dtype(visible[column]):
            visible[column] = visible[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    lines = [
        "| " + " | ".join(visible.columns) + " |",
        "| " + " | ".join("---" for _column in visible.columns) + " |",
    ]
    for row in visible.itertuples(index=False):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _report_text(tables: TreeConsensusTables, *, source_path: Path | None) -> str:
    summary = tables.summary.iloc[0]
    source_line = f"- Source rows: `{source_path}`" if source_path is not None else ""
    selection_cols = [
        "case_id",
        "selected_tree_inference",
        "selected_found_clusters",
        "selected_ari",
        "average_ari",
        "weighted_ari",
        "selector_status",
    ]
    return "\n".join(
        [
            "# Tree Consensus Gate",
            "",
            source_line,
            f"- Gate status: `{summary['gate_status']}`",
            f"- Result rows: `{int(summary['result_rows'])}`",
            f"- Expected result rows: `{int(summary['expected_result_rows'])}`",
            f"- Label rows: `{int(summary['label_rows'])}`",
            f"- Expected label rows: `{int(summary['expected_label_rows'])}`",
            "",
            "## Gate Checks",
            "",
            _markdown_table(
                tables.summary[
                    [
                        "gate_status",
                        "completeness_pass",
                        "label_integrity_pass",
                        "label_free_integrity_pass",
                        "metric_mean_pass",
                        "metric_median_pass",
                        "category_gate_pass",
                    ]
                ]
            ),
            "",
            "## Selected Topologies",
            "",
            _markdown_table(tables.selection[selection_cols], max_rows=200),
            "",
        ]
    )


def write_tree_consensus_artifacts(
    results: pd.DataFrame,
    labels_dir: Path | str,
    output_dir: Path | str,
    *,
    source_path: Path | str | None = None,
) -> TreeConsensusArtifacts:
    """Write tree-consensus artifacts from benchmark results and label files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = _read_label_assignments(labels_dir)
    tables = build_tree_consensus_tables(results, labels)

    label_assignments_csv = output / "tree_consensus_label_assignments.csv"
    pairwise_agreement_csv = output / "tree_consensus_pairwise_agreement.csv"
    stability_csv = output / "tree_consensus_stability.csv"
    rankings_csv = output / "tree_consensus_rankings.csv"
    selection_csv = output / "tree_consensus_selection.csv"
    summary_csv = output / "tree_consensus_summary.csv"
    report_md = output / "tree_consensus_report.md"

    tables.label_assignments.to_csv(label_assignments_csv, index=False)
    tables.pairwise_agreement.to_csv(pairwise_agreement_csv, index=False)
    tables.stability.to_csv(stability_csv, index=False)
    tables.rankings.to_csv(rankings_csv, index=False)
    tables.selection.to_csv(selection_csv, index=False)
    tables.summary.to_csv(summary_csv, index=False)
    report_md.write_text(
        _report_text(
            tables,
            source_path=None if source_path is None else Path(source_path),
        ),
        encoding="utf-8",
    )

    return TreeConsensusArtifacts(
        label_assignments_csv=label_assignments_csv,
        pairwise_agreement_csv=pairwise_agreement_csv,
        stability_csv=stability_csv,
        rankings_csv=rankings_csv,
        selection_csv=selection_csv,
        summary_csv=summary_csv,
        report_md=report_md,
    )


def export_tree_consensus_label_files(
    computed_results: Iterable[object],
    output_dir: Path | str,
) -> list[Path]:
    """Export one atomic label CSV per successful tree-consensus run cell."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for result in computed_results:
        if getattr(result, "method", None) != TARGET_TREE_CONSENSUS_METHOD:
            continue
        if getattr(result, "benchmark_grid", None) != TARGET_TREE_CONSENSUS_GRID:
            continue
        raw_labels = getattr(result, "labels", None)
        data = getattr(result, "data", None)
        if raw_labels is None or data is None or not hasattr(data, "index"):
            raise ValueError(
                "Cannot export tree consensus labels: computed result must include "
                "labels and a data index."
            )
        labels = np.asarray(raw_labels)
        sample_ids = list(data.index.astype(str))
        if labels.size != len(sample_ids):
            raise ValueError(
                "Cannot export tree consensus labels: labels do not align to data index."
            )
        case_id = str(getattr(result, "meta", {}).get("name", f"case_{result.test_case_num}"))
        run_id = str(getattr(result, "run_id"))
        frame = pd.DataFrame(
            {
                "test_case": int(getattr(result, "test_case_num")),
                "case_id": case_id,
                "method": str(getattr(result, "method")),
                "run_id": run_id,
                "sample_id": sample_ids,
                "cluster_label": labels.astype(int),
            },
            columns=LABEL_COLUMNS,
        )
        filename = f"case_{int(result.test_case_num):04d}__{_slugify(run_id)}.csv"
        path = output / filename
        tmp_path = output / f".{filename}.tmp"
        frame.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
        paths.append(path)
    return paths


__all__ = [
    "TREE_CONSENSUS_STRATEGIES",
    "TARGET_TREE_CONSENSUS_GRID",
    "TARGET_TREE_CONSENSUS_METHOD",
    "TreeConsensusArtifacts",
    "TreeConsensusTables",
    "build_tree_consensus_tables",
    "export_tree_consensus_label_files",
    "tree_inference_from_run_id",
    "write_tree_consensus_artifacts",
]
