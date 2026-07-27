#!/usr/bin/env python3
"""Audit whether GO annotation clusters are internally biologically meaningful."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

GO_RE = re.compile(r"(GO:\d+)")


@dataclass(frozen=True)
class Dataset:
    label: str
    feature_matrix: Path
    annotation_root: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--feature-matrix", required=True, type=Path)
    parser.add_argument("--annotation-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-term-hits", type=int, default=2)
    parser.add_argument("--strong-q", type=float, default=0.01)
    parser.add_argument("--strong-lift", type=float, default=2.0)
    parser.add_argument("--moderate-lift", type=float, default=1.5)
    parser.add_argument("--null-iterations", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=20260619)
    parser.add_argument("--top-examples", type=int, default=30)
    return parser.parse_args()


def load_feature_matrix(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_csv(path, sep="\t")
    gene_column = frame.columns[0]
    data = frame.set_index(gene_column)
    data.index = data.index.astype(str)
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)
    data = (data > 0).astype(np.uint8)
    go_ids = []
    term_names = []
    for column in data.columns:
        match = GO_RE.search(str(column))
        go_ids.append(match.group(1) if match else "")
        term_names.append(GO_RE.sub("", str(column)).strip(" ()"))
    return data, term_names, go_ids


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.ones_like(p_values)
    finite = np.isfinite(p_values)
    if not finite.any():
        return q_values
    finite_indices = np.flatnonzero(finite)
    p = p_values[finite]
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0, 1)
    q_values[finite_indices[order]] = ranked
    return q_values


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def term_stats(
    cluster_counts: np.ndarray,
    *,
    cluster_size: int,
    background_counts: np.ndarray,
    population_size: int,
    term_names: list[str],
    go_ids: list[str],
    alpha: float,
    min_term_hits: int,
    strong_q: float,
    strong_lift: float,
    moderate_lift: float,
) -> dict[str, Any]:
    valid = background_counts > 0
    p_values = np.ones_like(background_counts, dtype=float)
    p_values[valid] = hypergeom.sf(
        cluster_counts[valid] - 1,
        population_size,
        background_counts[valid],
        cluster_size,
    )
    q_values = bh_qvalues(p_values)
    interpretable = cluster_counts >= min_term_hits
    significant = (q_values <= alpha) & interpretable
    candidates = np.flatnonzero(interpretable)

    if len(candidates) == 0:
        return {
            "n_significant_terms": 0,
            "best_q_value": math.nan,
            "top_term": "",
            "top_go_id": "",
            "top_hits": 0,
            "top_background_hits": 0,
            "top_cluster_prevalence": math.nan,
            "top_background_prevalence": math.nan,
            "top_prevalence_delta": math.nan,
            "top_lift": math.nan,
            "meaningfulness_label": "untested",
        }

    best = candidates[np.lexsort((-cluster_counts[candidates], q_values[candidates]))][0]
    cluster_prev = float(cluster_counts[best] / cluster_size)
    background_prev = float(background_counts[best] / population_size)
    lift = float(cluster_prev / background_prev) if background_prev > 0 else math.inf
    best_q = float(q_values[best])
    n_significant = int(significant.sum())
    if best_q <= strong_q and lift >= strong_lift and cluster_counts[best] >= max(3, min_term_hits):
        label = "strong"
    elif best_q <= alpha and lift >= moderate_lift and cluster_counts[best] >= min_term_hits:
        label = "moderate"
    elif best_q <= alpha:
        label = "statistical_but_broad"
    else:
        label = "weak_or_not_enriched"
    return {
        "n_significant_terms": n_significant,
        "best_q_value": best_q,
        "top_term": term_names[best],
        "top_go_id": go_ids[best],
        "top_hits": int(cluster_counts[best]),
        "top_background_hits": int(background_counts[best]),
        "top_cluster_prevalence": cluster_prev,
        "top_background_prevalence": background_prev,
        "top_prevalence_delta": cluster_prev - background_prev,
        "top_lift": lift,
        "meaningfulness_label": label,
    }


def audit_cluster_table(
    *,
    data: pd.DataFrame,
    term_names: list[str],
    go_ids: list[str],
    membership: pd.DataFrame,
    alpha: float,
    min_cluster_size: int,
    min_term_hits: int,
    strong_q: float,
    strong_lift: float,
    moderate_lift: float,
) -> pd.DataFrame:
    x = data.to_numpy(dtype=np.uint8, copy=False)
    gene_to_position = {gene: index for index, gene in enumerate(data.index.astype(str))}
    background_counts = x.sum(axis=0)
    population_size = len(data)
    rows: list[dict[str, Any]] = []

    group_columns = [
        "run_id",
        "specificity_aware_rank",
        "display_rank",
        "weighting",
        "block_name",
        "status",
        "assignment_source",
        "cluster_id",
    ]
    for group_values, cluster_frame in membership.groupby(group_columns, dropna=False, sort=False):
        meta = dict(zip(group_columns, group_values, strict=True))
        genes = sorted(set(cluster_frame["gene"].astype(str)) & set(gene_to_position))
        cluster_size = len(genes)
        row: dict[str, Any] = {
            **meta,
            "cluster_size": cluster_size,
            "tested": cluster_size >= min_cluster_size,
        }
        if cluster_size < min_cluster_size:
            row.update(
                {
                    "n_significant_terms": 0,
                    "best_q_value": math.nan,
                    "top_term": "",
                    "top_go_id": "",
                    "top_hits": 0,
                    "top_background_hits": 0,
                    "top_cluster_prevalence": math.nan,
                    "top_background_prevalence": math.nan,
                    "top_prevalence_delta": math.nan,
                    "top_lift": math.nan,
                    "meaningfulness_label": "too_small",
                }
            )
        else:
            positions = np.array([gene_to_position[gene] for gene in genes], dtype=int)
            cluster_counts = x[positions].sum(axis=0)
            row.update(
                term_stats(
                    cluster_counts,
                    cluster_size=cluster_size,
                    background_counts=background_counts,
                    population_size=population_size,
                    term_names=term_names,
                    go_ids=go_ids,
                    alpha=alpha,
                    min_term_hits=min_term_hits,
                    strong_q=strong_q,
                    strong_lift=strong_lift,
                    moderate_lift=moderate_lift,
                )
            )
        row["member_genes"] = ";".join(genes)
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame["meaningfulness_label"].value_counts().sort_index().items()}


def summarize_subspaces(cluster_stats: pd.DataFrame, null_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, frame in cluster_stats.groupby("run_id", sort=False):
        tested = frame[frame["tested"]]
        first = frame.iloc[0]
        strong = tested["meaningfulness_label"].eq("strong")
        enriched = tested["n_significant_terms"].gt(0)
        weak = tested["meaningfulness_label"].isin(["weak_or_not_enriched", "statistical_but_broad"])
        null_frame = null_stats[null_stats["run_id"].eq(run_id)]
        observed_enriched_fraction = float(enriched.mean()) if len(tested) else math.nan
        observed_strong_fraction = float(strong.mean()) if len(tested) else math.nan
        if not null_frame.empty and np.isfinite(observed_enriched_fraction):
            null_enriched_mean = float(null_frame["null_enriched_fraction"].mean())
            null_enriched_p95 = float(null_frame["null_enriched_fraction"].quantile(0.95))
            empirical_p_enriched = float(
                (1 + null_frame["null_enriched_fraction"].ge(observed_enriched_fraction).sum())
                / (len(null_frame) + 1)
            )
            null_strong_mean = float(null_frame["null_strong_fraction"].mean())
            null_strong_p95 = float(null_frame["null_strong_fraction"].quantile(0.95))
            empirical_p_strong = float(
                (1 + null_frame["null_strong_fraction"].ge(observed_strong_fraction).sum())
                / (len(null_frame) + 1)
            )
        else:
            null_enriched_mean = math.nan
            null_enriched_p95 = math.nan
            empirical_p_enriched = math.nan
            null_strong_mean = math.nan
            null_strong_p95 = math.nan
            empirical_p_strong = math.nan
        if len(tested) < 3:
            eigenband_coherence = "insufficient_tested_clusters"
        elif observed_enriched_fraction > null_enriched_p95 and observed_strong_fraction > null_strong_p95:
            eigenband_coherence = "coherent"
        elif observed_enriched_fraction > null_enriched_p95:
            eigenband_coherence = "coherent_enrichment_only"
        elif observed_enriched_fraction > null_enriched_mean:
            eigenband_coherence = "above_null_mean_not_strong"
        else:
            eigenband_coherence = "null_like_or_weak"
        rows.append(
            {
                "run_id": run_id,
                "specificity_aware_rank": safe_float(first.get("specificity_aware_rank")),
                "display_rank": safe_float(first.get("display_rank")),
                "weighting": first.get("weighting", ""),
                "block_name": first.get("block_name", ""),
                "status": first.get("status", ""),
                "assignment_source": first.get("assignment_source", ""),
                "n_clusters": len(frame),
                "n_tested_clusters": len(tested),
                "n_strong_clusters": int(strong.sum()),
                "n_enriched_clusters": int(enriched.sum()),
                "n_weak_or_broad_clusters": int(weak.sum()),
                "strong_fraction": observed_strong_fraction,
                "enriched_fraction": observed_enriched_fraction,
                "weak_or_broad_fraction": float(weak.mean()) if len(tested) else math.nan,
                "median_best_q_value": float(tested["best_q_value"].median()) if len(tested) else math.nan,
                "median_top_lift": float(tested["top_lift"].replace([np.inf, -np.inf], np.nan).median())
                if len(tested)
                else math.nan,
                "median_significant_terms": float(tested["n_significant_terms"].median()) if len(tested) else math.nan,
                "null_enriched_fraction_mean": null_enriched_mean,
                "null_enriched_fraction_p95": null_enriched_p95,
                "empirical_p_enriched_fraction": empirical_p_enriched,
                "null_strong_fraction_mean": null_strong_mean,
                "null_strong_fraction_p95": null_strong_p95,
                "empirical_p_strong_fraction": empirical_p_strong,
                "eigenband_coherence": eigenband_coherence,
                "top_terms": ";".join(
                    tested.sort_values(["meaningfulness_label", "best_q_value"])
                    .loc[lambda d: d["top_term"].ne("")]
                    .head(8)["top_term"]
                    .astype(str)
                ),
            }
        )
    return pd.DataFrame(rows)


def null_partition_stats(
    *,
    data: pd.DataFrame,
    term_names: list[str],
    go_ids: list[str],
    membership: pd.DataFrame,
    alpha: float,
    min_cluster_size: int,
    min_term_hits: int,
    strong_q: float,
    strong_lift: float,
    moderate_lift: float,
    iterations: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    x = data.to_numpy(dtype=np.uint8, copy=False)
    gene_to_position = {gene: index for index, gene in enumerate(data.index.astype(str))}
    background_counts = x.sum(axis=0)
    population_size = len(data)
    rows = []
    for run_id, run_frame in membership.groupby("run_id", sort=False):
        first = run_frame.iloc[0]
        cluster_sizes = run_frame.groupby("cluster_id", sort=False)["gene"].nunique().astype(int).tolist()
        genes = sorted(set(run_frame["gene"].astype(str)) & set(gene_to_position))
        if sum(cluster_sizes) != len(genes):
            cluster_sizes = sorted(cluster_sizes, reverse=True)
            cluster_sizes[-1] += len(genes) - sum(cluster_sizes)
        positions = np.array([gene_to_position[gene] for gene in genes], dtype=int)
        for iteration in range(iterations):
            shuffled = rng.permutation(positions)
            offset = 0
            labels = []
            n_significant = []
            for cluster_size in cluster_sizes:
                cluster_positions = shuffled[offset : offset + cluster_size]
                offset += cluster_size
                if cluster_size < min_cluster_size:
                    continue
                cluster_counts = x[cluster_positions].sum(axis=0)
                stats = term_stats(
                    cluster_counts,
                    cluster_size=cluster_size,
                    background_counts=background_counts,
                    population_size=population_size,
                    term_names=term_names,
                    go_ids=go_ids,
                    alpha=alpha,
                    min_term_hits=min_term_hits,
                    strong_q=strong_q,
                    strong_lift=strong_lift,
                    moderate_lift=moderate_lift,
                )
                labels.append(stats["meaningfulness_label"])
                n_significant.append(stats["n_significant_terms"])
            labels_series = pd.Series(labels, dtype="object")
            sig_series = pd.Series(n_significant, dtype="float64")
            rows.append(
                {
                    "run_id": run_id,
                    "iteration": iteration,
                    "weighting": first.get("weighting", ""),
                    "block_name": first.get("block_name", ""),
                    "assignment_source": first.get("assignment_source", ""),
                    "n_tested_clusters": len(labels),
                    "null_strong_fraction": float(labels_series.eq("strong").mean()) if len(labels) else math.nan,
                    "null_enriched_fraction": float(sig_series.gt(0).mean()) if len(sig_series) else math.nan,
                    "null_median_significant_terms": float(sig_series.median()) if len(sig_series) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_by_assignment(subspace_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for assignment_source, frame in subspace_summary.groupby("assignment_source", dropna=False):
        rows.append(
            {
                "assignment_source": assignment_source,
                "n_subspaces": len(frame),
                "mean_strong_fraction": float(frame["strong_fraction"].mean()),
                "mean_enriched_fraction": float(frame["enriched_fraction"].mean()),
                "median_empirical_p_enriched_fraction": float(frame["empirical_p_enriched_fraction"].median()),
                "median_empirical_p_strong_fraction": float(frame["empirical_p_strong_fraction"].median()),
                "mean_null_enriched_fraction": float(frame["null_enriched_fraction_mean"].mean()),
                "mean_null_strong_fraction": float(frame["null_strong_fraction_mean"].mean()),
                "eigenband_coherence_counts": ";".join(
                    f"{key}:{value}"
                    for key, value in frame["eigenband_coherence"].value_counts().sort_index().items()
                ),
            }
        )
    return pd.DataFrame(rows)


def report_lines(
    *,
    dataset: Dataset,
    cluster_stats: pd.DataFrame,
    subspace_summary: pd.DataFrame,
    assignment_summary: pd.DataFrame,
    top_examples: pd.DataFrame,
    weak_examples: pd.DataFrame,
    args: argparse.Namespace,
) -> list[str]:
    lines = [
        f"# {dataset.label} cluster meaningfulness audit",
        "",
        "## Method",
        "",
        "This is an internal GO-coherence audit. For each cluster, the script tests whether",
        "GO features are over-represented relative to the full feature matrix using a",
        "one-sided hypergeometric test with Benjamini-Hochberg correction within the",
        "cluster. It then compares each subspace to random partitions that preserve the",
        "observed cluster-size distribution.",
        "",
        f"- Minimum tested cluster size: `{args.min_cluster_size}` genes.",
        f"- Minimum genes supporting an interpreted GO term: `{args.min_term_hits}`.",
        f"- Significant enrichment threshold: `q <= {args.alpha}`.",
        f"- Strong label: `q <= {args.strong_q}`, lift `>= {args.strong_lift}`, and at least `3` supporting genes.",
        f"- Null iterations per subspace: `{args.null_iterations}`.",
        "",
        "Rows with `assignment_source=diagnostic_linkage_cut` are failed-gate diagnostics,",
        "not accepted final TBS assignments.",
        "",
        "## Assignment-source summary",
        "",
    ]
    if assignment_summary.empty:
        lines.append("No assignment-source summaries were produced.")
    else:
        lines.append(
            assignment_summary.to_markdown(
                index=False,
                floatfmt=".3g",
            )
        )
    lines.extend(
        [
            "",
            "## Subspace summary",
            "",
            subspace_summary[
                [
                    "display_rank",
                    "weighting",
                    "block_name",
                    "assignment_source",
                    "n_tested_clusters",
                    "strong_fraction",
                    "enriched_fraction",
                    "null_enriched_fraction_mean",
                    "empirical_p_enriched_fraction",
                    "empirical_p_strong_fraction",
                    "eigenband_coherence",
                ]
            ]
            .sort_values(["assignment_source", "display_rank", "block_name"], na_position="last")
            .to_markdown(index=False, floatfmt=".3g"),
            "",
            "## Strong examples",
            "",
            top_examples[
                [
                    "display_rank",
                    "weighting",
                    "block_name",
                    "assignment_source",
                    "cluster_id",
                    "cluster_size",
                    "top_term",
                    "top_go_id",
                    "best_q_value",
                    "top_lift",
                    "top_hits",
                ]
            ].to_markdown(index=False, floatfmt=".3g")
            if not top_examples.empty
            else "No strong examples found.",
            "",
            "## Weak or broad examples",
            "",
            weak_examples[
                [
                    "display_rank",
                    "weighting",
                    "block_name",
                    "assignment_source",
                    "cluster_id",
                    "cluster_size",
                    "meaningfulness_label",
                    "top_term",
                    "top_go_id",
                    "best_q_value",
                    "top_lift",
                ]
            ].to_markdown(index=False, floatfmt=".3g")
            if not weak_examples.empty
            else "No weak examples found.",
            "",
            "## Files",
            "",
            "- `cluster_meaningfulness.csv`: one row per cluster.",
            "- `subspace_meaningfulness_summary.csv`: one row per subspace/eigenband.",
            "- `assignment_source_meaningfulness_summary.csv`: accepted TBS versus diagnostic linkage-cut summary.",
            "- `null_partition_summary.csv`: size-preserving null partition summaries.",
            "- `top_meaningful_cluster_examples.csv`: strongest cluster examples.",
            "- `weak_or_broad_cluster_examples.csv`: weakest or broadest examples.",
            "- `quickgo_top_term_ids.txt`: GO IDs worth checking in QuickGO for top examples.",
        ]
    )
    label_counts = cluster_label_counts(cluster_stats)
    lines.extend(["", "## Cluster label counts", "", str(label_counts), ""])
    return lines


def main() -> None:
    args = parse_args()
    annotation_root = args.annotation_root.resolve()
    output_dir = args.output_dir or annotation_root / "cluster_meaningfulness_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(
        label=args.dataset_label,
        feature_matrix=args.feature_matrix.resolve(),
        annotation_root=annotation_root,
        output_dir=output_dir.resolve(),
    )

    data, term_names, go_ids = load_feature_matrix(dataset.feature_matrix)
    membership = pd.read_csv(annotation_root / "subspace_gene_membership_long.csv")
    membership["gene"] = membership["gene"].astype(str)

    cluster_stats = audit_cluster_table(
        data=data,
        term_names=term_names,
        go_ids=go_ids,
        membership=membership,
        alpha=args.alpha,
        min_cluster_size=args.min_cluster_size,
        min_term_hits=args.min_term_hits,
        strong_q=args.strong_q,
        strong_lift=args.strong_lift,
        moderate_lift=args.moderate_lift,
    )
    null_stats = null_partition_stats(
        data=data,
        term_names=term_names,
        go_ids=go_ids,
        membership=membership,
        alpha=args.alpha,
        min_cluster_size=args.min_cluster_size,
        min_term_hits=args.min_term_hits,
        strong_q=args.strong_q,
        strong_lift=args.strong_lift,
        moderate_lift=args.moderate_lift,
        iterations=args.null_iterations,
        random_seed=args.random_seed,
    )
    subspace_summary = summarize_subspaces(cluster_stats, null_stats)
    assignment_summary = summarize_by_assignment(subspace_summary)

    top_examples = (
        cluster_stats[cluster_stats["meaningfulness_label"].eq("strong")]
        .sort_values(["best_q_value", "top_lift"], ascending=[True, False])
        .head(args.top_examples)
    )
    weak_examples = (
        cluster_stats[
            cluster_stats["tested"]
            & cluster_stats["meaningfulness_label"].isin(["weak_or_not_enriched", "statistical_but_broad"])
        ]
        .sort_values(["best_q_value", "top_lift"], ascending=[False, True], na_position="last")
        .head(args.top_examples)
    )

    cluster_stats.to_csv(output_dir / "cluster_meaningfulness.csv", index=False)
    null_stats.to_csv(output_dir / "null_partition_summary.csv", index=False)
    subspace_summary.to_csv(output_dir / "subspace_meaningfulness_summary.csv", index=False)
    assignment_summary.to_csv(output_dir / "assignment_source_meaningfulness_summary.csv", index=False)
    top_examples.to_csv(output_dir / "top_meaningful_cluster_examples.csv", index=False)
    weak_examples.to_csv(output_dir / "weak_or_broad_cluster_examples.csv", index=False)
    top_go_ids = [
        go_id for go_id in top_examples["top_go_id"].dropna().astype(str).drop_duplicates().tolist() if go_id
    ][:50]
    (output_dir / "quickgo_top_term_ids.txt").write_text(",".join(top_go_ids) + "\n", encoding="utf-8")
    (output_dir / "meaningfulness_report.md").write_text(
        "\n".join(
            report_lines(
                dataset=dataset,
                cluster_stats=cluster_stats,
                subspace_summary=subspace_summary,
                assignment_summary=assignment_summary,
                top_examples=top_examples,
                weak_examples=weak_examples,
                args=args,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        {
            "dataset": dataset.label,
            "clusters": len(cluster_stats),
            "subspaces": len(subspace_summary),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
