"""Selected sibling Bernoulli deviance diagnostic.

This diagnostic adds an LRT-like sibling statistic beside the existing
projected-Wald sibling annotations. It is descriptive only: the nominal
chi-square tail is for fixed sibling pairs and is not a selected-tree
calibration rule.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.stats import chi2
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmarks.shared.runners.kl_diffusion_runner import _build_diffusion_distance
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from kl_clustering_analysis.tree.io import tree_from_linkage


DEFAULT_GRID: tuple[tuple[int, int], ...] = ((15, 3), (15, 5), (30, 3))
SCHEMA_VERSION = "selected_sibling_lrt_diagnostic/v1"


@dataclass(frozen=True)
class DiffusionRunResult:
    run_id: str
    k_neighbors: int
    diffusion_time: int
    n_clusters: int
    assignment_labels: pd.Series
    sibling_rows: pd.DataFrame


def load_binary_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    if data.empty:
        raise ValueError(f"Input matrix is empty: {path}")
    numeric = data.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Selected sibling LRT diagnostic currently requires binary features.")
    return numeric.astype(int)


def _node_depths(tree: nx.DiGraph) -> dict[object, int]:
    root = tree.root() if hasattr(tree, "root") else next(
        node for node, degree in tree.in_degree() if degree == 0
    )
    return {
        node: int(depth)
        for node, depth in nx.single_source_shortest_path_length(tree, root).items()
    }


def _finite_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _p_value(value: object) -> float:
    parsed = _finite_float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        return math.nan
    return parsed


def _bernoulli_kl(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Coordinatewise Bernoulli KL, allowing empirical probabilities 0 or 1."""
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0 - eps)
    first = np.zeros_like(p, dtype=float)
    second = np.zeros_like(p, dtype=float)
    mask = p > 0.0
    first[mask] = p[mask] * np.log(np.clip(p[mask], eps, 1.0) / q[mask])
    mask = p < 1.0
    one_minus_p = 1.0 - p[mask]
    second[mask] = one_minus_p * np.log(
        np.clip(one_minus_p, eps, 1.0) / (1.0 - q[mask])
    )
    return first + second


def bernoulli_sibling_deviance(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    *,
    n_left: int,
    n_right: int,
) -> dict[str, float]:
    """Return two-child Bernoulli deviance against a shared parent model."""
    left = np.asarray(left_distribution, dtype=float)
    right = np.asarray(right_distribution, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Sibling distributions must have the same shape.")
    if n_left <= 0 or n_right <= 0:
        raise ValueError("Sibling leaf counts must be positive.")

    pooled = (float(n_left) * left + float(n_right) * right) / float(n_left + n_right)
    left_kl = _bernoulli_kl(left, pooled)
    right_kl = _bernoulli_kl(right, pooled)
    deviance = 2.0 * (float(n_left) * float(np.sum(left_kl)) + float(n_right) * float(np.sum(right_kl)))

    active = np.isfinite(pooled) & (pooled > 0.0) & (pooled < 1.0)
    changed = active & (np.abs(left - right) > 0.0)
    df_active = int(np.sum(active))
    df_changed = int(np.sum(changed))
    df_nominal = max(df_changed, 1)
    p_nominal = float(chi2.sf(deviance, df=df_nominal)) if math.isfinite(deviance) else math.nan
    return {
        "bernoulli_deviance_lrt": float(deviance),
        "bernoulli_deviance_per_active_feature": (
            float(deviance / df_active) if df_active > 0 else math.nan
        ),
        "bernoulli_deviance_per_changed_feature": (
            float(deviance / df_changed) if df_changed > 0 else math.nan
        ),
        "bernoulli_lrt_nominal_df_changed_features": float(df_changed),
        "bernoulli_lrt_active_parent_features": float(df_active),
        "bernoulli_lrt_nominal_fixed_pair_p": p_nominal,
    }


def _support_label(annotations: pd.DataFrame, left: object, right: object) -> str:
    left_sig = bool(annotations.at[left, "Child_Parent_Divergence_Significant"])
    right_sig = bool(annotations.at[right, "Child_Parent_Divergence_Significant"])
    left_blocked = bool(annotations.at[left, "Child_Parent_Divergence_Ancestor_Blocked"])
    right_blocked = bool(annotations.at[right, "Child_Parent_Divergence_Ancestor_Blocked"])
    if not left_sig and not right_sig:
        return "strict_null_like"
    if left_blocked or right_blocked:
        return "stopped_or_null_like"
    return "selected_nonnull_like"


def _calibration_readout(row: pd.Series) -> str:
    if bool(row.get("Sibling_Divergence_Skipped", False)):
        return "not_tested"
    corrected = _p_value(row.get("Sibling_Divergence_P_Value_Corrected", math.nan))
    raw = _p_value(row.get("Sibling_Divergence_P_Value", math.nan))
    if math.isfinite(corrected):
        return "fdr_corrected"
    if math.isfinite(raw):
        return "raw_only"
    return "no_p_value"


def extract_sibling_lrt_rows(
    *,
    tree: nx.DiGraph,
    annotations: pd.DataFrame,
    run_id: str,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    edge_alpha: float,
    sibling_alpha: float,
) -> pd.DataFrame:
    depths = _node_depths(tree)
    rows: list[dict[str, object]] = []
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        n_left = int(tree.nodes[left]["leaf_count"])
        n_right = int(tree.nodes[right]["leaf_count"])
        n_parent = int(tree.nodes[parent]["leaf_count"])
        deviance = bernoulli_sibling_deviance(
            tree.nodes[left]["distribution"],
            tree.nodes[right]["distribution"],
            n_left=n_left,
            n_right=n_right,
        )
        sibling_stat = _finite_float(annotations.at[parent, "Sibling_Test_Statistic"])
        sibling_df = _finite_float(annotations.at[parent, "Sibling_Degrees_of_Freedom"])
        selected_ratio = (
            float(sibling_stat / sibling_df)
            if math.isfinite(sibling_stat) and math.isfinite(sibling_df) and sibling_df > 0.0
            else math.nan
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "tree_strategy": "whole_space_fixed_diffusion",
                "k_neighbors": int(k_neighbors),
                "diffusion_time": int(diffusion_time),
                "n_components": int(n_components),
                "parent_id": str(parent),
                "left_child_id": str(left),
                "right_child_id": str(right),
                "depth": int(depths.get(parent, -1)),
                "n_parent": n_parent,
                "n_left": n_left,
                "n_right": n_right,
                "child_balance": float(min(n_left, n_right) / max(n_left + n_right, 1)),
                "edge_alpha": float(edge_alpha),
                "sibling_alpha": float(sibling_alpha),
                "left_edge_raw_p": _p_value(annotations.at[left, "Child_Parent_Divergence_P_Value"]),
                "right_edge_raw_p": _p_value(annotations.at[right, "Child_Parent_Divergence_P_Value"]),
                "left_edge_bh_p": _p_value(annotations.at[left, "Child_Parent_Divergence_P_Value_BH"]),
                "right_edge_bh_p": _p_value(annotations.at[right, "Child_Parent_Divergence_P_Value_BH"]),
                "left_edge_significant": bool(annotations.at[left, "Child_Parent_Divergence_Significant"]),
                "right_edge_significant": bool(annotations.at[right, "Child_Parent_Divergence_Significant"]),
                "support_label_from_edge_path": _support_label(annotations, left, right),
                "sibling_projected_wald_stat": sibling_stat,
                "sibling_projected_wald_df": sibling_df,
                "sibling_selected_ratio": selected_ratio,
                "sibling_raw_p": _p_value(annotations.at[parent, "Sibling_Divergence_P_Value"]),
                "sibling_bh_p": _p_value(annotations.at[parent, "Sibling_Divergence_P_Value_Corrected"]),
                "sibling_rejected": bool(annotations.at[parent, "Sibling_BH_Different"]),
                "sibling_skipped": bool(annotations.at[parent, "Sibling_Divergence_Skipped"]),
                "sibling_projection_dimension": _finite_float(
                    annotations.at[parent, "Sibling_Projection_Dimension"]
                ),
                "calibration_readout": _calibration_readout(annotations.loc[parent]),
                **deviance,
            }
        )
    return pd.DataFrame(rows)


def run_diffusion_lrt_cell(
    *,
    data: pd.DataFrame,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    edge_alpha: float,
    sibling_alpha: float,
) -> DiffusionRunResult:
    run_id = f"k{k_neighbors:02d}_t{diffusion_time}_c{n_components}"
    distance = _build_diffusion_distance(
        data,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
        n_components=n_components,
    )
    linkage_matrix = linkage(distance, method="average")
    tree = tree_from_linkage(linkage_matrix, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)
    gate_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        leaf_data=data,
    )
    decomposition = tree.decompose(
        gate_annotation_bundle=gate_bundle,
        leaf_data=data,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
    )
    assignments = build_sample_cluster_assignments(decomposition).loc[data.index]
    labels = assignments["cluster_id"].astype(int)
    rows = extract_sibling_lrt_rows(
        tree=tree,
        annotations=gate_bundle.annotated_df,
        run_id=run_id,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
        n_components=n_components,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
    )
    rows["n_final_clusters"] = int(labels.nunique())
    return DiffusionRunResult(
        run_id=run_id,
        k_neighbors=int(k_neighbors),
        diffusion_time=int(diffusion_time),
        n_clusters=int(labels.nunique()),
        assignment_labels=labels,
        sibling_rows=rows,
    )


def summarize_runs(results: Sequence[DiffusionRunResult]) -> pd.DataFrame:
    baseline = next((result for result in results if result.run_id == "k15_t3_c30"), None)
    rows: list[dict[str, object]] = []
    for result in results:
        labels = result.assignment_labels
        counts = labels.value_counts()
        table = result.sibling_rows
        if baseline is not None:
            common = baseline.assignment_labels.index.intersection(labels.index)
            ari = float(adjusted_rand_score(baseline.assignment_labels.loc[common], labels.loc[common]))
            nmi = float(normalized_mutual_info_score(baseline.assignment_labels.loc[common], labels.loc[common]))
        else:
            ari = math.nan
            nmi = math.nan
        rows.append(
            {
                "run_id": result.run_id,
                "k_neighbors": result.k_neighbors,
                "diffusion_time": result.diffusion_time,
                "n_clusters": result.n_clusters,
                "largest_cluster_size": int(counts.iloc[0]),
                "largest_cluster_fraction": float(counts.iloc[0] / len(labels)),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_cluster_fraction": float((counts == 1).sum() / len(counts)),
                "n_sibling_rows": int(len(table)),
                "n_selected_nonnull_like": int(
                    table["support_label_from_edge_path"].eq("selected_nonnull_like").sum()
                ),
                "n_strict_null_like": int(
                    table["support_label_from_edge_path"].eq("strict_null_like").sum()
                ),
                "median_deviance_per_changed_feature": float(
                    table["bernoulli_deviance_per_changed_feature"].replace([np.inf, -np.inf], np.nan).median()
                ),
                "median_projected_wald_selected_ratio": float(
                    table["sibling_selected_ratio"].replace([np.inf, -np.inf], np.nan).median()
                ),
                "nmi_vs_k15_t3": nmi,
                "ari_vs_k15_t3": ari,
            }
        )
    return pd.DataFrame(rows).sort_values(["k_neighbors", "diffusion_time"])


def write_plots(panel: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "n_clusters", "Final clusters"),
        (axes[1], "singleton_cluster_fraction", "Singleton cluster fraction"),
        (axes[2], "median_deviance_per_changed_feature", "Median deviance / changed feature"),
    ]:
        ax.bar(summary["run_id"], summary[metric], color="#4c78a8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Selected sibling LRT diagnostic: run summary", fontsize=15)
    fig.savefig(output_dir / "selected_sibling_lrt_run_summary.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (run_id, group) in zip(axes, panel.groupby("run_id", sort=True)):
        x = group["sibling_selected_ratio"].replace([np.inf, -np.inf], np.nan)
        y = group["bernoulli_deviance_per_changed_feature"].replace([np.inf, -np.inf], np.nan)
        colors = group["support_label_from_edge_path"].map(
            {
                "strict_null_like": "#1b9e77",
                "stopped_or_null_like": "#7570b3",
                "selected_nonnull_like": "#d95f02",
            }
        )
        ax.scatter(x, y, c=colors, s=16, alpha=0.65, linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(run_id)
        ax.set_xlabel("projected-Wald selected ratio")
        ax.set_ylabel("Bernoulli deviance / changed feature")
        ax.grid(alpha=0.15)
    fig.suptitle("LRT-like sibling deviance vs projected-Wald evidence", fontsize=15)
    fig.savefig(output_dir / "selected_sibling_lrt_vs_projected_wald.png", dpi=200)
    plt.close(fig)

    labels = ("strict_null_like", "stopped_or_null_like", "selected_nonnull_like")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True, sharey=True)
    for ax, (run_id, group) in zip(axes, panel.groupby("run_id", sort=True)):
        data = [
            group.loc[group["support_label_from_edge_path"].eq(label), "bernoulli_deviance_per_changed_feature"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            for label in labels
        ]
        ax.boxplot(data, tick_labels=["strict", "stopped", "nonnull"], showfliers=False)
        ax.set_yscale("log")
        ax.set_title(run_id)
        ax.set_ylabel("deviance / changed feature")
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Sibling deviance by edge-path support label", fontsize=15)
    fig.savefig(output_dir / "selected_sibling_lrt_support_boxes.png", dpi=200)
    plt.close(fig)


def parse_grid(raw: str) -> tuple[tuple[int, int], ...]:
    if not raw.strip():
        return DEFAULT_GRID
    parsed: list[tuple[int, int]] = []
    for token in raw.split(","):
        left, right = token.strip().split(":", 1)
        parsed.append((int(left), int(right)))
    return tuple(parsed)


def run_selected_sibling_lrt_diagnostic(
    *,
    input_path: Path,
    output_dir: Path,
    grid: Sequence[tuple[int, int]] = DEFAULT_GRID,
    n_components: int = 30,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_binary_matrix(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[DiffusionRunResult] = []
    for k_neighbors, diffusion_time in grid:
        result = run_diffusion_lrt_cell(
            data=data,
            k_neighbors=int(k_neighbors),
            diffusion_time=int(diffusion_time),
            n_components=int(n_components),
            edge_alpha=float(edge_alpha),
            sibling_alpha=float(sibling_alpha),
        )
        cell_dir = output_dir / result.run_id
        cell_dir.mkdir(exist_ok=True)
        result.sibling_rows.to_csv(cell_dir / "selected_sibling_lrt_node_panel.csv", index=False)
        pd.DataFrame({"gene": result.assignment_labels.index, "cluster_id": result.assignment_labels.to_numpy()}).to_csv(
            cell_dir / "cluster_assignments.csv",
            index=False,
        )
        results.append(result)

    panel = pd.concat([result.sibling_rows for result in results], ignore_index=True)
    summary = summarize_runs(results)
    panel.to_csv(output_dir / "selected_sibling_lrt_node_panel.csv", index=False)
    summary.to_csv(output_dir / "selected_sibling_lrt_summary.csv", index=False)
    write_plots(panel, summary, output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "role": "diagnostic_only_not_production_calibration",
        "input_path": str(input_path),
        "grid": [{"k_neighbors": int(k), "diffusion_time": int(t)} for k, t in grid],
        "n_components": int(n_components),
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "outputs": {
            "node_panel": str(output_dir / "selected_sibling_lrt_node_panel.csv"),
            "summary": str(output_dir / "selected_sibling_lrt_summary.csv"),
            "run_summary_plot": str(output_dir / "selected_sibling_lrt_run_summary.png"),
            "lrt_vs_projected_wald_plot": str(output_dir / "selected_sibling_lrt_vs_projected_wald.png"),
            "support_boxes_plot": str(output_dir / "selected_sibling_lrt_support_boxes.png"),
        },
        "interpretation": (
            "Bernoulli deviance is an LRT-like fixed-sibling statistic. "
            "Nominal chi-square p-values in this output are not selected-tree calibrated."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Selected Sibling LRT Diagnostic",
                "",
                "Diagnostic-only Bernoulli sibling deviance panel for fixed diffusion KL trees.",
                "The nominal LRT tail is fixed-pair only and is not a selected-tree calibration rule.",
                "",
                summary.to_markdown(index=False),
                "",
            ]
        )
    )
    return panel, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/"
            "29_selected_sibling_lrt_diagnostic_20260612"
        ),
    )
    parser.add_argument(
        "--grid",
        default=",".join(f"{k}:{t}" for k, t in DEFAULT_GRID),
        help="Comma-separated k:t diffusion grid, e.g. '15:3,15:5,30:3'.",
    )
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    args = parser.parse_args()

    _, summary = run_selected_sibling_lrt_diagnostic(
        input_path=args.input,
        output_dir=args.output_dir,
        grid=parse_grid(args.grid),
        n_components=int(args.n_components),
        edge_alpha=float(args.edge_alpha),
        sibling_alpha=float(args.sibling_alpha),
    )
    print(summary.to_string(index=False))
    print(f"Wrote selected sibling LRT diagnostic: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
