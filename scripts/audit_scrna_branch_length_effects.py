"""Audit branch-length effects in the scRNA TBS benchmark outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "raw"
    / "assets"
    / "benchmark-results"
    / "scrna_branch_length_effect_audit_20260624"
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    label: str
    output_dir: Path


@dataclass(frozen=True)
class MethodSpec:
    geometry: str
    variant: str
    method_label: str
    assignment_key: str
    edge_csv: str


DATASETS = [
    DatasetSpec(
        dataset="adult_pancreas",
        label="Adult pancreas",
        output_dir=PROJECT_ROOT
        / "raw"
        / "assets"
        / "benchmark-results"
        / "pancreas_scrna_cluster_benchmark_20260623",
    ),
    DatasetSpec(
        dataset="goncalves_fetal",
        label="Goncalves fetal pancreas",
        output_dir=PROJECT_ROOT
        / "raw"
        / "assets"
        / "benchmark-results"
        / "goncalves_fetal_pancreas_progenitor_benchmark_20260624",
    ),
]

METHODS = [
    MethodSpec(
        geometry="pca_linkage",
        variant="topology_only",
        method_label="TBS topology projected adaptive-k90 alpha=0.01 edge=0.001",
        assignment_key="tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001",
        edge_csv="tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    ),
    MethodSpec(
        geometry="pca_linkage",
        variant="branch_time_nnls",
        method_label="TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001",
        assignment_key="tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001",
        edge_csv="tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    ),
    MethodSpec(
        geometry="pca_linkage",
        variant="branch_time_raw_linkage",
        method_label="TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001",
        assignment_key="tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001",
        edge_csv="tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    ),
    MethodSpec(
        geometry="adaptive_diffusion",
        variant="topology_only",
        method_label="TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001",
        assignment_key="tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001",
        edge_csv="tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    ),
    MethodSpec(
        geometry="adaptive_diffusion",
        variant="branch_time_nnls",
        method_label=(
            "TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 "
            "alpha=0.01 edge=0.001"
        ),
        assignment_key=(
            "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_"
            "alpha0p01_edge0p001"
        ),
        edge_csv=(
            "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_"
            "alpha0p01_edge0p001_tree_edges.csv"
        ),
    ),
    MethodSpec(
        geometry="adaptive_diffusion",
        variant="branch_time_raw_linkage",
        method_label=(
            "TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 "
            "alpha=0.01 edge=0.001"
        ),
        assignment_key=(
            "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_"
            "k90_alpha0p01_edge0p001"
        ),
        edge_csv=(
            "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_"
            "k90_alpha0p01_edge0p001_tree_edges.csv"
        ),
    ),
]
VARIANT_ORDER = [
    "pca_linkage:topology_only",
    "pca_linkage:branch_time_nnls",
    "pca_linkage:branch_time_raw_linkage",
    "adaptive_diffusion:topology_only",
    "adaptive_diffusion:branch_time_nnls",
    "adaptive_diffusion:branch_time_raw_linkage",
]
VARIANT_LABELS = {
    "pca_linkage:topology_only": "PCA topo",
    "pca_linkage:branch_time_nnls": "PCA NNLS",
    "pca_linkage:branch_time_raw_linkage": "PCA raw",
    "adaptive_diffusion:topology_only": "AD topo",
    "adaptive_diffusion:branch_time_nnls": "AD NNLS",
    "adaptive_diffusion:branch_time_raw_linkage": "AD raw",
}


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _metric_row(metrics: pd.DataFrame, method_label: str) -> pd.Series:
    matches = metrics.loc[metrics["method"] == method_label]
    if len(matches) != 1:
        raise ValueError(f"Expected one metrics row for {method_label!r}; found {len(matches)}")
    return matches.iloc[0]


def _tree_summary_row(tree_summary: pd.DataFrame, method_label: str) -> pd.Series:
    matches = tree_summary.loc[tree_summary["method"] == method_label]
    if len(matches) != 1:
        raise ValueError(f"Expected one tree-summary row for {method_label!r}; found {len(matches)}")
    return matches.iloc[0]


def _edge_summary(path: Path) -> dict[str, object]:
    edges = pd.read_csv(path)
    branch = pd.to_numeric(edges["branch_length"], errors="coerce")
    significant = _bool_series(edges["Child_Parent_Divergence_Significant"])
    tested = _bool_series(edges["Child_Parent_Divergence_Tested"])
    blocked = _bool_series(edges["Child_Parent_Divergence_Ancestor_Blocked"])
    valid = branch.notna()
    corr = np.nan
    if valid.sum() > 2 and significant.nunique() > 1:
        corr = float(spearmanr(branch[valid], significant[valid].astype(int)).statistic)
    return {
        "edge_count": int(len(edges)),
        "edge_open_count": int(significant.sum()),
        "edge_tested_count": int(tested.sum()),
        "edge_ancestor_blocked_count": int(blocked.sum()),
        "branch_length_mean_all_edges": float(branch.mean()),
        "branch_length_median_all_edges": float(branch.median()),
        "branch_length_max_all_edges": float(branch.max()),
        "branch_length_mean_open_edges": float(branch[significant].mean())
        if significant.any()
        else np.nan,
        "branch_length_mean_closed_edges": float(branch[~significant].mean())
        if (~significant).any()
        else np.nan,
        "branch_length_open_spearman": corr,
    }


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _collect_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(spec.output_dir / "method_metrics.csv")
    assignments = pd.read_csv(spec.output_dir / "method_assignments.csv")
    tree_summary = pd.read_csv(spec.output_dir / "tbs_tree_branch_length_summary.csv")

    rows: list[dict[str, object]] = []
    for method in METHODS:
        metric = _metric_row(metrics, method.method_label)
        tree_row = _tree_summary_row(tree_summary, method.method_label)
        edge_row = _edge_summary(spec.output_dir / method.edge_csv)
        rows.append(
            {
                "dataset": spec.dataset,
                "dataset_label": spec.label,
                "geometry": method.geometry,
                "variant": method.variant,
                "method": method.method_label,
                "assignment_key": method.assignment_key,
                "edge_csv": str(spec.output_dir / method.edge_csv),
                "n_clusters": int(metric["n_clusters"]),
                "ari_vs_celltype": float(metric["ari"]),
                "v_measure_vs_celltype": float(metric["v_measure"]),
                "weighted_cluster_purity": float(metric["weighted_cluster_purity"]),
                "weighted_label_dominant_cluster_recall": float(
                    metric["weighted_label_dominant_cluster_recall"]
                ),
                "merge_error_rate": float(metric["merge_error_rate"]),
                "split_error_rate": float(metric["split_error_rate"]),
                "edge_branch_length_variance_policy": tree_row[
                    "edge_branch_length_variance_policy"
                ],
                "branch_length_optimization_method": tree_row[
                    "branch_length_optimization_method"
                ],
                "branch_length_mean": float(tree_row["branch_length_mean"]),
                "branch_length_median": float(tree_row["branch_length_median"]),
                "branch_length_max": float(tree_row["branch_length_max"]),
                "branch_length_optimization_residual_rmse": tree_row.get(
                    "branch_length_optimization_residual_rmse",
                    np.nan,
                ),
                "branch_length_optimization_residual_mae": tree_row.get(
                    "branch_length_optimization_residual_mae",
                    np.nan,
                ),
                **edge_row,
            }
        )
    effects = pd.DataFrame(rows)

    for geometry, group in effects.groupby("geometry"):
        baseline = group.loc[group["variant"] == "topology_only"].iloc[0]
        baseline_labels = pd.to_numeric(
            assignments[str(baseline["assignment_key"])],
            errors="coerce",
        ).to_numpy()
        for index in group.index:
            labels = pd.to_numeric(
                assignments[str(effects.loc[index, "assignment_key"])],
                errors="coerce",
            ).to_numpy()
            effects.loc[index, "delta_n_clusters_vs_topology"] = (
                int(effects.loc[index, "n_clusters"]) - int(baseline["n_clusters"])
            )
            effects.loc[index, "delta_v_measure_vs_topology"] = (
                float(effects.loc[index, "v_measure_vs_celltype"])
                - float(baseline["v_measure_vs_celltype"])
            )
            effects.loc[index, "delta_purity_vs_topology"] = (
                float(effects.loc[index, "weighted_cluster_purity"])
                - float(baseline["weighted_cluster_purity"])
            )
            effects.loc[index, "delta_recall_vs_topology"] = (
                float(effects.loc[index, "weighted_label_dominant_cluster_recall"])
                - float(baseline["weighted_label_dominant_cluster_recall"])
            )
            effects.loc[index, "assignment_ari_vs_topology"] = adjusted_rand_score(
                baseline_labels,
                labels,
            )
            effects.loc[index, "assignment_nmi_vs_topology"] = normalized_mutual_info_score(
                baseline_labels,
                labels,
            )
            effects.loc[index, "assignment_exact_match_fraction_vs_topology"] = float(
                np.mean(baseline_labels == labels)
            )

    pair_rows: list[dict[str, object]] = []
    for left in METHODS:
        left_labels = pd.to_numeric(assignments[left.assignment_key], errors="coerce").to_numpy()
        for right in METHODS:
            right_labels = pd.to_numeric(
                assignments[right.assignment_key],
                errors="coerce",
            ).to_numpy()
            pair_rows.append(
                {
                    "dataset": spec.dataset,
                    "left_variant": f"{left.geometry}:{left.variant}",
                    "right_variant": f"{right.geometry}:{right.variant}",
                    "ari": adjusted_rand_score(left_labels, right_labels),
                    "nmi": normalized_mutual_info_score(left_labels, right_labels),
                    "exact_match_fraction": float(np.mean(left_labels == right_labels)),
                }
            )

    sensitivity_path = spec.output_dir / "tbs_branch_time_sensitivity.csv"
    sensitivity = pd.read_csv(sensitivity_path)
    sensitivity.insert(0, "dataset", spec.dataset)
    sensitivity.insert(1, "dataset_label", spec.label)
    sensitivity["geometry"] = np.where(
        sensitivity["method"].str.contains("adaptive diffusion"),
        "adaptive_diffusion",
        "pca_linkage",
    )
    sensitivity["source_csv"] = str(sensitivity_path)

    return effects, pd.DataFrame(pair_rows), sensitivity


def _write_effect_plot(effects: pd.DataFrame) -> None:
    variant_order = ["topology_only", "branch_time_nnls", "branch_time_raw_linkage"]
    metric_specs = [
        ("n_clusters", "Clusters"),
        ("v_measure_vs_celltype", "V-measure"),
        ("weighted_cluster_purity", "Purity"),
        ("weighted_label_dominant_cluster_recall", "Dominant recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (metric, title) in zip(axes.ravel(), metric_specs, strict=True):
        for (dataset_label, geometry), group in effects.groupby(["dataset_label", "geometry"]):
            group = group.set_index("variant").loc[variant_order]
            x = np.arange(len(variant_order))
            ax.plot(x, group[metric], marker="o", label=f"{dataset_label}, {geometry}")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(variant_order)), ["topology", "NNLS", "raw"], rotation=20)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Branch-length policy changes TBS split/merge pattern", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "branch_length_cluster_effects.png", dpi=220)
    plt.close(fig)


def _write_similarity_plot(pairwise: pd.DataFrame) -> None:
    datasets = list(pairwise["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2 * len(datasets), 5.8))
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        frame = pairwise[pairwise["dataset"] == dataset]
        matrix = frame.pivot(index="left_variant", columns="right_variant", values="ari")
        matrix = matrix.loc[VARIANT_ORDER, VARIANT_ORDER]
        image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis")
        ax.set_title(dataset)
        ax.set_xticks(
            np.arange(matrix.shape[1]),
            [VARIANT_LABELS[value] for value in matrix.columns],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(
            np.arange(matrix.shape[0]),
            [VARIANT_LABELS[value] for value in matrix.index],
            fontsize=8,
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, label="Assignment ARI")
    fig.suptitle("Pairwise similarity among TBS branch-length variants", fontweight="bold")
    fig.savefig(OUTPUT_ROOT / "branch_length_assignment_similarity_heatmap.png", dpi=220)
    plt.close(fig)


def _write_sensitivity_plot(sensitivity: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["n_clusters", "v_measure"], strict=True):
        for (dataset_label, geometry), group in sensitivity.groupby(["dataset_label", "geometry"]):
            group = group.sort_values("median_variance_multiplier")
            ax.plot(
                group["median_variance_multiplier"],
                group[metric],
                marker="o",
                linewidth=1.2,
                label=f"{dataset_label}, {geometry}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Median branch-time variance multiplier")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fixed-topology branch-time sensitivity scan", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "branch_time_sensitivity_effects.png", dpi=220)
    plt.close(fig)


def _artifact_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if path.suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
        record["rows"] = int(len(frame))
        record["columns"] = int(len(frame.columns))
    return record


def _write_manifest(generated_at: str) -> None:
    artifact_paths = [
        OUTPUT_ROOT / "branch_length_method_effects.csv",
        OUTPUT_ROOT / "branch_length_assignment_similarity.csv",
        OUTPUT_ROOT / "branch_time_sensitivity_combined.csv",
        OUTPUT_ROOT / "branch_length_cluster_effects.png",
        OUTPUT_ROOT / "branch_length_assignment_similarity_heatmap.png",
        OUTPUT_ROOT / "branch_time_sensitivity_effects.png",
        OUTPUT_ROOT / "scrna_branch_length_effect_audit.md",
    ]
    manifest = {
        "manifest_schema_version": "static_artifact_provenance/v1",
        "bundle": "scrna_branch_length_effect_audit_20260624",
        "generated_at": generated_at,
        "source_script": "scripts/audit_scrna_branch_length_effects.py",
        "source_inputs": [
            str((spec.output_dir / "manifest.json").relative_to(PROJECT_ROOT))
            for spec in DATASETS
        ],
        "artifacts": [_artifact_record(path) for path in artifact_paths],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_report(
    effects: pd.DataFrame,
    sensitivity: pd.DataFrame,
    manifests: dict[str, dict[str, object]],
    generated_at: str | None = None,
) -> str:
    generated_at = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    display_cols = [
        "dataset_label",
        "geometry",
        "variant",
        "n_clusters",
        "v_measure_vs_celltype",
        "weighted_cluster_purity",
        "weighted_label_dominant_cluster_recall",
        "delta_n_clusters_vs_topology",
        "assignment_ari_vs_topology",
        "edge_open_count",
        "branch_length_median",
        "branch_length_max",
        "branch_length_optimization_method",
    ]
    effect_table = effects[display_cols].to_markdown(index=False, floatfmt=".4f")

    best_sensitivity = (
        sensitivity.sort_values(
            ["dataset", "geometry", "v_measure", "weighted_label_dominant_cluster_recall"],
            ascending=[True, True, False, False],
        )
        .groupby(["dataset_label", "geometry"], as_index=False)
        .head(3)
    )
    sensitivity_table = best_sensitivity[
        [
            "dataset_label",
            "geometry",
            "length_model",
            "edge_open_count",
            "n_clusters",
            "weighted_cluster_purity",
            "weighted_label_dominant_cluster_recall",
            "v_measure",
            "median_variance_multiplier",
            "max_variance_multiplier",
        ]
    ].to_markdown(index=False, floatfmt=".4f")

    manifest_text = "\n".join(
        f"- {dataset}: elapsed {manifest.get('elapsed_sec', 'unknown')} seconds, "
        f"seed {manifest.get('seed', 'unknown')}, max cells {manifest.get('max_cells', 'unknown')}"
        for dataset, manifest in manifests.items()
    )

    report = f"""# scRNA Branch-Length Effect Audit

Generated at: {generated_at}

## Summary

The rerun confirms that branch lengths are not needed to place the cells on the
tree: topology is inferred first from a distance matrix, then branch lengths are
stored or refit. Branch lengths change clustering only when the edge gate is run
with `edge_branch_length_variance_policy="normalized_branch_length"`.

The stable interpretation is:

- Adaptive-diffusion topology is the safest current tree for the Goncalves
  progenitor review. Its topology-only, NNLS branch-time, and raw-linkage
  branch-time rows give the same `24` fetal TBS clusters in the rerun.
- Adult pancreas also keeps the same adaptive-diffusion topology-only and NNLS
  clusters (`43` clusters); raw-linkage branch-time changes that row only mildly
  (`41` clusters).
- Standardized-PCA linkage is branch-time sensitive. Adult raw-linkage
  branch-time collapses from `45` to `9` clusters. Goncalves standardized-PCA
  branch-time, including the NNLS row, collapses from `40` to `1` cluster.

So the wrong conclusion would be "branch lengths define the correct clusters."
The evidence says the tree topology and the edge/sibling gates define the
current clusters; branch lengths are a variance-scaling sensitivity dial unless
we can justify a real stochastic-time model.

## How The Tree Is Inferred

1. The benchmark builds a PCA matrix from the current scRNA workflow.
2. For standardized-PCA topology, it computes pairwise Euclidean distances in
   PCA space and runs average-linkage hierarchical clustering.
3. For adaptive-diffusion topology, it builds a variable-bandwidth diffusion
   distance from the PCA kNN graph (`k=15`, diffusion time `3`) and runs the
   same average-linkage tree builder.
4. Linkage heights become diagnostic branch lengths. In NNLS rows, the topology
   is held fixed and non-negative least squares refits edge lengths to better
   approximate continuous pairwise distances.
5. TBS traverses that fixed tree using child-parent edge gates and sibling
   gates. Branch lengths affect the edge gate only under the explicit
   normalized branch-time variance policy.

## Rerun Manifests

{manifest_text}

## Branch-Length Effect Table

{effect_table}

## Branch-Time Sensitivity Scan

{sensitivity_table}

## Output Plots

- `branch_length_cluster_effects.png`
- `branch_length_assignment_similarity_heatmap.png`
- `branch_time_sensitivity_effects.png`

## Source Tables

- Adult pancreas: `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/`
- Goncalves fetal pancreas:
  `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/`
"""
    (OUTPUT_ROOT / "scrna_branch_length_effect_audit.md").write_text(report)
    return generated_at


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    effect_frames = []
    pairwise_frames = []
    sensitivity_frames = []
    manifests = {}
    for spec in DATASETS:
        effects, pairwise, sensitivity = _collect_dataset(spec)
        effect_frames.append(effects)
        pairwise_frames.append(pairwise)
        sensitivity_frames.append(sensitivity)
        manifests[spec.dataset] = _read_manifest(spec.output_dir / "manifest.json")

    effects = pd.concat(effect_frames, ignore_index=True)
    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    effects.to_csv(OUTPUT_ROOT / "branch_length_method_effects.csv", index=False)
    pairwise.to_csv(OUTPUT_ROOT / "branch_length_assignment_similarity.csv", index=False)
    sensitivity.to_csv(OUTPUT_ROOT / "branch_time_sensitivity_combined.csv", index=False)

    _write_effect_plot(effects)
    _write_similarity_plot(pairwise)
    _write_sensitivity_plot(sensitivity)
    _write_report(effects, sensitivity, manifests, generated_at)
    _write_manifest(generated_at)
    print(f"Wrote branch-length audit outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
