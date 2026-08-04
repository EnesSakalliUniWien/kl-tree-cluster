"""Analyze Goncalves fetal-pancreas clusters and nodes for progenitor states."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kl_te_cluster_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/kl_te_cluster_numba")

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import sparse

from applications.scrna._shared import json_default, top_counts_text

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "raw"
    / "assets"
    / "benchmark-results"
    / "goncalves_fetal_pancreas_progenitor_benchmark_20260624"
)
METHOD_KEY = "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001"
PROGENITOR_POPULATIONS = {"trunk", "tip", "proliferating"}
SIGNATURES: dict[str, list[str]] = {
    "endocrine_progenitor_core": ["NEUROG3", "PAX4"],
    "endocrine_commitment": ["RFX6", "NEUROD1", "ISL1", "PAX6", "NKX2-2"],
    "trunk_progenitor": ["PDX1", "SOX9", "HNF1B", "HES1", "NKX6-1", "KRT19", "SPP1"],
    "tip_progenitor": ["PTF1A", "CPA1", "CPA2", "RBPJL", "CEL", "CTRB2", "GATA4", "NR5A2"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "CDK1"],
    "mesenchyme": ["VIM", "DCN", "COL1A1", "COL3A1"],
    "mature_endocrine": ["CHGA", "INS", "GCG", "SST", "PPY"],
    "neuronal": ["PRPH", "GAP43"],
    "blood": ["ALAS2", "HBG1"],
}
ALL_MARKERS = sorted({marker for markers in SIGNATURES.values() for marker in markers})


def _query_symbol(symbol: str) -> list[str]:
    response = requests.get(
        "https://mygene.info/v3/query",
        params={
            "q": f"symbol:{symbol}",
            "species": "human",
            "fields": "symbol,ensembl.gene",
            "size": 5,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    ensembl_ids: list[str] = []
    for hit in payload.get("hits", []):
        if str(hit.get("symbol", "")).upper() != symbol.upper():
            continue
        ensembl = hit.get("ensembl")
        if isinstance(ensembl, dict):
            gene = ensembl.get("gene")
            if gene:
                ensembl_ids.append(str(gene))
        elif isinstance(ensembl, list):
            for item in ensembl:
                if isinstance(item, dict) and item.get("gene"):
                    ensembl_ids.append(str(item["gene"]))
    return sorted(set(ensembl_ids))


def load_marker_mapping(
    raw_var_names: pd.Index,
    output_dir: Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    mapping_path = output_dir / "goncalves_marker_ensembl_mapping.csv"
    if mapping_path.exists() and not refresh:
        return pd.read_csv(mapping_path)

    raw_genes = set(map(str, raw_var_names))
    rows: list[dict[str, object]] = []
    for symbol in ALL_MARKERS:
        ensembl_ids = _query_symbol(symbol)
        present = [gene for gene in ensembl_ids if gene in raw_genes]
        rows.append(
            {
                "symbol": symbol,
                "ensembl_candidates": ",".join(ensembl_ids),
                "ensembl_id": present[0] if present else "",
                "present_in_raw": bool(present),
            }
        )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(mapping_path, index=False)
    return mapping


def load_marker_expression(adata: ad.AnnData, mapping: pd.DataFrame) -> pd.DataFrame:
    if adata.raw is None:
        raise RuntimeError("Expected Goncalves AnnData raw layer to contain the full gene matrix.")
    raw_index = {str(gene): index for index, gene in enumerate(adata.raw.var_names)}
    present = mapping[mapping["present_in_raw"]].copy()
    indices = [raw_index[str(gene)] for gene in present["ensembl_id"]]
    matrix = adata.raw.X[:, indices]
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().toarray()
    expression = pd.DataFrame(
        np.asarray(matrix, dtype=float),
        index=adata.obs_names.astype(str),
        columns=present["symbol"].astype(str).to_list(),
    )
    return expression


def add_signature_scores(expression: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = expression.copy()
    means = expression.mean(axis=0)
    stds = expression.std(axis=0, ddof=0).replace(0.0, 1.0)
    zscores = (expression - means) / stds
    signature_rows: list[dict[str, object]] = []
    for signature, genes in SIGNATURES.items():
        available = [gene for gene in genes if gene in zscores.columns]
        missing = sorted(set(genes) - set(available))
        if available:
            scored[f"{signature}_score"] = zscores[available].mean(axis=1)
        else:
            scored[f"{signature}_score"] = np.nan
        signature_rows.append(
            {
                "signature": signature,
                "requested_genes": ",".join(genes),
                "available_genes": ",".join(available),
                "missing_genes": ",".join(missing),
                "n_available": len(available),
            }
        )
    return scored, pd.DataFrame(signature_rows)


def entropy(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def summarize_group(
    *,
    label: str,
    group_type: str,
    cell_ids: list[str],
    scored: pd.DataFrame,
    populations: pd.Series,
    clusters: pd.Series | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    frame = scored.loc[cell_ids]
    group_populations = populations.loc[cell_ids].astype(str)
    population_counts = group_populations.value_counts()
    n_cells = int(len(frame))
    row: dict[str, object] = {
        "label": label,
        "group_type": group_type,
        "n_cells": n_cells,
        "dominant_population": str(population_counts.index[0]),
        "dominant_population_fraction": float(population_counts.iloc[0] / n_cells),
        "n_populations": int(len(population_counts)),
        "effective_populations": float(np.exp(entropy(population_counts.to_numpy()))),
        "top_populations": top_counts_text(group_populations),
        "progenitor_population_fraction": float(
            group_populations.isin(PROGENITOR_POPULATIONS).mean()
        ),
        "contains_trunk": bool((group_populations == "trunk").any()),
        "contains_tip": bool((group_populations == "tip").any()),
        "contains_proliferating": bool((group_populations == "proliferating").any()),
        "contains_endocrine": bool((group_populations == "endocrine").any()),
    }
    if clusters is not None:
        group_clusters = clusters.loc[cell_ids].astype(str)
        cluster_counts = group_clusters.value_counts()
        row.update(
            {
                "n_tbs_clusters": int(len(cluster_counts)),
                "dominant_tbs_cluster": str(cluster_counts.index[0]),
                "dominant_tbs_cluster_fraction": float(cluster_counts.iloc[0] / n_cells),
                "top_tbs_clusters": top_counts_text(group_clusters),
            }
        )
    if extra:
        row.update(extra)
    for signature in SIGNATURES:
        row[f"{signature}_score"] = float(frame[f"{signature}_score"].mean())
    for marker in ["NEUROG3", "PAX4", "PDX1", "SOX9", "HNF1B", "PTF1A", "CPA1", "MKI67"]:
        if marker in frame:
            row[f"{marker}_scaled_mean"] = float(frame[marker].mean())
    row["progenitor_interpretation"] = interpret_group(row)
    return row


def interpret_group(row: dict[str, object]) -> str:
    progenitor_fraction = float(row["progenitor_population_fraction"])
    dominant = str(row["dominant_population"])
    dominant_fraction = float(row["dominant_population_fraction"])
    n_populations = int(row["n_populations"])
    effective_populations = float(row.get("effective_populations", n_populations))
    endocrine_core = float(row.get("endocrine_progenitor_core_score", math.nan))
    trunk_score = float(row.get("trunk_progenitor_score", math.nan))
    tip_score = float(row.get("tip_progenitor_score", math.nan))
    proliferation = float(row.get("proliferation_score", math.nan))
    endocrine = float(row.get("mature_endocrine_score", math.nan))

    if progenitor_fraction >= 0.80 and n_populations >= 2:
        if dominant_fraction < 0.70 or n_populations > 3 or effective_populations > 2.5:
            return "broad mixed progenitor-rich neighborhood"
        return "mixed fetal progenitor-state grouping"
    if progenitor_fraction >= 0.80 and dominant in PROGENITOR_POPULATIONS:
        if dominant == "trunk" and trunk_score >= tip_score:
            return "trunk-progenitor-enriched grouping"
        if dominant == "tip" and tip_score >= trunk_score:
            return "tip-progenitor-enriched grouping"
        if dominant == "proliferating" and proliferation > 0:
            return "proliferating progenitor-enriched grouping"
        return "fetal progenitor-enriched grouping"
    if dominant == "endocrine" or endocrine_core > 0.5:
        return "endocrine/endocrine-progenitor-like grouping"
    if endocrine > 0.5:
        return "mature endocrine-like grouping"
    if progenitor_fraction >= 0.30:
        return "mixed grouping with partial progenitor content"
    return "non-progenitor or broad mixed grouping"


def build_tree_helpers(
    edges: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, str], str, Any]:
    children: dict[str, list[str]] = {}
    parents = set(edges["parent"].astype(str))
    child_nodes = set(edges["child"].astype(str))
    for parent, child in zip(edges["parent"].astype(str), edges["child"].astype(str), strict=True):
        children.setdefault(parent, []).append(child)
    parent_of = {child: parent for parent, child_list in children.items() for child in child_list}
    root = next(iter(parents - child_nodes))

    @lru_cache(maxsize=None)
    def descendant_leaves(node: str) -> tuple[int, ...]:
        if node.startswith("L"):
            return (int(node[1:]),)
        leaves: list[int] = []
        for child in children[node]:
            leaves.extend(descendant_leaves(child))
        return tuple(sorted(leaves))

    return children, parent_of, root, descendant_leaves


def depth_from_root(node: str, parent_of: dict[str, str]) -> int:
    depth = 0
    while node in parent_of:
        depth += 1
        node = parent_of[node]
    return depth


def add_generated_at(fig: plt.Figure, generated_at: str) -> None:
    fig.text(
        0.99,
        0.01,
        f"Generated at: {generated_at}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#6b7280",
    )


def save_timestamped_figure(
    fig: plt.Figure, output_path: Path, generated_at: str, *, dpi: int = 240
) -> None:
    add_generated_at(fig, generated_at)
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(
        output_path.with_suffix(".pdf"),
        metadata={"Subject": f"Generated at: {generated_at}"},
    )


def write_umap_signature_plot(
    assignments: pd.DataFrame,
    scored: pd.DataFrame,
    output_path: Path,
    generated_at: str,
) -> None:
    plot_frame = assignments[["cell_id", "celltype", "umap1", "umap2"]].copy()
    plot_frame = plot_frame.join(scored, on="cell_id")
    panels = [
        ("celltype", "Population", "categorical"),
        ("trunk_progenitor_score", "Trunk progenitor score", "continuous"),
        ("tip_progenitor_score", "Tip progenitor score", "continuous"),
        ("proliferation_score", "Proliferation score", "continuous"),
        ("endocrine_progenitor_core_score", "Endocrine progenitor core score", "continuous"),
        ("endocrine_commitment_score", "Endocrine commitment score", "continuous"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (column, title, kind) in zip(axes.ravel(), panels, strict=True):
        if kind == "categorical":
            values = plot_frame[column].astype("category")
            scatter = ax.scatter(
                plot_frame["umap1"],
                plot_frame["umap2"],
                c=values.cat.codes,
                cmap="tab20",
                s=8,
                linewidths=0,
            )
            del scatter
            for label, group in plot_frame.groupby(column):
                ax.text(
                    float(group["umap1"].median()),
                    float(group["umap2"].median()),
                    str(label),
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
                )
        else:
            scatter = ax.scatter(
                plot_frame["umap1"],
                plot_frame["umap2"],
                c=plot_frame[column],
                cmap="viridis",
                s=8,
                linewidths=0,
            )
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Goncalves fetal pancreas progenitor signatures", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_timestamped_figure(fig, output_path, generated_at)
    plt.close(fig)


def write_signature_heatmap(summary: pd.DataFrame, output_path: Path, generated_at: str) -> None:
    rows = summary.copy()
    rows["sort_group"] = np.where(rows["group_type"] == "population", 0, 1)
    rows = rows.sort_values(
        ["sort_group", "progenitor_population_fraction", "n_cells"],
        ascending=[True, False, False],
    )
    rows = pd.concat(
        [
            rows[rows["group_type"] == "population"],
            rows[(rows["group_type"] == "tbs_cluster") & (rows["n_cells"] >= 25)].head(30),
        ],
        ignore_index=True,
    )
    score_cols = [f"{signature}_score" for signature in SIGNATURES]
    matrix = rows[score_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(rows))))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels([col.replace("_score", "") for col in score_cols], rotation=45, ha="right")
    labels = [
        f"{row.group_type}:{row.label} n={int(row.n_cells)}" for row in rows.itertuples(index=False)
    ]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, label="Mean marker z-score")
    fig.tight_layout()
    save_timestamped_figure(fig, output_path, generated_at)
    plt.close(fig)


def write_meeting_plot(meetings: pd.DataFrame, output_path: Path, generated_at: str) -> None:
    if meetings.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    color = meetings["trunk_progenitor_score"].fillna(0) + meetings["tip_progenitor_score"].fillna(
        0
    )
    scatter = ax.scatter(
        meetings["n_cells"],
        meetings["progenitor_population_fraction"],
        c=color,
        s=35 + 8 * meetings["n_tbs_clusters"].astype(float),
        cmap="viridis",
        alpha=0.78,
        edgecolors="#111827",
        linewidths=0.4,
    )
    label_rows = meetings.sort_values(
        ["progenitor_population_fraction", "n_cells"],
        ascending=[False, False],
    ).head(18)
    for row in label_rows.itertuples(index=False):
        ax.annotate(
            str(row.label),
            (float(row.n_cells), float(row.progenitor_population_fraction)),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Meeting-node descendant cells")
    ax.set_ylabel("Fraction trunk/tip/proliferating")
    fig.colorbar(scatter, ax=ax, label="Trunk + tip progenitor score")
    fig.tight_layout()
    save_timestamped_figure(fig, output_path, generated_at)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-marker-mapping", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    start = time.perf_counter()
    adata_path = output_dir / "goncalves_fetal_pancreas_classical_pipeline.h5ad"
    assignments = pd.read_csv(output_dir / "method_assignments.csv")
    assignments["cell_id"] = assignments["cell_id"].astype(str)
    assignments = assignments.set_index("cell_id", drop=False)
    adata = ad.read_h5ad(adata_path)
    mapping = load_marker_mapping(
        adata.raw.var_names,
        output_dir,
        refresh=args.refresh_marker_mapping,
    )
    expression = load_marker_expression(adata, mapping)
    scored, signature_defs = add_signature_scores(expression)
    signature_defs.to_csv(
        output_dir / "goncalves_progenitor_signature_definitions.csv", index=False
    )
    scored.to_csv(output_dir / "goncalves_cell_marker_signature_scores.csv")

    populations = adata.obs["celltype"].astype(str)
    populations.index = adata.obs_names.astype(str)
    clusters = assignments[METHOD_KEY].astype(str)
    clusters.index = assignments.index

    population_rows = [
        summarize_group(
            label=str(population),
            group_type="population",
            cell_ids=group.index.astype(str).to_list(),
            scored=scored,
            populations=populations,
        )
        for population, group in populations.groupby(populations)
    ]
    population_df = pd.DataFrame(population_rows).sort_values("n_cells", ascending=False)
    population_df.to_csv(
        output_dir / "goncalves_population_progenitor_signature_scores.csv", index=False
    )

    cluster_rows = [
        summarize_group(
            label=f"C{cluster_id}",
            group_type="tbs_cluster",
            cell_ids=group.index.astype(str).to_list(),
            scored=scored,
            populations=populations,
            clusters=clusters,
            extra={"tbs_cluster": str(cluster_id)},
        )
        for cluster_id, group in clusters.groupby(clusters)
    ]
    cluster_df = pd.DataFrame(cluster_rows).sort_values(
        ["progenitor_population_fraction", "n_cells"],
        ascending=[False, False],
    )
    cluster_df.to_csv(
        output_dir / "goncalves_tbs_cluster_progenitor_signature_scores.csv", index=False
    )

    edges = pd.read_csv(output_dir / f"{METHOD_KEY}_tree_edges.csv")
    children, parent_of, root, descendant_leaves = build_tree_helpers(edges)
    internal_nodes = sorted(children, key=lambda node: int(node[1:]))
    cell_ids_by_position = assignments["cell_id"].to_numpy()
    node_rows: list[dict[str, object]] = []
    for node in internal_nodes:
        leaf_indices = np.array(descendant_leaves(node), dtype=int)
        cell_ids = cell_ids_by_position[leaf_indices].astype(str).tolist()
        pop_values = populations.loc[cell_ids].astype(str)
        cluster_values = clusters.loc[cell_ids].astype(str)
        role = (
            "root"
            if node == root
            else "within_final_cluster"
            if cluster_values.nunique() == 1
            else "shared_ancestor_across_final_clusters"
        )
        node_rows.append(
            summarize_group(
                label=node,
                group_type="inner_node",
                cell_ids=cell_ids,
                scored=scored,
                populations=populations,
                clusters=clusters,
                extra={
                    "node": node,
                    "depth_from_root": depth_from_root(node, parent_of),
                    "role": role,
                    "child_nodes": ",".join(children[node]),
                    "child_count": len(children[node]),
                    "is_root": node == root,
                    "population_entropy": entropy(pop_values.value_counts().to_numpy()),
                },
            )
        )
    node_df = pd.DataFrame(node_rows)
    node_df.to_csv(
        output_dir / "goncalves_tbs_inner_node_progenitor_signature_scores.csv", index=False
    )

    cluster_to_leaves = {
        str(cluster_id): frozenset(np.flatnonzero(clusters.to_numpy() == str(cluster_id)).tolist())
        for cluster_id in sorted(clusters.unique(), key=lambda value: int(value))
    }
    meeting_rows: list[dict[str, object]] = []
    meeting_child_rows: list[dict[str, object]] = []
    for node in internal_nodes:
        child_cluster_sets: list[set[str]] = []
        valid_children = True
        for child in children[node]:
            child_leaves = frozenset(descendant_leaves(child))
            child_clusters = {str(clusters.iloc[index]) for index in child_leaves}
            union = frozenset().union(*(cluster_to_leaves[cluster] for cluster in child_clusters))
            if union != child_leaves:
                valid_children = False
                break
            child_cluster_sets.append(child_clusters)
        if not valid_children:
            continue
        node_clusters = set().union(*child_cluster_sets)
        if len(node_clusters) <= 1:
            continue
        leaf_indices = np.array(descendant_leaves(node), dtype=int)
        cell_ids = cell_ids_by_position[leaf_indices].astype(str).tolist()
        row = summarize_group(
            label=node,
            group_type="monophyletic_meeting_node",
            cell_ids=cell_ids,
            scored=scored,
            populations=populations,
            clusters=clusters,
            extra={
                "node": node,
                "depth_from_root": depth_from_root(node, parent_of),
                "child_count": len(children[node]),
                "child_cluster_sets": " | ".join(
                    ",".join(f"C{cluster}" for cluster in sorted(child_set, key=int))
                    for child_set in child_cluster_sets
                ),
                "meeting_clusters": ",".join(
                    f"C{cluster}" for cluster in sorted(node_clusters, key=int)
                ),
            },
        )
        meeting_rows.append(row)
        for child, child_set in zip(children[node], child_cluster_sets, strict=True):
            child_indices = np.array(descendant_leaves(child), dtype=int)
            child_cell_ids = cell_ids_by_position[child_indices].astype(str).tolist()
            meeting_child_rows.append(
                summarize_group(
                    label=f"{node}->{child}",
                    group_type="monophyletic_meeting_child",
                    cell_ids=child_cell_ids,
                    scored=scored,
                    populations=populations,
                    clusters=clusters,
                    extra={
                        "parent_node": node,
                        "child_node": child,
                        "child_clusters": ",".join(
                            f"C{cluster}" for cluster in sorted(child_set, key=int)
                        ),
                    },
                )
            )
    meeting_df = pd.DataFrame(meeting_rows).sort_values(
        ["progenitor_population_fraction", "n_cells"],
        ascending=[False, False],
    )
    meeting_child_df = pd.DataFrame(meeting_child_rows)
    meeting_df.to_csv(
        output_dir / "goncalves_tbs_monophyletic_meeting_progenitor_signature_scores.csv",
        index=False,
    )
    meeting_child_df.to_csv(
        output_dir / "goncalves_tbs_monophyletic_meeting_child_progenitor_signature_scores.csv",
        index=False,
    )

    combined_summary = pd.concat([population_df, cluster_df], ignore_index=True)
    write_umap_signature_plot(
        assignments.reset_index(drop=True),
        scored,
        output_dir / "goncalves_progenitor_signature_umap.png",
        generated_at,
    )
    write_signature_heatmap(
        combined_summary,
        output_dir / "goncalves_population_tbs_cluster_signature_heatmap.png",
        generated_at,
    )
    write_meeting_plot(
        meeting_df,
        output_dir / "goncalves_tbs_monophyletic_meeting_progenitor_plot.png",
        generated_at,
    )

    manifest = {
        "generated_at": generated_at,
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "method_key": METHOD_KEY,
        "elapsed_sec": time.perf_counter() - start,
        "cells": int(adata.n_obs),
        "raw_genes": int(adata.raw.shape[1]),
        "mapped_markers": int(mapping["present_in_raw"].sum()),
        "requested_markers": int(len(mapping)),
        "inner_nodes": int(len(node_df)),
        "monophyletic_meeting_nodes": int(len(meeting_df)),
        "progenitor_populations": sorted(PROGENITOR_POPULATIONS),
    }
    (output_dir / "goncalves_progenitor_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_default)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=json_default))
    print("\nTop population rows:")
    print(
        population_df[
            [
                "label",
                "n_cells",
                "trunk_progenitor_score",
                "tip_progenitor_score",
                "proliferation_score",
                "endocrine_progenitor_core_score",
                "progenitor_interpretation",
            ]
        ].to_string(index=False)
    )
    print("\nTop monophyletic progenitor meetings:")
    print(
        meeting_df[
            [
                "label",
                "n_cells",
                "n_tbs_clusters",
                "progenitor_population_fraction",
                "top_populations",
                "meeting_clusters",
                "progenitor_interpretation",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
