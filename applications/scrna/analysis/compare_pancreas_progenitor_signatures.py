"""Compare pancreas junctions and cell states with progenitor signatures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

METHOD_KEY = "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001"

SIGNATURES: dict[str, list[str]] = {
    "endocrine_progenitor_core": ["NEUROG3", "PAX4"],
    "endocrine_commitment_tf": [
        "RFX6",
        "NEUROD1",
        "ISL1",
        "PAX6",
        "NKX2-2",
        "FOXA2",
    ],
    "pancreatic_trunk_progenitor": [
        "PDX1",
        "SOX9",
        "HNF1B",
        "HES1",
        "ONECUT1",
        "NKX6-1",
        "KRT19",
    ],
    "adult_ductal_state": ["SOX9", "HNF1B", "ONECUT1", "KRT19", "EPCAM", "MUC1", "CFTR"],
    "tip_acinar_progenitor_or_acinar": ["PTF1A", "CPA1", "PRSS1", "REG1A"],
    "mature_endocrine_hormone": ["INS", "GCG", "SST", "PPY", "GHRL", "CHGA", "CHGB"],
    "mature_exocrine_enzyme": ["PRSS1", "CPA1", "REG1A"],
}
SIGNATURE_NOTES = {
    "endocrine_progenitor_core": (
        "Strict endocrine-progenitor evidence; NEUROG3 is required for a strong call."
    ),
    "endocrine_commitment_tf": (
        "Endocrine differentiation transcription factors; many persist in mature endocrine cells."
    ),
    "pancreatic_trunk_progenitor": (
        "Fetal trunk/progenitor-associated markers that overlap adult ductal programs."
    ),
    "adult_ductal_state": (
        "Adult ductal epithelial state markers; high score alone is not progenitor evidence."
    ),
    "tip_acinar_progenitor_or_acinar": (
        "Tip/acinar-associated markers; in adult pancreas this is mostly mature acinar signal."
    ),
    "mature_endocrine_hormone": "Mature endocrine hormone/secretory markers.",
    "mature_exocrine_enzyme": "Mature exocrine enzyme markers.",
}
MARKERS = sorted({gene for genes in SIGNATURES.values() for gene in genes})
KEY_MARKERS = [
    "NEUROG3",
    "PAX4",
    "RFX6",
    "NEUROD1",
    "PDX1",
    "SOX9",
    "HNF1B",
    "ONECUT1",
    "NKX6-1",
    "KRT19",
    "EPCAM",
    "PTF1A",
    "INS",
    "GCG",
    "SST",
    "PPY",
    "GHRL",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def output_dir() -> Path:
    return (
        project_root()
        / "raw"
        / "assets"
        / "benchmark-results"
        / "pancreas_scrna_cluster_benchmark_20260623"
    )


def top_counts_text(values: pd.Series, n: int = 8) -> str:
    counts = values.astype(str).value_counts()
    return "; ".join(f"{key}:{value}" for key, value in counts.head(n).items())


def read_expression(adata_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    adata = ad.read_h5ad(adata_path)
    raw = adata.raw
    if raw is None:
        raise RuntimeError("Expected raw expression layer in pancreas AnnData.")
    raw_index = {str(gene): index for index, gene in enumerate(raw.var_names)}
    missing = sorted(set(MARKERS) - set(raw_index))
    if missing:
        raise RuntimeError(f"Missing signature genes from raw layer: {missing}")

    matrix = raw.X[:, [raw_index[gene] for gene in MARKERS]]
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().toarray()
    expression = pd.DataFrame(
        np.asarray(matrix), index=adata.obs_names.astype(str), columns=MARKERS
    )
    celltypes = adata.obs["celltype"].astype(str)
    celltypes.index = adata.obs_names.astype(str)
    return expression, celltypes


def add_signature_scores(expression: pd.DataFrame) -> pd.DataFrame:
    scored = expression.copy()
    means = expression.mean(axis=0)
    stds = expression.std(axis=0, ddof=0).replace(0, 1.0)
    zscores = (expression - means) / stds
    for signature, genes in SIGNATURES.items():
        scored[f"{signature}_score"] = zscores[genes].mean(axis=1)
    return scored


def summarize_cells(
    *,
    label: str,
    group_type: str,
    cell_ids: list[str],
    scored: pd.DataFrame,
    celltypes: pd.Series,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    frame = scored.loc[cell_ids]
    group_celltypes = celltypes.loc[cell_ids]
    n_cells = int(len(frame))
    row: dict[str, object] = {
        "label": label,
        "group_type": group_type,
        "n_cells": n_cells,
        "dominant_celltype": str(group_celltypes.value_counts().index[0]),
        "dominant_celltype_fraction": float(group_celltypes.value_counts().iloc[0] / n_cells),
        "top_celltypes": top_counts_text(group_celltypes),
    }
    if extra:
        row.update(extra)

    for signature in SIGNATURES:
        row[f"{signature}_score"] = float(frame[f"{signature}_score"].mean())
    for marker in KEY_MARKERS:
        row[f"{marker}_mean"] = float(frame[marker].mean())
        row[f"{marker}_pct_positive"] = float((frame[marker] > 0).mean())

    row["progenitor_call"] = progenitor_call(row)
    return row


def progenitor_call(row: dict[str, object]) -> str:
    n_cells = int(row["n_cells"])
    neurog3_pct = float(row["NEUROG3_pct_positive"])
    neurog3_mean = float(row["NEUROG3_mean"])
    endocrine_core = float(row["endocrine_progenitor_core_score"])
    endocrine_commit = float(row["endocrine_commitment_tf_score"])
    trunk = float(row["pancreatic_trunk_progenitor_score"])
    ductal = float(row["adult_ductal_state_score"])
    tip_acinar = float(row["tip_acinar_progenitor_or_acinar_score"])
    mature_endocrine = float(row["mature_endocrine_hormone_score"])

    if n_cells < 10 and neurog3_pct > 0:
        return "too few cells for a stable progenitor-state call despite NEUROG3 signal"
    if neurog3_pct >= 0.01 and neurog3_mean > 0 and endocrine_core > 0:
        return "putative NEUROG3+ endocrine-progenitor-like signal; requires manual validation"
    if mature_endocrine > 0.5 and endocrine_commit > 0 and neurog3_pct == 0:
        return "endocrine-committed or mature endocrine state; no NEUROG3 progenitor support"
    if ductal > 0.5 and trunk > 0.5 and neurog3_pct == 0:
        return "adult ductal/trunk-marker overlap; not endocrine progenitor evidence"
    if tip_acinar > 0.8 and neurog3_pct == 0:
        return "acinar/tip-marker overlap dominated by adult exocrine state"
    if endocrine_core <= 0 and neurog3_pct == 0:
        return "no core endocrine-progenitor signal"
    return "weak or ambiguous progenitor-like marker overlap without decisive NEUROG3 support"


def build_tree_helpers(edges: pd.DataFrame) -> tuple[dict[str, list[str]], object]:
    children: dict[str, list[str]] = {}
    for parent, child in zip(edges["parent"].astype(str), edges["child"].astype(str), strict=True):
        children.setdefault(parent, []).append(child)

    @lru_cache(maxsize=None)
    def descendant_leaves(node: str) -> tuple[int, ...]:
        if node.startswith("L"):
            return (int(node[1:]),)
        leaves: list[int] = []
        for child in children[node]:
            leaves.extend(descendant_leaves(child))
        return tuple(leaves)

    return children, descendant_leaves


def write_signature_definition(output_path: Path) -> None:
    rows = [
        {
            "signature": signature,
            "genes": ",".join(genes),
            "interpretation_note": SIGNATURE_NOTES[signature],
        }
        for signature, genes in SIGNATURES.items()
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_dataset_summary(
    *,
    scored: pd.DataFrame,
    scored_subset: pd.DataFrame,
    output_path: Path,
) -> None:
    rows = []
    for label, frame in [
        ("full_h5ad", scored),
        ("benchmark_subset", scored_subset),
    ]:
        rows.append(
            {
                "scope": label,
                "n_cells": int(len(frame)),
                "NEUROG3_positive_cells": int((frame["NEUROG3"] > 0).sum()),
                "NEUROG3_positive_fraction": float((frame["NEUROG3"] > 0).mean()),
                "PAX4_positive_cells": int((frame["PAX4"] > 0).sum()),
                "PAX4_positive_fraction": float((frame["PAX4"] > 0).mean()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    heat_rows = summary[
        (summary["group_type"].isin(["reference", "meeting_node", "benchmark_celltype"]))
        & (
            summary["label"].isin(
                [
                    "full_NEUROG3_positive",
                    "N4963",
                    "N4917",
                    "N4957",
                    "N4964",
                    "N4922",
                    "N4965",
                    "N4857",
                    "alpha",
                    "beta",
                    "delta",
                    "gamma",
                    "epsilon",
                    "ductal",
                    "acinar",
                    "PSC",
                    "endothelial",
                ]
            )
        )
    ].copy()
    score_cols = [f"{signature}_score" for signature in SIGNATURES]
    heat_rows["display_label"] = heat_rows.apply(
        lambda row: f"{row['label']} ({row['group_type']}, n={int(row['n_cells'])})",
        axis=1,
    )
    matrix = heat_rows[score_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12.8, max(5.5, 0.38 * len(heat_rows))))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_yticks(np.arange(len(heat_rows)))
    ax.set_yticklabels(heat_rows["display_label"], fontsize=8)
    ax.set_xticks(np.arange(len(score_cols)))
    ax.set_xticklabels(
        [col.removesuffix("_score") for col in score_cols],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_title("Pancreas progenitor and mature-state signature scores")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="mean gene z-score")
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=240)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def write_umap(scored_subset: pd.DataFrame, output_path: Path) -> None:
    panels = [
        ("NEUROG3", "NEUROG3 expression"),
        ("endocrine_progenitor_core_score", "Endocrine progenitor core"),
        ("pancreatic_trunk_progenitor_score", "Trunk/progenitor markers"),
        ("mature_endocrine_hormone_score", "Mature endocrine hormones"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), squeeze=False)
    for ax, (column, title) in zip(axes.ravel(), panels, strict=True):
        values = scored_subset[column].to_numpy(dtype=float)
        if np.nanmax(values) == np.nanmin(values):
            vmin, vmax = float(np.nanmin(values)), float(np.nanmin(values) + 1.0)
        else:
            vmin, vmax = np.nanpercentile(values, [2, 98])
        scatter = ax.scatter(
            scored_subset["umap1"],
            scored_subset["umap2"],
            c=values,
            s=10,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=0.9,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Benchmark subset progenitor-signature UMAPs", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path.with_suffix(".png"), dpi=240)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    out = output_dir()
    assignments = pd.read_csv(out / "method_assignments.csv")
    edges = pd.read_csv(out / f"{METHOD_KEY}_tree_edges.csv")
    meeting_review = pd.read_csv(out / "tbs_adaptive_monophyletic_subtree_meeting_review.csv")
    children, descendant_leaves = build_tree_helpers(edges)

    expression, celltypes = read_expression(out / "pancreas_classical_pipeline.h5ad")
    scored = add_signature_scores(expression)
    benchmark_cell_ids = assignments["cell_id"].astype(str).tolist()
    scored_subset = scored.loc[benchmark_cell_ids].copy()
    scored_subset["celltype"] = assignments["celltype"].astype(str).to_numpy()
    scored_subset[METHOD_KEY] = assignments[METHOD_KEY].astype(int).to_numpy()
    scored_subset["umap1"] = assignments["umap1"].to_numpy()
    scored_subset["umap2"] = assignments["umap2"].to_numpy()

    write_signature_definition(out / "pancreas_progenitor_signature_definitions.csv")
    write_dataset_summary(
        scored=scored,
        scored_subset=scored_subset,
        output_path=out / "pancreas_progenitor_signature_dataset_summary.csv",
    )

    summary_rows: list[dict[str, object]] = []
    neurog3_positive = scored.index[scored["NEUROG3"] > 0].tolist()
    if neurog3_positive:
        neurog3_positive_frame = scored.loc[
            neurog3_positive,
            MARKERS + [f"{signature}_score" for signature in SIGNATURES],
        ].copy()
        neurog3_positive_frame.insert(
            0,
            "celltype",
            celltypes.loc[neurog3_positive].astype(str).to_numpy(),
        )
        neurog3_positive_frame.insert(0, "cell_id", neurog3_positive_frame.index)
        neurog3_positive_frame.to_csv(
            out / "pancreas_full_neurog3_positive_cells.csv",
            index=False,
        )
        summary_rows.append(
            summarize_cells(
                label="full_NEUROG3_positive",
                group_type="reference",
                cell_ids=neurog3_positive,
                scored=scored,
                celltypes=celltypes,
                extra={"scope": "full_h5ad"},
            )
        )

    focus_nodes = meeting_review.loc[meeting_review["display_focus"].astype(bool), "node"].astype(
        str
    )
    node_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    for node in focus_nodes:
        node_cell_ids = (
            assignments.iloc[list(descendant_leaves(node))]["cell_id"].astype(str).tolist()
        )
        row = summarize_cells(
            label=node,
            group_type="meeting_node",
            cell_ids=node_cell_ids,
            scored=scored,
            celltypes=celltypes,
            extra={"scope": "benchmark_subset"},
        )
        node_rows.append(row)
        summary_rows.append(row)
        for child in children[node]:
            child_cell_ids = (
                assignments.iloc[list(descendant_leaves(child))]["cell_id"].astype(str).tolist()
            )
            child_row = summarize_cells(
                label=f"{node}:{child}",
                group_type="meeting_child_branch",
                cell_ids=child_cell_ids,
                scored=scored,
                celltypes=celltypes,
                extra={"scope": "benchmark_subset", "meeting_node": node, "child_node": child},
            )
            child_rows.append(child_row)

    full_celltype_rows = [
        summarize_cells(
            label=str(celltype),
            group_type="full_celltype",
            cell_ids=group.index.astype(str).tolist(),
            scored=scored,
            celltypes=celltypes,
            extra={"scope": "full_h5ad"},
        )
        for celltype, group in celltypes.groupby(celltypes)
    ]
    benchmark_celltype_rows = [
        summarize_cells(
            label=str(celltype),
            group_type="benchmark_celltype",
            cell_ids=group.index.astype(str).tolist(),
            scored=scored_subset,
            celltypes=scored_subset["celltype"],
            extra={"scope": "benchmark_subset"},
        )
        for celltype, group in scored_subset.groupby("celltype")
    ]
    cluster_rows = [
        summarize_cells(
            label=f"C{int(cluster_id)}",
            group_type="benchmark_tbs_cluster",
            cell_ids=group.index.astype(str).tolist(),
            scored=scored_subset,
            celltypes=scored_subset["celltype"],
            extra={"scope": "benchmark_subset", "cluster_id": int(cluster_id)},
        )
        for cluster_id, group in scored_subset.groupby(METHOD_KEY)
    ]
    summary_rows.extend(benchmark_celltype_rows)

    node_df = pd.DataFrame(node_rows)
    child_df = pd.DataFrame(child_rows)
    full_celltype_df = pd.DataFrame(full_celltype_rows).sort_values(
        ["endocrine_progenitor_core_score", "n_cells"],
        ascending=[False, False],
    )
    benchmark_celltype_df = pd.DataFrame(benchmark_celltype_rows).sort_values(
        ["endocrine_progenitor_core_score", "n_cells"],
        ascending=[False, False],
    )
    cluster_df = pd.DataFrame(cluster_rows).sort_values(
        ["endocrine_progenitor_core_score", "n_cells"],
        ascending=[False, False],
    )
    summary_df = pd.DataFrame(summary_rows)

    node_df.to_csv(
        out / "tbs_adaptive_monophyletic_meeting_progenitor_signature_comparison.csv",
        index=False,
    )
    child_df.to_csv(
        out / "tbs_adaptive_monophyletic_meeting_child_progenitor_signature_comparison.csv",
        index=False,
    )
    full_celltype_df.to_csv(
        out / "pancreas_full_celltype_progenitor_signature_scores.csv", index=False
    )
    benchmark_celltype_df.to_csv(
        out / "pancreas_benchmark_celltype_progenitor_signature_scores.csv",
        index=False,
    )
    cluster_df.to_csv(out / "tbs_adaptive_cluster_progenitor_signature_scores.csv", index=False)
    summary_df.to_csv(out / "pancreas_progenitor_signature_summary_rows.csv", index=False)

    write_heatmap(summary_df, out / "pancreas_progenitor_signature_comparison")
    write_umap(scored_subset, out / "pancreas_progenitor_signature_umap")


if __name__ == "__main__":
    main()
