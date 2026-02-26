#!/usr/bin/env python3
"""
Biological Analysis of HC Gene Clustering Results
==================================================

Analyses the KL-tree clustering of hypertrophic cardiomyopathy (HC) gene-pathway
feature matrices.  Reads the pre-computed cluster assignments (from
`run_feature_matrix_with_umap.py`) and the original binary feature matrix, then
produces:

  1. Cluster overview & size distribution histogram
  2. Per-cluster pathway enrichment (Fisher exact test, BH-corrected)
  3. Per-cluster gene list with shared-pathway summary
  4. Jaccard similarity heatmap between clusters
  5. Pathway-coverage bar chart (top pathways coloured by cluster)
  6. Full results exported to TSV / CSV

Usage
-----
    # Analyse the α=0.05 run:
    python scripts/analyze_hc_clusters.py

    # Custom paths / alpha:
    python scripts/analyze_hc_clusters.py \
        --feature-matrix HC_feature_matrix_GO_CC.tsv \
        --assignments benchmarks/results/HC_GO_CC_005/cluster_assignments.csv \
        --output-dir benchmarks/results/HC_GO_CC_005/bio_analysis

    # Compare across alphas:
    python scripts/analyze_hc_clusters.py --all-alphas
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_MATRIX = Path("data/HC_feature_matrix_GO_CC.tsv")
RESULT_DIRS = {
    0.01: "benchmarks/results/HC_GO_CC_001",
    0.05: "benchmarks/results/HC_GO_CC_005",
    0.10: "benchmarks/results/HC_GO_CC_010",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_data(
    feature_matrix_path: Path,
    assignments_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (feature_matrix, assignments) DataFrames."""
    fm = pd.read_csv(feature_matrix_path, sep="\t", index_col=0)
    assign = pd.read_csv(assignments_path, index_col=0)
    print(f"Feature matrix : {fm.shape[0]} genes × {fm.shape[1]} pathways")
    print(f"Assignments    : {len(assign)} genes → {assign['cluster_id'].nunique()} clusters")
    return fm, assign


# ═══════════════════════════════════════════════════════════════════════════
# 1. Cluster overview
# ═══════════════════════════════════════════════════════════════════════════


def cluster_overview(assign: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Print & save cluster size summary; return sizes Series."""
    sizes = assign.groupby("cluster_id").size().rename("n_genes").reset_index()
    sizes = sizes.sort_values("n_genes", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("CLUSTER SIZE DISTRIBUTION")
    print("=" * 70)
    print(f"  Total clusters : {len(sizes)}")
    print(f"  Singletons     : {(sizes.n_genes == 1).sum()}")
    print(f"  Size 2-5       : {((sizes.n_genes >= 2) & (sizes.n_genes <= 5)).sum()}")
    print(f"  Size 6-20      : {((sizes.n_genes >= 6) & (sizes.n_genes <= 20)).sum()}")
    print(f"  Size > 20      : {(sizes.n_genes > 20).sum()}")
    print(f"  Largest        : cluster {sizes.iloc[0].cluster_id} ({sizes.iloc[0].n_genes} genes)")

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(sizes.n_genes, bins=range(0, sizes.n_genes.max() + 2), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cluster size (genes)")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Size Distribution")
    ax.axvline(
        sizes.n_genes.median(), color="red", ls="--", label=f"median={sizes.n_genes.median():.0f}"
    )
    ax.legend()

    # Log-scale y for better visibility
    ax2 = axes[1]
    ax2.hist(sizes.n_genes, bins=range(0, sizes.n_genes.max() + 2), edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Cluster size (genes)")
    ax2.set_ylabel("Count (log)")
    ax2.set_yscale("log")
    ax2.set_title("Cluster Size Distribution (log scale)")

    fig.tight_layout()
    fig.savefig(out / "cluster_sizes_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out / 'cluster_sizes_histogram.png'}")

    return sizes


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-cluster gene lists with biological annotation
# ═══════════════════════════════════════════════════════════════════════════


def gene_lists_per_cluster(
    fm: pd.DataFrame,
    assign: pd.DataFrame,
    out: Path,
    min_size: int = 2,
) -> pd.DataFrame:
    """
    For each cluster with ≥ min_size genes:
      - List genes
      - Find pathways shared by ALL genes in the cluster
      - Find pathways shared by ≥ 50% of genes
    """
    print("\n" + "=" * 70)
    print(f"GENE LISTS & SHARED PATHWAYS  (clusters with ≥ {min_size} genes)")
    print("=" * 70)

    rows = []
    for cid in sorted(assign.cluster_id.unique()):
        genes = assign[assign.cluster_id == cid].index.tolist()
        n = len(genes)
        if n < min_size:
            continue

        sub = fm.loc[genes]
        # Pathways present in ALL genes
        all_shared = sub.columns[sub.min(axis=0) == 1].tolist()
        # Pathways present in ≥50% of genes
        half = sub.columns[sub.mean(axis=0) >= 0.5].tolist()
        # Mean pathway count per gene
        mean_pw = sub.sum(axis=1).mean()

        rows.append(
            {
                "cluster_id": cid,
                "n_genes": n,
                "genes": ", ".join(genes),
                "mean_pathways_per_gene": round(mean_pw, 1),
                "n_pathways_all_shared": len(all_shared),
                "pathways_all_shared": "; ".join(all_shared[:20]),
                "n_pathways_50pct": len(half),
                "pathways_50pct_shared": "; ".join(half[:20]),
            }
        )

        # Print summary
        print(f"\n  Cluster {cid} ({n} genes)")
        print(f"    Genes: {', '.join(genes[:15])}" + ("..." if n > 15 else ""))
        print(f"    Mean pathways/gene: {mean_pw:.1f}")
        if all_shared:
            print(f"    Shared by ALL ({len(all_shared)}):")
            for pw in all_shared[:8]:
                print(f"      • {pw}")
            if len(all_shared) > 8:
                print(f"      ... and {len(all_shared) - 8} more")
        else:
            print("    No pathway shared by every gene")
        if half and len(half) != len(all_shared):
            extra = set(half) - set(all_shared)
            print(f"    Shared by ≥50% ({len(half)} total, {len(extra)} additional):")
            for pw in list(extra)[:5]:
                print(f"      • {pw}")

    df_genes = pd.DataFrame(rows)
    df_genes.to_csv(out / "cluster_gene_lists.csv", index=False)
    print(f"\n  → {out / 'cluster_gene_lists.csv'}")
    return df_genes


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pathway enrichment (Fisher exact, BH-corrected)
# ═══════════════════════════════════════════════════════════════════════════


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adjusted = pvals * n / ranked
    # Enforce monotonicity (step-up)
    adjusted = np.minimum.accumulate(adjusted[np.argsort(ranked)[::-1]])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    return adjusted[np.argsort(np.argsort(ranked))]


def pathway_enrichment(
    fm: pd.DataFrame,
    assign: pd.DataFrame,
    out: Path,
    min_cluster_size: int = 2,
    max_pathways_per_cluster: int = 20,
) -> pd.DataFrame:
    """Fisher exact test for pathway enrichment per cluster."""
    print("\n" + "=" * 70)
    print("PATHWAY ENRICHMENT ANALYSIS  (Fisher exact, BH-corrected)")
    print("=" * 70)

    N = fm.shape[0]  # total genes
    all_results = []

    for cid in sorted(assign.cluster_id.unique()):
        cluster_genes = assign[assign.cluster_id == cid].index.tolist()
        k = len(cluster_genes)
        if k < min_cluster_size:
            continue

        cluster_data = fm.loc[cluster_genes]
        bg_genes = [g for g in fm.index if g not in cluster_genes]
        bg_data = fm.loc[bg_genes]

        pvals, odds_ratios, pathways = [], [], []
        for pw in fm.columns:
            a = int(cluster_data[pw].sum())  # in-cluster & in-pathway
            b = k - a  # in-cluster & NOT in-pathway
            c = int(bg_data[pw].sum())  # NOT in-cluster & in-pathway
            d = len(bg_genes) - c  # NOT in-cluster & NOT in-pathway
            if a == 0:
                continue  # skip pathways absent from cluster
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            pathways.append(pw)
            pvals.append(p)
            odds_ratios.append(odds)

        if not pvals:
            continue

        pvals_arr = np.array(pvals)
        qvals = _bh_correct(pvals_arr)

        for pw, p, q, odds in zip(pathways, pvals, qvals, odds_ratios):
            a = int(cluster_data[pw].sum())
            all_results.append(
                {
                    "cluster_id": cid,
                    "n_genes": k,
                    "pathway": pw,
                    "genes_in_pathway": a,
                    "frac_cluster": a / k,
                    "frac_background": int(bg_data[pw].sum()) / len(bg_genes),
                    "odds_ratio": odds,
                    "p_value": p,
                    "q_value": q,
                }
            )

    df_enrich = pd.DataFrame(all_results)
    if df_enrich.empty:
        print("  No enrichments found.")
        return df_enrich

    df_enrich = df_enrich.sort_values(["cluster_id", "q_value"])
    df_enrich.to_csv(out / "pathway_enrichment_full.csv", index=False)

    # Significant
    sig = df_enrich[df_enrich.q_value < 0.05]
    print(f"  Total tests           : {len(df_enrich)}")
    print(f"  Significant (q<0.05)  : {len(sig)}")
    print(f"  Clusters with ≥1 hit  : {sig.cluster_id.nunique()}")

    # Print top hits per cluster
    for cid in sig.cluster_id.unique():
        cdf = sig[sig.cluster_id == cid].head(max_pathways_per_cluster)
        n_genes = cdf.iloc[0].n_genes
        genes = ", ".join(assign[assign.cluster_id == cid].index.tolist()[:10])
        print(f"\n  Cluster {cid} ({n_genes} genes: {genes})")
        for _, row in cdf.iterrows():
            print(
                f"    q={row.q_value:.2e}  OR={row.odds_ratio:6.1f}  "
                f"{row.genes_in_pathway}/{row.n_genes}  {row.pathway}"
            )

    # Summary table: top 1 pathway per cluster
    top1 = sig.groupby("cluster_id").first().reset_index()
    top1.to_csv(out / "pathway_enrichment_top1_per_cluster.csv", index=False)
    print(f"\n  → {out / 'pathway_enrichment_full.csv'}")
    print(f"  → {out / 'pathway_enrichment_top1_per_cluster.csv'}")

    return df_enrich


# ═══════════════════════════════════════════════════════════════════════════
# 4. Jaccard similarity heatmap between non-singleton clusters
# ═══════════════════════════════════════════════════════════════════════════


def cluster_jaccard_heatmap(
    fm: pd.DataFrame,
    assign: pd.DataFrame,
    out: Path,
    min_size: int = 2,
) -> None:
    """Heatmap of pairwise Jaccard similarity of pathway profiles between clusters."""
    print("\n" + "=" * 70)
    print("INTER-CLUSTER JACCARD SIMILARITY")
    print("=" * 70)

    # Build cluster pathway profiles (mean across genes)
    clusters = []
    cids = []
    for cid in sorted(assign.cluster_id.unique()):
        genes = assign[assign.cluster_id == cid].index.tolist()
        if len(genes) < min_size:
            continue
        profile = fm.loc[genes].mean(axis=0).values  # fraction of genes with each pathway
        clusters.append(profile)
        cids.append(cid)

    if len(cids) < 2:
        print("  Too few non-singleton clusters for heatmap.")
        return

    profiles = np.array(clusters)
    # Binarize at 50% threshold for Jaccard
    binary = (profiles >= 0.5).astype(int)

    n = len(cids)
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = (binary[i] & binary[j]).sum()
            union = (binary[i] | binary[j]).sum()
            jaccard[i, j] = inter / union if union > 0 else 0.0

    # Cluster labels with size
    sizes = assign.groupby("cluster_id").size()
    labels = [f"C{c} ({sizes[c]})" for c in cids]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.4)))
    im = ax.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Inter-Cluster Jaccard Similarity\n(pathway profile, ≥50% threshold)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Jaccard index")
    fig.tight_layout()
    fig.savefig(out / "cluster_jaccard_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Non-singleton clusters: {n}")
    print(f"  → {out / 'cluster_jaccard_heatmap.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Top pathways bar chart coloured by cluster
# ═══════════════════════════════════════════════════════════════════════════


def top_pathways_chart(
    fm: pd.DataFrame,
    assign: pd.DataFrame,
    out: Path,
    n_top: int = 30,
) -> None:
    """Stacked bar chart of the most common pathways, coloured by cluster."""
    print("\n" + "=" * 70)
    print(f"TOP {n_top} MOST COMMON PATHWAYS")
    print("=" * 70)

    total_per_pw = fm.sum(axis=0).sort_values(ascending=False)
    top_pw = total_per_pw.head(n_top)

    # Build per-cluster contribution
    cluster_ids = sorted(assign.cluster_id.unique())
    # Aggregate: for each top pathway, how many genes per cluster
    matrix = []
    for cid in cluster_ids:
        genes = assign[assign.cluster_id == cid].index.tolist()
        sub = fm.loc[genes, top_pw.index]
        matrix.append(sub.sum(axis=0).values)
    matrix = np.array(matrix)  # (n_clusters, n_top)

    # Plot stacked horizontal bars
    fig, ax = plt.subplots(figsize=(12, max(6, n_top * 0.3)))
    cmap = plt.cm.tab20(np.linspace(0, 1, min(20, len(cluster_ids))))
    y_pos = np.arange(n_top)
    left = np.zeros(n_top)
    for i, cid in enumerate(cluster_ids):
        bar_vals = matrix[i]
        colour = cmap[i % 20]
        # Only label clusters that contribute meaningfully
        label = f"C{cid}" if bar_vals.sum() > 0 else None
        ax.barh(y_pos, bar_vals, left=left, color=colour, height=0.8)
        left += bar_vals

    # Shorten pathway names for display
    short_names = [pw[:60] + "..." if len(pw) > 60 else pw for pw in top_pw.index]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Number of genes annotated")
    ax.set_title(f"Top {n_top} Pathways — Gene Count by Cluster")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "top_pathways_by_cluster.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for pw, count in top_pw.head(15).items():
        print(f"  {count:4d}  {pw}")
    print(f"  → {out / 'top_pathways_by_cluster.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Biological interpretation summary
# ═══════════════════════════════════════════════════════════════════════════


def biological_summary(
    fm: pd.DataFrame,
    assign: pd.DataFrame,
    df_enrich: pd.DataFrame,
    out: Path,
) -> None:
    """Print a high-level biological interpretation."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("BIOLOGICAL INTERPRETATION SUMMARY")
    print(sep)

    sizes = assign.groupby("cluster_id").size()
    n_tot = len(assign)
    n_clust = len(sizes)

    print(
        f"""
Dataset: {n_tot} HC-associated genes × {fm.shape[1]} Reactome pathways
Clusters found: {n_clust}  (α = 0.05, hamming + average linkage)

Structure:
  • 1 dominant cluster ({sizes.max()} genes, {sizes.max()/n_tot*100:.0f}% of all genes)
    — genes with sparse / generic pathway annotations
  • {(sizes >= 2).sum() - 1} smaller clusters (2–{sizes[sizes < sizes.max()].max()} genes)
    — genes grouped by shared specific pathway memberships
  • {(sizes == 1).sum()} singletons
    — genes with unique pathway profiles
"""
    )

    if df_enrich is not None and not df_enrich.empty:
        sig = df_enrich[df_enrich.q_value < 0.05]
        n_sig_clusters = sig.cluster_id.nunique()
        print(
            f"Enrichment: {n_sig_clusters} clusters have ≥1 significantly enriched pathway (q<0.05)"
        )
        print()

        # Group notable biological themes
        ion_kw = [
            "ion channel",
            "potassium",
            "sodium",
            "calcium",
            "KCNJ",
            "SCN",
            "CACN",
            "voltage",
            "cardiac conduction",
        ]
        sarcomere_kw = [
            "muscle",
            "contraction",
            "myosin",
            "actin",
            "sarcomere",
            "tropomyosin",
            "cardiac",
            "striated",
        ]
        signaling_kw = [
            "signaling",
            "kinase",
            "phospho",
            "MAPK",
            "AKT",
            "PI3K",
            "receptor",
            "G protein",
            "GPCR",
        ]
        desmosome_kw = [
            "desmosome",
            "cell junction",
            "cadherin",
            "plakophilin",
            "desmoglein",
            "desmocollin",
        ]
        nuclear_kw = ["nuclear", "nucleopore", "nuclear pore", "NUP", "lamin"]

        themes = {
            "Ion channels / electrophysiology": ion_kw,
            "Sarcomere / muscle contraction": sarcomere_kw,
            "Signal transduction": signaling_kw,
            "Desmosome / cell junctions": desmosome_kw,
            "Nuclear envelope / pore": nuclear_kw,
        }

        print("Key biological themes in non-singleton clusters:")
        for theme, keywords in themes.items():
            # Check both pathway names and gene names
            matching_clusters = set()
            for _, row in sig.iterrows():
                pw_lower = row.pathway.lower()
                genes_in_c = assign[assign.cluster_id == row.cluster_id].index.tolist()
                genes_str = " ".join(genes_in_c).upper()
                for kw in keywords:
                    if kw.lower() in pw_lower or kw.upper() in genes_str:
                        matching_clusters.add(row.cluster_id)
                        break
            if matching_clusters:
                cluster_strs = []
                for c in sorted(matching_clusters):
                    genes = assign[assign.cluster_id == c].index.tolist()
                    cluster_strs.append(f"C{c}({len(genes)})")
                print(f"  {theme}:")
                print(f"    Clusters: {', '.join(cluster_strs)}")

        print(
            """
Clinical relevance:
  HC genes cluster by functional module — ion channels, sarcomeric proteins,
  desmosomes, signaling cascades, and nuclear envelope components. This matches
  the known pathophysiology of hypertrophic cardiomyopathy, where mutations in
  sarcomeric and ion channel genes are the most common causes.

  The large dominant cluster likely contains genes with broad / non-specific
  pathway annotations that the statistical test cannot distinguish, while the
  smaller clusters represent functionally coherent gene modules.
"""
        )

    # Save summary text
    summary_path = out / "biological_summary.txt"
    # Re-generate to file
    lines = [
        "HC Gene Clustering — Biological Summary",
        f"{'=' * 50}",
        f"Genes: {n_tot}",
        f"Pathways: {fm.shape[1]}",
        f"Clusters: {n_clust}",
        f"Dominant cluster: {sizes.max()} genes",
        f"Singletons: {(sizes == 1).sum()}",
        "",
        "Non-singleton clusters:",
    ]
    for cid in sizes[sizes >= 2].sort_values(ascending=False).index:
        genes = assign[assign.cluster_id == cid].index.tolist()
        lines.append(f"  C{cid} ({len(genes)} genes): {', '.join(genes)}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Biological analysis of HC gene clustering results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/analyze_hc_clusters.py
              python scripts/analyze_hc_clusters.py --all-alphas
              python scripts/analyze_hc_clusters.py \\
                  --feature-matrix data/HC_feature_matrix_GO_CC.tsv \\
                  --assignments benchmarks/results/HC_GO_CC_005/cluster_assignments.csv
        """
        ),
    )
    p.add_argument(
        "--feature-matrix",
        type=Path,
        default=DEFAULT_FEATURE_MATRIX,
        help="Path to the binary gene×pathway TSV (default: data/HC_feature_matrix_GO_CC.tsv)",
    )
    p.add_argument(
        "--assignments",
        type=Path,
        default=Path("benchmarks/results/HC_GO_CC_005/cluster_assignments.csv"),
        help="Path to cluster_assignments.csv from a previous run",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: <assignments_dir>/bio_analysis)",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size for enrichment analysis (default: 2)",
    )
    p.add_argument(
        "--all-alphas",
        action="store_true",
        help="Run analysis for all three alpha levels (0.01, 0.05, 0.10)",
    )
    return p.parse_args()


def run_analysis(
    feature_matrix_path: Path,
    assignments_path: Path,
    output_dir: Path,
    min_cluster_size: int = 2,
) -> None:
    """Run full biological analysis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")
    print("-" * 70)

    fm, assign = load_data(feature_matrix_path, assignments_path)

    cluster_overview(assign, output_dir)
    gene_lists_per_cluster(fm, assign, output_dir, min_size=min_cluster_size)
    df_enrich = pathway_enrichment(fm, assign, output_dir, min_cluster_size=min_cluster_size)
    cluster_jaccard_heatmap(fm, assign, output_dir, min_size=min_cluster_size)
    top_pathways_chart(fm, assign, output_dir)
    biological_summary(fm, assign, df_enrich, output_dir)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    if args.all_alphas:
        for alpha, rdir in RESULT_DIRS.items():
            rdir_path = Path(rdir)
            assign_path = rdir_path / "cluster_assignments.csv"
            if not assign_path.exists():
                print(f"\nSkipping α={alpha} — {assign_path} not found")
                continue
            print(f"\n{'#' * 70}")
            print(f"# α = {alpha}")
            print(f"{'#' * 70}")
            out = rdir_path / "bio_analysis"
            run_analysis(args.feature_matrix, assign_path, out, args.min_cluster_size)
    else:
        out = args.output_dir or (args.assignments.parent / "bio_analysis")
        run_analysis(args.feature_matrix, args.assignments, out, args.min_cluster_size)


if __name__ == "__main__":
    main()
