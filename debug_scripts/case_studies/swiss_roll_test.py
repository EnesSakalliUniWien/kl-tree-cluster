#!/usr/bin/env python
"""
Swiss Roll Dataset Clustering Test
===================================
Demonstrates KL-tree clustering on the Swiss Roll manifold.

1. Generates a Swiss Roll dataset (3D continuous → binarized)
2. Shows that naive 3-feature binarization fragments the manifold
3. Creates richer binary features (quantile, radial, angular, height)
4. Re-runs clustering — more coherent spatial clusters
5. Produces comparison figures, separated cluster views, and tree plot

Usage
-----
    python scripts/swiss_roll_test.py                   # defaults
    python scripts/swiss_roll_test.py -n 1000 --bins 8  # more samples, finer bins
    python scripts/swiss_roll_test.py -o results/swiss   # custom output dir
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_swiss_roll

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kl_clustering_analysis import config
from kl_clustering_analysis.plot.cluster_tree_visualization import plot_tree_with_clusters
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════


def create_swiss_roll_dataset(
    n_samples: int = 300,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate Swiss roll and binarize with a simple median threshold (3 features)."""
    X, y_true = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    X_binary = (X > np.median(X, axis=0)).astype(int)
    data = pd.DataFrame(
        X_binary,
        index=[f"R{j}" for j in range(len(X))],
        columns=[f"F{j}" for j in range(X.shape[1])],
    )
    return data, y_true, X


def create_rich_swiss_roll_features(
    X: np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Create richer binary features from Swiss Roll 3-D coordinates.

    Feature families:
      * Quantile thresholds per dimension
      * Distance-from-centre quantiles
      * Radial quantiles  (XZ plane)
      * Angular quantiles (arctan2 in XZ plane)
      * Height (Y) quantiles
    """
    features: list[np.ndarray] = []
    feature_names: list[str] = []

    # 1. Quantile-based per dimension
    for dim in range(X.shape[1]):
        percentiles = np.percentile(X[:, dim], np.linspace(0, 100, n_bins + 1)[1:-1])
        for i, p in enumerate(percentiles):
            features.append((X[:, dim] > p).astype(int))
            feature_names.append(f"Dim{dim}_Q{i + 1}")

    # 2. Distance from centre
    centre = X.mean(axis=0)
    dist = np.sqrt(((X - centre) ** 2).sum(axis=1))
    for q in (25, 50, 75):
        features.append((dist > np.percentile(dist, q)).astype(int))
        feature_names.append(f"Dist_Q{q}")

    # 3. Radial features (XZ plane — the "roll" direction)
    radius = np.sqrt(X[:, 0] ** 2 + X[:, 2] ** 2)
    for q in (20, 40, 60, 80):
        features.append((radius > np.percentile(radius, q)).astype(int))
        feature_names.append(f"Radius_Q{q}")

    # 4. Angular features
    angle = np.arctan2(X[:, 2], X[:, 0])
    for q in (25, 50, 75):
        features.append((angle > np.percentile(angle, q)).astype(int))
        feature_names.append(f"Angle_Q{q}")

    # 5. Height (Y)
    for q in (20, 40, 60, 80):
        features.append((X[:, 1] > np.percentile(X[:, 1], q)).astype(int))
        feature_names.append(f"Height_Q{q}")

    return pd.DataFrame(
        np.column_stack(features),
        index=[f"R{j}" for j in range(len(X))],
        columns=feature_names,
    )


def run_kl_clustering(
    data: pd.DataFrame,
    label: str = "",
) -> tuple[PosetTree, dict, np.ndarray]:
    """Build tree + decompose. Returns (tree, result_dict, cluster_id_array)."""
    print(
        f"[{label}] Building tree  (metric={config.TREE_DISTANCE_METRIC}, "
        f"linkage={config.TREE_LINKAGE_METHOD}) ..."
    )
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

    print(f"[{label}] Running KL decomposition ...")
    result = tree.decompose(leaf_data=data)

    # Extract per-sample cluster IDs
    cluster_ids = np.zeros(len(data), dtype=int)
    for cid, info in result.get("cluster_assignments", {}).items():
        for leaf in info["leaves"]:
            idx = int(leaf[1:])  # "R42" → 42
            cluster_ids[idx] = cid

    print(f"[{label}] Clusters found: {result['num_clusters']}")
    return tree, result, cluster_ids


# ═══════════════════════════════════════════════════════════════════════════
# Diagnosis: why naive binarization fragments the manifold
# ═══════════════════════════════════════════════════════════════════════════


def diagnose_binarization(
    data: pd.DataFrame,
    cluster_ids: np.ndarray,
) -> None:
    """Print diagnostic info about naive 3-feature binarization."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("DIAGNOSIS: Why clusters don't follow the manifold structure")
    print(sep)

    n_unique = len(data.drop_duplicates())
    print("\n1. BINARIZATION")
    print(f"   Features:          {data.shape[1]}")
    print(f"   Max possible:      2^{data.shape[1]} = {2 ** data.shape[1]}")
    print(f"   Unique patterns:   {n_unique}")

    print("\n2. UNIQUE PATTERNS:")
    print(data.drop_duplicates().to_string())

    print("\n3. CLUSTER → PATTERN MAPPING:")
    for cid in sorted(np.unique(cluster_ids)):
        mask = cluster_ids == cid
        pats = data.iloc[mask].drop_duplicates()
        print(f"   Cluster {cid}: {len(pats)} pattern(s)")
        for _, row in pats.iterrows():
            pat = "".join(row.astype(str).values)
            count = ((data.iloc[mask] == row).all(axis=1)).sum()
            print(f"      [{pat}]: {count} samples")

    print(
        "\n   ⇒ With only 3 binary features, points are grouped by which\n"
        "     OCTANT of 3-D space they occupy — not by their position\n"
        "     on the Swiss Roll manifold.\n"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════


def plot_comparison(
    coords: np.ndarray,
    labels: np.ndarray,
    ids_orig: np.ndarray,
    result_orig: dict,
    ids_rich: np.ndarray,
    result_rich: dict,
    save_path: str | Path,
) -> None:
    """Side-by-side: ground truth vs. naive vs. rich-feature clustering."""
    fig = plt.figure(figsize=(18, 6))

    # Ground truth (continuous position on roll)
    ax1 = fig.add_subplot(131, projection="3d")
    sc1 = ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=labels,
        cmap="viridis",
        alpha=0.7,
        s=30,
    )
    ax1.set_title("Ground Truth\n(position on roll)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Original (3 features)
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=ids_orig,
        cmap="tab20",
        alpha=0.7,
        s=30,
    )
    ax2.set_title(
        f"Original: 3 Features\n({result_orig['num_clusters']} clusters — FRAGMENTED)",
        fontsize=12,
        fontweight="bold",
        color="red",
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Rich features
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=ids_rich,
        cmap="tab20",
        alpha=0.7,
        s=30,
    )
    ax3.set_title(
        f"Improved: {result_rich['num_clusters']} Features\n"
        f"({result_rich['num_clusters']} clusters — CONNECTED)",
        fontsize=12,
        fontweight="bold",
        color="green",
    )
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    fig.suptitle("Impact of Feature Engineering on Cluster Quality", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Comparison plot  → {save_path}")


def plot_clusters_separated(
    coords: np.ndarray,
    cluster_ids: np.ndarray,
    n_clusters: int,
    save_path: str | Path,
    max_display: int | None = None,
) -> None:
    """Grid of 3-D subplots — one per cluster, highlighted against grey background."""
    unique = np.unique(cluster_ids)
    display = sorted(unique) if max_display is None else sorted(unique)[:max_display]

    n_cols = 4
    n_rows = (len(display) + 1 + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # Overview
    ax0 = fig.add_subplot(n_rows, n_cols, 1, projection="3d")
    ax0.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=cluster_ids,
        cmap="tab20",
        alpha=0.6,
        s=20,
    )
    ax0.set_title(f"All {n_clusters} Clusters", fontsize=11, fontweight="bold")
    ax0.set_xlabel("X")
    ax0.set_ylabel("Y")
    ax0.set_zlabel("Z")

    colours = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, cid in enumerate(display):
        ax = fig.add_subplot(n_rows, n_cols, i + 2, projection="3d")
        mask = cluster_ids == cid
        ax.scatter(
            coords[~mask, 0],
            coords[~mask, 1],
            coords[~mask, 2],
            c="lightgray",
            alpha=0.01,
            s=10,
        )
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            c=[colours[cid % 20]],
            alpha=0.95,
            s=60,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_title(f"Cluster {cid} ({mask.sum()})", fontsize=10, fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    fig.suptitle(
        f"Swiss Roll: {n_clusters} Clusters with Rich Features " f"(showing {len(display)})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Separated clusters → {save_path}")


def plot_tree(
    tree: PosetTree,
    result: dict,
    alpha: float,
    save_path: str | Path,
) -> None:
    """Plot the cluster tree with edge / node significance styling."""
    n_clusters = result["num_clusters"]
    n_leaves = sum(1 for n in tree.nodes() if tree.out_degree(n) == 0)
    h = max(10, n_leaves * 0.06)

    fig, ax = plt.subplots(figsize=(16, h))
    plot_tree_with_clusters(
        tree,
        result,
        results_df=tree.stats_df,
        layout="rectangular",
        title=f"KL Tree — {n_clusters} clusters (α={alpha})",
        ax=ax,
        node_size=12,
        font_size=9,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Tree plot        → {save_path}")


def print_significance_summary(tree: PosetTree) -> None:
    """Print a quick summary of the gate test statistics."""
    df = tree.stats_df
    sep = "=" * 60
    print(f"\n{sep}")
    print("SIGNIFICANCE TEST SUMMARY")
    print(sep)
    print(f"Stats DataFrame shape: {df.shape}")

    pval_cols = [c for c in df.columns if "p_value" in c.lower()]
    for col in pval_cols:
        vals = df[col].dropna()
        if vals.empty:
            continue
        print(f"\n  {col}")
        print(f"    count  = {len(vals)}")
        print(f"    median = {vals.median():.4g}")
        print(f"    <0.05  = {(vals < 0.05).sum()} / {len(vals)}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swiss Roll clustering test")
    p.add_argument(
        "-n", "--n-samples", type=int, default=2000, help="Number of samples (default: 2000)"
    )
    p.add_argument("--noise", type=float, default=0.2, help="Swiss roll noise σ (default: 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--bins", type=int, default=5, help="Quantile bins for rich features (default: 5)"
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/swiss_roll_<timestamp>)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Output directory
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results") / f"swiss_roll_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}\n")

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    print("─── Step 1: Generate Swiss Roll ───")
    data_naive, labels, coords = create_swiss_roll_dataset(
        n_samples=args.n_samples,
        noise=args.noise,
        seed=args.seed,
    )
    print(f"  Samples:  {len(data_naive)}")
    print(f"  Features: {data_naive.shape[1]}  (naive median binarization)")

    # ------------------------------------------------------------------
    # 2. Cluster with naive features
    # ------------------------------------------------------------------
    print("\n─── Step 2: KL clustering — naive 3 features ───")
    tree_naive, result_naive, ids_naive = run_kl_clustering(data_naive, label="Naive")

    # ------------------------------------------------------------------
    # 3. Diagnose fragmentation
    # ------------------------------------------------------------------
    diagnose_binarization(data_naive, ids_naive)

    # ------------------------------------------------------------------
    # 4. Create rich features & recluster
    # ------------------------------------------------------------------
    print("─── Step 3: Create rich binary features ───")
    data_rich = create_rich_swiss_roll_features(coords, n_bins=args.bins)
    n_unique_naive = len(data_naive.drop_duplicates())
    n_unique_rich = len(data_rich.drop_duplicates())
    print(f"  Naive:  {data_naive.shape[1]} features → {n_unique_naive} unique patterns")
    print(f"  Rich:   {data_rich.shape[1]} features → {n_unique_rich} unique patterns")
    print(f"  Features: {', '.join(data_rich.columns)}")

    print("\n─── Step 4: KL clustering — rich features ───")
    tree_rich, result_rich, ids_rich = run_kl_clustering(data_rich, label="Rich")

    # Significance summary
    print_significance_summary(tree_rich)

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    print("\n─── Step 5: Generating plots ───")
    plot_comparison(
        coords,
        labels,
        ids_naive,
        result_naive,
        ids_rich,
        result_rich,
        save_path=out / "comparison.png",
    )
    plot_clusters_separated(
        coords,
        ids_rich,
        result_rich["num_clusters"],
        save_path=out / "clusters_separated.png",
    )
    plot_tree(
        tree_rich,
        result_rich,
        alpha=config.SIBLING_ALPHA,
        save_path=out / "tree_clusters.png",
    )

    # ------------------------------------------------------------------
    # 6. Save CSVs
    # ------------------------------------------------------------------
    print("\n─── Step 6: Saving results ───")

    # Cluster assignments
    rows = []
    for cid, info in result_rich.get("cluster_assignments", {}).items():
        for leaf in info["leaves"]:
            rows.append({"sample_id": leaf, "cluster_id": cid, "cluster_size": info["size"]})
    report = pd.DataFrame(rows).set_index("sample_id")
    report.to_csv(out / "cluster_assignments.csv")
    print(f"  ✓ Assignments      → {out / 'cluster_assignments.csv'}")

    # Data + cluster IDs
    annotated = data_rich.copy()
    annotated.insert(0, "cluster_id", ids_rich)
    annotated = annotated.sort_values("cluster_id")
    annotated.to_csv(out / "data_with_clusters.tsv", sep="\t")
    print(f"  ✓ Data + clusters  → {out / 'data_with_clusters.tsv'}")

    # Cluster sizes
    sizes = report.groupby("cluster_id")["cluster_size"].first().reset_index()
    sizes.to_csv(out / "cluster_sizes.csv", index=False)
    print(f"  ✓ Cluster sizes    → {out / 'cluster_sizes.csv'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    sep = "=" * 60
    print(f"\n{sep}")
    print("CONCLUSION")
    print(sep)
    print(
        f"""
  NAIVE  ({data_naive.shape[1]} features) → {result_naive['num_clusters']} clusters
    Only captures which octant of space a point is in.
    Clusters are fragmented and don't respect the manifold.

  RICH   ({data_rich.shape[1]} features)  → {result_rich['num_clusters']} clusters
    Captures radius, angle, height, and distance information.
    Clusters follow the natural structure of the Swiss Roll.

  ⇒ Quality of clustering depends critically on feature engineering.
    Binary features must encode meaningful information about the
    underlying data structure.
"""
    )


if __name__ == "__main__":
    main()
