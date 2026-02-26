#!/usr/bin/env python3
"""Sweep alpha (significance level) and show how K changes.

Usage:
    python scripts/sweep_alpha.py                          # feature_matrix.tsv
    python scripts/sweep_alpha.py --input feature_matrix.tsv
    python scripts/sweep_alpha.py --synthetic              # 80-sample 4-cluster synthetic
    python scripts/sweep_alpha.py --alphas 0.001 0.01 0.05 0.1 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

DEFAULT_ALPHAS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def _load_feature_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    return data.apply(pd.to_numeric, errors="raise").astype(int)


def _make_synthetic() -> tuple[pd.DataFrame, int]:
    X, y = make_blobs(n_samples=80, n_features=30, centers=4, cluster_std=1.2, random_state=42)
    X_bin = (X > np.median(X, axis=0)).astype(int)
    df = pd.DataFrame(X_bin, index=[f"S{i}" for i in range(len(X_bin))])
    return df, 4


def _run_once(data: pd.DataFrame, alpha: float) -> dict:
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    result = tree.decompose(leaf_data=data, alpha_local=alpha, sibling_alpha=alpha)
    k = result.get("num_clusters", 0)
    sizes = sorted(
        [info["size"] for info in result.get("cluster_assignments", {}).values()],
        reverse=True,
    )
    return {"alpha": alpha, "K": k, "sizes": sizes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep alpha and show K.")
    parser.add_argument(
        "--input", type=Path, default=None, help="TSV feature matrix (default: feature_matrix.tsv)"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic 4-cluster data instead of a file."
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="Alpha values to sweep (default: 10 values from 0.001 to 0.50)",
    )
    args = parser.parse_args()

    alphas = args.alphas if args.alphas else DEFAULT_ALPHAS

    if args.synthetic:
        data, true_k = _make_synthetic()
        label = f"Synthetic (n={data.shape[0]}, d={data.shape[1]}, true K={true_k})"
    else:
        path = args.input or Path("feature_matrix.tsv")
        if not path.exists():
            print(f"File not found: {path}. Use --synthetic or --input <file>.")
            sys.exit(1)
        data = _load_feature_matrix(path)
        true_k = None
        label = f"{path.name} (n={data.shape[0]}, d={data.shape[1]})"

    print(f"Dataset: {label}")
    print(
        f"Config:  SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}, "
        f"FELSENSTEIN_SCALING={config.FELSENSTEIN_SCALING}, "
        f"SPECTRAL_METHOD={config.SPECTRAL_METHOD}"
    )
    print(f"Sweeping {len(alphas)} alpha values...\n")
    print(f"{'alpha':>8}  {'K':>4}  {'cluster sizes (top 10)'}")
    print("-" * 60)

    results = []
    for alpha in sorted(alphas):
        r = _run_once(data, alpha)
        results.append(r)
        top_sizes = r["sizes"][:10]
        sizes_str = ", ".join(str(s) for s in top_sizes)
        if len(r["sizes"]) > 10:
            sizes_str += f", ... (+{len(r['sizes']) - 10} more)"
        print(f"{alpha:>8.4f}  {r['K']:>4}  [{sizes_str}]")

    print("-" * 60)
    ks = [r["K"] for r in results]
    print(f"K range: {min(ks)} – {max(ks)}")
    if true_k is not None:
        exact = [r["alpha"] for r in results if r["K"] == true_k]
        print(f"Exact K={true_k} at alpha: {exact if exact else 'none'}")


if __name__ == "__main__":
    main()
