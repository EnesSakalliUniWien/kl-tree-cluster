#!/usr/bin/env python3
"""
Diagnose binarization degeneracy on the hardest validation case.

What it does:
- Generates the last test case from tests/test_cluster_validation.py
  (100 samples, 2000 features, 30 clusters, std=2, seed=44)
- Builds a PosetTree and computes node distributions via KL pipeline
- For each parent with exactly two children, computes:
  - Triad Otsu threshold on concatenated (parent, child1, child2) distributions
  - Fixed threshold at 0.5
  - Whether any of the three binarized vectors is degenerate (all 0 or all 1)
- Prints summary counts and fractions for both methods, plus basic stats

Run:
  python misc/diagnose_binarization.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

from tree.poset_tree import PosetTree
from hierarchy_analysis import calculate_hierarchy_kl_divergence
from misc.thresholding import compute_otsu_threshold
from hierarchy_analysis.cmi import _perm_test_cmi_binary


def _binarize(arr: np.ndarray, thr: float) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return (a >= float(thr)).astype(np.uint8)


def build_last_case():
    n_samples = 100
    n_features = 2000
    n_clusters = 30
    std = 2.0
    seed = 44
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=std,
        random_state=seed,
    )
    X_bin = (X > np.median(X, axis=0)).astype(int)
    df = pd.DataFrame(X_bin, index=[f"S{i}" for i in range(n_samples)])
    Z = linkage(pdist(df.values, metric="hamming"), method="complete")
    tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())
    stats = calculate_hierarchy_kl_divergence(tree, df)
    return tree, stats


def diagnose(tree, stats_df, *, permutations: int = 60, random_state: int = 0):
    # Pull distributions for quick access
    dist = stats_df.get("distribution")
    if dist is None:
        raise RuntimeError("statistics dataframe missing 'distribution' column")
    dist_map = dist.to_dict()

    parents = []
    otsu_thr_list = []
    deg_otsu = 0
    deg_fixed = 0
    # CMI uninformative tallies (p close to 1)
    uninf_otsu = 0
    uninf_fixed = 0
    # Collect a small sample of triad diagnostics
    sample_rows = []
    total = 0

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        p = dist_map.get(parent)
        c1 = dist_map.get(children[0])
        c2 = dist_map.get(children[1])
        if p is None or c1 is None or c2 is None:
            continue
        if not (len(p) and len(c1) and len(c2)):
            continue
        if not (np.asarray(p).size == np.asarray(c1).size == np.asarray(c2).size):
            continue

        total += 1
        vals = np.concatenate([np.asarray(p, float), np.asarray(c1, float), np.asarray(c2, float)])
        thr_otsu = compute_otsu_threshold(vals)
        otsu_thr_list.append(thr_otsu)

        # Otsu binarization
        z_o = _binarize(p, thr_otsu)
        x_o = _binarize(c1, thr_otsu)
        y_o = _binarize(c2, thr_otsu)
        deg_o = (z_o.sum() in (0, z_o.size)) or (x_o.sum() in (0, x_o.size)) or (y_o.sum() in (0, y_o.size))
        if deg_o:
            deg_otsu += 1

        # Fixed 0.5 binarization
        z_f = _binarize(p, 0.5)
        x_f = _binarize(c1, 0.5)
        y_f = _binarize(c2, 0.5)
        deg_f = (z_f.sum() in (0, z_f.size)) or (x_f.sum() in (0, x_f.size)) or (y_f.sum() in (0, y_f.size))
        if deg_f:
            deg_fixed += 1

        # CMI permutation tests (moderate permutations for speed)
        cmi_o, p_o = _perm_test_cmi_binary(x_o, y_o, z_o, permutations=permutations, random_state=random_state)
        cmi_f, p_f = _perm_test_cmi_binary(x_f, y_f, z_f, permutations=permutations, random_state=random_state)
        if p_o >= 0.95:
            uninf_otsu += 1
        if p_f >= 0.95:
            uninf_fixed += 1

        parents.append(parent)
        if len(sample_rows) < 8:
            sample_rows.append({
                "parent": parent,
                "thr_otsu": float(thr_otsu),
                "deg_otsu": bool(deg_o),
                "deg_fixed": bool(deg_f),
                "cmi_otsu": float(cmi_o),
                "p_otsu": float(p_o),
                "cmi_fixed": float(cmi_f),
                "p_fixed": float(p_f),
            })

    return {
        "num_parents": total,
        "deg_otsu": deg_otsu,
        "deg_fixed": deg_fixed,
        "uninf_otsu": uninf_otsu,
        "uninf_fixed": uninf_fixed,
        "otsu_thresholds": np.array(otsu_thr_list, dtype=float),
        "sample": sample_rows,
    }


def main():
    tree, stats = build_last_case()
    res = diagnose(tree, stats)
    n = res["num_parents"]
    d_o = res["deg_otsu"]
    d_f = res["deg_fixed"]
    u_o = res["uninf_otsu"]
    u_f = res["uninf_fixed"]
    otsu = res["otsu_thresholds"]

    print("Triad count:", n)
    print("Degenerate (Otsu):", d_o, f"({(d_o / n * 100) if n else 0:.1f}% )")
    print("Degenerate (0.5): ", d_f, f"({(d_f / n * 100) if n else 0:.1f}% )")
    print("Uninformative CMI p>=0.95 (Otsu):", u_o, f"({(u_o / n * 100) if n else 0:.1f}% )")
    print("Uninformative CMI p>=0.95 (0.5): ", u_f, f"({(u_f / n * 100) if n else 0:.1f}% )")
    if otsu.size:
        print(
            "Otsu thresholds — mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}".format(
                float(otsu.mean()), float(otsu.std()), float(otsu.min()), float(otsu.max())
            )
        )
    # Show a small sample for inspection
    if res.get("sample"):
        print("\nSample triads (parent, thr_otsu, deg_otsu, deg_fixed, cmi/p):")
        for row in res["sample"]:
            print(
                f"  {row['parent']}: thr={row['thr_otsu']:.3f}, deg(Otsu)={row['deg_otsu']}, deg(0.5)={row['deg_fixed']}, "
                f"CMI_Otsu={row['cmi_otsu']:.4f}, p_Otsu={row['p_otsu']:.3f}, CMI_Fixed={row['cmi_fixed']:.4f}, p_Fixed={row['p_fixed']:.3f}"
            )


if __name__ == "__main__":
    main()
