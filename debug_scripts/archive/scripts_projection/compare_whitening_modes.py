#!/usr/bin/env python3
"""Compare eigenvalue whitening vs Satterthwaite on key test cases.

Runs the pipeline on:
- The 3 FAILING test cases (clear, balanced 72×40, unbalanced 96×36)
- The feature_matrix.tsv real dataset
under both EIGENVALUE_WHITENING=True (whitened) and False (Satterthwaite).
"""

import sys
import os

# Bootstrap: add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.generators import generate_random_feature_matrix
from benchmarks.shared.util.decomposition import _labels_from_decomposition
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def run_pipeline(data_df, significance_level=0.05):
    """Minimal pipeline helper."""
    Z = linkage(pdist(data_df.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    decomposition = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=significance_level,
    )
    return decomposition


def test_balanced():
    """72×40 balanced 4-cluster case."""
    data_dict, true_clusters = generate_random_feature_matrix(
        n_rows=72, n_cols=40, entropy_param=0.1,
        n_clusters=4, random_seed=314, balanced_clusters=True,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
    decomp = run_pipeline(data_df)
    predicted = _labels_from_decomposition(decomp, data_df.index.tolist())
    true_labels = [true_clusters[name] for name in data_df.index]
    assigned = np.array(predicted) != -1
    if assigned.any():
        ari = adjusted_rand_score(np.array(true_labels)[assigned], np.array(predicted)[assigned])
    else:
        ari = 0.0
    return decomp["num_clusters"], ari, "balanced_72x40"


def test_unbalanced():
    """96×36 unbalanced 4-cluster case."""
    data_dict, true_clusters = generate_random_feature_matrix(
        n_rows=96, n_cols=36, entropy_param=0.25,
        n_clusters=4, random_seed=2024, balanced_clusters=False,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
    decomp = run_pipeline(data_df)
    predicted = _labels_from_decomposition(decomp, data_df.index.tolist())
    true_labels = [true_clusters[name] for name in data_df.index]
    assigned = np.array(predicted) != -1
    if assigned.any():
        ari = adjusted_rand_score(np.array(true_labels)[assigned], np.array(predicted)[assigned])
    else:
        ari = 0.0
    return decomp["num_clusters"], ari, "unbalanced_96x36"


def test_clear_case():
    """'clear' case from SMALL_TEST_CASES (std=0.4)."""
    try:
        from tests.test_cases_config import SMALL_TEST_CASES
    except ImportError:
        from benchmarks.shared.cases import SMALL_TEST_CASES
    from benchmarks.shared.pipeline import benchmark_cluster_algorithm

    case = next(c for c in SMALL_TEST_CASES if c["name"] == "clear")
    df, _ = benchmark_cluster_algorithm(
        test_cases=[case.copy()], verbose=False, plot_umap=False, methods=["kl"],
    )
    row = df[df["Method"] == "KL Divergence"].iloc[0]
    return int(row["Found"]), float(row["ARI"]), "clear_case"


def test_feature_matrix():
    """Real feature_matrix.tsv."""
    fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "feature_matrix.tsv")
    if not os.path.exists(fpath):
        return None, None, "feature_matrix (NOT FOUND)"
    data_df = pd.read_csv(fpath, sep="\t", index_col=0)
    decomp = run_pipeline(data_df)
    return decomp["num_clusters"], None, "feature_matrix.tsv"


def main():
    tests = [test_balanced, test_unbalanced, test_clear_case, test_feature_matrix]
    results = []

    for whitening in [True, False]:
        config.EIGENVALUE_WHITENING = whitening
        mode = "WHITENED" if whitening else "SATTERTHWAITE"
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")

        for test_fn in tests:
            try:
                k, ari, name = test_fn()
                ari_str = f"{ari:.3f}" if ari is not None else "N/A"
                print(f"  {name:25s}  K={k}  ARI={ari_str}")
                results.append({"mode": mode, "case": name, "K": k, "ARI": ari})
            except Exception as e:
                print(f"  {test_fn.__name__:25s}  ERROR: {e}")
                results.append({"mode": mode, "case": test_fn.__name__, "K": None, "ARI": None})

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Case':25s} {'Whitened K':>10s} {'Satt K':>8s} {'Whitened ARI':>13s} {'Satt ARI':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*13} {'-'*10}")
    cases = list(dict.fromkeys(r["case"] for r in results))
    for case in cases:
        w = next((r for r in results if r["case"] == case and r["mode"] == "WHITENED"), {})
        s = next((r for r in results if r["case"] == case and r["mode"] == "SATTERTHWAITE"), {})
        wk = str(w.get("K", "?"))
        sk = str(s.get("K", "?"))
        wa = f"{w['ARI']:.3f}" if w.get("ARI") is not None else "N/A"
        sa = f"{s['ARI']:.3f}" if s.get("ARI") is not None else "N/A"
        print(f"  {case:25s} {wk:>10s} {sk:>8s} {wa:>13s} {sa:>10s}")


if __name__ == "__main__":
    main()
