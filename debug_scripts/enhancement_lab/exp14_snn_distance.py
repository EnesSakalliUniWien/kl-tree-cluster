"""Exp 14 — SNN (Shared Nearest Neighbor) Distance for Tree Construction.

Hypothesis
----------
The root cause of power loss on overlapping cases is that raw Hamming
distances fail to separate clusters, producing trees where true boundaries
sit at tiny n=2–5 leaf pairs.  SNN distance amplifies local neighborhood
structure:  SNN(i,j) = |NN_k(i) ∩ NN_k(j)| / k, which widens the gap
between intra- and inter-cluster distances.

Experiment
----------
For each failure case (+ regression guards):
  1. Generate binary data
  2. Build k-NN graph (Hamming), compute SNN similarity & distance
  3. Build linkage tree from SNN distance
  4. Populate distributions and decompose with standard gates
  5. Compare ARI, K, gate statistics vs. baseline (raw Hamming)
  6. Sweep k ∈ {5, 10, 15, 20, 30} to find sensitivity
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lab_helpers import (
    FAILURE_CASES,
    REGRESSION_GUARD_CASES,
    get_case,
)
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.shared.generators import generate_case_data  # noqa: E402
from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.tree.poset_tree import PosetTree  # noqa: E402


def compute_snn_distance(
    X: np.ndarray,
    *,
    k: int = 15,
    metric: str = "hamming",
) -> np.ndarray:
    """Return an SNN distance matrix in ``[0, 1]`` for a binary feature matrix.

    Parameters
    ----------
    X : array (n, d)
        Binary feature matrix.
    k : int
        Number of nearest neighbors.
    metric : str
        Distance metric for k-NN.

    Returns
    -------
    snn_distance : array (n, n)
        Symmetric distance matrix in [0, 1].
    """
    n = X.shape[0]
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric, algorithm="brute")
    nn.fit(X)
    # Binary connectivity matrix (sparse, n×n)
    knn_graph = nn.kneighbors_graph(mode="connectivity")
    # Shared neighbor count: entry (i,j) = |NN(i) ∩ NN(j)|
    snn_counts = (knn_graph @ knn_graph.T).toarray().astype(np.float64)
    snn_similarity = snn_counts / k_eff
    # Clip to [0,1] (numerical safety)
    np.clip(snn_similarity, 0.0, 1.0, out=snn_similarity)
    snn_distance = 1.0 - snn_similarity
    # Ensure exact zeros on diagonal
    np.fill_diagonal(snn_distance, 0.0)
    # Symmetrize (should already be symmetric, but be safe)
    snn_distance = (snn_distance + snn_distance.T) / 2.0
    return snn_distance


# ── Run a single case ────────────────────────────────────────────────


def run_case(
    case_name: str,
    *,
    k_neighbors: int = 15,
    linkage_method: str = "average",
    use_snn: bool = True,
) -> dict:
    """Run one case with either SNN or baseline distance.

    Returns dict with case, true_k, found_k, ari, method, k, and
    tree structure diagnostics.
    """
    tc = get_case(case_name)
    data_df, y_true, _, _ = generate_case_data(tc)
    true_k = tc.get("n_clusters", 1)
    n_samples = len(data_df)

    if use_snn:
        snn_dist = compute_snn_distance(data_df.values, k=k_neighbors)
        dist_condensed = squareform(snn_dist, checks=False)
        method_label = f"snn_k{k_neighbors}"
    else:
        dist_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
        method_label = "baseline_hamming"

    Z = linkage(dist_condensed, method=linkage_method)
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(data_df)

    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
    )

    found_k = decomp["num_clusters"]

    # Compute ARI
    y_pred = np.full(n_samples, -1, dtype=int)
    for cid, cinfo in decomp["cluster_assignments"].items():
        for leaf in cinfo["leaves"]:
            idx = data_df.index.get_loc(leaf)
            y_pred[idx] = cid
    ari = adjusted_rand_score(y_true, y_pred) if y_true is not None else float("nan")

    # Tree structure diagnostics: how many true-split nodes and at what sizes
    true_split_sizes = _analyze_tree_structure(tree, data_df, y_true)

    return {
        "case": case_name,
        "method": method_label,
        "true_k": true_k,
        "found_k": found_k,
        "ari": round(ari, 4),
        "n_samples": n_samples,
        "max_true_split_n": max(true_split_sizes) if true_split_sizes else 0,
        "median_true_split_n": int(np.median(true_split_sizes)) if true_split_sizes else 0,
        "n_true_splits": len(true_split_sizes),
    }


def _analyze_tree_structure(
    tree: PosetTree,
    data_df: pd.DataFrame,
    y_true: np.ndarray | None,
) -> list[int]:
    """Return list of n_parent for each true-split internal node."""
    if y_true is None:
        return []

    leaf_label_map = {}
    for i, sample_id in enumerate(data_df.index):
        leaf_label_map[sample_id] = int(y_true[i])

    desc_sets = tree.compute_descendant_sets(use_labels=True)
    sizes = []
    for nd in tree.nodes():
        children = list(tree.successors(nd))
        if len(children) != 2:
            continue
        left, right = children
        left_leaves = desc_sets.get(left, {left})
        right_leaves = desc_sets.get(right, {right})
        left_labels = {leaf_label_map.get(lf) for lf in left_leaves if lf in leaf_label_map}
        right_labels = {leaf_label_map.get(lf) for lf in right_leaves if lf in leaf_label_map}
        left_labels.discard(None)
        right_labels.discard(None)
        if left_labels and right_labels and left_labels.isdisjoint(right_labels):
            sizes.append(len(left_leaves) + len(right_leaves))
    return sizes


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    ALL_CASES = FAILURE_CASES + REGRESSION_GUARD_CASES
    K_VALUES = [5, 10, 15, 20, 30]

    print("=" * 80)
    print("Exp 14: SNN Distance for Tree Construction")
    print("=" * 80)

    # ── Part 1: Baseline vs SNN (k=15) ──
    print("\n── Part 1: Baseline (Hamming) vs SNN (k=15) ──\n")
    rows = []
    for case_name in ALL_CASES:
        print(f"  {case_name}...", end=" ", flush=True)
        baseline = run_case(case_name, use_snn=False)
        snn = run_case(case_name, k_neighbors=15, use_snn=True)
        rows.append(baseline)
        rows.append(snn)
        delta_ari = snn["ari"] - baseline["ari"]
        print(
            f"baseline K={baseline['found_k']} ARI={baseline['ari']:.3f}  |  "
            f"SNN K={snn['found_k']} ARI={snn['ari']:.3f}  (Δ={delta_ari:+.3f})"
        )

    df1 = pd.DataFrame(rows)
    print("\nSummary Table (Part 1):")
    pivot = df1.pivot(index="case", columns="method", values=["found_k", "ari"])
    print(pivot.to_string())

    # ── Part 2: k sweep on failure cases ──
    print("\n\n── Part 2: k-neighbor sweep on failure cases ──\n")
    sweep_rows = []
    for case_name in FAILURE_CASES:
        print(f"  {case_name}:")
        baseline = run_case(case_name, use_snn=False)
        sweep_rows.append(baseline)
        for k in K_VALUES:
            result = run_case(case_name, k_neighbors=k, use_snn=True)
            sweep_rows.append(result)
            print(
                f"    k={k:2d}  K={result['found_k']:3d}  ARI={result['ari']:.3f}  "
                f"max_split_n={result['max_true_split_n']:4d}  "
                f"med_split_n={result['median_true_split_n']:3d}  "
                f"n_true_splits={result['n_true_splits']}"
            )

    df2 = pd.DataFrame(sweep_rows)

    # ── Part 3: Summary statistics ──
    print("\n\n── Part 3: Grand Summary ──\n")
    failure_baseline = df1[
        (df1["method"] == "baseline_hamming") & (df1["case"].isin(FAILURE_CASES))
    ]
    failure_snn = df1[
        (df1["method"] == "snn_k15") & (df1["case"].isin(FAILURE_CASES))
    ]
    guard_baseline = df1[
        (df1["method"] == "baseline_hamming") & (df1["case"].isin(REGRESSION_GUARD_CASES))
    ]
    guard_snn = df1[
        (df1["method"] == "snn_k15") & (df1["case"].isin(REGRESSION_GUARD_CASES))
    ]

    print("FAILURE CASES:")
    print(f"  Baseline — mean ARI: {failure_baseline['ari'].mean():.3f}, "
          f"median K: {failure_baseline['found_k'].median():.0f}")
    print(f"  SNN k=15 — mean ARI: {failure_snn['ari'].mean():.3f}, "
          f"median K: {failure_snn['found_k'].median():.0f}")

    print("\nREGRESSION GUARD CASES:")
    print(f"  Baseline — mean ARI: {guard_baseline['ari'].mean():.3f}, "
          f"exact K: {(guard_baseline['found_k'] == guard_baseline['true_k']).sum()}"
          f"/{len(guard_baseline)}")
    print(f"  SNN k=15 — mean ARI: {guard_snn['ari'].mean():.3f}, "
          f"exact K: {(guard_snn['found_k'] == guard_snn['true_k']).sum()}"
          f"/{len(guard_snn)}")

    # ── Part 4: Tree structure comparison ──
    print("\n\n── Part 4: Tree Structure Quality ──\n")
    print(f"{'Case':<35} {'Method':<20} {'max_split_n':>12} {'med_split_n':>12} {'n_splits':>9}")
    print("-" * 90)
    for _, row in df1[df1["case"].isin(FAILURE_CASES)].iterrows():
        print(
            f"{row['case']:<35} {row['method']:<20} "
            f"{row['max_true_split_n']:>12} {row['median_true_split_n']:>12} "
            f"{row['n_true_splits']:>9}"
        )


if __name__ == "__main__":
    main()
