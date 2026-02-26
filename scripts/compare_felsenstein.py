"""Compare decomposition with and without Felsenstein branch-length scaling.

Runs a subset of benchmark cases under both regimes, printing ARI + K side-by-side.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

# Import test case generators
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def run_case_with_felsenstein(case, use_felsenstein: bool):
    """Run a single case, optionally disabling Felsenstein scaling."""
    try:
        binary, labels_true, _, _ = generate_case_data(case)
    except Exception:
        return None, None, None

    k_true = len(set(labels_true))

    Z = linkage(
        pdist(binary.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD
    )
    tree = PosetTree.from_linkage(Z, leaf_names=binary.index.tolist())

    if not use_felsenstein:
        # Remove all branch_length attributes → Felsenstein scaling disabled
        for u, v in list(tree.edges()):
            if "branch_length" in tree.edges[u, v]:
                del tree.edges[u, v]["branch_length"]

    results = tree.decompose(leaf_data=binary, alpha_local=0.05, sibling_alpha=0.05)

    cluster_assignments = results.get("cluster_assignments", {})
    cluster_map = {}
    for cid, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            cluster_map[leaf] = cid

    pred = [cluster_map.get(idx, -1) for idx in binary.index]
    le = LabelEncoder()
    pred_enc = le.fit_transform(pred)

    k_found = len(set(pred_enc))
    ari = adjusted_rand_score(labels_true, pred_enc)
    return ari, k_found, k_true


def main():
    cases = get_default_test_cases()
    # Pick a representative subset
    subset_names = [
        "gaussian_clear_1",
        "gaussian_clear_2",
        "gaussian_clear_3",
        "gaussian_mixed_1",
        "gaussian_mixed_2",
        "gauss_clear_small",
        "gauss_clear_medium",
        "gauss_clear_large",
        "gauss_moderate_3c",
        "gauss_moderate_5c",
        "gauss_noisy_3c",
        "gauss_noisy_many",
        "gaussian_extreme_noise_1",
        "gaussian_extreme_noise_3",
        "binary_perfect_2c",
        "binary_perfect_4c",
        "trivial_2c",
        "block_4c",
        "sparse_72x72",
    ]

    case_dict = {c["name"]: c for c in cases}
    selected = [(n, case_dict[n]) for n in subset_names if n in case_dict]

    print(
        f"{'Case':<30} {'K_true':>6} │ {'ARI_F':>7} {'K_F':>5} │ {'ARI_noF':>7} {'K_noF':>5} │ {'Δ ARI':>7}"
    )
    print("─" * 90)

    ari_with, ari_without = [], []
    for name, case in selected:
        ari_f, k_f, k_true = run_case_with_felsenstein(case, use_felsenstein=True)
        ari_nf, k_nf, _ = run_case_with_felsenstein(case, use_felsenstein=False)
        if ari_f is None:
            continue

        delta = (ari_nf or 0) - (ari_f or 0)
        ari_with.append(ari_f)
        ari_without.append(ari_nf)
        print(
            f"{name:<30} {k_true:>6} │ {ari_f:>7.3f} {k_f:>5} │ {ari_nf:>7.3f} {k_nf:>5} │ {delta:>+7.3f}"
        )

    print("─" * 90)
    print(
        f"{'Mean':<30} {'':>6} │ {np.mean(ari_with):>7.3f} {'':>5} │ {np.mean(ari_without):>7.3f} {'':>5} │ {np.mean(ari_without)-np.mean(ari_with):>+7.3f}"
    )
    wins = sum(1 for a, b in zip(ari_with, ari_without) if b > a + 0.01)
    losses = sum(1 for a, b in zip(ari_with, ari_without) if a > b + 0.01)
    ties = len(ari_with) - wins - losses
    print(f"\nNo-Felsenstein wins: {wins}, losses: {losses}, ties: {ties}")


if __name__ == "__main__":
    main()
    main()
