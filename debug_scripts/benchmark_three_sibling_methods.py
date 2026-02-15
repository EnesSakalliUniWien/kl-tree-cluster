"""Full benchmark comparing all three sibling test methods.

Runs every default test case with kl method under each sibling test:
  - wald
  - cousin_adjusted_wald
  - cousin_ftest

Outputs a CSV + summary table to stdout.
"""

import gc
import sys
import time
from pathlib import Path


import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SIBLING_METHODS = ["wald", "cousin_adjusted_wald", "cousin_ftest"]


def _compute_ari(labels_true, labels_pred):
    """Adjusted Rand Index."""
    from sklearn.metrics import adjusted_rand_score

    return adjusted_rand_score(labels_true, labels_pred)


def _compute_nmi(labels_true, labels_pred):
    """Normalized Mutual Information."""
    from sklearn.metrics import normalized_mutual_info_score

    return normalized_mutual_info_score(labels_true, labels_pred)


def _labels_from_decomposition(decomp, leaf_names):
    """Extract integer labels from decomposition result.

    cluster_assignments format: {cluster_id: {"root_node": str, "leaves": [...], "size": int}}
    """
    assignments = decomp.get("cluster_assignments", {})
    if not assignments:
        return [0] * len(leaf_names)

    label_map = {}
    for cluster_id, info in assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cluster_id

    return [label_map.get(name, -1) for name in leaf_names]


def run_single_case(case, method_name):
    """Run one case with a specific sibling method. Returns dict of results."""
    # Set sibling method
    config.SIBLING_TEST_METHOD = method_name

    # Generate data (returns 4 values: data_df, labels, x_original, metadata)
    data_df, true_labels, x_original, metadata = generate_case_data(case)
    n_clusters_true = case.get("n_clusters", len(set(true_labels)))

    # Compute distance and linkage
    case_type = case.get("type", "gaussian")
    if case_type == "sbm" and metadata.get("distance_condensed") is not None:
        distance_condensed = metadata["distance_condensed"]
    else:
        distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)

    Z = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
    )

    labels_pred = _labels_from_decomposition(decomp, data_df.index.tolist())
    k_found = decomp.get("num_clusters", 0)

    ari = _compute_ari(true_labels, labels_pred)
    nmi = _compute_nmi(true_labels, labels_pred)

    # Get calibration audit info if available
    audit = {}
    if hasattr(tree, "stats_df") and tree.stats_df is not None:
        sdf = tree.stats_df
        if hasattr(sdf, "attrs"):
            audit = sdf.attrs.get("sibling_divergence_audit", {})

    return {
        "case_name": case.get("name", "unknown"),
        "case_type": case.get("type", "unknown"),
        "sibling_method": method_name,
        "n_samples": len(data_df),
        "n_features": data_df.shape[1],
        "true_k": n_clusters_true,
        "found_k": k_found,
        "ari": ari,
        "nmi": nmi,
        "k_match": k_found == n_clusters_true,
        "calibration_method": audit.get("calibration_method", ""),
        "calibration_n": audit.get("calibration_n", ""),
        "global_c_hat": audit.get("global_c_hat", ""),
    }


def main():
    print("=" * 80)
    print("FULL BENCHMARK: Three Sibling Test Methods")
    print("=" * 80)

    test_cases = get_default_test_cases()
    n_cases = len(test_cases)
    print(f"Total test cases: {n_cases}")
    print(f"Sibling methods: {SIBLING_METHODS}")
    print(f"Total runs: {n_cases * len(SIBLING_METHODS)}")
    print()

    all_results = []
    t_start = time.time()

    for i, case in enumerate(test_cases):
        case_name = case.get("name", f"case_{i}")
        case_type = case.get("type", "unknown")
        true_k = case.get("n_clusters", "?")
        print(
            f"[{i+1}/{n_cases}] {case_name} (type={case_type}, true_k={true_k})", end="", flush=True
        )

        for method in SIBLING_METHODS:
            try:
                result = run_single_case(case, method)
                all_results.append(result)
                print(
                    f"  {method}:K={result['found_k']},ARI={result['ari']:.3f}", end="", flush=True
                )
            except Exception as e:
                print(f"  {method}:FAIL({e})", end="", flush=True)
                all_results.append(
                    {
                        "case_name": case_name,
                        "case_type": case_type,
                        "sibling_method": method,
                        "n_samples": 0,
                        "n_features": 0,
                        "true_k": true_k,
                        "found_k": -1,
                        "ari": np.nan,
                        "nmi": np.nan,
                        "k_match": False,
                        "calibration_method": "",
                        "calibration_n": "",
                        "global_c_hat": "",
                    }
                )
            gc.collect()

        print()  # newline after each case

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")

    # Build results DataFrame
    df = pd.DataFrame(all_results)

    # Save CSV
    output_dir = repo_root / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sibling_method_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Mean ARI by Sibling Method")
    print("=" * 80)
    valid = df[df["found_k"] >= 0]
    summary = (
        valid.groupby("sibling_method")
        .agg(
            mean_ari=("ari", "mean"),
            median_ari=("ari", "median"),
            mean_nmi=("nmi", "mean"),
            exact_k=("k_match", "sum"),
            n_cases=("ari", "count"),
        )
        .round(3)
    )
    print(summary.to_string())

    # Summary by case type
    print("\n" + "=" * 80)
    print("SUMMARY: Mean ARI by Case Type × Sibling Method")
    print("=" * 80)
    type_summary = (
        valid.groupby(["case_type", "sibling_method"])
        .agg(
            mean_ari=("ari", "mean"),
            exact_k=("k_match", "sum"),
            n_cases=("ari", "count"),
        )
        .round(3)
    )
    print(type_summary.to_string())

    # Pivot: cases where methods disagree on K
    print("\n" + "=" * 80)
    print("CASES WHERE METHODS DISAGREE ON K")
    print("=" * 80)
    pivot_k = valid.pivot_table(
        index="case_name", columns="sibling_method", values="found_k", aggfunc="first"
    )
    if all(m in pivot_k.columns for m in SIBLING_METHODS):
        disagree = pivot_k[
            (pivot_k["wald"] != pivot_k["cousin_adjusted_wald"])
            | (pivot_k["wald"] != pivot_k["cousin_ftest"])
        ]
        # Add true_k
        true_k_map = valid.drop_duplicates("case_name").set_index("case_name")["true_k"]
        disagree = disagree.copy()
        disagree.insert(0, "true_k", disagree.index.map(true_k_map))
        if len(disagree) > 0:
            print(disagree.to_string())
        else:
            print("All methods agree on K for all cases!")

    # Pivot: ARI comparison
    print("\n" + "=" * 80)
    print("ARI COMPARISON (Wald vs Adj-Wald vs Cousin-F)")
    print("=" * 80)
    pivot_ari = valid.pivot_table(
        index="case_name", columns="sibling_method", values="ari", aggfunc="first"
    )
    if all(m in pivot_ari.columns for m in SIBLING_METHODS):
        pivot_ari = pivot_ari.copy()
        pivot_ari.insert(0, "true_k", pivot_ari.index.map(true_k_map))
        # Sort by biggest difference
        pivot_ari["max_diff"] = pivot_ari[SIBLING_METHODS].max(axis=1) - pivot_ari[
            SIBLING_METHODS
        ].min(axis=1)
        pivot_ari = pivot_ari.sort_values("max_diff", ascending=False)
        print(pivot_ari.round(3).to_string())

    # Win/loss/tie
    print("\n" + "=" * 80)
    print("WIN / LOSS / TIE (by ARI, tolerance=0.01)")
    print("=" * 80)
    if all(m in pivot_ari.columns for m in SIBLING_METHODS):
        for m1 in SIBLING_METHODS:
            for m2 in SIBLING_METHODS:
                if m1 >= m2:
                    continue
                wins = (pivot_ari[m1] > pivot_ari[m2] + 0.01).sum()
                losses = (pivot_ari[m2] > pivot_ari[m1] + 0.01).sum()
                ties = len(pivot_ari) - wins - losses
                print(f"  {m1} vs {m2}: {wins}W / {losses}L / {ties}T")


if __name__ == "__main__":
    main()
