#!/usr/bin/env python
"""Compare hamming vs rogerstanimoto distance metrics for tree inference.

Uses the same pipeline flow as benchmark_cluster_algorithm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmarks.shared.generators import generate_case_data
from benchmarks.shared.cases import SMALL_TEST_CASES
from benchmarks.shared.cases.gaussian import GAUSSIAN_CASES
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def run_kl_with_metric(
    data_df: pd.DataFrame,
    distance_metric: str,
    linkage_method: str = "average",
    significance_level: float = 0.05,
) -> dict:
    """Run KL decomposition with specified distance metric."""
    distance_condensed = pdist(data_df.values, metric=distance_metric)
    Z = linkage(distance_condensed, method=linkage_method)
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=significance_level,
    )
    return {
        "tree": tree,
        "decomposition": decomp,
        "num_clusters": decomp.get("num_clusters", 0),
        "cluster_assignments": decomp.get("cluster_assignments", {}),
    }


def labels_from_decomposition(decomp: dict, leaf_names: list) -> np.ndarray:
    """Convert cluster assignments to label array."""
    assignments = decomp.get("cluster_assignments", {})
    label_map = {name: -1 for name in leaf_names}
    
    # cluster_assignments values are dicts with 'leaves' list
    for cluster_id, info in assignments.items():
        leaves = info.get("leaves", [])
        for leaf in leaves:
            if leaf in label_map:
                label_map[leaf] = cluster_id
            
    return np.array([label_map[name] for name in leaf_names])


def flatten_cases(cases_dict: dict) -> list:
    """Flatten nested case dict to list of cases with names."""
    result = []
    for category_name, cases in cases_dict.items():
        # Handle if cases is not a list but a single dict (though gaussian usually lists)
        if isinstance(cases, dict):
            cases = [cases]
        elif not isinstance(cases, list):
            continue
            
        for i, case in enumerate(cases):
            # Ensure case is a dict
            if not isinstance(case, dict):
                continue
            case_with_name = case.copy()
            if "name" not in case_with_name:
                case_with_name["name"] = f"{category_name}_{i}"
            result.append(case_with_name)
    return result


def main():
    metrics = ["hamming", "rogerstanimoto"]
    
    results = []
    
    # Combine small test cases with gaussian cases
    gaussian_flat = flatten_cases(GAUSSIAN_CASES)
    test_cases = SMALL_TEST_CASES + gaussian_flat
    
    print(f"Running comparison on {len(test_cases)} test cases...")
    print("=" * 80)
    
    for i, tc in enumerate(test_cases, 1):
        case_name = tc.get("name", f"Case {i}")
        
        # Generate data using same method as pipeline
        data_df, y_true, X_original, meta = generate_case_data(tc)
        
        row = {
            "case_id": i,
            "case_name": case_name,
            "n_samples": len(data_df),
            "n_features": data_df.shape[1],
            "n_clusters_true": len(np.unique(y_true)),
        }
        
        for metric in metrics:
            result = run_kl_with_metric(
                data_df,
                distance_metric=metric,
                linkage_method="average",
                significance_level=config.SIBLING_ALPHA,
            )
            
            labels = labels_from_decomposition(result["decomposition"], data_df.index.tolist())
            n_clusters = result["num_clusters"]
            
            if n_clusters > 0 and len(np.unique(labels)) > 1:
                ari = adjusted_rand_score(y_true, labels)
                nmi = normalized_mutual_info_score(y_true, labels)
            else:
                ari = 0.0
                nmi = 0.0
            
            row[f"{metric}_ari"] = ari
            row[f"{metric}_nmi"] = nmi
            row[f"{metric}_clusters"] = n_clusters
        
        # Calculate difference (hamming - rogerstanimoto)
        row["ari_diff"] = row["hamming_ari"] - row["rogerstanimoto_ari"]
        
        results.append(row)
        
        print(f"{i:3d}. {case_name:30s} | "
              f"hamming: ARI={row['hamming_ari']:.3f} k={row['hamming_clusters']} | "
              f"rogerstanimoto: ARI={row['rogerstanimoto_ari']:.3f} k={row['rogerstanimoto_clusters']} | "
              f"diff={row['ari_diff']:+.3f}")
    
    # Summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nMean ARI:")
    print(f"  hamming:        {df['hamming_ari'].mean():.4f} (std={df['hamming_ari'].std():.4f})")
    print(f"  rogerstanimoto: {df['rogerstanimoto_ari'].mean():.4f} (std={df['rogerstanimoto_ari'].std():.4f})")
    
    print(f"\nMean NMI:")
    print(f"  hamming:        {df['hamming_nmi'].mean():.4f}")
    print(f"  rogerstanimoto: {df['rogerstanimoto_nmi'].mean():.4f}")
    
    # Cases where methods differ
    diff_cases = df[df['ari_diff'].abs() > 0.01]
    if len(diff_cases) > 0:
        print(f"\nCases with |ARI diff| > 0.01: {len(diff_cases)}")
        print(diff_cases[['case_name', 'hamming_ari', 'rogerstanimoto_ari', 'ari_diff']].to_string(index=False))
    else:
        print("\nNo cases with |ARI diff| > 0.01")
    
    # Win/Loss/Tie
    wins_hamming = (df['ari_diff'] > 0.01).sum()
    wins_rogers = (df['ari_diff'] < -0.01).sum()
    ties = len(df) - wins_hamming - wins_rogers
    print(f"\nWins: hamming={wins_hamming}, rogerstanimoto={wins_rogers}, ties={ties}")


if __name__ == "__main__":
    main()
