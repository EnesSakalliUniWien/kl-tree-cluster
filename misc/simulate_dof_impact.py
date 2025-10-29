#!/usr/bin/env python3
"""
Simulation to compare the impact of different degrees-of-freedom (dof)
calculation modes on cluster recovery.

Compares:
1.  dof = total_number_of_features (current default)
2.  dof = effective_number_of_features (heuristic based on |p_child - p_parent| > eps)

Metrics:
- Number of clusters found
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

Run:
  python misc/simulate_dof_impact.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from statsmodels.stats.multitest import multipletests

# Add root to sys.path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tree.poset_tree import PosetTree
from hierarchy_analysis import calculate_hierarchy_kl_divergence
from hierarchy_analysis.cluster_decomposition import ClusterDecomposer
from hierarchy_analysis.decomposition_utils import generate_decomposition_report

# --- Configuration ---
TEST_CASES = [
    # Clear cases
    {"n_samples": 50, "n_features": 20, "n_clusters": 3, "cluster_std": 0.1, "seed": 100},
    {"n_samples": 70, "n_features": 30, "n_clusters": 4, "cluster_std": 0.15, "seed": 101},
    # Moderate cases
    {"n_samples": 60, "n_features": 25, "n_clusters": 3, "cluster_std": 0.2, "seed": 102},
    {"n_samples": 80, "n_features": 35, "n_clusters": 5, "cluster_std": 0.25, "seed": 103},
    # Noisy cases
    {"n_samples": 50, "n_features": 20, "n_clusters": 2, "cluster_std": 0.3, "seed": 104},
    {"n_samples": 100, "n_features": 40, "n_clusters": 4, "cluster_std": 0.35, "seed": 105},
    # Very noisy cases (where differences might be more pronounced)
    {"n_samples": 100, "n_features": 50, "n_clusters": 5, "cluster_std": 0.4, "seed": 106},
    {"n_samples": 120, "n_features": 60, "n_clusters": 6, "cluster_std": 0.45, "seed": 107},
]
ALPHA = 0.05
EPS_EFFECTIVE_DF = 1e-5  # Threshold for |p_child - p_parent| for effective DOF


# --- Helper Functions (adapted from statistical_tests.py and debug_local_gate.py) ---

def _calculate_chi_square_p_value(
    kl_divergence: float, number_of_leaves: int, dof: int
) -> float:
    """Calculates p-value for a single chi-square test."""
    if dof <= 0:
        if kl_divergence > 1e-9: # If there's some KL, ensure at least 1 DOF
            dof = 1
        else:
            return 1.0 # No KL, no DOF, not significant
    chi2_statistic = 2.0 * float(number_of_leaves) * float(kl_divergence)
    return float(chi2.sf(chi2_statistic, df=dof))


def apply_benjamini_hochberg_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Applies BH/FDR correction and returns boolean rejection array."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=bool)
    reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return reject.astype(bool)


def _binary_threshold(arr: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Return uint8(0/1) vector for thresholded probabilities."""
    a = np.asarray(arr, dtype=float)
    return (a >= float(thr)).astype(np.uint8, copy=False)


def get_local_significance(
    tree: PosetTree,
    stats_df: pd.DataFrame,
    total_n_features: int,
    alpha: float,
    dof_mode: str, # "total" or "effective"
    eps_effective_df: float = EPS_EFFECTIVE_DF,
) -> pd.Series:
    """
    Determines local significance (child vs parent) for each child node.
    Returns a Series of boolean (True if significant, False otherwise).
    """
    edges = list(tree.edges())
    if not edges:
        return pd.Series(False, index=[])

    children_nodes = [v for u, v in edges]
    parent_nodes = [u for u, v in edges]

    kl_local = (
        stats_df.get("kl_divergence_local", pd.Series(index=stats_df.index, dtype=float))
        .reindex(children_nodes)
        .to_numpy()
    )
    leaf_cnt = (
        stats_df.get("leaf_count", pd.Series(index=stats_df.index, dtype=float))
        .reindex(children_nodes)
        .to_numpy()
    )

    p_values = np.full(len(edges), np.nan, dtype=float)
    dofs_for_edges = np.zeros(len(edges), dtype=int)

    if dof_mode == "total":
        dofs_for_edges = np.full(len(edges), total_n_features, dtype=int)
    elif dof_mode == "effective":
        dist_dict = stats_df["distribution"].to_dict()
        for i, (parent, child) in enumerate(edges):
            pc = np.asarray(dist_dict.get(child), float)
            pp = np.asarray(dist_dict.get(parent), float)
            if pc is None or pp is None or pc.size != pp.size:
                dofs_for_edges[i] = 0
            else:
                dofs_for_edges[i] = int(np.sum(np.abs(pc - pp) > eps_effective_df))
    else:
        raise ValueError(f"Unknown dof_mode: {dof_mode}")

    # Calculate p-values for valid edges
    valid_mask = np.isfinite(kl_local) & (leaf_cnt > 0) & (dofs_for_edges > 0)
    
    # Debug prints
    print(f"      Debug - kl_local (first 5): {kl_local[:5]}")
    print(f"      Debug - leaf_cnt (first 5): {leaf_cnt[:5]}")
    print(f"      Debug - dofs_for_edges (first 5): {dofs_for_edges[:5]}")
    print(f"      Debug - valid_mask (sum): {np.sum(valid_mask)}")

    for i in np.flatnonzero(valid_mask):
        p_values[i] = _calculate_chi_square_p_value(
            kl_local[i], leaf_cnt[i], dofs_for_edges[i]
        )

    # Debug prints after p-value calculation
    print(f"      Debug - p_values (first 5 valid): {p_values[np.isfinite(p_values)][:5]}")
    print(f"      Debug - p_values (median valid): {np.median(p_values[np.isfinite(p_values)]) if np.isfinite(p_values).any() else 'N/A'}")

    # Apply BH correction to valid p-values
    valid_p_values = p_values[np.isfinite(p_values)]
    if valid_p_values.size > 0:
        rejections = apply_benjamini_hochberg_correction(valid_p_values, alpha=alpha)
        
        # Debug prints after BH correction
        print(f"      Debug - BH rejections (sum): {np.sum(rejections)}")

        # Map rejections back to original edge indices
        full_rejections = np.full(len(edges), False, dtype=bool)
        valid_indices = np.flatnonzero(np.isfinite(p_values))
        full_rejections[valid_indices] = rejections
    else:
        full_rejections = np.full(len(edges), False, dtype=bool)

    # Create a Series indexed by child nodes for easy lookup
    return pd.Series(full_rejections, index=children_nodes)


# --- Main Simulation Logic ---

def run_simulation():
    results_data = []

    print("Starting simulation for DOF impact on cluster recovery...")
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Significance Alpha: {ALPHA}")
    print(f"Effective DOF Epsilon: {EPS_EFFECTIVE_DF}\n")

    for i, tc in enumerate(TEST_CASES):
        print(f"--- Running Test Case {i+1}/{len(TEST_CASES)} ---")
        print(f"  Params: {tc}")

        # Generate data
        X, y_true = make_blobs(
            n_samples=tc["n_samples"],
            n_features=tc["n_features"],
            centers=tc["n_clusters"],
            cluster_std=tc["cluster_std"],
            random_state=tc["seed"],
        )
        X_bin = (X > np.median(X, axis=0)).astype(int)
        data_df = pd.DataFrame(
            X_bin,
            index=[f"S{j}" for j in range(tc["n_samples"])],
            columns=[f"F{j}" for j in range(tc["n_features"])],
        )

        # Build tree and calculate statistics
        Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
        stats_df = calculate_hierarchy_kl_divergence(tree, data_df)

        # --- Run with dof_mode = "total" ---
        print("  Processing with dof_mode = 'total'...")
        local_sig_total = get_local_significance(
            tree, stats_df, tc["n_features"], ALPHA, dof_mode="total"
        )
        
        # Annotate stats_df with the local significance for the decomposer
        stats_df_total = stats_df.copy()
        stats_df_total["Local_BH_Significant"] = False
        stats_df_total.loc[local_sig_total.index, "Local_BH_Significant"] = local_sig_total

        decomposer_total = ClusterDecomposer(
            tree=tree,
            results_df=stats_df_total, # Pass the annotated stats_df
            significance_column="Are_Features_Dependent", # This column is not used for local gate
            parent_gate="off", # Ensure parent gate is off for this comparison
        )
        decomp_total = decomposer_total.decompose_tree()
        report_total = generate_decomposition_report(decomp_total)

        ari_total, nmi_total, found_clusters_total = 0, 0, 0
        if decomp_total["num_clusters"] > 0 and not report_total.empty:
            true_label_map = {name: label for name, label in zip(data_df.index, y_true)}
            report_total["true_cluster"] = report_total.index.map(true_label_map)
            ari_total = adjusted_rand_score(report_total["true_cluster"], report_total["cluster_id"])
            nmi_total = normalized_mutual_info_score(report_total["true_cluster"], report_total["cluster_id"])
            found_clusters_total = decomp_total["num_clusters"]
        
        print(f"    Total - Found Clusters: {found_clusters_total}, ARI: {ari_total:.3f}, NMI: {nmi_total:.3f}")


        # --- Run with dof_mode = "effective" ---
        print("  Processing with dof_mode = 'effective'...")
        local_sig_effective = get_local_significance(
            tree, stats_df, tc["n_features"], ALPHA, dof_mode="effective", eps_effective_df=EPS_EFFECTIVE_DF
        )

        # Annotate stats_df with the local significance for the decomposer
        stats_df_effective = stats_df.copy()
        stats_df_effective["Local_BH_Significant"] = False
        stats_df_effective.loc[local_sig_effective.index, "Local_BH_Significant"] = local_sig_effective

        decomposer_effective = ClusterDecomposer(
            tree=tree,
            results_df=stats_df_effective, # Pass the annotated stats_df
            significance_column="Are_Features_Dependent", # This column is not used for local gate
            parent_gate="off", # Ensure parent gate is off for this comparison
        )
        decomp_effective = decomposer_effective.decompose_tree()
        report_effective = generate_decomposition_report(decomp_effective)

        ari_effective, nmi_effective, found_clusters_effective = 0, 0, 0
        if decomp_effective["num_clusters"] > 0 and not report_effective.empty:
            true_label_map = {name: label for name, label in zip(data_df.index, y_true)}
            report_effective["true_cluster"] = report_effective.index.map(true_label_map)
            ari_effective = adjusted_rand_score(report_effective["true_cluster"], report_effective["cluster_id"])
            nmi_effective = normalized_mutual_info_score(report_effective["true_cluster"], report_effective["cluster_id"])
            found_clusters_effective = decomp_effective["num_clusters"]

        print(f"    Effective - Found Clusters: {found_clusters_effective}, ARI: {ari_effective:.3f}, NMI: {nmi_effective:.3f}")

        results_data.append({
            "Test_Case": i + 1,
            "True_Clusters": tc["n_clusters"],
            "N_Samples": tc["n_samples"],
            "N_Features": tc["n_features"],
            "Cluster_Std": tc["cluster_std"],
            "Found_Clusters_Total": found_clusters_total,
            "ARI_Total": ari_total,
            "NMI_Total": nmi_total,
            "Found_Clusters_Effective": found_clusters_effective,
            "ARI_Effective": ari_effective,
            "NMI_Effective": nmi_effective,
        })

    results_df = pd.DataFrame(results_data)
    print("\n--- Simulation Summary ---")
    print(results_df.to_string(index=False))

    print("\n--- Average Performance ---")
    avg_perf = results_df[[
        "True_Clusters",
        "Found_Clusters_Total", "ARI_Total", "NMI_Total",
        "Found_Clusters_Effective", "ARI_Effective", "NMI_Effective",
    ]].mean().to_frame(name="Average")
    print(avg_perf.to_string())

if __name__ == "__main__":
    run_simulation()
