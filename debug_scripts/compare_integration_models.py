#!/usr/bin/env python3
"""
Mathematical Models for Branch Length Integration
==================================================

This script explores different mathematical models for integrating
tree structure (branch lengths) into our KL-divergence clustering method.

Based on literature and our empirical findings, we have several options:

MODEL 1: Height-Based Prior (Bayesian)
--------------------------------------
P(split | height) ∝ sigmoid(α * (height - threshold))

Nodes at higher heights have higher prior probability of being true splits.

MODEL 2: Expected Divergence Calibration (Jukes-Cantor-style)
-------------------------------------------------------------
Under evolutionary models, expected divergence increases with branch length:
E[D | b] = D_max * (1 - exp(-λ * b))

We can compare observed divergence to expected divergence.

MODEL 3: Inconsistency Coefficient (Classical)
----------------------------------------------
I(k) = (h_k - μ_local) / σ_local

Uses local neighborhood to determine if a height is unusually high.

MODEL 4: Combined Score (Ensemble)
----------------------------------
Score = w1 * normalized_height + w2 * normalized_KL + w3 * inconsistency

Let's test which model works best empirically.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, inconsistent as scipy_inconsistent
from scipy.spatial.distance import pdist
from scipy import stats
from scipy.special import expit  # sigmoid
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.benchmarking.generators.generate_phylogenetic import (
    generate_phylogenetic_data,
)


# =============================================================================
# MODEL IMPLEMENTATIONS
# =============================================================================


def model_1_height_prior(stats_df, threshold_percentile=50, steepness=10):
    """
    Model 1: Height-based Prior

    P(split) = sigmoid(steepness * (normalized_height - threshold))

    Higher heights → higher probability of split.
    """
    internal = stats_df[~stats_df["is_leaf"]].copy()

    # Normalize height to [0, 1]
    h_min, h_max = internal["height"].min(), internal["height"].max()
    internal["norm_height"] = (internal["height"] - h_min) / (h_max - h_min + 1e-10)

    # Threshold based on percentile
    threshold = np.percentile(internal["norm_height"], threshold_percentile)

    # Sigmoid probability
    internal["height_prior"] = expit(steepness * (internal["norm_height"] - threshold))

    return internal["height_prior"]


def model_2_calibrated_divergence(stats_df, d_max=100, lambda_param=5):
    """
    Model 2: Expected Divergence Calibration

    Expected: E[D|h] = D_max * (1 - exp(-λ * h))
    Score = observed_D / expected_D  (or residual)

    If observed >> expected, likely true split.
    """
    internal = stats_df[~stats_df["is_leaf"]].copy()
    internal = internal.dropna(subset=["height", "kl_divergence_local"])

    # Expected divergence under model
    internal["expected_kl"] = d_max * (1 - np.exp(-lambda_param * internal["height"]))

    # Ratio of observed to expected
    internal["kl_ratio"] = internal["kl_divergence_local"] / (
        internal["expected_kl"] + 1e-10
    )

    # Residual (observed - expected)
    internal["kl_residual"] = internal["kl_divergence_local"] - internal["expected_kl"]

    return internal[["expected_kl", "kl_ratio", "kl_residual"]]


def model_3_inconsistency(linkage_matrix, stats_df, depth=2):
    """
    Model 3: Inconsistency Coefficient (classical scipy approach)

    I(k) = (h_k - μ_local) / σ_local

    High inconsistency → unusual height jump → likely cluster boundary.
    """
    R = scipy_inconsistent(linkage_matrix, d=depth)
    n_leaves = linkage_matrix.shape[0] + 1

    inconsistency_map = {}
    for merge_idx in range(len(linkage_matrix)):
        node_id = f"N{n_leaves + merge_idx}"
        inconsistency_map[node_id] = R[merge_idx, 3]

    return pd.Series(inconsistency_map, name="inconsistency")


def model_4_combined_score(stats_df, weights=(0.4, 0.3, 0.3)):
    """
    Model 4: Combined Score

    Score = w1 * norm_height + w2 * norm_kl + w3 * norm_inconsistency

    Ensemble of multiple signals.
    """
    internal = stats_df[~stats_df["is_leaf"]].copy()

    w_h, w_kl, w_inc = weights

    # Normalize each metric to [0, 1]
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)

    internal["norm_height"] = normalize(internal["height"])

    if "inconsistency" in internal.columns:
        internal["norm_inconsistency"] = normalize(internal["inconsistency"].fillna(0))
    else:
        internal["norm_inconsistency"] = 0

    # For KL, we found it's NEGATIVELY correlated, so we might invert or use differently
    # Actually, let's use height and inconsistency which are positively correlated

    internal["combined_score"] = (
        w_h * internal["norm_height"] + (1 - w_h) * internal["norm_inconsistency"]
    )

    return internal["combined_score"]


def model_5_path_length_ratio(stats_df, tree):
    """
    Model 5: Path Length Ratio

    Nodes with longer path lengths to leaves (relative to their height)
    are more likely to be true cluster boundaries.

    Score = path_to_leaves / height
    """
    internal = stats_df[~stats_df["is_leaf"]].copy()

    def get_total_path_to_leaves(tree, node_id, accumulated=0):
        children = list(tree.successors(node_id))
        if len(children) == 0:
            return accumulated

        total = 0
        for child_id in children:
            child_height = (
                0
                if tree.nodes[child_id].get("is_leaf")
                else tree.nodes[child_id].get("height", 0)
            )
            node_height = tree.nodes[node_id].get("height", 0)
            edge_len = node_height - child_height
            total += get_total_path_to_leaves(tree, child_id, accumulated + edge_len)
        return total

    path_lengths = {}
    for node_id in internal.index:
        path_lengths[node_id] = get_total_path_to_leaves(tree, node_id, 0)

    internal["path_to_leaves"] = pd.Series(path_lengths)
    internal["path_ratio"] = internal["path_to_leaves"] / (internal["height"] + 1e-10)

    return internal[["path_to_leaves", "path_ratio"]]


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_model(predictions, true_labels, model_name):
    """Evaluate a model's predictions."""
    valid_mask = ~(predictions.isna() | true_labels.isna())
    pred = predictions[valid_mask]
    true = true_labels[valid_mask]

    if len(pred) < 10 or true.sum() < 3:
        return None

    try:
        auc_roc = roc_auc_score(true, pred)
        precision, recall, _ = precision_recall_curve(true, pred)
        auc_pr = auc(recall, precision)
        corr, _ = stats.pointbiserialr(true.astype(int), pred)
    except:
        return None

    return {
        "model": model_name,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "correlation": corr,
    }


def run_comparison(
    n_clusters=4, n_per_cluster=50, n_features=100, divergence=0.3, seed=42
):
    """Compare all models on one dataset."""

    # Generate data
    sample_dict, cluster_assignments, _, _ = generate_phylogenetic_data(
        n_taxa=n_clusters,
        n_features=n_features,
        n_categories=4,
        samples_per_taxon=n_per_cluster,
        mutation_rate=divergence,
        random_seed=seed,
    )

    sample_names = list(sample_dict.keys())
    data = np.array([sample_dict[s] for s in sample_names])
    labels = np.array([cluster_assignments[s] for s in sample_names])

    # Build tree
    distances = pdist(data, metric="hamming")
    Z = linkage(distances, method="weighted")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Decompose
    K = int(data.max()) + 1
    n_samples, n_feats = data.shape
    prob_data = np.zeros((n_samples, n_feats * K))
    for i in range(n_samples):
        for j in range(n_feats):
            prob_data[i, j * K + data[i, j]] = 1.0
    prob_df = pd.DataFrame(prob_data, index=sample_names)
    tree.decompose(leaf_data=prob_df)
    stats_df = tree.stats_df.copy()

    # Add true labels
    leaf_labels = {name: lbl for name, lbl in zip(sample_names, labels)}
    for node_id in tree.nodes():
        if tree.nodes[node_id].get("is_leaf"):
            sample_name = tree.nodes[node_id].get("label")
            if sample_name in leaf_labels:
                leaf_labels[node_id] = leaf_labels[sample_name]

    def get_leaves(t, n):
        if t.out_degree(n) == 0:
            return [n]
        return sum([get_leaves(t, c) for c in t.successors(n)], [])

    should_split = {}
    for node_id in stats_df.index:
        if stats_df.loc[node_id, "is_leaf"]:
            should_split[node_id] = False
        else:
            leaves = get_leaves(tree, node_id)
            lbls = [leaf_labels.get(l) for l in leaves if l in leaf_labels]
            should_split[node_id] = len(set(lbls)) > 1

    stats_df["should_split"] = pd.Series(should_split)

    # Add inconsistency
    inconsistency = model_3_inconsistency(Z, stats_df)
    stats_df["inconsistency"] = inconsistency

    # Evaluate models
    results = []
    internal = stats_df[~stats_df["is_leaf"]]
    true_labels = internal["should_split"]

    # Model 1: Height Prior
    height_prior = model_1_height_prior(stats_df)
    stats_df.loc[height_prior.index, "height_prior"] = height_prior
    res = evaluate_model(
        stats_df.loc[internal.index, "height_prior"], true_labels, "M1: Height Prior"
    )
    if res:
        results.append(res)

    # Model 2: Calibrated Divergence
    calib = model_2_calibrated_divergence(stats_df)
    for col in calib.columns:
        stats_df.loc[calib.index, col] = calib[col]
    res = evaluate_model(
        stats_df.loc[internal.index, "kl_ratio"], true_labels, "M2: KL Ratio"
    )
    if res:
        results.append(res)

    # Model 3: Inconsistency
    res = evaluate_model(internal["inconsistency"], true_labels, "M3: Inconsistency")
    if res:
        results.append(res)

    # Model 4: Combined
    combined = model_4_combined_score(stats_df)
    stats_df.loc[combined.index, "combined_score"] = combined
    res = evaluate_model(
        stats_df.loc[internal.index, "combined_score"], true_labels, "M4: Combined"
    )
    if res:
        results.append(res)

    # Model 5: Path Length
    path_metrics = model_5_path_length_ratio(stats_df, tree)
    for col in path_metrics.columns:
        stats_df.loc[path_metrics.index, col] = path_metrics[col]
    res = evaluate_model(
        stats_df.loc[internal.index, "path_to_leaves"], true_labels, "M5: Path Length"
    )
    if res:
        results.append(res)

    # Baselines
    res = evaluate_model(internal["height"], true_labels, "Baseline: Height")
    if res:
        results.append(res)

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("MATHEMATICAL MODELS FOR BRANCH LENGTH INTEGRATION")
    print("=" * 80)

    print("""
Models being compared:
----------------------
M1: Height Prior      - P(split) = sigmoid(normalized_height)
M2: KL Ratio          - observed_KL / expected_KL(height)
M3: Inconsistency     - Classical (h - μ_local) / σ_local
M4: Combined          - Weighted sum of height + inconsistency
M5: Path Length       - Total path to descendant leaves

Baseline: Raw Height
""")

    scenarios = [
        (4, 50, 100, 0.3, "Base"),
        (2, 50, 100, 0.3, "2 clusters"),
        (8, 25, 100, 0.3, "8 clusters"),
        (4, 50, 100, 0.1, "Low div"),
        (4, 50, 100, 0.5, "High div"),
    ]

    all_results = []

    for n_c, n_per, n_f, div, desc in scenarios:
        print(f"\n--- Scenario: {desc} ---")
        results = run_comparison(n_c, n_per, n_f, div, seed=42)
        results["scenario"] = desc
        all_results.append(results)

        print(
            results[["model", "auc_roc", "auc_pr", "correlation"]].to_string(
                index=False
            )
        )

    # Summary
    all_df = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 80)
    print("SUMMARY: MEAN AUC-ROC ACROSS SCENARIOS")
    print("=" * 80)

    summary = (
        all_df.groupby("model")["auc_roc"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best_model = summary["mean"].idxmax()
    best_auc = summary.loc[best_model, "mean"]

    print(f"""
Best performing model: {best_model} (AUC = {best_auc:.3f})

INTEGRATION OPTIONS:

1. PRE-FILTER: Use height or path_length to skip low-value nodes
   - Only run expensive statistical tests on nodes above threshold
   - Reduces computation and false positives from low-height nodes

2. PRIOR/WEIGHT: Use height as Bayesian prior
   - P(split | data) ∝ P(data | split) * P(split | height)
   - Modulates the statistical test with tree structure

3. COMBINED CRITERION: Require BOTH statistical significance AND high height
   - Node is cluster boundary if:
     * Child-parent KL test is significant, AND
     * Height > threshold (or inconsistency > threshold)

4. ADAPTIVE THRESHOLD: Use height to set adaptive α for testing
   - Lower height → stricter threshold (lower α)
   - Higher height → standard threshold
""")


if __name__ == "__main__":
    main()
