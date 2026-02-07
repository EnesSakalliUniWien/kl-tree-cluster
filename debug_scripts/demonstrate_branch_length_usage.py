#!/usr/bin/env python3
"""
How to Use Branch Lengths with Our KL-Divergence Method
========================================================

This script demonstrates the relationship between different branch length
metrics and how they can complement our KL-divergence statistical tests.

Key findings from our benchmarks:
- `height` (merge distance): AUC=0.75, positively correlated with true splits
- `sibling_branch_sum`: AUC=0.25, NEGATIVELY correlated (not useful alone)
- `inconsistency`: AUC=0.62, weakly positive

This script shows:
1. What each metric captures
2. How they relate to KL-divergence test decisions
3. Practical ways to combine them
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, inconsistent
from scipy.spatial.distance import pdist
from scipy import stats

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.benchmarking.generators.generate_phylogenetic import (
    generate_phylogenetic_data,
)


def generate_example_data(
    n_clusters=4, n_per_cluster=50, n_features=100, divergence=0.3, seed=42
):
    """Generate test data with known cluster structure."""
    sample_dict, cluster_assignments, distributions, metadata = (
        generate_phylogenetic_data(
            n_taxa=n_clusters,
            n_features=n_features,
            n_categories=4,
            samples_per_taxon=n_per_cluster,
            mutation_rate=divergence,
            random_seed=seed,
        )
    )

    sample_names = list(sample_dict.keys())
    data = np.array([sample_dict[s] for s in sample_names])
    labels = np.array([cluster_assignments[s] for s in sample_names])

    return data, labels, sample_names


def analyze_branch_metrics(data, labels, sample_names):
    """Analyze different branch length metrics."""

    # Build linkage and tree
    distances = pdist(data, metric="hamming")
    Z = linkage(distances, method="weighted")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Compute inconsistency coefficients
    R = inconsistent(Z, d=2)

    # Create probability data and run decomposition
    K = int(data.max()) + 1
    n_samples, n_features = data.shape
    prob_data = np.zeros((n_samples, n_features * K))
    for i in range(n_samples):
        for j in range(n_features):
            prob_data[i, j * K + data[i, j]] = 1.0

    prob_df = pd.DataFrame(prob_data, index=sample_names)

    # Run decomposition to get KL-divergence stats
    decomp_results = tree.decompose(leaf_data=prob_df)
    stats_df = tree.stats_df.copy()

    # Map labels to leaf nodes
    leaf_labels = {name: lbl for name, lbl in zip(sample_names, labels)}
    for node_id in tree.nodes():
        if tree.nodes[node_id].get("is_leaf"):
            sample_name = tree.nodes[node_id].get("label")
            if sample_name and sample_name in leaf_labels:
                leaf_labels[node_id] = leaf_labels[sample_name]

    # Compute ground truth for each internal node
    def get_leaves_under(tree, node_id):
        if tree.out_degree(node_id) == 0:
            return [node_id]
        leaves = []
        for child in tree.successors(node_id):
            leaves.extend(get_leaves_under(tree, child))
        return leaves

    # Build comprehensive analysis dataframe
    n_leaves = len(sample_names)
    analysis_data = []

    for node_id in stats_df.index:
        if stats_df.loc[node_id, "is_leaf"]:
            continue

        # Get merge_idx for this node
        node_idx = int(node_id[1:])  # Remove 'N' prefix
        merge_idx = node_idx - n_leaves

        if merge_idx < 0 or merge_idx >= len(Z):
            continue

        # Get metrics
        height = Z[merge_idx, 2]
        incons = R[merge_idx, 3] if R[merge_idx, 1] > 0 else 0  # Avoid div by zero
        sibling_branch = stats_df.loc[node_id, "sibling_branch_sum"]

        # Get KL test results
        kl_local = stats_df.loc[node_id, "kl_divergence_local"]
        cp_sig = (
            stats_df.loc[node_id, "Child_Parent_Divergence_Significant"]
            if "Child_Parent_Divergence_Significant" in stats_df.columns
            else np.nan
        )

        # Ground truth
        leaves = get_leaves_under(tree, node_id)
        leaf_lbls = [leaf_labels.get(l) for l in leaves if l in leaf_labels]
        n_true_clusters = len(set(leaf_lbls)) if leaf_lbls else 0
        should_split = n_true_clusters > 1

        analysis_data.append(
            {
                "node_id": node_id,
                "height": height,
                "inconsistency": incons,
                "sibling_branch_sum": sibling_branch
                if not np.isnan(sibling_branch)
                else 0,
                "kl_divergence_local": kl_local,
                "cp_significant": cp_sig,
                "n_true_clusters": n_true_clusters,
                "should_split": should_split,
            }
        )

    return pd.DataFrame(analysis_data)


def demonstrate_usage():
    """Main demonstration."""

    print("=" * 80)
    print("HOW TO USE BRANCH LENGTHS WITH KL-DIVERGENCE METHOD")
    print("=" * 80)

    # Generate data
    print("\n1. GENERATING TEST DATA")
    print("-" * 40)
    data, labels, sample_names = generate_example_data(
        n_clusters=4, n_per_cluster=50, n_features=100, divergence=0.3, seed=42
    )
    print(f"   Generated {len(sample_names)} samples from 4 clusters")

    # Analyze
    print("\n2. ANALYZING BRANCH LENGTH METRICS")
    print("-" * 40)
    df = analyze_branch_metrics(data, labels, sample_names)

    # Summary statistics
    true_splits = df[df["should_split"] == True]
    no_splits = df[df["should_split"] == False]

    print(f"\n   Total internal nodes: {len(df)}")
    print(f"   Nodes that SHOULD split: {len(true_splits)}")
    print(f"   Nodes that should NOT split: {len(no_splits)}")

    # Compare metrics
    print("\n3. METRIC COMPARISON (mean values)")
    print("-" * 40)
    print(f"                        Should Split    Should NOT Split    Direction")
    print(
        f"   height:              {true_splits['height'].mean():.4f}          {no_splits['height'].mean():.4f}              {'✓ HIGHER' if true_splits['height'].mean() > no_splits['height'].mean() else '✗ LOWER'}"
    )
    print(
        f"   inconsistency:       {true_splits['inconsistency'].mean():.4f}          {no_splits['inconsistency'].mean():.4f}              {'✓ HIGHER' if true_splits['inconsistency'].mean() > no_splits['inconsistency'].mean() else '✗ LOWER'}"
    )
    print(
        f"   sibling_branch_sum:  {true_splits['sibling_branch_sum'].mean():.4f}          {no_splits['sibling_branch_sum'].mean():.4f}              {'✓ HIGHER' if true_splits['sibling_branch_sum'].mean() > no_splits['sibling_branch_sum'].mean() else '✗ LOWER'}"
    )
    print(
        f"   kl_divergence_local: {true_splits['kl_divergence_local'].mean():.4f}          {no_splits['kl_divergence_local'].mean():.4f}              {'✓ HIGHER' if true_splits['kl_divergence_local'].mean() > no_splits['kl_divergence_local'].mean() else '✗ LOWER'}"
    )

    # ROC-AUC scores
    from sklearn.metrics import roc_auc_score

    print("\n4. PREDICTIVE POWER (AUC-ROC)")
    print("-" * 40)

    try:
        auc_height = roc_auc_score(df["should_split"], df["height"])
        print(
            f"   height:              AUC = {auc_height:.3f}  {'✓ GOOD' if auc_height > 0.6 else '○ WEAK' if auc_height > 0.5 else '✗ INVERTED'}"
        )
    except:
        pass

    try:
        auc_incons = roc_auc_score(df["should_split"], df["inconsistency"])
        print(
            f"   inconsistency:       AUC = {auc_incons:.3f}  {'✓ GOOD' if auc_incons > 0.6 else '○ WEAK' if auc_incons > 0.5 else '✗ INVERTED'}"
        )
    except:
        pass

    try:
        auc_sibling = roc_auc_score(df["should_split"], df["sibling_branch_sum"])
        print(
            f"   sibling_branch_sum:  AUC = {auc_sibling:.3f}  {'✓ GOOD' if auc_sibling > 0.6 else '○ WEAK' if auc_sibling > 0.5 else '✗ INVERTED'}"
        )
    except:
        pass

    try:
        auc_kl = roc_auc_score(df["should_split"], df["kl_divergence_local"])
        print(
            f"   kl_divergence_local: AUC = {auc_kl:.3f}  {'✓ GOOD' if auc_kl > 0.6 else '○ WEAK' if auc_kl > 0.5 else '✗ INVERTED'}"
        )
    except:
        pass

    # Correlation matrix
    print("\n5. CORRELATION BETWEEN METRICS")
    print("-" * 40)
    corr_cols = ["height", "inconsistency", "sibling_branch_sum", "kl_divergence_local"]
    corr_matrix = df[corr_cols].corr()
    print(corr_matrix.round(3).to_string())

    # Key insight
    print("\n6. KEY INSIGHT: height vs sibling_branch_sum")
    print("-" * 40)
    print("""
   The difference explained:
   
   • height: The merge DISTANCE at which this node was formed
     - HIGH height = late merge = samples are far apart = likely DIFFERENT clusters
     - This is what dendrogram cutting uses
   
   • sibling_branch_sum: Sum of branch lengths going DOWN to children
     - Measures how much the subtree "descends" before reaching next merges
     - NOT the same as the node's position in the tree
     - Anti-correlated with true boundaries because true splits happen at
       HEIGHT JUMPS, and the branches going down from those nodes are SHORT
   
   • inconsistency: (height - local_mean) / local_std
     - Measures whether this merge is UNUSUAL compared to nearby merges
     - A height JUMP relative to the local neighborhood
""")

    # Practical recommendations
    print("\n7. HOW TO USE BRANCH LENGTHS")
    print("=" * 80)
    print("""
   RECOMMENDED APPROACHES:
   
   A) Use HEIGHT as a pre-filter:
      - Only apply expensive KL tests to nodes with height > threshold
      - Low height nodes are unlikely to be true cluster boundaries
      
      Example:
      ```python
      height_threshold = np.percentile(all_heights, 50)  # Top 50%
      candidates = [n for n in nodes if n.height > height_threshold]
      for node in candidates:
          run_kl_divergence_test(node)
      ```
   
   B) Use HEIGHT to calibrate expected divergence:
      - Higher nodes should have higher expected KL divergence
      - Normalize KL by height: ratio = kl_divergence / height
      
      Example:
      ```python
      kl_height_ratio = node.kl_divergence_local / node.height
      # Nodes with high ratio = surprising divergence for their height
      ```
   
   C) Combine HEIGHT with KL p-value:
      - Require BOTH: significant KL test AND high height
      
      Example:
      ```python
      is_split = (p_value < alpha) AND (height > height_threshold)
      ```
   
   D) Use inconsistency for detecting "jumps":
      - High inconsistency = this merge is unusually high for its neighborhood
      - Good for identifying sudden transitions in the tree
   
   NOT RECOMMENDED:
   
   ✗ Using sibling_branch_sum alone for split decisions
     (anti-correlated with true boundaries)
   
   ✗ Ignoring height entirely
     (wastes useful structural information)
""")

    # Show example of height-based filtering
    print("\n8. EXAMPLE: HEIGHT-BASED FILTERING EFFECTIVENESS")
    print("-" * 40)

    # Try different height thresholds
    percentiles = [25, 50, 75, 90]
    for pct in percentiles:
        threshold = np.percentile(df["height"], pct)
        above = df[df["height"] >= threshold]
        if len(above) > 0:
            precision = above["should_split"].sum() / len(above)
            recall = (
                above["should_split"].sum() / df["should_split"].sum()
                if df["should_split"].sum() > 0
                else 0
            )
            print(f"   Height >= {pct}th percentile ({threshold:.4f}):")
            print(f"      Nodes remaining: {len(above)} / {len(df)}")
            print(f"      Precision (true splits / selected): {precision:.2%}")
            print(f"      Recall (found true splits / all true): {recall:.2%}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
   Branch lengths ARE useful, but use the RIGHT metric:
   
   ✓ HEIGHT (merge distance) - positively correlated, use for filtering
   ✓ INCONSISTENCY - captures local jumps, complementary information
   ✗ SIBLING_BRANCH_SUM - anti-correlated, not directly useful for decisions
   
   The KL-divergence test measures DISTRIBUTIONAL differences.
   HEIGHT measures STRUCTURAL position in the tree.
   These capture complementary information and can be combined.
""")

    return df


if __name__ == "__main__":
    df = demonstrate_usage()
