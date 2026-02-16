"""
Purpose: Debug script to analyze sibling testing, iteration, and cluster merging logic.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/pipeline_gates/q_sibling_iteration_and_merge_logic__pipeline_gates__diagnostic.py
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.tree.poset_tree import PosetTree


def trace_decomposition_logic():
    """Trace through the decomposition step by step."""
    print("=" * 70)
    print("ANALYSIS: Decomposition Iteration Logic")
    print("=" * 70)

    print("""
DECOMPOSITION ALGORITHM OVERVIEW:
=================================

The algorithm performs TOP-DOWN traversal using a stack (LIFO):

1. Start at ROOT
2. For each node, evaluate _should_split():
   
   Gate 1: BINARY STRUCTURE
   - Must have exactly 2 children
   - Otherwise: MERGE (collect all leaves as one cluster)
   
   Gate 2: CHILD-PARENT DIVERGENCE (at least one child)
   - Tests: "Did either child diverge from parent?"
   - At least one child must have Child_Parent_Divergence_Significant = True
   - If NEITHER diverges: MERGE (no signal, it's just noise)
   
   Gate 3: SIBLING DIVERGENCE (the two children)
   - Tests: "Are the siblings significantly different from EACH OTHER?"
   - Uses Sibling_BH_Different = True/False
   - If siblings are SAME: MERGE (they belong to same cluster)
   - If siblings are DIFFERENT: SPLIT (continue down both branches)

3. If SPLIT: Push both children onto stack, continue
4. If MERGE: Collect all leaves under this node as ONE cluster

VISUAL:
              ROOT
             /    \\
           A        B       <- Sibling test compares A vs B
          / \\      / \\
         C   D    E   F     <- If A and B are SAME: merge all C,D,E,F
                            <- If A and B are DIFFERENT: 
                               continue testing C vs D, and E vs F
""")


def analyze_sibling_only_testing():
    """Analyze whether only checking siblings is correct."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Is Only Checking Siblings Correct?")
    print("=" * 70)

    print("""
WHY ONLY SIBLINGS?
==================

The algorithm relies on a key property of agglomerative hierarchical clustering:

PROPERTY: In a dendrogram, if two nodes are siblings, their LOWEST COMMON 
ANCESTOR (LCA) is their immediate parent. This means the split decision
at any node ONLY affects its immediate children.

ARGUMENT FOR SIBLING-ONLY TESTING:
----------------------------------
1. At each internal node, we ask: "Should these two subtrees be ONE cluster 
   or TWO clusters?"

2. This is precisely what the sibling test answers: "Are the distributions
   of the left and right subtrees significantly different?"

3. If we split, we recursively ask the same question for each child.

4. The tree structure naturally handles transitivity:
   - If A differs from B at level 1
   - And C differs from D (under A) at level 2
   - We get 3 clusters: C, D, B (not 4, because B wasn't split)

POTENTIAL ISSUE: NON-ADJACENT COMPARISONS
-----------------------------------------
Consider this tree:
              ROOT
             /    \\
           A        B
          / \\      / \\
         C   D    E   F

If A vs B are NOT significantly different (siblings same):
  -> We merge A and B -> all C,D,E,F in one cluster

But what if:
  - C vs D: DIFFERENT
  - E vs F: DIFFERENT
  - C vs E: DIFFERENT
  - C vs F: DIFFERENT
  
The current algorithm would NOT split C/D or E/F because we stop at A/B level.

IS THIS A PROBLEM?
------------------
MAYBE NOT, because:
1. If A vs B are truly the same, then the differences C/D and E/F might be
   noise within the same larger cluster.
   
2. The child-parent gate (Gate 2) helps: if no child diverges from parent,
   we don't even do the sibling test.

MAYBE YES, because:
1. We might miss fine-grained structure when high-level test is conservative.
2. FDR correction at high level might be too stringent.

POST-HOC MERGE ADDRESSES THIS:
-----------------------------
The algorithm has a post-hoc merge phase that:
1. First splits aggressively (over-splits)
2. Then merges clusters that are NOT significantly different
3. This catches cases where we over-split and need to recombine
""")


def trace_posthoc_merge():
    """Explain how post-hoc merge works."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Post-Hoc Merge Logic")
    print("=" * 70)

    print("""
POST-HOC MERGE ALGORITHM:
=========================

After initial decomposition (which may over-split), we:

1. COLLECT all sibling-boundary cluster pairs
   - For each internal node with 2 children:
   - Find clusters under left child
   - Find clusters under right child
   - Form all pairs (left_cluster, right_cluster)

2. TEST each pair with sibling_divergence_test
   - Get p-values for all pairs

3. FDR CORRECTION (single Benjamini-Hochberg across ALL pairs)
   - Controls false discovery rate globally

4. BLOCK RULE: If ANY pair at a given LCA rejects H0, block ALL merges
   under that LCA
   - This prevents inconsistent merging

5. GREEDY MERGE (highest p-value first = most similar)
   - Merge clusters that failed to reject H0
   - Skip if either cluster already merged
   - Add LCA as new cluster root

EXAMPLE:
--------
Initial clusters after over-splitting: C, D, E, F

              ROOT
             /    \\
           A        B
          / \\      / \\
         C   D    E   F
         
Post-hoc comparisons:
- C vs E (across A-B boundary): p = 0.8 (similar)
- C vs F (across A-B boundary): p = 0.7 (similar)  
- D vs E (across A-B boundary): p = 0.01 (different)
- D vs F (across A-B boundary): p = 0.02 (different)

After FDR: D vs E and D vs F reject H0 -> block all merges at ROOT
Result: Keep C, D, E, F as separate clusters

Alternative scenario:
- C vs E: p = 0.8
- C vs F: p = 0.7
- D vs E: p = 0.6
- D vs F: p = 0.5

After FDR: None reject -> can merge
Greedy: Merge C & E first (p=0.8), then D & F (p=0.7)
Result: 2 clusters instead of 4
""")


def run_example():
    """Run a concrete example to trace the logic."""
    print("\n" + "=" * 70)
    print("CONCRETE EXAMPLE: Tracing Decomposition")
    print("=" * 70)

    # Generate simple binary data with 3 clusters
    np.random.seed(42)

    # 3 clusters with different patterns
    cluster1 = np.array([[1, 1, 1, 0, 0, 0, 0, 0]] * 4)  # samples 0-3
    cluster2 = np.array([[0, 0, 0, 1, 1, 1, 0, 0]] * 4)  # samples 4-7
    cluster3 = np.array([[0, 0, 0, 0, 0, 0, 1, 1]] * 4)  # samples 8-11

    data = np.vstack([cluster1, cluster2, cluster3])
    data_df = pd.DataFrame(data, index=[f"s{i}" for i in range(12)])

    print(f"Data: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"True clusters: 3 (samples 0-3, 4-7, 8-11)")

    # Build tree
    Z = linkage(pdist(data_df.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    print(f"\nTree structure: {len(tree.nodes)} nodes, {len(tree.edges)} edges")

    # Run decomposition with verbose output
    print("\nRunning decomposition...")
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
    )

    stats_df = tree.stats_df

    # Show child-parent test results
    print("\n--- Child-Parent Divergence Results ---")
    cp_cols = ["Child_Parent_Divergence_Significant", "Child_Parent_Divergence_P_Value"]
    cp_sig = stats_df[stats_df["Child_Parent_Divergence_Significant"] == True]
    print(f"Significant child-parent divergences: {len(cp_sig)}")

    # Show sibling test results
    print("\n--- Sibling Divergence Results ---")
    if "Sibling_BH_Different" in stats_df.columns:
        sib_tested = stats_df["Sibling_Divergence_P_Value"].notna().sum()
        sib_diff = stats_df["Sibling_BH_Different"].sum()
        print(f"Sibling tests performed: {sib_tested}")
        print(f"Siblings different: {sib_diff}")

        # Show details
        tested_nodes = stats_df[stats_df["Sibling_Divergence_P_Value"].notna()]
        for node_id, row in tested_nodes.iterrows():
            p_val = row["Sibling_Divergence_P_Value"]
            is_diff = row["Sibling_BH_Different"]
            children = list(tree.successors(node_id))
            print(
                f"  {node_id}: children={children}, p={p_val:.4f}, different={is_diff}"
            )

    # Show final clusters
    print("\n--- Final Clusters ---")
    cluster_assignments = decomp.get("cluster_assignments", {})
    for cid, info in cluster_assignments.items():
        print(
            f"  Cluster {cid}: root={info['root_node']}, size={info['size']}, leaves={info['leaves']}"
        )

    # Check if clusters match ground truth
    print("\n--- Ground Truth Check ---")
    gt_clusters = {
        "s0": 0,
        "s1": 0,
        "s2": 0,
        "s3": 0,
        "s4": 1,
        "s5": 1,
        "s6": 1,
        "s7": 1,
        "s8": 2,
        "s9": 2,
        "s10": 2,
        "s11": 2,
    }

    pred_clusters = {}
    for cid, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            pred_clusters[leaf] = cid

    # Count matches
    correct = sum(1 for s in gt_clusters if pred_clusters.get(s, -1) == gt_clusters[s])

    from sklearn.metrics import adjusted_rand_score

    gt_labels = [gt_clusters[f"s{i}"] for i in range(12)]
    pred_labels = [pred_clusters.get(f"s{i}", -1) for i in range(12)]
    ari = adjusted_rand_score(gt_labels, pred_labels)
    print(f"ARI: {ari:.4f}")


def summarize_concerns():
    """Summarize potential concerns with the current approach."""
    print("\n" + "=" * 70)
    print("SUMMARY: Concerns and Recommendations")
    print("=" * 70)

    print("""
CURRENT APPROACH:
================
1. Top-down traversal with 3 gates
2. Sibling-only testing at each split
3. Post-hoc merge to reduce over-splitting

POTENTIAL CONCERNS:
==================

1. GATE ORDER DEPENDENCY
   - Gate 2 (child-parent) runs before Gate 3 (sibling)
   - If child-parent test is too conservative, sibling test never runs
   - This could miss real splits

2. SIBLING-ONLY LIMITATION
   - Only tests immediate siblings
   - Doesn't test across multiple levels (e.g., cousins)
   - Relies on transitivity via tree structure

3. FDR CORRECTION SCOPE
   - Sibling tests use BH correction across all tested pairs
   - But tests are run AFTER filtering by child-parent gate
   - This might affect power

4. POST-HOC MERGE COMPLEXITY
   - Compares all cross-boundary pairs
   - N^2 comparisons in worst case
   - Block rule might be too conservative

RECOMMENDATIONS:
===============

1. CONSIDER REMOVING GATE 2 (child-parent)
   - Let sibling test alone decide splits
   - Simplifies logic
   - Might increase sensitivity
   
2. ADD OPTION FOR NON-SIBLING COMPARISONS
   - Compare clusters at different tree levels
   - Could catch missed structure
   
3. TUNE FDR CORRECTION
   - Consider hierarchical FDR methods
   - Match correction scope to tree structure

4. VERIFY ON KNOWN GROUND TRUTH
   - Run benchmarks with synthetic data
   - Check that all true clusters are found
""")


if __name__ == "__main__":
    trace_decomposition_logic()
    analyze_sibling_only_testing()
    trace_posthoc_merge()
    run_example()
    summarize_concerns()

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
