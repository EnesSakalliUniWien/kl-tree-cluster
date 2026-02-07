"""Debug script for clustering pipeline failures.

Investigates why clusters are not being found (returning 1 cluster) or
why sibling divergence tests are being skipped.
"""

import sys
from pathlib import Path

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    annotate_sibling_divergence,
)


def debug_clustering():
    print("=" * 80)
    print("DEBUGGING CLUSTERING PIPELINE")
    print("=" * 80)

    # 1. Create dataset
    print("\n1. Creating synthetic dataset (3 blobs, 50 features)...")
    # Using more features to ensure separability after binarization
    X, y = make_blobs(
        n_samples=150, n_features=50, centers=3, cluster_std=2.0, random_state=42
    )
    # Simple binarization
    X_binary = (X > np.median(X, axis=0)).astype(int)

    sample_names = [f"Sample_{i}" for i in range(len(X))]
    leaf_data = pd.DataFrame(X_binary, index=sample_names)
    print(f"   Data shape: {leaf_data.shape}")

    # 2. Build Tree
    print("\n2. Building hierarchical tree...")
    linkage_matrix = linkage(
        pdist(leaf_data.values, metric="rogerstanimoto"), method="average"
    )
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=sample_names)
    print(f"   Tree built with {tree.number_of_nodes()} nodes")

    # 3. Populate distributions
    print("\n3. Populating node distributions...")
    tree.populate_node_divergences(leaf_data)

    # 4. Check Root Node
    root = [n for n, d in tree.in_degree() if d == 0][0]
    print(f"   Root node: {root}")
    print(f"   Root sample size: {tree.nodes[root].get('leaf_count', 'MISSING')}")
    print(f"   Root distribution shape: {tree.nodes[root]['distribution'].shape}")

    # 5. Initialize stats DataFrame
    print("\n5. Initializing stats DataFrame...")
    if hasattr(tree, "stats_df"):
        stats_df = tree.stats_df.copy()
        print(
            f"   Using existing tree.stats_df with {len(stats_df)} rows and columns: {list(stats_df.columns)}"
        )
    else:
        print("   WARNING: tree.stats_df not found! Creating from node attributes...")
        data = {}
        for n in tree.nodes:
            data[n] = tree.nodes[n]
        stats_df = pd.DataFrame.from_dict(data, orient="index")
        print(f"   Created stats_df with columns: {list(stats_df.columns)}")

    # 6. Run Child-Parent Divergence
    print("\n6. Running Child-Parent Divergence Test...")
    try:
        from kl_clustering_analysis import config

        alpha_local = 0.05

        # Manually run the annotation
        stats_df = annotate_child_parent_divergence(
            tree=tree,
            nodes_statistics_dataframe=stats_df,
            significance_level_alpha=alpha_local,
        )

        sig_count = stats_df["Child_Parent_Divergence_Significant"].sum()
        total_count = len(stats_df)
        print(f"   Significant nodes: {sig_count} / {total_count}")

        # DEBUG: Print significant nodes info
        sig_nodes = stats_df[
            stats_df["Child_Parent_Divergence_Significant"]
        ].index.tolist()
        print(f"   Significant Node IDs: {sig_nodes}")
        for nid in sig_nodes:
            print(
                f"     - {nid}: leaf_count={tree.nodes[nid].get('leaf_count')}, pval={stats_df.loc[nid, 'Child_Parent_Divergence_P_Value']:.4f}"
            )

        # Check root children specifically
        print("   Root Children Analysis:")
        children = list(tree.successors(root))
        for child in children:
            print(
                f"     Child {child}: leaf_count={tree.nodes[child].get('leaf_count')}"
            )
            print(f"       KL Local: {stats_df.loc[child, 'kl_divergence_local']:.4f}")
            print(
                f"       P-Value: {stats_df.loc[child, 'Child_Parent_Divergence_P_Value']:.4f}"
            )
            print(
                f"       Significant: {stats_df.loc[child, 'Child_Parent_Divergence_Significant']}"
            )

        if sig_count == 0:
            print("   CRITICAL: No nodes found significant in Child-Parent test!")
            print("   This causes Sibling test to be skipped for all nodes.")

            # Print p-values to see if they are close
            print("   P-value stats:")
            print(stats_df["Child_Parent_Divergence_P_Value"].describe())

    except Exception as e:
        print(f"   ERROR in Child-Parent test: {e}")
        import traceback

        traceback.print_exc()
        return

    # 7. Run Sibling Divergence
    print("\n7. Running Sibling Divergence Test...")
    try:
        # Check eligibility manually first
        parents_eligible = []
        for node in tree.nodes:
            children = list(tree.successors(node))
            if len(children) != 2:
                continue

            l, r = children
            # Condition 1: Check significance map
            sig_map = stats_df["Child_Parent_Divergence_Significant"].to_dict()
            is_sig = sig_map.get(l, False) or sig_map.get(r, False)

            # Condition 2: Check sample sizes
            n_l = tree.nodes[l].get("leaf_count", 0)
            n_r = tree.nodes[r].get("leaf_count", 0)
            has_samples = n_l >= 2 and n_r >= 2

            if is_sig and has_samples:
                parents_eligible.append(node)

        print(f"   Manually found {len(parents_eligible)} eligible parent nodes.")

        if len(parents_eligible) == 0:
            print("   Checking failed conditions for top nodes:")
            # Check root's children
            children = list(tree.successors(root))
            if len(children) == 2:
                l, r = children
                print(f"     Root children: {l}, {r}")
                print(
                    f"     Sig(L): {stats_df.loc[l, 'Child_Parent_Divergence_Significant']}"
                )
                print(
                    f"     Sig(R): {stats_df.loc[r, 'Child_Parent_Divergence_Significant']}"
                )
                print(
                    f"     P-val(L): {stats_df.loc[l, 'Child_Parent_Divergence_P_Value']}"
                )
                print(
                    f"     P-val(R): {stats_df.loc[r, 'Child_Parent_Divergence_P_Value']}"
                )
                print(f"     Count(L): {tree.nodes[l].get('leaf_count')}")
                print(f"     Count(R): {tree.nodes[r].get('leaf_count')}")

        sibling_alpha = 0.05
        stats_df = annotate_sibling_divergence(
            tree=tree,
            nodes_statistics_dataframe=stats_df,
            significance_level_alpha=sibling_alpha,
        )

        diff_count = (
            stats_df["Sibling_BH_Different"].sum()
            if "Sibling_BH_Different" in stats_df.columns
            else 0
        )
        print(f"   Sibling different pairs: {diff_count}")

    except Exception as e:
        print(f"   ERROR in Sibling test: {e}")
        import traceback

        traceback.print_exc()

    # 8. Full Decompose Integration
    print("\n8. Testing full decompose() integration...")
    try:
        results = tree.decompose(leaf_data=leaf_data)
        print(f"   Clusters found: {results.get('num_clusters')}")
        assignments = results.get("cluster_assignments", {})
        lens = [len(v) for v in assignments.values()]
        print(f"   Cluster sizes: {lens}")

        # DEBUG: Print exact assignments
        print("   Cluster Assignments dump:")
        actual_lens = []
        for cid, info in assignments.items():
            # info should be a dict with key 'leaves'
            leaves = info.get("leaves", [])
            actual_lens.append(len(leaves))
            print(
                f"     Cluster {cid}: {len(leaves)} leaves. Info keys: {list(info.keys())}"
            )

        print(f"   corrected cluster sizes: {actual_lens}")
        total_assigned = sum(actual_lens)

        # 9. Broken Chain Analysis
        print("\n9. Broken Chain Analysis...")
        sig_map = stats_df["Child_Parent_Divergence_Significant"].to_dict()
        broken_chains = []

        # Traverse downwards
        queue = [root]
        reachable_significant = set()

        while queue:
            node = queue.pop(0)

            # Check children
            children = list(tree.successors(node))
            if not children:
                continue

            # Decomposition logic: Stop if neither child is significant
            # (Simplification: _should_split logic)
            l_sig = sig_map.get(children[0], False) if len(children) > 0 else False
            r_sig = sig_map.get(children[1], False) if len(children) > 1 else False

            should_continue = l_sig or r_sig

            if should_continue:
                queue.extend(children)
                if l_sig:
                    reachable_significant.add(children[0])
                if r_sig:
                    reachable_significant.add(children[1])
            else:
                # Branch stops here.
                # Check if there are significant nodes hidden below?
                descendants = nx.descendants(tree, node)
                hidden_sigs = [d for d in descendants if sig_map.get(d, False)]
                if hidden_sigs:
                    broken_chains.append((node, len(hidden_sigs)))

        print(f"   Reachable significant nodes: {len(reachable_significant)}")
        print(
            f"   Broken chains (stop node -> count of missed significant descendants):"
        )
        for node, count in broken_chains:
            print(
                f"     STOP at {node}: Missed {count} significant descendants underneath."
            )

        if not broken_chains and len(reachable_significant) < 10:
            print(
                "   Chain is intact but simply ends too early. Test is too conservative?"
            )

    except Exception as e:
        print(f"   ERROR in decompose(): {e}")


if __name__ == "__main__":
    debug_clustering()

if __name__ == "__main__":
    debug_clustering()
