"""Diagnose v2 signal localization ARI=0.0 failures.

Runs a simple 2-cluster case through both v1 and v2, printing detailed
intermediate state: split points, localization results, similarity/difference
graphs, and final cluster assignments.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def make_simple_2cluster_data(n: int = 60, d: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.standard_normal((n // 2, d)) + 2, rng.standard_normal((n // 2, d)) - 2])
    X_binary = (X > np.median(X, axis=0)).astype(int)
    data = pd.DataFrame(
        X_binary,
        index=[f"S{i}" for i in range(n)],
        columns=[f"F{j}" for j in range(d)],
    )
    true_labels = np.array([0] * (n // 2) + [1] * (n // 2))
    return data, true_labels


def run_v1(data: pd.DataFrame) -> dict:
    Z = linkage(pdist(data.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    return tree.decompose(leaf_data=data, alpha_local=0.05, sibling_alpha=0.05)


def run_v2(data: pd.DataFrame) -> dict:
    Z = linkage(pdist(data.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    return tree.decompose(
        leaf_data=data,
        alpha_local=0.05,
        sibling_alpha=0.05,
        use_signal_localization=True,
    )


def print_result(label: str, result: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  num_clusters = {result['num_clusters']}")

    ca = result.get("cluster_assignments", {})
    for cid, info in sorted(ca.items()):
        leaves = info.get("leaves", [])
        print(
            f"  Cluster {cid}: size={info['size']}, root={info['root_node']}, "
            f"leaves={sorted(leaves)[:5]}{'...' if len(leaves) > 5 else ''}"
        )

    # V2-specific
    if "split_points" in result:
        print(f"\n  split_points ({len(result['split_points'])}):")
        for sp in result["split_points"]:
            print(f"    {sp}")

    loc = result.get("localization_results", {})
    if loc:
        print(f"\n  localization_results ({len(loc)} nodes):")
        for node_id, lr in loc.items():
            print(f"    Node {node_id}:")
            print(f"      difference_pairs: {lr.difference_pairs}")
            print(f"      similarity_edges: {len(lr.similarity_edges)}")
            for se in lr.similarity_edges:
                print(f"        {se.node_a} -- {se.node_b}  p={se.p_value:.4f}")
            print(f"      nodes_tested: {lr.nodes_tested}, depth_reached: {lr.depth_reached}")

    sim = result.get("similarity_graph")
    if sim is not None:
        print(
            f"\n  similarity_graph: {sim.number_of_nodes()} nodes, "
            f"{sim.number_of_edges()} edges"
        )
        for u, v, d in sim.edges(data=True):
            print(f"    {u} -- {v}  p={d.get('p_value', '?'):.4f}")

    diff = result.get("difference_graph")
    if diff is not None:
        print(
            f"\n  difference_graph: {diff.number_of_nodes()} nodes, "
            f"{diff.number_of_edges()} edges"
        )
        for u, v, d in diff.edges(data=True):
            p = d.get("p_value", "?")
            p_str = f"{p:.4f}" if isinstance(p, (int, float)) else str(p)
            print(f"    {u} -- {v}  p={p_str}")

    audit = result.get("posthoc_merge_audit", [])
    if audit:
        print(f"\n  posthoc_merge_audit ({len(audit)} entries):")
        for entry in audit:
            print(f"    {entry}")


def main():
    print("Config:")
    print(f"  SIBLING_TEST_METHOD = {config.SIBLING_TEST_METHOD}")
    print(f"  LOCALIZATION_MAX_DEPTH = {config.LOCALIZATION_MAX_DEPTH}")
    print(f"  LOCALIZATION_MAX_PAIRS = {config.LOCALIZATION_MAX_PAIRS}")
    print(f"  POSTHOC_MERGE = {config.POSTHOC_MERGE}")

    data, true_labels = make_simple_2cluster_data()
    print(f"\nData: {data.shape}, true K=2")

    r1 = run_v1(data)
    print_result("V1 (hard split + posthoc merge)", r1)

    r2 = run_v2(data)
    print_result("V2 (signal localization)", r2)

    # Quick ARI
    try:
        from sklearn.metrics import adjusted_rand_score

        def labels_from_result(result, index):
            ca = result.get("cluster_assignments", {})
            label_map = {}
            for cid, info in ca.items():
                for leaf in info.get("leaves", []):
                    label_map[leaf] = cid
            return np.array([label_map.get(s, -1) for s in index])

        l1 = labels_from_result(r1, data.index)
        l2 = labels_from_result(r2, data.index)
        print(f"\nARI v1 = {adjusted_rand_score(true_labels, l1):.4f}")
        print(f"ARI v2 = {adjusted_rand_score(true_labels, l2):.4f}")
    except ImportError:
        print("\nsklearn not available — skipping ARI computation")


if __name__ == "__main__":
    main()
