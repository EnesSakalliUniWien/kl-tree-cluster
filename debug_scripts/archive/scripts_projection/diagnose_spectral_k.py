#!/usr/bin/env python3
"""Diagnose spectral_k (effective rank) at each tree node.

Shows:
1. Data matrix composition (n_leaves, n_internal, n_total) per node
2. Effective rank with leaves-only vs leaves+internal nodes
3. Children's spectral dims vs parent's spectral dims
4. JL dimension comparison
5. Impact on sibling test T and p-value

This reveals WHY spectral_k is too low at the root for the sibling test.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    compute_projection_dimension,
    sibling_divergence_test,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral_dimension import (
    compute_sibling_spectral_dimensions,
    compute_spectral_decomposition,
    effective_rank,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def build_tree_and_data(case):
    """Build PosetTree from a benchmark case."""
    data_df, true_labels, X_raw, metadata = generate_case_data(case)
    true_k = len(set(true_labels))

    # data_df is already binarized by the generator
    data = data_df

    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)

    return tree, data, true_labels, true_k


def analyze_spectral_k(tree, leaf_data):
    """Detailed per-node spectral dimension analysis."""
    d = leaf_data.shape[1]
    X = leaf_data.values.astype(np.float64)
    label_to_idx = {label: i for i, label in enumerate(leaf_data.index)}

    def _is_leaf(node_id):
        is_leaf_attr = tree.nodes[node_id].get("is_leaf")
        if is_leaf_attr is not None:
            return bool(is_leaf_attr)
        return tree.out_degree(node_id) == 0

    # --- Precompute descendant info bottom-up ---
    desc_leaf_indices = {}
    desc_internal_nodes = {}
    for node_id in reversed(list(nx.topological_sort(tree))):
        if _is_leaf(node_id):
            lbl = tree.nodes[node_id].get("label", node_id)
            desc_leaf_indices[node_id] = [label_to_idx[lbl]] if lbl in label_to_idx else []
            desc_internal_nodes[node_id] = []
        else:
            indices = []
            internals = []
            for child in tree.successors(node_id):
                indices.extend(desc_leaf_indices.get(child, []))
                if not _is_leaf(child):
                    internals.append(child)
                internals.extend(desc_internal_nodes.get(child, []))
            desc_leaf_indices[node_id] = indices
            desc_internal_nodes[node_id] = internals

    results = []
    for node_id in tree.nodes:
        if _is_leaf(node_id):
            continue

        row_indices = desc_leaf_indices[node_id]
        n_leaves = len(row_indices)
        if n_leaves < 2:
            continue

        leaf_rows = X[row_indices, :]

        # Collect internal node distributions
        internal_rows = []
        for inode in desc_internal_nodes[node_id]:
            dist = tree.nodes[inode].get("distribution")
            if dist is not None:
                dist_arr = np.asarray(dist, dtype=np.float64)
                if dist_arr.shape == (d,):
                    internal_rows.append(dist_arr)
        n_internal = len(internal_rows)

        # --- Effective rank: leaves only ---
        col_var_lo = np.var(leaf_rows, axis=0)
        d_active_lo = int(np.sum(col_var_lo > 0))
        if d_active_lo >= 2:
            if n_leaves < d_active_lo:
                X_std = leaf_rows[:, col_var_lo > 0] - leaf_rows[:, col_var_lo > 0].mean(axis=0)
                stds = leaf_rows[:, col_var_lo > 0].std(axis=0, ddof=0)
                stds[stds == 0] = 1.0
                X_std /= stds
                gram = X_std @ X_std.T / d_active_lo
                evals_lo = np.sort(np.linalg.eigvalsh(gram))[::-1]
            else:
                corr = np.corrcoef(leaf_rows[:, col_var_lo > 0].T)
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
                evals_lo = np.sort(np.linalg.eigvalsh(corr))[::-1]
            evals_lo = np.maximum(evals_lo, 0.0)
            erank_lo = effective_rank(evals_lo)
        else:
            erank_lo = 1.0
            d_active_lo = max(d_active_lo, 1)

        # --- Effective rank: leaves + internal ---
        if internal_rows:
            data_sub = np.vstack([leaf_rows, np.array(internal_rows)])
        else:
            data_sub = leaf_rows
        n_total = data_sub.shape[0]
        col_var = np.var(data_sub, axis=0)
        d_active = int(np.sum(col_var > 0))
        if d_active >= 2:
            if n_total < d_active:
                X_std = data_sub[:, col_var > 0] - data_sub[:, col_var > 0].mean(axis=0)
                stds = data_sub[:, col_var > 0].std(axis=0, ddof=0)
                stds[stds == 0] = 1.0
                X_std /= stds
                gram = X_std @ X_std.T / d_active
                evals = np.sort(np.linalg.eigvalsh(gram))[::-1]
            else:
                corr = np.corrcoef(data_sub[:, col_var > 0].T)
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
                evals = np.sort(np.linalg.eigvalsh(corr))[::-1]
            evals = np.maximum(evals, 0.0)
            erank_full = effective_rank(evals)
        else:
            erank_full = 1.0
            d_active = max(d_active, 1)

        # --- JL dimension ---
        from scipy.stats import hmean

        children = list(tree.successors(node_id))
        if len(children) == 2:
            lc = tree.nodes[children[0]].get("leaf_count", 1)
            rc = tree.nodes[children[1]].get("leaf_count", 1)
            n_eff = hmean([float(lc), float(rc)])
        else:
            n_eff = float(n_leaves)
        k_jl = compute_projection_dimension(int(n_eff), d)

        # --- Children spectral dims ---
        children_eranks = {}
        for child in children:
            if _is_leaf(child):
                children_eranks[child] = 0
                continue
            child_indices = desc_leaf_indices[child]
            nc = len(child_indices)
            if nc < 2:
                children_eranks[child] = 1
                continue
            child_rows = X[child_indices, :]
            cv = np.var(child_rows, axis=0)
            da = int(np.sum(cv > 0))
            if da < 2:
                children_eranks[child] = 1
                continue
            if nc < da:
                Xs = child_rows[:, cv > 0] - child_rows[:, cv > 0].mean(axis=0)
                ss = child_rows[:, cv > 0].std(axis=0, ddof=0)
                ss[ss == 0] = 1.0
                Xs /= ss
                g = Xs @ Xs.T / da
                ev = np.sort(np.linalg.eigvalsh(g))[::-1]
            else:
                c = np.corrcoef(child_rows[:, cv > 0].T)
                c = np.nan_to_num(c, nan=0.0)
                np.fill_diagonal(c, 1.0)
                ev = np.sort(np.linalg.eigvalsh(c))[::-1]
            ev = np.maximum(ev, 0.0)
            children_eranks[child] = effective_rank(ev)

        # --- Pooled within-cluster effective rank ---
        if len(children) == 2:
            left_idx = desc_leaf_indices[children[0]]
            right_idx = desc_leaf_indices[children[1]]
            n_l, n_r = len(left_idx), len(right_idx)
            if n_l >= 2 and n_r >= 2:
                left_rows = X[left_idx, :]
                right_rows = X[right_idx, :]
                # Pool within-cluster covariances
                mean_l = left_rows.mean(axis=0)
                mean_r = right_rows.mean(axis=0)
                resid_l = left_rows - mean_l
                resid_r = right_rows - mean_r
                pooled_resid = np.vstack([resid_l, resid_r])
                # Compute correlation of pooled residuals
                pv = np.var(pooled_resid, axis=0)
                da_p = int(np.sum(pv > 0))
                if da_p >= 2:
                    if pooled_resid.shape[0] < da_p:
                        Xs = pooled_resid[:, pv > 0] - pooled_resid[:, pv > 0].mean(axis=0)
                        ss = pooled_resid[:, pv > 0].std(axis=0, ddof=0)
                        ss[ss == 0] = 1.0
                        Xs /= ss
                        g = Xs @ Xs.T / da_p
                        ev = np.sort(np.linalg.eigvalsh(g))[::-1]
                    else:
                        c = np.corrcoef(pooled_resid[:, pv > 0].T)
                        c = np.nan_to_num(c, nan=0.0)
                        np.fill_diagonal(c, 1.0)
                        ev = np.sort(np.linalg.eigvalsh(c))[::-1]
                    ev = np.maximum(ev, 0.0)
                    erank_pooled = effective_rank(ev)
                else:
                    erank_pooled = 1.0
            else:
                erank_pooled = None
        else:
            erank_pooled = None

        depth = nx.shortest_path_length(
            tree, tree.graph.get("root", list(nx.topological_sort(tree))[0]), node_id
        )

        results.append(
            {
                "node": node_id,
                "depth": depth,
                "n_leaves": n_leaves,
                "n_internal": n_internal,
                "n_total": n_total,
                "d_active": d_active,
                "erank_leaves_only": round(erank_lo, 1),
                "erank_leaves+internal": round(erank_full, 1),
                "erank_pooled_within": round(erank_pooled, 1) if erank_pooled else None,
                "children_eranks": {k: round(v, 1) for k, v in children_eranks.items()},
                "max_child_erank": (
                    round(max(children_eranks.values()), 1) if children_eranks else None
                ),
                "k_jl": k_jl,
                "n_children": len(children),
            }
        )

    return results


def sibling_test_comparison(tree, leaf_data, node_info, sibling_dims=None):
    """Compare sibling test with different k values at binary internal nodes."""
    print("\n" + "=" * 90)
    print(
        "SIBLING TEST COMPARISON (parent spectral_k vs children max vs JL vs pooled_within vs sibling_k)"
    )
    print("=" * 90)

    for info in node_info:
        if info["n_children"] != 2:
            continue
        node_id = info["node"]
        children = list(tree.successors(node_id))
        left, right = children[0], children[1]

        left_dist = np.asarray(tree.nodes[left]["distribution"])
        right_dist = np.asarray(tree.nodes[right]["distribution"])
        n_left = float(tree.nodes[left].get("leaf_count", 1))
        n_right = float(tree.nodes[right].get("leaf_count", 1))

        # Test with different k values
        k_values = {
            "parent_erank": int(round(info["erank_leaves+internal"])),
            "parent_erank_lo": int(round(info["erank_leaves_only"])),
            "max_child_erank": (
                int(round(info["max_child_erank"])) if info["max_child_erank"] else None
            ),
            "pooled_within": (
                int(round(info["erank_pooled_within"])) if info["erank_pooled_within"] else None
            ),
            "sibling_k (pipeline)": sibling_dims.get(node_id) if sibling_dims else None,
            "JL": info["k_jl"],
        }

        print(
            f"\n--- {node_id} (depth={info['depth']}, n_leaves={info['n_leaves']}, "
            f"left={left} n={int(n_left)}, right={right} n={int(n_right)}) ---"
        )
        print(f"  d_active={info['d_active']}")
        print(f"  Children eranks: {info['children_eranks']}")

        for label, k in k_values.items():
            if k is None or k < 1:
                print(f"  {label:25s}: k=N/A")
                continue
            k = max(k, 1)
            # Run sibling test with this k (using random projection, no PCA)
            T, df, p = sibling_divergence_test(
                left_dist,
                right_dist,
                n_left,
                n_right,
                spectral_k=k,
                pca_projection=None,
                pca_eigenvalues=None,
            )
            verdict = "REJECT" if p < 0.05 else "SAME"
            marker = " << PIPELINE" if label == "sibling_k (pipeline)" else ""
            print(
                f"  {label:25s}: k={k:3d}  T={T:8.2f}  df={df:5.0f}  p={p:.6f}  -> {verdict}{marker}"
            )


def main():
    test_cases = [
        {
            "name": "gauss_clear_small",
            "generator": "blobs",
            "n_samples": 30,
            "n_features": 100,
            "n_clusters": 3,
            "cluster_std": 1.0,
            "seed": 42,
        },
        {
            "name": "gauss_clear_med",
            "generator": "blobs",
            "n_samples": 100,
            "n_features": 100,
            "n_clusters": 3,
            "cluster_std": 1.0,
            "seed": 42,
        },
        {
            "name": "block_4c",
            "generator": "binary",
            "n_rows": 80,
            "n_cols": 100,
            "n_clusters": 4,
            "entropy_param": 0.3,
            "seed": 42,
        },
        {
            "name": "trivial_2c",
            "generator": "blobs",
            "n_samples": 40,
            "n_features": 50,
            "n_clusters": 2,
            "cluster_std": 1.0,
            "seed": 42,
        },
    ]

    for case in test_cases:
        print("\n" + "#" * 90)
        params = case.get("params", case)
        true_k_param = params.get("k", params.get("n_clusters", "?"))
        n_param = params.get("n", params.get("n_samples", "?"))
        d_param = params.get("d", params.get("n_features", "?"))
        print(f"# CASE: {case['name']} (true K={true_k_param}, n={n_param}, d={d_param})")
        print("#" * 90)

        tree, data, true_labels, true_k = build_tree_and_data(case)

        # Get pipeline spectral dims for comparison
        pipeline_dims, _, _ = compute_spectral_decomposition(
            tree,
            data,
            method="effective_rank",
            min_k=1,
            compute_projections=True,
        )

        # Get sibling-specific spectral dims (pooled within-cluster)
        sibling_dims = compute_sibling_spectral_dimensions(
            tree,
            data,
            method="effective_rank",
            min_k=1,
        )

        node_info = analyze_spectral_k(tree, data)

        # Sort by depth
        node_info.sort(key=lambda x: (x["depth"], -x["n_leaves"]))

        print(
            f"\n{'Node':<8} {'Depth':>5} {'nLeaf':>5} {'nInt':>5} {'nTot':>5} "
            f"{'dAct':>5} {'erank_LO':>9} {'erank_LI':>9} {'erank_PW':>9} "
            f"{'maxChild':>9} {'k_JL':>5} {'edge_k':>7} {'sib_k':>6}"
        )
        print("-" * 120)

        for info in node_info:
            p_k = pipeline_dims.get(info["node"], "?")
            s_k = sibling_dims.get(info["node"], "?")
            print(
                f"{info['node']:<8} {info['depth']:>5} {info['n_leaves']:>5} "
                f"{info['n_internal']:>5} {info['n_total']:>5} {info['d_active']:>5} "
                f"{info['erank_leaves_only']:>9.1f} {info['erank_leaves+internal']:>9.1f} "
                f"{str(info['erank_pooled_within'] or 'N/A'):>9} "
                f"{str(info['max_child_erank'] or 'N/A'):>9} "
                f"{info['k_jl']:>5} {str(p_k):>7} {str(s_k):>6}"
            )

        # Show eigenvalue spectrum at root
        root = tree.graph.get("root", list(nx.topological_sort(tree))[0])
        root_info = next((i for i in node_info if i["node"] == root), None)
        if root_info:
            print(f"\n  ROOT ANALYSIS ({root}):")
            print(f"    Edge spectral_k (pipeline)     = {pipeline_dims.get(root, '?')}")
            print(f"    Sibling spectral_k (pooled PW) = {sibling_dims.get(root, '?')}")
            print(f"    Effective rank (leaves only)    = {root_info['erank_leaves_only']}")
            print(f"    Effective rank (leaves+internal)= {root_info['erank_leaves+internal']}")
            print(f"    Effective rank (pooled within)  = {root_info['erank_pooled_within']}")
            print(f"    Max child erank                = {root_info['max_child_erank']}")
            print(f"    JL dimension                   = {root_info['k_jl']}")
            print(f"    Children eranks: {root_info['children_eranks']}")
            print(
                f"    Data matrix: {root_info['n_leaves']} leaves + {root_info['n_internal']} "
                f"internal = {root_info['n_total']} rows x {root_info['d_active']} active features"
            )
            print(
                f"    Rank bound: min(n_total, d_active) = {min(root_info['n_total'], root_info['d_active'])}"
            )
            if root_info["n_internal"] > 0:
                print("    ! Internal node distributions are convex combinations of leaf data")
                print("      -> they do NOT increase rank, but shift mean toward global average")
                print(
                    "      -> this can DECREASE effective rank by concentrating variance in top PCs"
                )

        # Sibling test comparison at all binary nodes (include sibling_k column)
        sibling_test_comparison(tree, data, node_info, sibling_dims)

    # --- End-to-end pipeline K comparison ---
    print("\n" + "#" * 90)
    print("# END-TO-END PIPELINE COMPARISON: decompose() with sibling pooled-within dims")
    print("#" * 90)
    from sklearn.metrics import adjusted_rand_score

    for case in test_cases:
        params = case.get("params", case)
        true_k_param = params.get("k", params.get("n_clusters", "?"))
        tree, data, true_labels, true_k = build_tree_and_data(case)
        result = tree.decompose(
            leaf_data=data,
            alpha_local=config.ALPHA_LOCAL,
            sibling_alpha=config.SIBLING_ALPHA,
        )
        found_k = result["num_clusters"]
        # Build per-sample mapping: sample_id -> cluster_index
        sample_to_cluster = {}
        for cid, cinfo in result["cluster_assignments"].items():
            for leaf in cinfo["leaves"]:
                sample_to_cluster[leaf] = cid
        # Match sample names to true labels
        sample_names = list(data.index)
        pred = [sample_to_cluster.get(s, -1) for s in sample_names]
        true = list(true_labels)
        ari = adjusted_rand_score(true, pred)
        status = "OK" if found_k == true_k else "MISS"
        print(
            f"  {case['name']:25s}  true_K={true_k_param}  found_K={found_k}  ARI={ari:.3f}  {status}"
        )


if __name__ == "__main__":
    main()
    main()
