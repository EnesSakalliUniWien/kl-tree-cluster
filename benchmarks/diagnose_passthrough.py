#!/usr/bin/env python3
"""
Diagnostic: Gate 3 failures blocking descendant signal.

For each benchmark subset case, runs the full KL decomposition pipeline, then
inspects the annotated stats_df to identify internal nodes where:
  - Gates 1+2 PASS (binary structure, at least one child diverges from parent)
  - Gate 3 FAILS (siblings declared "same" or test skipped)
  - BUT some descendant node has Sibling_BH_Different == True

These are "blocked" nodes — the greedy top-down DFS stops here, even though
deeper structure exists.  A pass-through strategy would continue the DFS
past these nodes.

Additionally, the script simulates what pass-through traversal would produce
(continuing DFS past blocked nodes instead of stopping) and compares the
cluster count and ARI against the current (v1) result.

Output: per-case table + summary statistics.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Suppress noisy warnings during batch runs

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

os.environ.setdefault("KL_TE_N_JOBS", "1")


# ── Benchmark subset (same as run_subset.py) ────────────────────────
SUBSET_NAMES = {
    "gauss_clear_small",
    "gauss_clear_large",
    "gauss_moderate_3c",
    "binary_perfect_4c",
    "binary_low_noise_4c",
    "binary_moderate_6c",
    "sparse_features_72x72",
    "cat_clear_3cat_4c",
    "sbm_clear_small",
    "overlap_mod_4c_small",
    "overlap_heavy_4c_small_feat",
    "gauss_overlap_3c_small",
    "binary_2clusters",
    "binary_many_features",
    "feature_matrix_go_terms",
}


def _get_internal_nodes(tree):
    """Return set of internal (non-leaf) node IDs."""
    return {n for n in tree.nodes() if tree.out_degree(n) > 0}


def _get_descendants(tree, node):
    """Return all descendant node IDs (not just leaves)."""
    import networkx as nx

    return nx.descendants(tree, node)


def _evaluate_gates_for_diagnostic(tree, stats_df, node):
    """Evaluate Gates 1-3 for a single node. Returns dict with gate results."""
    children = list(tree.successors(node))

    result = {
        "node": node,
        "n_children": len(children),
        "gate1_pass": len(children) == 2,
        "gate2_pass": False,
        "gate3_pass": False,
        "gate3_skipped": False,
        "left_child": None,
        "right_child": None,
        "left_diverges": None,
        "right_diverges": None,
        "sibling_different": None,
        "sibling_p_value": None,
    }

    if len(children) != 2:
        return result

    left, right = children
    result["left_child"] = left
    result["right_child"] = right

    # Gate 2: child-parent divergence
    left_sig = (
        stats_df.loc[left, "Child_Parent_Divergence_Significant"]
        if left in stats_df.index
        else False
    )
    right_sig = (
        stats_df.loc[right, "Child_Parent_Divergence_Significant"]
        if right in stats_df.index
        else False
    )
    result["left_diverges"] = bool(left_sig)
    result["right_diverges"] = bool(right_sig)
    result["gate2_pass"] = bool(left_sig or right_sig)

    if not result["gate2_pass"]:
        return result

    # Gate 3: sibling divergence
    if node in stats_df.index:
        skipped = (
            bool(stats_df.loc[node, "Sibling_Divergence_Skipped"])
            if "Sibling_Divergence_Skipped" in stats_df.columns
            else False
        )
        different = (
            bool(stats_df.loc[node, "Sibling_BH_Different"])
            if "Sibling_BH_Different" in stats_df.columns
            else False
        )
        result["gate3_skipped"] = skipped
        result["sibling_different"] = different
        result["gate3_pass"] = different and not skipped

        if "Sibling_Divergence_P_Value_Corrected" in stats_df.columns:
            pval = stats_df.loc[node, "Sibling_Divergence_P_Value_Corrected"]
            result["sibling_p_value"] = float(pval) if pd.notna(pval) else None

    return result


def _has_descendant_with_signal(tree, stats_df, node):
    """Check if any descendant internal node has Sibling_BH_Different == True."""
    descendants = _get_descendants(tree, node)
    internal_descendants = [d for d in descendants if tree.out_degree(d) > 0]

    signal_nodes = []
    for desc in internal_descendants:
        if desc in stats_df.index:
            if "Sibling_BH_Different" in stats_df.columns:
                if bool(stats_df.loc[desc, "Sibling_BH_Different"]):
                    # Also check it's not skipped
                    skipped = (
                        bool(stats_df.loc[desc, "Sibling_Divergence_Skipped"])
                        if "Sibling_Divergence_Skipped" in stats_df.columns
                        else False
                    )
                    if not skipped:
                        signal_nodes.append(desc)

    return signal_nodes


def _simulate_passthrough(tree, stats_df):
    """Simulate pass-through traversal and return cluster leaf sets.

    Like v1, but when Gates 1+2 pass and Gate 3 fails, check if any descendant
    has Sibling_BH_Different==True (AND not skipped). If yes, continue DFS
    instead of stopping.
    """

    root = next(n for n, deg in tree.in_degree() if deg == 0)
    descendant_leaf_sets = tree.compute_descendant_sets(use_labels=True)

    nodes_to_visit = [root]
    processed = set()
    final_leaf_sets = []

    while nodes_to_visit:
        node = nodes_to_visit.pop()
        if node in processed:
            continue
        processed.add(node)

        gate_result = _evaluate_gates_for_diagnostic(tree, stats_df, node)

        if gate_result["gate1_pass"] and gate_result["gate2_pass"] and gate_result["gate3_pass"]:
            # SPLIT — push children
            children = list(tree.successors(node))
            nodes_to_visit.append(children[1])  # right
            nodes_to_visit.append(children[0])  # left

        elif (
            gate_result["gate1_pass"]
            and gate_result["gate2_pass"]
            and not gate_result["gate3_pass"]
        ):
            # Gates 1+2 pass but Gate 3 fails — check for descendant signal
            signal_below = _has_descendant_with_signal(tree, stats_df, node)
            if signal_below:
                # PASS-THROUGH: continue DFS
                children = list(tree.successors(node))
                nodes_to_visit.append(children[1])
                nodes_to_visit.append(children[0])
            else:
                # No signal below — MERGE
                final_leaf_sets.append(set(descendant_leaf_sets.get(node, set())))
        else:
            # Gates 1 or 2 fail — MERGE
            final_leaf_sets.append(set(descendant_leaf_sets.get(node, set())))

    return final_leaf_sets


def _compute_ari(labels_true, labels_pred, sample_index):
    """Compute ARI between two label arrays."""
    from sklearn.metrics import adjusted_rand_score

    if labels_true is None or len(labels_true) == 0:
        return float("nan")
    return adjusted_rand_score(labels_true, labels_pred)


def run_diagnostic_for_case(tc):
    """Run full diagnostic for one test case. Returns dict of results."""
    case_name = tc.get("name", "unnamed")
    true_k = tc.get("n_clusters", 0)

    try:
        # Generate data
        data_t, y_t, x_original, meta = generate_case_data(tc)

        # Check for precomputed distances (SBM cases)
        precomputed = meta.get("precomputed_distance_condensed")
        if precomputed is not None:
            distance_condensed = np.asarray(precomputed, dtype=float)
        else:
            distance_condensed = pdist(data_t.values, metric=config.TREE_DISTANCE_METRIC)

        # Build tree
        Z = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data_t.index.tolist())

        # Run decomposition (this populates stats_df)
        decomp = tree.decompose(
            leaf_data=data_t,
            alpha_local=config.EDGE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )

        stats_df = tree.stats_df
        v1_k = decomp["num_clusters"]

        # Analyze all internal nodes
        internal_nodes = _get_internal_nodes(tree)
        n_internal = len(internal_nodes)

        blocked_nodes = []  # Gates 1+2 pass, Gate 3 fails
        blocked_with_signal = []  # Above + descendant has signal

        for node in internal_nodes:
            gate_result = _evaluate_gates_for_diagnostic(tree, stats_df, node)
            if (
                gate_result["gate1_pass"]
                and gate_result["gate2_pass"]
                and not gate_result["gate3_pass"]
            ):
                signal_below = _has_descendant_with_signal(tree, stats_df, node)
                blocked_nodes.append(
                    {
                        **gate_result,
                        "descendant_signal_nodes": signal_below,
                        "n_descendant_signals": len(signal_below),
                    }
                )
                if signal_below:
                    blocked_with_signal.append(
                        {
                            **gate_result,
                            "descendant_signal_nodes": signal_below,
                            "n_descendant_signals": len(signal_below),
                        }
                    )

        # Count nodes where Gate 3 passes (actual splits in v1)
        n_gate3_pass = sum(
            1
            for node in internal_nodes
            if _evaluate_gates_for_diagnostic(tree, stats_df, node)["gate3_pass"]
        )

        # Simulate pass-through
        pt_leaf_sets = _simulate_passthrough(tree, stats_df)
        pt_k = len(pt_leaf_sets)

        # Compute labels for ARI comparison
        from sklearn.metrics import adjusted_rand_score

        # v1 labels
        v1_assignments = decomp.get("cluster_assignments", {})
        v1_label_map = {}
        for cluster_id, info in v1_assignments.items():
            for leaf in info.get("leaves", []):
                v1_label_map[leaf] = cluster_id

        # pass-through labels
        pt_label_map = {}
        for i, leaf_set in enumerate(pt_leaf_sets):
            for leaf in leaf_set:
                pt_label_map[leaf] = i

        # Compute ARI against true labels if available
        sample_names = data_t.index.tolist()
        v1_ari = float("nan")
        pt_ari = float("nan")
        has_truth = (
            y_t is not None
            and len(y_t) > 0
            and not np.all(np.isnan(np.asarray(y_t, dtype=float)))
        )
        if has_truth:
            v1_labels = [v1_label_map.get(s, -1) for s in sample_names]
            pt_labels = [pt_label_map.get(s, -1) for s in sample_names]
            v1_ari = adjusted_rand_score(y_t, v1_labels)
            pt_ari = adjusted_rand_score(y_t, pt_labels)

        return {
            "case": case_name,
            "true_k": true_k,
            "v1_k": v1_k,
            "pt_k": pt_k,
            "v1_ari": v1_ari,
            "pt_ari": pt_ari,
            "n_internal": n_internal,
            "n_gate3_pass": n_gate3_pass,
            "n_blocked": len(blocked_nodes),
            "n_blocked_with_signal": len(blocked_with_signal),
            "blocked_details": blocked_with_signal,
            "status": "ok",
        }

    except Exception as e:
        return {
            "case": case_name,
            "true_k": true_k,
            "v1_k": -1,
            "pt_k": -1,
            "v1_ari": float("nan"),
            "pt_ari": float("nan"),
            "n_internal": 0,
            "n_gate3_pass": 0,
            "n_blocked": 0,
            "n_blocked_with_signal": 0,
            "blocked_details": [],
            "status": f"ERROR: {e}",
        }


def main():
    all_cases = get_default_test_cases()
    subset = [c for c in all_cases if c["name"] in SUBSET_NAMES]
    print(f"Pass-Through Diagnostic — {len(subset)} cases")
    print("=" * 100)
    print()

    results = []
    for i, tc in enumerate(subset, 1):
        name = tc.get("name", "unnamed")
        print(f"  [{i:2d}/{len(subset)}] {name:<36s} ... ", end="", flush=True)
        result = run_diagnostic_for_case(tc)
        results.append(result)

        status = result["status"]
        if status == "ok":
            flag = ""
            if result["n_blocked_with_signal"] > 0:
                flag = " *** BLOCKED SIGNAL ***"
            print(
                f"v1 K={result['v1_k']:>2d}  pt K={result['pt_k']:>2d}  "
                f"blocked={result['n_blocked']:>2d}  "
                f"blocked+signal={result['n_blocked_with_signal']:>2d}  "
                f"v1 ARI={result['v1_ari']:.3f}  pt ARI={result['pt_ari']:.3f}"
                f"{flag}"
            )
        else:
            print(f"  {status}")

    # Summary table
    print()
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print()
    header = (
        f"{'Case':<36s} {'True':>4s} {'v1 K':>4s} {'pt K':>4s} "
        f"{'v1ARI':>6s} {'ptARI':>6s} {'Δ ARI':>6s} "
        f"{'intern':>6s} {'G3yes':>5s} {'block':>5s} {'blk+s':>5s}"
    )
    print(header)
    print("-" * len(header))

    n_changed = 0
    n_improved = 0
    n_degraded = 0
    n_blocked_signal_total = 0

    for r in sorted(results, key=lambda x: x["case"]):
        if r["status"] != "ok":
            print(f"{r['case']:<36s} {'ERR':>4s}")
            continue

        true_str = f"{r['true_k']:>4d}" if r["true_k"] and r["true_k"] > 0 else " N/A"
        ari_v1 = f"{r['v1_ari']:.3f}" if not np.isnan(r["v1_ari"]) else "  N/A"
        ari_pt = f"{r['pt_ari']:.3f}" if not np.isnan(r["pt_ari"]) else "  N/A"

        delta_ari_val = (
            r["pt_ari"] - r["v1_ari"]
            if not (np.isnan(r["v1_ari"]) or np.isnan(r["pt_ari"]))
            else float("nan")
        )
        delta_str = f"{delta_ari_val:>+.3f}" if not np.isnan(delta_ari_val) else "  N/A"

        marker = ""
        if r["v1_k"] != r["pt_k"]:
            marker = " *"
            n_changed += 1
            if not np.isnan(delta_ari_val):
                if delta_ari_val > 0.01:
                    n_improved += 1
                elif delta_ari_val < -0.01:
                    n_degraded += 1

        if r["n_blocked_with_signal"] > 0:
            n_blocked_signal_total += 1

        print(
            f"{r['case']:<36s} {true_str} {r['v1_k']:>4d} {r['pt_k']:>4d} "
            f"{ari_v1:>6s} {ari_pt:>6s} {delta_str:>6s} "
            f"{r['n_internal']:>6d} {r['n_gate3_pass']:>5d} {r['n_blocked']:>5d} {r['n_blocked_with_signal']:>5d}"
            f"{marker}"
        )

    print()
    print("Legend:")
    print("  v1 K    = clusters found by current DFS (stops at Gate 3 failure)")
    print("  pt K    = clusters found by pass-through (continues past blocked nodes)")
    print("  intern  = total internal nodes in tree")
    print("  G3yes   = nodes where Gate 3 passes (siblings declared different)")
    print("  block   = nodes where Gates 1+2 pass but Gate 3 fails")
    print("  blk+s   = blocked nodes with descendant signal (would trigger pass-through)")
    print("  *       = case where pass-through changes K")
    print()
    print(f"Cases with blocked signal: {n_blocked_signal_total}/{len(results)}")
    print(f"Cases where K changes:     {n_changed}/{len(results)}")
    print(f"  Improved (ΔARI > +0.01): {n_improved}")
    print(f"  Degraded (ΔARI < -0.01): {n_degraded}")

    # Detailed report for blocked-with-signal cases
    blocked_cases = [r for r in results if r["n_blocked_with_signal"] > 0]
    if blocked_cases:
        print()
        print("=" * 100)
        print("DETAILED BLOCKED-SIGNAL NODES")
        print("=" * 100)
        for r in blocked_cases:
            print(
                f"\n  Case: {r['case']}  (true K={r['true_k']}, v1 K={r['v1_k']}, pt K={r['pt_k']})"
            )
            for detail in r["blocked_details"]:
                node = detail["node"]
                n_sigs = detail["n_descendant_signals"]
                sig_nodes = detail["descendant_signal_nodes"][:5]  # show first 5
                pval = detail.get("sibling_p_value")
                pval_str = f"p={pval:.4f}" if pval is not None else "p=N/A"
                print(
                    f"    Node {node}:  Gate3 {pval_str}  |  "
                    f"{n_sigs} descendant(s) with signal: {sig_nodes}"
                    f"{'...' if n_sigs > 5 else ''}"
                )


if __name__ == "__main__":
    main()
