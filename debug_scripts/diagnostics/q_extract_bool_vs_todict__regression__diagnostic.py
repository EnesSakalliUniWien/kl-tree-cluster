"""
Purpose: Diagnose whether the .to_dict() → extract_bool_column_dict() change
         in sibling divergence methods causes a behavioral regression.
         Also checks branch-length resolution changes.

Inputs:  Benchmark overlap cases (gauss_overlap_3c_small, overlap_heavy_4c_small_feat,
         overlap_mod_4c_small) via shared benchmark infrastructure.

Outputs: Console comparison of:
         1. sig_map contents from .to_dict() vs extract_bool_column_dict()
         2. is_null_like classification differences
         3. Calibration weight differences
         4. Final K and ARI for each path

Expected runtime: ~30-120 seconds.
How to run:
    python debug_scripts/diagnostics/q_extract_bool_vs_todict__regression__diagnostic.py
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import extract_bool_column_dict
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald import (
    _collect_weighted_pairs,
    _either_child_significant,
    _get_binary_children,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ---------------------------------------------------------------------------


def compare_sig_maps(results_df: pd.DataFrame):
    """Compare .to_dict() vs extract_bool_column_dict on
    the Child_Parent_Divergence_Significant column.

    Returns (sig_map_old, sig_map_new, diff_nodes).
    """
    col = "Child_Parent_Divergence_Significant"

    # OLD path: raw .to_dict() — as it was at HEAD
    sig_map_old = results_df[col].to_dict()

    # NEW path: extract_bool_column_dict — as we changed it
    sig_map_new = extract_bool_column_dict(results_df, col)

    diff_nodes = []
    for node in sig_map_old:
        old_val = sig_map_old[node]
        new_val = sig_map_new.get(node)

        # Check both value and type
        if old_val != new_val or type(old_val) != type(new_val):
            diff_nodes.append(
                {
                    "node": node,
                    "old_val": old_val,
                    "old_type": type(old_val).__name__,
                    "new_val": new_val,
                    "new_type": type(new_val).__name__,
                }
            )

    return sig_map_old, sig_map_new, diff_nodes


def compare_null_like_classification(tree, results_df: pd.DataFrame, sig_map_old, sig_map_new):
    """Compare is_null_like classification for each parent using old vs new sig_map.

    Returns list of dicts with differing nodes.
    """
    diffs = []
    for parent in tree.nodes:
        children = _get_binary_children(tree, parent)
        if children is None:
            continue
        left, right = children

        null_old = not _either_child_significant(left, right, sig_map_old)
        null_new = not _either_child_significant(left, right, sig_map_new)

        if null_old != null_new:
            diffs.append(
                {
                    "parent": parent,
                    "left": left,
                    "right": right,
                    "old_sig_L": sig_map_old.get(left),
                    "old_sig_R": sig_map_old.get(right),
                    "new_sig_L": sig_map_new.get(left),
                    "new_sig_R": sig_map_new.get(right),
                    "old_null_like": null_old,
                    "new_null_like": null_new,
                }
            )

    return diffs


def check_column_dtype_and_values(results_df: pd.DataFrame):
    """Report the raw dtype and value types in the Significant column."""
    col = "Child_Parent_Divergence_Significant"
    series = results_df[col]

    # Collect unique value types
    type_counts = {}
    for val in series:
        t = type(val).__name__
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "dtype": str(series.dtype),
        "value_types": type_counts,
        "unique_values": sorted(set(str(v) for v in series.unique())),
        "has_nan": bool(series.isna().any()),
        "n_true": int(series.sum()) if series.dtype == bool else "N/A",
    }


def check_branch_lengths(tree):
    """Report branch length statistics and any missing/invalid edges."""
    bl_edges = []
    missing_bl = 0
    zero_bl = 0
    neg_bl = 0
    total_edges = tree.number_of_edges()

    for u, v in tree.edges():
        raw = tree.edges[u, v].get("branch_length")
        if raw is None:
            missing_bl += 1
        elif raw <= 0:
            if raw == 0:
                zero_bl += 1
            else:
                neg_bl += 1
        else:
            bl_edges.append(float(raw))

    mean_bl = compute_mean_branch_length(tree)
    felsenstein_active = config.FELSENSTEIN_SCALING

    return {
        "total_edges": total_edges,
        "valid_bl": len(bl_edges),
        "missing_bl": missing_bl,
        "zero_bl": zero_bl,
        "neg_bl": neg_bl,
        "mean_bl": mean_bl,
        "min_bl": min(bl_edges) if bl_edges else None,
        "max_bl": max(bl_edges) if bl_edges else None,
        "felsenstein_active": felsenstein_active,
    }


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------


def diagnose_case(name: str, data: pd.DataFrame, true_labels: np.ndarray):
    """Full comparison for one benchmark case."""
    true_k = len(set(true_labels))
    print(f"\n{'='*80}")
    print(f"  CASE: {name}   (n={len(data)}, p={data.shape[1]}, true_K={true_k})")
    print(f"{'='*80}")

    # 1. Build tree
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)

    # 2. Branch lengths
    bl_info = check_branch_lengths(tree)
    print(
        f"\n  Branch lengths: {bl_info['valid_bl']}/{bl_info['total_edges']} valid, "
        f"missing={bl_info['missing_bl']}, zero={bl_info['zero_bl']}, neg={bl_info['neg_bl']}"
    )
    print(f"  mean_bl={bl_info['mean_bl']}, min={bl_info['min_bl']}, max={bl_info['max_bl']}")
    print(f"  FELSENSTEIN_SCALING={bl_info['felsenstein_active']}")

    # 3. Edge test
    results_df = (
        tree.stats_df.copy() if tree.stats_df is not None else pd.DataFrame(index=list(tree.nodes))
    )
    results_df = annotate_child_parent_divergence(
        tree,
        results_df,
        significance_level_alpha=config.SIGNIFICANCE_ALPHA,
    )

    # 4. Column dtype check
    col_info = check_column_dtype_and_values(results_df)
    print(f"\n  Column dtype: {col_info['dtype']}")
    print(f"  Value types: {col_info['value_types']}")
    print(f"  Unique values: {col_info['unique_values']}")
    print(f"  Has NaN: {col_info['has_nan']}")
    print(f"  n_significant: {col_info['n_true']}")

    # 5. Compare sig_maps
    sig_map_old, sig_map_new, diff_nodes = compare_sig_maps(results_df)
    if diff_nodes:
        print(f"\n  **SIG_MAP DIFFERS** in {len(diff_nodes)} nodes:")
        for d in diff_nodes[:10]:
            print(
                f"    {d['node']}: old={d['old_val']!r} ({d['old_type']}) "
                f"→ new={d['new_val']!r} ({d['new_type']})"
            )
        if len(diff_nodes) > 10:
            print(f"    ... and {len(diff_nodes) - 10} more")
    else:
        print("\n  sig_map: IDENTICAL (values match between .to_dict() and extract_bool)")

    # 6. Compare is_null_like classification
    null_diffs = compare_null_like_classification(tree, results_df, sig_map_old, sig_map_new)
    if null_diffs:
        print(f"\n  **NULL-LIKE CLASSIFICATION DIFFERS** in {len(null_diffs)} parents:")
        for d in null_diffs[:10]:
            print(
                f"    Parent {d['parent']}: "
                f"L({d['left']}) sig: old={d['old_sig_L']!r} new={d['new_sig_L']!r}, "
                f"R({d['right']}) sig: old={d['old_sig_R']!r} new={d['new_sig_R']!r} "
                f"→ null_like: old={d['old_null_like']} new={d['new_null_like']}"
            )
        if len(null_diffs) > 10:
            print(f"    ... and {len(null_diffs) - 10} more")
    else:
        print("  null_like classification: IDENTICAL")

    # 7. Compare weight computation impact
    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None
    try:
        records_new, non_binary = _collect_weighted_pairs(tree, results_df, mean_bl)
        print(
            f"\n  Weighted pairs (new): {len(records_new)} records, "
            f"{len(non_binary)} non-binary/leaf nodes"
        )
        null_new_count = sum(1 for r in records_new if r.is_null_like)
        focal_new_count = len(records_new) - null_new_count
        print(f"  null_like={null_new_count}, focal={focal_new_count}")

        # Compare weights
        if records_new:
            weights = [r.weight for r in records_new]
            print(
                f"  Weights: min={min(weights):.4e}, max={max(weights):.4e}, "
                f"mean={np.mean(weights):.4e}"
            )
    except Exception as e:
        print(f"\n  _collect_weighted_pairs (new) FAILED: {e}")

    # 8. Full decompose comparison
    print("\n  --- Full decompose comparison ---")

    # With current code (uses extract_bool_column_dict)
    config.SIBLING_TEST_METHOD = "cousin_weighted_wald"
    tree_cur = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    result_cur = tree_cur.decompose(leaf_data=data)
    k_cur = result_cur["num_clusters"]
    pred_cur = _extract_pred(result_cur, data)
    ari_cur = adjusted_rand_score(true_labels, pred_cur) if true_k > 1 else 0.0

    print(f"  Current (extract_bool):  K={k_cur}, ARI={ari_cur:.4f}")

    # Check if we have any calibration info
    stats = tree_cur.stats_df
    if stats is not None:
        audit = stats.attrs.get("sibling_divergence_audit", {})
        if audit:
            print(
                f"    Audit: method={audit.get('calibration_method', '?')}, "
                f"ĉ={audit.get('global_inflation_factor', '?')}, "
                f"n_cal={audit.get('calibration_n', '?')}"
            )
            diag = audit.get("diagnostics", {})
            if diag:
                print(f"    Diagnostics: {diag}")

    # Summary
    print(f"\n  SUMMARY: K={k_cur} (true={true_k}), ARI={ari_cur:.4f}")
    if diff_nodes or null_diffs:
        print("  *** BEHAVIORAL DIFFERENCE DETECTED ***")
        print(f"     sig_map diffs: {len(diff_nodes)}")
        print(f"     null_like diffs: {len(null_diffs)}")
    else:
        print("  No behavioral difference from extract_bool change")


def _extract_pred(result, data):
    """Extract prediction array from decompose result."""
    ca_raw = result["cluster_assignments"]
    if isinstance(ca_raw, dict):
        first_val = next(iter(ca_raw.values()), None)
        if isinstance(first_val, dict):
            pred_map = {}
            for cid, info in ca_raw.items():
                for leaf in info.get("leaves", []):
                    pred_map[leaf] = cid
            return np.array([pred_map.get(s, -1) for s in data.index])
        else:
            return np.array([ca_raw.get(s, -1) for s in data.index])
    return np.zeros(len(data), dtype=int)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")

    print(f"Config: metric={config.TREE_DISTANCE_METRIC}, " f"linkage={config.TREE_LINKAGE_METHOD}")
    print(f"Config: sig_alpha={config.SIGNIFICANCE_ALPHA}, " f"sib_alpha={config.SIBLING_ALPHA}")
    print(f"Config: FELSENSTEIN_SCALING={config.FELSENSTEIN_SCALING}")
    print(f"Config: SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}")
    print(f"Config: EIGENVALUE_WHITENING={config.EIGENVALUE_WHITENING}")

    pick = {
        "gauss_overlap_3c_small",
        "overlap_heavy_4c_small_feat",
        "overlap_mod_4c_small",
        # Include some known-good cases for comparison
        "trivial_2c",
        "block_4c",
        "gauss_clear_small",
    }

    try:
        from benchmarks.shared.cases import get_default_test_cases
        from benchmarks.shared.generators import generate_case_data

        all_cases = get_default_test_cases()
        for tc in all_cases:
            if tc["name"] in pick:
                data_df, true_labels, _, _ = generate_case_data(tc)
                diagnose_case(tc["name"], data_df, true_labels)
    except ImportError as e:
        print(f"\nBenchmark cases not available: {e}")
        return

    # Reset config
    config.SIBLING_TEST_METHOD = "cousin_weighted_wald"


if __name__ == "__main__":
    main()
