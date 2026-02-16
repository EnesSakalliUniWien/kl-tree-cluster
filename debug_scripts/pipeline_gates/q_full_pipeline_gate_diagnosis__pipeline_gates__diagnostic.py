"""
Purpose: Diagnose the full pipeline: edge test → sibling test → _should_split gates.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/pipeline_gates/q_full_pipeline_gate_diagnosis__pipeline_gates__diagnostic.py
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
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    annotate_sibling_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_adjusted_wald import (
    annotate_sibling_divergence_adjusted,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_calibrated_test import (
    annotate_sibling_divergence_cousin,
)


# ── Test data ──────────────────────────────────────────────────────────────


def make_trivial_2c(n_per=50, p=100, seed=42):
    """Trivially separable: cluster A ≈ all 1s, cluster B ≈ all 0s."""
    rng = np.random.default_rng(seed)
    X = np.vstack([
        rng.binomial(1, 0.9, size=(n_per, p)),
        rng.binomial(1, 0.1, size=(n_per, p)),
    ])
    labels = np.array([0] * n_per + [1] * n_per)
    df = pd.DataFrame(X, index=[f"S{i}" for i in range(2 * n_per)],
                       columns=[f"F{j}" for j in range(p)])
    return df, labels


def make_block_4c(n_per=50, p=200, seed=42):
    """Block-diagonal: 4 clusters, each owns p/4 features."""
    rng = np.random.default_rng(seed)
    block = p // 4
    rows, labels = [], []
    for c in range(4):
        X_c = rng.binomial(1, 0.1, size=(n_per, p))
        X_c[:, c * block : (c + 1) * block] = rng.binomial(
            1, 0.9, size=(n_per, block)
        )
        rows.append(X_c)
        labels.extend([c] * n_per)
    X = np.vstack(rows)
    df = pd.DataFrame(X, index=[f"S{i}" for i in range(4 * n_per)],
                       columns=[f"F{j}" for j in range(p)])
    return df, np.array(labels)


# ── Gate tracing ───────────────────────────────────────────────────────────


def trace_should_split_v1(decomp: TreeDecomposition, node: str) -> dict:
    """Manually trace the three gates in _should_split (v1).

    Returns a dict with each gate's outcome and the final decision.
    """
    result = {
        "node": node,
        "version": "v1",
        "gate1_binary": None,
        "gate2_edge_left": None,
        "gate2_edge_right": None,
        "gate2_pass": None,
        "gate3_sibling_different": None,
        "gate3_sibling_skipped": None,
        "gate3_pass": None,
        "final": None,
    }

    children = decomp._children.get(node, [])
    if len(children) != 2:
        result["gate1_binary"] = False
        result["final"] = False
        return result
    result["gate1_binary"] = True

    left, right = children
    left_div = decomp._local_significant.get(left)
    right_div = decomp._local_significant.get(right)
    result["gate2_edge_left"] = left_div
    result["gate2_edge_right"] = right_div
    result["gate2_pass"] = bool(left_div or right_div) if left_div is not None and right_div is not None else None

    if not result["gate2_pass"]:
        result["final"] = False
        return result

    is_different = decomp._sibling_different.get(node)
    is_skipped = decomp._sibling_skipped.get(node, False)
    result["gate3_sibling_different"] = is_different
    result["gate3_sibling_skipped"] = is_skipped

    if is_skipped:
        result["gate3_pass"] = False
        result["final"] = False
    elif is_different is None:
        result["gate3_pass"] = None
        result["final"] = None  # error state
    else:
        result["gate3_pass"] = bool(is_different)
        result["final"] = bool(is_different)

    return result


def trace_should_split_v2(decomp: TreeDecomposition, node: str) -> dict:
    """Manually trace the gates in _should_split_v2.

    Same gates as v1 but with localization on gate3 pass.
    """
    result = trace_should_split_v1(decomp, node)
    result["version"] = "v2"
    # v2 is identical up to localization — same gate logic
    # If v1 says split, v2 would also split (plus run localization)
    return result


def _format_gate(val):
    """Format a gate value for display."""
    if val is None:
        return "N/A"
    if val is True:
        return "PASS"
    if val is False:
        return "FAIL"
    return str(val)


# ── Diagnostic runner ──────────────────────────────────────────────────────


SIBLING_METHODS = {
    "wald": annotate_sibling_divergence,
    "cousin_adjusted_wald": annotate_sibling_divergence_adjusted,
    "cousin_ftest": annotate_sibling_divergence_cousin,
}


def diagnose(name: str, data: pd.DataFrame, true_labels: np.ndarray):
    """Full diagnostic trace for a single case."""
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

    root = tree.root()
    n_internal = sum(1 for n in tree.nodes if list(tree.successors(n)))
    n_leaves = sum(1 for n in tree.nodes if not list(tree.successors(n)))
    print(f"\n  Tree: {n_internal} internal, {n_leaves} leaves, root={root}")

    # 2. Branch lengths
    bl_edges = [
        tree.edges[u, v]["branch_length"]
        for u, v in tree.edges()
        if "branch_length" in tree.edges[u, v]
    ]
    if bl_edges:
        print(f"  BL: n={len(bl_edges)}, min={min(bl_edges):.4f}, "
              f"med={np.median(bl_edges):.4f}, max={max(bl_edges):.4f}, "
              f"mean={np.mean(bl_edges):.4f}")
    else:
        print("  BL: NONE")

    # 3. Edge test (shared across all sibling methods)
    # Use tree.stats_df which has leaf_count and other required columns
    results_df = tree.stats_df.copy() if tree.stats_df is not None else pd.DataFrame(index=list(tree.nodes))
    results_df = annotate_child_parent_divergence(
        tree, results_df, significance_level_alpha=config.SIGNIFICANCE_ALPHA
    )

    edge_sig = results_df["Child_Parent_Divergence_Significant"]
    n_edge_sig = int(edge_sig.sum())
    print(f"\n  EDGE TEST: {n_edge_sig}/{len(results_df)} nodes edge-significant")

    # Root children
    root_children = list(tree.successors(root))
    for ch in root_children:
        sig = edge_sig.get(ch, "?")
        pval = results_df.loc[ch].get("Child_Parent_Divergence_P_Value", "?")
        n_desc = len(list(tree.successors(ch))) if list(tree.successors(ch)) else 0
        print(f"    Root→{ch}: edge_sig={sig}, p={pval}, descendants={n_desc}")

    # Top 5 smallest edge p-values
    if "Child_Parent_Divergence_P_Value" in results_df.columns:
        edge_pvals = results_df["Child_Parent_Divergence_P_Value"].dropna().sort_values()
        print(f"  Top 5 edge p-values:")
        for node, pval in edge_pvals.head(5).items():
            print(f"    {node}: p={pval:.4e}, sig={edge_sig.get(node, '?')}")

    # 4. For each sibling method: annotate, build TreeDecomposition, trace gates
    for method_name, annotate_fn in SIBLING_METHODS.items():
        print(f"\n  --- {method_name.upper()} ---")
        df_m = results_df.copy()
        df_m = annotate_fn(tree, df_m, significance_level_alpha=config.SIBLING_ALPHA)

        # Print calibration audit if available
        audit = df_m.attrs.get("sibling_divergence_audit", {})
        if audit:
            cal_m = audit.get("calibration_method", audit.get("test_method", "?"))
            c_hat = audit.get("global_c_hat", None)
            n_cal = audit.get("calibration_n", audit.get("total_tests", 0))
            null_pairs = audit.get("null_like_pairs", "")
            focal = audit.get("focal_pairs", "")
            c_str = f"ĉ={c_hat:.3f}" if c_hat is not None else ""
            extra = f", null={null_pairs}, focal={focal}" if null_pairs else ""
            print(f"    Audit: {cal_m}, n={n_cal} {c_str}{extra}")
            diag = audit.get("diagnostics", {})
            if diag.get("beta"):
                print(f"    Regression: β={[f'{b:.3f}' for b in diag['beta']]}, "
                      f"R²={diag.get('r_squared', 0):.3f}")

        # Sibling summary
        _print_sibling_summary(df_m, method_name)

        # Build TreeDecomposition to access _should_split internals
        config.SIBLING_TEST_METHOD = method_name
        decomp = TreeDecomposition(
            tree=tree,
            results_df=df_m,
            alpha_local=config.SIGNIFICANCE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
            use_signal_localization=False,
        )

        # Trace gates on root and its children (top 2 levels)
        print(f"    Gate trace (v1 = _should_split, v2 = _should_split_v2):")
        nodes_to_trace = [root]
        for ch in root_children:
            if list(tree.successors(ch)):
                nodes_to_trace.append(ch)

        for node in nodes_to_trace:
            g1 = trace_should_split_v1(decomp, node)
            print(f"      {node} [v1]: "
                  f"G1(binary)={_format_gate(g1['gate1_binary'])} "
                  f"G2(edge L={_format_gate(g1['gate2_edge_left'])}, "
                  f"R={_format_gate(g1['gate2_edge_right'])})={_format_gate(g1['gate2_pass'])} "
                  f"G3(sib_diff={_format_gate(g1['gate3_sibling_different'])}, "
                  f"skip={_format_gate(g1['gate3_sibling_skipped'])})={_format_gate(g1['gate3_pass'])} "
                  f"=> SPLIT={_format_gate(g1['final'])}")

        # Also show v2 localization toggle
        decomp_v2 = TreeDecomposition(
            tree=tree,
            results_df=df_m,
            alpha_local=config.SIGNIFICANCE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
            use_signal_localization=True,
        )
        for node in nodes_to_trace:
            g2 = trace_should_split_v2(decomp_v2, node)
            # v2 gate logic is identical to v1; difference is localization after pass
            # Just confirm they agree
            if g2["final"] != trace_should_split_v1(decomp, node)["final"]:
                print(f"      {node} [v2]: DIFFERS from v1!")

        # Full decompose
        config.SIBLING_TEST_METHOD = method_name
        tree2 = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
        result = tree2.decompose(leaf_data=data)

        # Extract cluster assignments — handle both dict-of-label and dict-of-dict formats
        ca_raw = result["cluster_assignments"]
        if isinstance(ca_raw, dict):
            first_val = next(iter(ca_raw.values()), None)
            if isinstance(first_val, dict):
                # {cluster_id: {"leaves": [...], ...}} format
                pred_map = {}
                for cid, info in ca_raw.items():
                    for leaf in info.get("leaves", []):
                        pred_map[leaf] = cid
                pred = np.array([pred_map.get(s, -1) for s in data.index])
            else:
                # {leaf: cluster_id} format
                pred = np.array([ca_raw.get(s, -1) for s in data.index])
        else:
            pred = np.zeros(len(data), dtype=int)

        k_found = len(set(pred))
        ari = adjusted_rand_score(true_labels, pred) if true_k > 1 else 0.0
        print(f"    RESULT: K={k_found}, ARI={ari:.4f}")

    config.SIBLING_TEST_METHOD = "wald"  # reset


def _print_sibling_summary(df: pd.DataFrame, label: str):
    """Print sibling test summary from annotated DataFrame."""
    if "Sibling_BH_Different" not in df.columns:
        print(f"    {label}: No sibling columns found")
        return

    sib_diff = df["Sibling_BH_Different"]
    sib_same = df["Sibling_BH_Same"]
    sib_skip = df.get("Sibling_Divergence_Skipped", pd.Series(dtype=bool))

    n_diff = int(sib_diff.sum()) if not sib_diff.isna().all() else 0
    n_same = int(sib_same.sum()) if not sib_same.isna().all() else 0
    n_skip = int(sib_skip.sum()) if not sib_skip.isna().all() else 0

    print(f"    Sibling: different={n_diff}, same={n_same}, skipped={n_skip}")

    # Top 5 tested p-values
    if "Sibling_Divergence_P_Value" in df.columns:
        tested = df["Sibling_Divergence_P_Value"].dropna()
        if len(tested) > 0:
            sorted_p = tested.sort_values()
            print(f"    Top 5 sibling p-values (of {len(tested)} tested):")
            for node, pval in sorted_p.head(5).items():
                diff = sib_diff.get(node, "?")
                stat = df.loc[node].get("Sibling_Test_Statistic", "?")
                dof = df.loc[node].get("Sibling_Degrees_of_Freedom", "?")
                meth = df.loc[node].get("Sibling_Test_Method", "")
                print(f"      {node}: T={stat:.2f}, k={dof}, p={pval:.4e}, "
                      f"diff={diff} [{meth}]"
                      if isinstance(stat, float) else
                      f"      {node}: T={stat}, k={dof}, p={pval:.4e}, diff={diff}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")

    print(f"Config: metric={config.TREE_DISTANCE_METRIC}, "
          f"linkage={config.TREE_LINKAGE_METHOD}")
    print(f"Config: sig_alpha={config.SIGNIFICANCE_ALPHA}, "
          f"sib_alpha={config.SIBLING_ALPHA}")

    data_2c, labels_2c = make_trivial_2c()
    diagnose("trivial_2c", data_2c, labels_2c)

    data_4c, labels_4c = make_block_4c()
    diagnose("block_4c", data_4c, labels_4c)

    # Benchmark cases
    try:
        from benchmarks.shared.cases import get_default_test_cases
        from benchmarks.shared.generators import generate_case_data

        pick = {"sparse_features_72x72", "binary_perfect_2c", "binary_perfect_4c",
                "gauss_clear_small", "gauss_moderate_3c"}
        all_cases = get_default_test_cases()
        for tc in all_cases:
            if tc["name"] in pick:
                data_df, true_labels, _, _ = generate_case_data(tc)
                diagnose(tc["name"], data_df, true_labels)
    except ImportError:
        print("\n(Benchmark cases not available)")


if __name__ == "__main__":
    main()
