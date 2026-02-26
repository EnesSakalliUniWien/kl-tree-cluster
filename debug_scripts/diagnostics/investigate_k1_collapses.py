#!/usr/bin/env python3
"""Investigate why 7 specific cases collapse to K=1 under soft edge calibration.

For each case, runs decomposition WITH and WITHOUT edge calibration, then
traces the root node's gate statistics to pinpoint the failure mechanism:

- Gate 2: raw T, weight w_i, ĉ_local, ĉ_soft, deflated T, raw p, deflated p
- Gate 3: sibling BH different, sibling p-value
- Final decision: split or merge

The 7 cases (all K=1 under soft_local, non-K=1 under baseline):
  1. binary_perfect_8c     (True K=8, baseline K=6)
  2. cat_highcard_20cat_4c  (True K=4, baseline K=4)
  3. binary_perfect_4c      (True K=4, baseline K=2)
  4. overlap_heavy_4c_med_feat (True K=4, baseline K=3)
  5. overlap_heavy_8c_large_feat (True K=8, baseline K=3)
  6. overlap_extreme_6c_highd (True K=6, baseline K=11)
  7. overlap_extreme_4c      (True K=4, baseline K=32)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_calibration import (
    EdgeCalibrationModel,
    compute_edge_null_weight,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

# The 7 cases that collapse to K=1 under soft_local calibration
TARGET_CASES = [
    "binary_perfect_8c",
    "cat_highcard_20cat_4c",
    "binary_perfect_4c",
    "overlap_heavy_4c_med_feat",
    "overlap_heavy_8c_large_feat",
    "overlap_extreme_6c_highd",
    "overlap_extreme_4c",
]


def find_case_config(name: str) -> dict | None:
    """Find a test case config by name from the default benchmark suite."""
    for case in get_default_test_cases():
        if case.get("name") == name:
            return case
    return None


def run_decompose(data_df: pd.DataFrame, edge_calibration: bool) -> dict:
    """Run decomposition with or without edge calibration."""
    orig = config.EDGE_CALIBRATION
    try:
        config.EDGE_CALIBRATION = edge_calibration
        Z = linkage(
            pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
            method=config.TREE_LINKAGE_METHOD,
        )
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
        results = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.SIGNIFICANCE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )
        return {"tree": tree, "results": results, "stats_df": tree.stats_df}
    finally:
        config.EDGE_CALIBRATION = orig


def trace_sibling_calibration(stats_df: pd.DataFrame, label: str):
    """Print sibling (Gate 3) calibration audit info."""
    audit = stats_df.attrs.get("sibling_divergence_audit", {})
    if not audit:
        print(f"    [{label}] Sibling calibration: NO AUDIT DATA")
        return
    c_hat = audit.get("global_c_hat", "?")
    method = audit.get("calibration_method", "?")
    n_cal = audit.get("calibration_n", "?")
    max_ratio = audit.get("max_observed_ratio", "?")
    diag = audit.get("diagnostics", {})
    print(
        f"    [{label}] Sibling calibration: method={method}, n={n_cal}, "
        f"ĉ_sibling={c_hat}, max_ratio={max_ratio}"
    )
    if diag:
        for k, v in diag.items():
            if isinstance(v, float):
                print(f"      {k} = {v:.4f}")
            else:
                print(f"      {k} = {v}")


def trace_root_gates(tree: PosetTree, stats_df: pd.DataFrame, label: str):
    """Print gate trace for the root and its children."""
    root = tree.root()
    children = list(tree.successors(root))
    print(f"\n  [{label}] Root = {root}, Children = {children}")

    if len(children) != 2:
        print(f"    Gate 1: FAIL (non-binary, {len(children)} children)")
        return

    left, right = children

    # Gate 2: child-parent divergence
    for child_label, child in [("Left", left), ("Right", right)]:
        row = stats_df.loc[child] if child in stats_df.index else None
        if row is None:
            print(f"    Gate 2 ({child_label} = {child}): NO STATS ROW")
            continue

        sig = row.get("Child_Parent_Divergence_Significant", None)
        p_bh = row.get("Child_Parent_Divergence_P_Value_BH", None)
        p_raw = row.get("Child_Parent_Divergence_P_Value", None)
        df = row.get("Child_Parent_Divergence_df", None)
        print(
            f"    Gate 2 ({child_label} = {child}): sig={sig}, p_raw={p_raw:.4e}, p_BH={p_bh:.4e}, df={df}"
        )

    left_sig = (
        stats_df.loc[left].get("Child_Parent_Divergence_Significant", False)
        if left in stats_df.index
        else False
    )
    right_sig = (
        stats_df.loc[right].get("Child_Parent_Divergence_Significant", False)
        if right in stats_df.index
        else False
    )
    gate2_pass = left_sig or right_sig
    print(f"    Gate 2 overall: {'PASS' if gate2_pass else 'FAIL'} (L={left_sig}, R={right_sig})")

    if not gate2_pass:
        print("    → K=1 because Gate 2 blocks the root split (no child diverges from root).")
        return

    # Gate 3: sibling divergence
    root_row = stats_df.loc[root] if root in stats_df.index else None
    if root_row is not None:
        sib_diff = root_row.get("Sibling_BH_Different", None)
        sib_p = root_row.get("Sibling_Divergence_P_Value_Corrected", None)
        sib_p_raw = root_row.get("Sibling_Divergence_P_Value", None)
        sib_skip = root_row.get("Sibling_Divergence_Skipped", None)
        sib_stat = root_row.get("Sibling_Test_Statistic", None)
        sib_df = root_row.get("Sibling_Degrees_of_Freedom", None)
        sib_method = root_row.get("Sibling_Test_Method", None)
        print(
            f"    Gate 3: different={sib_diff}, skipped={sib_skip}, "
            f"T={sib_stat:.2f}, df={sib_df}, p_raw={sib_p_raw:.4e}, p_BH={sib_p:.4e}, "
            f"method={sib_method}"
        )
        gate3_pass = bool(sib_diff) and not bool(sib_skip)
        print(f"    Gate 3 overall: {'PASS' if gate3_pass else 'FAIL'}")
    else:
        print(f"    Gate 3: NO STATS ROW for root {root}")


def trace_calibration_detail(tree: PosetTree, stats_df: pd.DataFrame):
    """Print edge calibration details for root's children."""
    root = tree.root()
    children = list(tree.successors(root))
    if len(children) != 2:
        return

    # Get calibration model from stats_df.attrs
    cal_model: EdgeCalibrationModel | None = stats_df.attrs.get("edge_calibration_model", None)
    if cal_model is None:
        print("    Calibration: NO MODEL (calibration disabled or failed)")
        return

    print(
        f"    Calibration model: method={cal_model.method}, n_cal={cal_model.n_calibration}, "
        f"global_c_hat={cal_model.global_c_hat:.4f}, max_ratio={cal_model.max_observed_ratio:.4f}"
    )
    if cal_model.depth_c_hats:
        for depth, c in sorted(cal_model.depth_c_hats.items()):
            print(f"      depth {depth}: ĉ_local = {c:.4f}")

    # Detail on root's children
    node_depths = compute_node_depths(tree)

    pca_eigenvalues = stats_df.attrs.get("_pca_eigenvalues", {})
    spectral_dims = stats_df.attrs.get("_spectral_dims", {})

    for child in children:
        n_child = len(tree.get_leaves(child))
        n_parent = len(tree.get_leaves(root))
        depth = node_depths.get(child, 0)

        parent_eig = pca_eigenvalues.get(root)
        parent_k = spectral_dims.get(root)

        w = compute_edge_null_weight(
            n_child=n_child,
            n_parent=n_parent,
            parent_eigenvalues=parent_eig,
            spectral_k=parent_k,
        )

        c_level = cal_model.depth_c_hats.get(depth, cal_model.global_c_hat)
        c_soft = 1.0 + w * (c_level - 1.0)

        # Compute leaf ratio and eigenvalue ratio components
        leaf_ratio = n_child / n_parent if n_parent > 0 else 0.0
        from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral_dimension import (
            effective_rank,
        )

        if parent_eig is not None and len(parent_eig) > 0:
            k = parent_k if parent_k is not None and parent_k > 0 else len(parent_eig)
            er = effective_rank(parent_eig)
            eig_ratio = min(er / k, 1.0)
        else:
            er = None
            eig_ratio = None

        print(f"    Child {child}: depth={depth}, n_child={n_child}, n_parent={n_parent}")
        print(
            f"      leaf_ratio={leaf_ratio:.4f}, eff_rank={er}, spectral_k={parent_k}, "
            f"eig_ratio={eig_ratio}"
        )
        print(f"      w={w:.4f}, ĉ_local(depth={depth})={c_level:.4f}, ĉ_soft={c_soft:.4f}")


def trace_all_internal_nodes(tree: PosetTree, stats_df: pd.DataFrame, max_nodes: int = 10):
    """Print gate trace for the first few internal nodes (BFS from root)."""
    root = tree.root()
    from collections import deque

    queue = deque([root])
    visited = set()
    count = 0

    while queue and count < max_nodes:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        children = list(tree.successors(node))
        if len(children) == 0:
            continue  # leaf

        count += 1
        if len(children) != 2:
            print(f"    {node}: non-binary ({len(children)} children) → MERGE")
            for c in children:
                queue.append(c)
            continue

        left, right = children
        row = stats_df.loc[node] if node in stats_df.index else None

        # Gate 2
        left_sig = (
            stats_df.loc[left].get("Child_Parent_Divergence_Significant", False)
            if left in stats_df.index
            else False
        )
        right_sig = (
            stats_df.loc[right].get("Child_Parent_Divergence_Significant", False)
            if right in stats_df.index
            else False
        )
        g2 = left_sig or right_sig

        # Gate 3
        sib_diff = row.get("Sibling_BH_Different", False) if row is not None else False
        sib_skip = row.get("Sibling_Divergence_Skipped", False) if row is not None else False
        sib_p_bh = (
            row.get("Sibling_Divergence_P_Value_Corrected", None) if row is not None else None
        )
        g3 = bool(sib_diff) and not bool(sib_skip)

        split = g2 and g3
        marker = "SPLIT" if split else "MERGE"

        left_p = (
            stats_df.loc[left].get("Child_Parent_Divergence_P_Value_BH", None)
            if left in stats_df.index
            else None
        )
        right_p = (
            stats_df.loc[right].get("Child_Parent_Divergence_P_Value_BH", None)
            if right in stats_df.index
            else None
        )

        left_p_str = f"{left_p:.3e}" if left_p is not None else "?"
        right_p_str = f"{right_p:.3e}" if right_p is not None else "?"
        sib_p_str = f"{sib_p_bh:.3e}" if sib_p_bh is not None else "?"

        print(
            f"    {node}: G2={'PASS' if g2 else 'FAIL'}(L={left_p_str},R={right_p_str}) "
            f"G3={'PASS' if g3 else 'FAIL'}(p_BH={sib_p_str}) → {marker}"
        )

        if split:
            for c in children:
                queue.append(c)


def investigate_case(case_name: str):
    """Full investigation of a single case."""
    print(f"\n{'='*80}")
    print(f"CASE: {case_name}")
    print(f"{'='*80}")

    case_cfg = find_case_config(case_name)
    if case_cfg is None:
        print("  *** Case not found in default test cases! ***")
        return

    true_k = case_cfg.get("n_clusters", "?")
    print(f"  True K = {true_k}")
    print(
        f"  Config: n_rows={case_cfg.get('n_rows')}, n_cols={case_cfg.get('n_cols')}, "
        f"entropy={case_cfg.get('entropy_param', '?')}, generator={case_cfg.get('generator')}"
    )

    # Generate data
    data_df, y, x_original, metadata = generate_case_data(case_cfg)
    print(f"  Data shape: {data_df.shape}")

    # Run WITHOUT calibration (baseline)
    print(f"\n  {'─'*60}")
    print("  RUN 1: Edge calibration OFF (baseline)")
    print(f"  {'─'*60}")
    baseline = run_decompose(data_df, edge_calibration=False)
    k_baseline = baseline["results"]["num_clusters"]
    print(f"  K found = {k_baseline}")
    trace_root_gates(baseline["tree"], baseline["stats_df"], "Baseline")
    trace_sibling_calibration(baseline["stats_df"], "Baseline")

    # Run WITH calibration (soft local)
    print(f"\n  {'─'*60}")
    print("  RUN 2: Edge calibration ON (soft local)")
    print(f"  {'─'*60}")
    calibrated = run_decompose(data_df, edge_calibration=True)
    k_calibrated = calibrated["results"]["num_clusters"]
    print(f"  K found = {k_calibrated}")
    trace_root_gates(calibrated["tree"], calibrated["stats_df"], "Calibrated")
    trace_sibling_calibration(calibrated["stats_df"], "Calibrated")
    trace_calibration_detail(calibrated["tree"], calibrated["stats_df"])

    # Show top-10 internal nodes for calibrated run to see where everything merges
    if k_calibrated == 1:
        print(f"\n  {'─'*60}")
        print("  GATE TRACE (calibrated, top-10 nodes BFS from root)")
        print(f"  {'─'*60}")
        trace_all_internal_nodes(calibrated["tree"], calibrated["stats_df"], max_nodes=10)

    # Compare edge calibration weights across ALL edges
    print(f"\n  {'─'*60}")
    print("  EDGE WEIGHT DISTRIBUTION (calibrated run)")
    print(f"  {'─'*60}")
    cal_model = calibrated["stats_df"].attrs.get("edge_calibration_model", None)
    if cal_model is not None and cal_model.diagnostics:
        diag = cal_model.diagnostics
        print(f"    Global ĉ (weighted mean)  = {diag.get('global_c_hat', '?'):.4f}")
        print(f"    Global ĉ (GLM)            = {diag.get('global_c_glm', '?'):.4f}")
        print(f"    Max observed ratio (clamp) = {diag.get('max_observed_ratio', '?'):.4f}")
        print(f"    N calibration edges        = {diag.get('n_calibration', '?')}")
        print(f"    N locally calibrated       = {diag.get('n_locally_calibrated', '?')}")
        tw = diag.get("total_weight", "?")
        en = diag.get("effective_n", "?")
        print(
            f"    Total weight               = {tw:.4f}"
            if isinstance(tw, (int, float))
            else f"    Total weight               = {tw}"
        )
        print(
            f"    Effective N                = {en:.2f}"
            if isinstance(en, (int, float))
            else f"    Effective N                = {en}"
        )
        dd = diag.get("depth_details", {})
        for depth in sorted(dd.keys()):
            info = dd[depth]
            c_val = info.get("c_hat", "?")
            mw_val = info.get("mean_weight", "?")
            c_str = f"{c_val:.4f}" if isinstance(c_val, (int, float)) else str(c_val)
            mw_str = f"{mw_val:.4f}" if isinstance(mw_val, (int, float)) else str(mw_val)
            print(
                f"    Depth {depth}: n={info.get('n_edges','?')}, "
                f"ĉ={c_str}, mean_w={mw_str}"
                f"{', fallback=global' if info.get('fallback') else ''}"
            )

    # Print summary
    print(f"\n  SUMMARY: True K={true_k}, Baseline K={k_baseline}, Calibrated K={k_calibrated}")
    if k_calibrated == 1 and k_baseline > 1:
        print("  *** REGRESSION: Calibration collapsed this case to K=1 ***")


def main():
    import logging

    logging.disable(logging.WARNING)  # suppress noisy logs

    print("=" * 80)
    print("K=1 COLLAPSE INVESTIGATION — SOFT EDGE CALIBRATION")
    print("=" * 80)
    print(
        f"Config: SIGNIFICANCE_ALPHA={config.SIGNIFICANCE_ALPHA}, "
        f"SIBLING_ALPHA={config.SIBLING_ALPHA}, "
        f"SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}"
    )
    print(
        f"        SPECTRAL_METHOD={config.SPECTRAL_METHOD}, "
        f"EDGE_CALIBRATION={config.EDGE_CALIBRATION}"
    )

    for case_name in TARGET_CASES:
        investigate_case(case_name)

    print(f"\n{'='*80}")
    print("INVESTIGATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
