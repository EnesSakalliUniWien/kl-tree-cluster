#!/usr/bin/env python3
"""Diagnose K=1 cases from the cousin_weighted_wald benchmark.

Reproduces each K=1 case through the exact pipeline path and traces:
  - Gate 2 (edge significance) decisions at the root
  - Gate 3 (sibling divergence) calibration model and decisions
  - Where the decision to NOT split occurs

Usage:
    python scripts/diagnose_k1_pipeline.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on path
repo = Path(__file__).resolve().parents[1]
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

# Suppress noisy warnings but keep our targeted logs
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s: %(message)s",
)

# ── Configuration snapshot ──────────────────────────────────────────────────
print("=" * 90)
print("PIPELINE CONFIGURATION")
print("=" * 90)
print(f"  SIBLING_TEST_METHOD   = {config.SIBLING_TEST_METHOD}")
print(f"  ALPHA_LOCAL           = {config.ALPHA_LOCAL}")
print(f"  SIBLING_ALPHA         = {config.SIBLING_ALPHA}")
print(f"  POSTHOC_MERGE         = {config.POSTHOC_MERGE}")
print(f"  TREE_DISTANCE_METRIC  = {config.TREE_DISTANCE_METRIC}")
print(f"  TREE_LINKAGE_METHOD   = {config.TREE_LINKAGE_METHOD}")
print(f"  PROJECTION_EPS        = {config.PROJECTION_EPS}")
print(f"  PROJECTION_MIN_K      = {config.PROJECTION_MIN_K}")

# ── Identify K=1 cases from benchmark results ──────────────────────────────
CSV = repo / "benchmarks" / "results" / "run_20260216_133601Z" / "full_benchmark_comparison.csv"
if not CSV.exists():
    print(f"\nERROR: Benchmark CSV not found at {CSV}")
    sys.exit(1)

bench_df = pd.read_csv(CSV)
kl_rows = bench_df[bench_df["method"] == "kl"]
k1_rows = kl_rows[kl_rows["found_clusters"] == 1].sort_values("test_case")

print(f"\nFound {len(k1_rows)} K=1 cases in the benchmark.\n")

# ── Build test-case lookup ──────────────────────────────────────────────────
all_cases = get_default_test_cases()
case_by_name = {c["name"]: c for c in all_cases}

# ── Trace each K=1 case ────────────────────────────────────────────────────
summary_rows = []

for _, brow in k1_rows.iterrows():
    case_name = brow["case_id"]
    true_k = int(brow["true_clusters"])
    n_samples = int(brow["Samples"])
    n_features = int(brow["Features"])

    tc = case_by_name.get(case_name)
    if tc is None:
        print(f"SKIP: {case_name} — not found in test case registry")
        continue

    print("=" * 90)
    print(f"CASE: {case_name}  (true K={true_k}, n={n_samples}, p={n_features})")
    print("=" * 90)

    # ── Generate data ───────────────────────────────────────────────────
    try:
        data_df, y_true, x_orig, meta = generate_case_data(tc)
    except Exception as e:
        print(f"  DATA ERROR: {e}\n")
        continue

    # ── Build tree ──────────────────────────────────────────────────────
    dist_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist_condensed, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    n_nodes = tree.number_of_nodes()
    n_internal = sum(1 for n in tree.nodes if tree.out_degree(n) > 0)
    root = [n for n in tree.nodes if tree.in_degree(n) == 0][0]

    print(f"  Tree: {n_nodes} nodes ({n_internal} internal), root={root}")

    # ── Run decomposition ───────────────────────────────────────────────
    result = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=config.SIBLING_ALPHA,
    )
    found_k = result.get("num_clusters", 0)
    cluster_sizes = {k: len(v) for k, v in result.get("cluster_assignments", {}).items()}

    print(f"  Result: found K={found_k}, sizes={cluster_sizes}")

    # ── Extract stats_df ────────────────────────────────────────────────
    sdf = tree.stats_df
    if sdf is None:
        print("  WARNING: stats_df is None\n")
        continue

    # ── Calibration audit ───────────────────────────────────────────────
    audit = sdf.attrs.get("sibling_divergence_audit", {})
    cal_method = audit.get("calibration_method", audit.get("test_method", "?"))
    cal_n = audit.get("calibration_n", "?")
    global_c = audit.get("global_c_hat", "?")
    n_null = audit.get("null_like_pairs", "?")
    n_focal = audit.get("focal_pairs", "?")
    diag = audit.get("diagnostics", {})

    print(f"\n  Calibration: method={cal_method}, n_cal={cal_n}, ĉ={global_c}")
    print(f"  Pairs: {n_null} null-like, {n_focal} focal, total={audit.get('total_pairs', '?')}")
    if diag:
        beta = diag.get("beta", None)
        r2 = diag.get("r_squared", None)
        max_obs = diag.get("max_observed_ratio", None)
        eff_n = diag.get("effective_n", None)
        if beta is not None:
            print(f"  Regression: β={beta}, R²={r2:.4f}" if r2 else f"  Regression: β={beta}")
        if max_obs is not None:
            print(f"  max_observed_ratio={max_obs:.4f}, effective_n={eff_n:.1f}" if eff_n else f"  max_observed_ratio={max_obs}")

    # ── Root node gate trace ────────────────────────────────────────────
    print(f"\n  ROOT NODE ({root}) gate trace:")
    root_row = sdf.loc[root] if root in sdf.index else None

    gate_cols = [
        ("Child_Parent_Divergence_Significant", "Gate 2 (edge) significant"),
        ("Child_Parent_Divergence_P_Value", "Gate 2 raw p-value"),
        ("Child_Parent_Divergence_P_Value_BH", "Gate 2 BH p-value"),
        ("Child_Parent_Divergence_Test_Statistic", "Gate 2 test stat"),
        ("Child_Parent_Divergence_DF", "Gate 2 degrees of freedom"),
        ("Sibling_Test_Statistic", "Gate 3 test stat"),
        ("Sibling_Degrees_of_Freedom", "Gate 3 degrees of freedom"),
        ("Sibling_Divergence_P_Value", "Gate 3 p-value (deflated)"),
        ("Sibling_Divergence_P_Value_Corrected", "Gate 3 BH p-value"),
        ("Sibling_BH_Different", "Gate 3 siblings different"),
        ("Sibling_BH_Same", "Gate 3 siblings same"),
        ("Sibling_Divergence_Skipped", "Gate 3 skipped"),
        ("Sibling_Test_Method", "Gate 3 method used"),
    ]

    if root_row is not None:
        for col, label in gate_cols:
            if col in sdf.columns:
                val = root_row[col]
                print(f"    {label:40s} = {val}")

    # ── Trace children of root ──────────────────────────────────────────
    root_children = list(tree.successors(root))
    print(f"\n  Root children: {root_children}")
    for child in root_children:
        if child in sdf.index:
            crow = sdf.loc[child]
            cp_sig = crow.get("Child_Parent_Divergence_Significant", None)
            cp_pval = crow.get("Child_Parent_Divergence_P_Value_BH", None)
            n_desc = crow.get("N_Descendants", None)
            print(f"    {child}: edge_sig={cp_sig}, edge_p_bh={cp_pval}, n_desc={n_desc}")

    # ── Check all internal nodes: how many passed each gate? ────────────
    internal_nodes = [n for n in tree.nodes if tree.out_degree(n) > 0]
    if "Child_Parent_Divergence_Significant" in sdf.columns:
        # Count how many children have edge-significant
        n_edge_sig = 0
        n_edge_total = 0
        for node in internal_nodes:
            children = list(tree.successors(node))
            for ch in children:
                if ch in sdf.index:
                    n_edge_total += 1
                    if sdf.loc[ch].get("Child_Parent_Divergence_Significant", False):
                        n_edge_sig += 1
        print(f"\n  Edge significance: {n_edge_sig}/{n_edge_total} children edge-significant")

    if "Sibling_BH_Different" in sdf.columns:
        sib_diff = sdf["Sibling_BH_Different"].sum()
        sib_same = sdf["Sibling_BH_Same"].sum() if "Sibling_BH_Same" in sdf.columns else "?"
        sib_skip = sdf["Sibling_Divergence_Skipped"].sum() if "Sibling_Divergence_Skipped" in sdf.columns else "?"
        print(f"  Sibling test: {sib_diff} different, {sib_same} same, {sib_skip} skipped")

    # ── Identify the blocking gate ──────────────────────────────────────
    if root_row is not None:
        root_skipped = root_row.get("Sibling_Divergence_Skipped", False)
        root_same = root_row.get("Sibling_BH_Same", False)
        root_diff = root_row.get("Sibling_BH_Different", False)

        # Check Gate 2 for root's children
        root_children_edge_sig = []
        for ch in root_children:
            if ch in sdf.index:
                sig = sdf.loc[ch].get("Child_Parent_Divergence_Significant", False)
                root_children_edge_sig.append((ch, sig))

        any_child_sig = any(s for _, s in root_children_edge_sig)
        both_children_sig = all(s for _, s in root_children_edge_sig) if len(root_children_edge_sig) == 2 else False

        if not any_child_sig:
            blocking = "GATE 2: Neither child edge-significant → skipped → MERGE"
        elif root_skipped:
            blocking = "GATE 3: Sibling test skipped → MERGE"
        elif root_same:
            blocking = "GATE 3: Siblings declared SAME (p > α after BH) → MERGE"
        elif root_diff:
            blocking = "NOT BLOCKED AT ROOT — split should occur"
        else:
            blocking = "UNKNOWN — check stats_df manually"

        print(f"\n  >>> ROOT BLOCKING REASON: {blocking}")

    # ── Summary row ─────────────────────────────────────────────────────
    row = {
        "case": case_name,
        "true_k": true_k,
        "found_k": found_k,
        "n": n_samples,
        "p": n_features,
        "cal_method": cal_method,
        "cal_n": cal_n,
        "global_c": global_c,
        "n_null": n_null,
        "n_focal": n_focal,
        "blocking": blocking if root_row is not None else "?",
    }
    summary_rows.append(row)
    print()

# ── Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SUMMARY: K=1 Root Causes")
print("=" * 90)
if summary_rows:
    sdf_summary = pd.DataFrame(summary_rows)
    # Count by blocking reason
    print("\nBlocking reason counts:")
    for reason, count in sdf_summary["blocking"].value_counts().items():
        print(f"  {count:3d}  {reason}")

    print(f"\nCalibration method distribution:")
    for method, count in sdf_summary["cal_method"].value_counts().items():
        print(f"  {count:3d}  {method}")

    print(f"\nDetailed table:")
    print(sdf_summary.to_string(index=False))
