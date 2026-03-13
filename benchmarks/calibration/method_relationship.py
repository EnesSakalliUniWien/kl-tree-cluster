"""Structural comparison: raw Wald vs cousin-adjusted Wald.

Runs BOTH methods on the SAME tree for each case and compares per-node
statistics to reveal the exact relationship between them.

Key questions answered:
  1. Are T_raw identical across both methods? (Yes — same kernel.)
  2. How does deflation (T_adj = T_raw / ĉ) change the p-value distribution?
  3. How does BH correction interact differently with raw vs deflated p-values?
  4. Where do the methods agree/disagree on SPLIT vs MERGE?

Usage
-----
    python benchmarks/calibration/method_relationship.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ---------------------------------------------------------------------------
# Cases — same as grid_search.py
# ---------------------------------------------------------------------------


def _make_block_diagonal(n_per: int, n_feat: int, k: int, noise: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    block_size = n_feat // k
    X = rng.binomial(1, noise, size=(n_per * k, n_feat)).astype(float)
    labels = np.repeat(np.arange(k), n_per)
    for c in range(k):
        rows = slice(c * n_per, (c + 1) * n_per)
        cols = slice(c * block_size, (c + 1) * block_size)
        X[rows, cols] = rng.binomial(1, 1.0 - noise, size=(n_per, block_size)).astype(float)
    idx = [f"S{i}" for i in range(len(labels))]
    feat = [f"F{j}" for j in range(n_feat)]
    return {"data": pd.DataFrame(X, index=idx, columns=feat), "labels": labels}


CASES = [
    {"name": "easy_2c", "n_per": 30, "n_feat": 40, "k": 2, "noise": 0.15, "seed": 0},
    {"name": "easy_3c", "n_per": 25, "n_feat": 60, "k": 3, "noise": 0.15, "seed": 1},
    {"name": "easy_4c", "n_per": 20, "n_feat": 80, "k": 4, "noise": 0.15, "seed": 2},
    {"name": "noisy_3c", "n_per": 25, "n_feat": 60, "k": 3, "noise": 0.30, "seed": 4},
    {"name": "large_5c", "n_per": 40, "n_feat": 100, "k": 5, "noise": 0.15, "seed": 8},
]


# ---------------------------------------------------------------------------
# Runner: extract per-node stats from a single decomposition
# ---------------------------------------------------------------------------


def _run_method(
    data: pd.DataFrame,
    y_true: np.ndarray,
    method: str,
    sibling_alpha: float = 0.01,
    edge_alpha: float = 0.01,
) -> dict:
    """Decompose and return per-node stats + cluster-level results."""
    orig = (config.SIBLING_TEST_METHOD, config.SIBLING_ALPHA, config.EDGE_ALPHA)
    try:
        config.SIBLING_TEST_METHOD = method
        config.SIBLING_ALPHA = sibling_alpha
        config.EDGE_ALPHA = edge_alpha

        dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
        Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomp = tree.decompose(
                leaf_data=data,
                alpha_local=edge_alpha,
                sibling_alpha=sibling_alpha,
            )

        stats = tree.stats_df
        k_found = decomp.get("num_clusters", 0)

        # Compute ARI
        assignments = decomp.get("cluster_assignments", {})
        labels_pred = np.full(len(data), -1, dtype=int)
        for cid, info in assignments.items():
            for leaf in info["leaves"]:
                labels_pred[data.index.get_loc(leaf)] = cid
        ari = adjusted_rand_score(y_true, labels_pred)

        audit = stats.attrs.get("sibling_divergence_audit", {}) if stats is not None else {}

        return {
            "stats_df": stats,
            "k_found": k_found,
            "ari": ari,
            "audit": audit,
        }
    finally:
        config.SIBLING_TEST_METHOD, config.SIBLING_ALPHA, config.EDGE_ALPHA = orig


# ---------------------------------------------------------------------------
# Per-node comparison
# ---------------------------------------------------------------------------


def _compare_nodes(wald_stats: pd.DataFrame, adj_stats: pd.DataFrame) -> pd.DataFrame:
    """Build a per-node comparison of Wald vs adjusted Wald statistics.

    Both DataFrames come from the SAME tree (same linkage), so internal
    nodes align by index.
    """
    # Identify tested nodes (not skipped, not leaf)
    wald_tested = wald_stats[wald_stats["Sibling_Divergence_Skipped"] == False].copy()  # noqa: E712
    adj_tested = adj_stats[adj_stats["Sibling_Divergence_Skipped"] == False].copy()  # noqa: E712

    # Intersect — should be identical sets on the same tree
    common = sorted(set(wald_tested.index) & set(adj_tested.index))

    rows = []
    for node in common:
        w = wald_tested.loc[node]
        a = adj_tested.loc[node]

        T_wald = w["Sibling_Test_Statistic"]
        T_adj = a["Sibling_Test_Statistic"]
        # Per-node inflation factor: ĉ_i = T_raw / T_adj
        # (raw wald stores T_raw; adjusted stores T_adj = T_raw / ĉ_i)
        c_hat_i = T_wald / T_adj if T_adj > 0 else np.nan

        rows.append(
            {
                "node": node,
                "T_wald": T_wald,
                "T_adj": T_adj,
                "c_hat_i": c_hat_i,
                "df": w["Sibling_Degrees_of_Freedom"],
                "p_raw": w["Sibling_Divergence_P_Value"],
                "p_deflated": a["Sibling_Divergence_P_Value"],
                "p_bh_wald": w.get("Sibling_Divergence_P_Value_Corrected", np.nan),
                "p_bh_adj": a.get("Sibling_Divergence_P_Value_Corrected", np.nan),
                "split_wald": bool(w["Sibling_BH_Different"]),
                "split_adj": bool(a["Sibling_BH_Different"]),
                "method_adj": a.get("Sibling_Test_Method", ""),
                "is_edge_sig_L": w["Child_Parent_Divergence_Significant"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    SIB_ALPHA = 0.01
    EDGE_ALPHA = 0.01

    all_node_rows = []

    for case in CASES:
        gen = _make_block_diagonal(
            case["n_per"], case["n_feat"], case["k"], case["noise"], case["seed"]
        )
        data, y_true = gen["data"], gen["labels"]
        name = case["name"]

        wald_res = _run_method(data, y_true, "wald", SIB_ALPHA, EDGE_ALPHA)
        adj_res = _run_method(data, y_true, "cousin_adjusted_wald", SIB_ALPHA, EDGE_ALPHA)

        print("=" * 80)
        print(f"CASE: {name}  (true_k={case['k']}, noise={case['noise']})")
        print(f"  Wald:     K={wald_res['k_found']}, ARI={wald_res['ari']:.3f}")
        print(f"  Adjusted: K={adj_res['k_found']}, ARI={adj_res['ari']:.3f}")

        # Calibration info
        audit = adj_res["audit"]
        c_hat_global = audit.get("global_inflation_factor", 1.0)
        cal_method = audit.get("calibration_method", "n/a")
        n_null = audit.get("null_like_pairs", 0)
        n_focal = audit.get("focal_pairs", 0)
        diag = audit.get("diagnostics", {})
        beta = diag.get("beta", None)
        r2 = diag.get("r_squared", None)
        max_ratio = diag.get("max_observed_ratio", None)

        print(f"  Calibration: method={cal_method}, null-like={n_null}, focal={n_focal}")
        print(
            f"    median(T/k) of null-like = {c_hat_global:.4f}  "
            f"(< 1 ⇒ tree groups similar things, suppressing null divergence)"
        )
        if max_ratio is not None:
            print(f"    max(T/k) of null-like = {max_ratio:.4f}  (clamp ceiling for regression)")
        if beta is not None:
            print(
                f"    Regression: log(T/k) = {beta[0]:.3f} + "
                f"{beta[1]:.3f}·log(BL) + {beta[2]:.3f}·log(n_parent),  R²={r2:.3f}"
            )
            print(
                f"    ĉ_i = exp({beta[0]:.3f} + {beta[1]:.3f}·log(BL_i) + "
                f"{beta[2]:.3f}·log(n_i)),  clamped to [1.0, {max_ratio:.3f}]"
            )

        # Per-node comparison
        comp = _compare_nodes(wald_res["stats_df"], adj_res["stats_df"])

        if comp.empty:
            print("  (no tested nodes to compare)")
            continue

        # Per-node ĉ_i values
        c_hat_vals = comp["c_hat_i"].dropna()
        print(
            f"\n  Per-node ĉ_i (= T_raw / T_adj):  "
            f"mean={c_hat_vals.mean():.3f}, "
            f"min={c_hat_vals.min():.3f}, max={c_hat_vals.max():.3f}"
        )
        if (c_hat_vals < 1.0).any():
            print(
                f"    ĉ_i < 1 at {(c_hat_vals < 1.0).sum()} nodes (clamped to 1.0 ⇒ no deflation)"
            )
        if (c_hat_vals >= 1.0).any():
            print(f"    ĉ_i ≥ 1 at {(c_hat_vals >= 1.0).sum()} nodes (actual deflation applied)")

        # Transformation chain for each node
        print(
            f"\n  {'node':>10}  {'T_raw':>8}  {'ĉ_i':>6}  {'T_adj':>8}  {'df':>4}  "
            f"{'p_raw':>10}  {'p_adj':>10}  {'p_bh_w':>10}  {'p_bh_a':>10}  "
            f"{'wald':>5}  {'adj':>5}"
        )
        for _, row in comp.iterrows():
            wald_dec = "SPLIT" if row["split_wald"] else "merge"
            adj_dec = "SPLIT" if row["split_adj"] else "merge"
            marker = " ◄" if row["split_wald"] != row["split_adj"] else ""
            print(
                f"  {row['node']:>10}  {row['T_wald']:8.2f}  {row['c_hat_i']:6.3f}  "
                f"{row['T_adj']:8.2f}  {row['df']:4.0f}  "
                f"{row['p_raw']:10.6f}  {row['p_deflated']:10.6f}  "
                f"{row['p_bh_wald']:10.6f}  {row['p_bh_adj']:10.6f}  "
                f"{wald_dec:>5}  {adj_dec:>5}{marker}"
            )

        # Summary
        agree = (comp["split_wald"] == comp["split_adj"]).sum()
        wald_only = ((comp["split_wald"]) & (~comp["split_adj"])).sum()
        adj_only = ((~comp["split_wald"]) & (comp["split_adj"])).sum()
        print(
            f"\n  Agree: {agree}/{len(comp)}, Wald-only splits: {wald_only}, "
            f"Adj-only splits: {adj_only}"
        )

        # Annotate for aggregation
        for _, row in comp.iterrows():
            all_node_rows.append({**row.to_dict(), "case": name, "true_k": case["k"]})

    # =========================================================================
    # Cross-case summary
    # =========================================================================
    all_df = pd.DataFrame(all_node_rows)

    if all_df.empty:
        print("\nNo tested nodes across any case.")
        return

    print("\n" + "=" * 80)
    print("CROSS-CASE SUMMARY")
    print("=" * 80)

    # 1. Per-node ĉ distribution
    c_vals = all_df["c_hat_i"].dropna()
    print(f"\n  Per-node ĉ_i across all {len(c_vals)} tested nodes:")
    print(
        f"    mean={c_vals.mean():.3f}, median={c_vals.median():.3f}, "
        f"std={c_vals.std():.3f}, range=[{c_vals.min():.3f}, {c_vals.max():.3f}]"
    )
    print(f"    ĉ_i = 1.0 (no deflation): {(abs(c_vals - 1.0) < 0.001).sum()} nodes")
    print(f"    ĉ_i > 1.0 (deflated):     {(c_vals > 1.001).sum()} nodes")

    # 2. How many decisions flip?
    total = len(all_df)
    agree = (all_df["split_wald"] == all_df["split_adj"]).sum()
    wald_splits = int(all_df["split_wald"].sum())
    adj_splits = int(all_df["split_adj"].sum())
    wald_over = int(((all_df["split_wald"]) & (~all_df["split_adj"])).sum())
    adj_over = int(((~all_df["split_wald"]) & (all_df["split_adj"])).sum())

    print(f"\n  Decisions across {total} tested nodes:")
    print(f"    Wald total splits:    {wald_splits}")
    print(f"    Adjusted total splits:{adj_splits}")
    print(f"    Agreement:            {agree} ({100*agree/total:.1f}%)")
    print(f"    Wald-only splits:     {wald_over} (over-splitting)")
    print(f"    Adj-only splits:      {adj_over}")

    # 3. The transformation chain
    print("\n  TRANSFORMATION CHAIN (the structural relationship):")
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  Both methods share the SAME Wald kernel:                      │")
    print("  │    T_raw = ||R · z||²  where z = (θ_L - θ_R) / √Var          │")
    print("  │                                                                │")
    print("  │  RAW WALD path:                                                │")
    print("  │    T_raw → p = P(χ²(k) > T_raw) → BH over focal pairs        │")
    print("  │                                                                │")
    print("  │  ADJUSTED WALD path:                                           │")
    print("  │    Step 1: Collect T_raw for ALL sibling pairs (same kernel)   │")
    print("  │    Step 2: Partition into null-like (no edge signal) vs focal  │")
    print("  │    Step 3: Fit regression on null-like pairs:                  │")
    print("  │            log(T/k) = β₀ + β₁·log(BL) + β₂·log(n_parent)    │")
    print("  │    Step 4: For each focal pair, predict per-node ĉ_i ≥ 1.0   │")
    print("  │    Step 5: T_adj = T_raw / ĉ_i  (deflation)                  │")
    print("  │    Step 6: p_adj = P(χ²(k) > T_adj)                          │")
    print("  │    Step 7: BH correction over focal pairs only                │")
    print("  │                                                                │")
    print("  │  ĉ_i ≥ 1 always (clamped), so T_adj ≤ T_raw, p_adj ≥ p_raw. │")
    print("  │  The adjusted path is STRICTLY more conservative.             │")
    print("  └─────────────────────────────────────────────────────────────────┘")

    # 4. Anatomy of over-splits
    if wald_over > 0:
        flipped = all_df[(all_df["split_wald"]) & (~all_df["split_adj"])]
        agreed = all_df[all_df["split_wald"] == all_df["split_adj"]]
        print(f"\n  ANATOMY OF {wald_over} WALD-ONLY SPLITS (the over-splits):")
        print(f"    Mean ĉ_i at flipped nodes: {flipped['c_hat_i'].mean():.3f}")
        print(f"    Mean ĉ_i at agreed nodes:  {agreed['c_hat_i'].dropna().mean():.3f}")
        print(f"    Mean p_raw  at flipped:    {flipped['p_raw'].mean():.6f}")
        print(f"    Mean p_adj  at flipped:    {flipped['p_deflated'].mean():.6f}")
        print("    → These nodes pass p_raw < α but fail after deflation (p_adj > α)")
        print("    → The calibration regression identifies these as post-selection artifacts")

    # Save
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "method_relationship_nodes.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\n  Node-level data saved to {csv_path}")


if __name__ == "__main__":
    main()
