"""Compare two candidate fixes for adjusted Wald over-deflation.

Fix A: Cap ĉ at the maximum observed T/k ratio from null-like calibration pairs.
Fix B: Drop β₂ (n_parent) from the regression, use only β₀ + β₁·log(BL_sum).

Runs all 95 default benchmarks with:
  1. Raw Wald (baseline)
  2. Current adj_wald (broken — has over-deflation)
  3. Fix A: adj_wald + cap ĉ at max observed
  4. Fix B: adj_wald + BL-only regression (no n_parent)

Does NOT modify production code. Instead, monkey-patches _predict_c and
_fit_inflation_model locally to simulate each fix.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import gc
import logging
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from benchmarks.shared.util.decomposition import _labels_from_decomposition
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)

# Import internals for monkey-patching
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_adjusted_wald import (
    _collect_all_pairs,
    _fit_inflation_model,
    _predict_c,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

logging.basicConfig(level=logging.WARNING)


# ─── Fix A: cap ĉ at max observed ratio ───────────────────────────────────

def _predict_c_fix_a(model, bl_sum, n_parent, max_ratio):
    """Like _predict_c but caps at max observed ratio."""
    c = _predict_c(model, bl_sum, n_parent)
    if model.method == "regression":
        c = min(c, max_ratio)
        c = max(c, 1.0)
    return c


# ─── Fix B: BL-only regression (drop β₂) ──────────────────────────────────

def _predict_c_fix_b(model, bl_sum, n_parent):
    """Like _predict_c but ignores β₂ (n_parent term)."""
    if model.method != "regression" or model.beta is None:
        return _predict_c(model, bl_sum, n_parent)

    if bl_sum <= 0:
        return max(model.global_c_hat, 1.0)

    # Only β₀ + β₁·log(BL)
    log_c = model.beta[0] + model.beta[1] * np.log(bl_sum)
    c_hat = float(np.exp(log_c))
    return max(c_hat, 1.0)


def run_with_wald(data_df, Z, true_labels):
    """Run with raw Wald sibling test."""
    config.SIBLING_TEST_METHOD = "wald"
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    decomp = tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)
    pred = _labels_from_decomposition(decomp, data_df.index.tolist())
    ari = adjusted_rand_score(true_labels, pred)
    return decomp["num_clusters"], ari


def run_with_adj_wald(data_df, Z, true_labels):
    """Run with current (broken) adjusted Wald."""
    config.SIBLING_TEST_METHOD = "cousin_adjusted_wald"
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    decomp = tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)
    pred = _labels_from_decomposition(decomp, data_df.index.tolist())
    ari = adjusted_rand_score(true_labels, pred)

    # Extract calibration data for fixes
    sdf = tree.stats_df
    mean_bl = compute_mean_branch_length(tree)
    records = _collect_all_pairs(tree, sdf, mean_bl)
    model = _fit_inflation_model(records)

    return decomp["num_clusters"], ari, records, model, tree, sdf


def simulate_fix(records, model, tree, sdf, predict_fn, alpha=0.05):
    """Re-run the deflation + BH + decomposition with a patched predict_c.

    Instead of re-running the whole pipeline, we:
    1. Re-compute T_adj and p-values with the patched predict_c
    2. Apply BH correction
    3. Re-run the tree decomposition with patched sibling annotations
    """
    from kl_clustering_analysis.core_utils.data_utils import (
        initialize_sibling_divergence_columns,
    )
    from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing import (
        benjamini_hochberg_correction,
    df = sdf.copy()

    # Reset sibling columns
    df = initialize_sibling_divergence_columns(df)

    focal_parents = []
    focal_stats = []
    focal_dfs = []
    focal_pvals = []
    skipped_parents = []

    for rec in records:
        if rec.is_null_like:
            skipped_parents.append(rec.parent)
            continue

        if not np.isfinite(rec.stat) or rec.df <= 0:
            focal_parents.append(rec.parent)
            focal_stats.append(np.nan)
            focal_dfs.append(np.nan)
            focal_pvals.append(np.nan)
            continue

        c_hat = predict_fn(model, rec.bl_sum, rec.n_parent)
        t_adj = rec.stat / c_hat
        p_adj = float(chi2.sf(t_adj, df=rec.df))

        focal_parents.append(rec.parent)
        focal_stats.append(t_adj)
        focal_dfs.append(float(rec.df))
        focal_pvals.append(p_adj)

    # Mark skipped
    if skipped_parents:
        df.loc[skipped_parents, "Sibling_Divergence_Skipped"] = True

    if not focal_parents:
        return df

    stats = np.array(focal_stats)
    dfs_arr = np.array(focal_dfs)
    pvals = np.array(focal_pvals)

    invalid_mask = ~np.isfinite(pvals)
    pvals_for_bh = np.where(np.isfinite(pvals), pvals, 1.0)
    reject, pvals_adj, _ = benjamini_hochberg_correction(pvals_for_bh, alpha=alpha)
    reject = np.where(invalid_mask, False, reject)

    df.loc[focal_parents, "Sibling_Test_Statistic"] = stats
    df.loc[focal_parents, "Sibling_Degrees_of_Freedom"] = dfs_arr
    df.loc[focal_parents, "Sibling_Divergence_P_Value"] = pvals
    df.loc[focal_parents, "Sibling_Divergence_P_Value_Corrected"] = pvals_adj
    df.loc[focal_parents, "Sibling_Divergence_Invalid"] = invalid_mask
    df.loc[focal_parents, "Sibling_BH_Different"] = reject
    df.loc[focal_parents, "Sibling_BH_Same"] = ~reject

    return df


def decompose_with_patched_sdf(tree, patched_sdf, data_df, true_labels):
    """Re-run tree decomposition using patched sibling annotations."""
    from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition

    # Inject patched stats_df back into tree
    tree._stats_df = patched_sdf

    decomposer = TreeDecomposition(
        tree,
        patched_sdf,
        alpha_local=0.05,
        sibling_alpha=0.05,
        posthoc_merge=config.POSTHOC_MERGE,
    )
    decomp_result = decomposer.decompose_tree()
    pred = _labels_from_decomposition(decomp_result, data_df.index.tolist())
    ari = adjusted_rand_score(true_labels, pred)
    return decomp_result["num_clusters"], ari


def main():
    print("=" * 95)
    print("COMPARING TWO ADJUSTED WALD FIXES ACROSS ALL BENCHMARK CASES")
    print("  Fix A: Cap ĉ at max observed ratio from null-like pairs")
    print("  Fix B: Drop β₂ (n_parent), use only β₀ + β₁·log(BL_sum)")
    print("=" * 95)

    cases = get_default_test_cases()
    n_cases = len(cases)
    print(f"\nTotal cases: {n_cases}")

    rows = []
    t0 = time.time()

    for i, case in enumerate(cases):
        name = case.get("name", f"case_{i}")
        true_k = case.get("n_clusters", -1)
        case_type = case.get("type", "unknown")

        try:
            data_df, true_labels, _, metadata = generate_case_data(case)
        except Exception as e:
            print(f"  [{i+1}/{n_cases}] {name}: GENERATE ERROR: {e}")
            continue

        # Build linkage once
        if case_type == "sbm" and metadata.get("distance_condensed") is not None:
            dist_c = metadata["distance_condensed"]
        else:
            dist_c = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
        Z = linkage(dist_c, method=config.TREE_LINKAGE_METHOD)

        # 1. Raw Wald
        try:
            k_wald, ari_wald = run_with_wald(data_df, Z, true_labels)
        except Exception:
            k_wald, ari_wald = -1, np.nan

        # 2. Current adj_wald (to get records + model for patching)
        try:
            k_adj, ari_adj, records, model, tree, sdf = run_with_adj_wald(
                data_df, Z, true_labels
            )
        except Exception as e:
            k_adj, ari_adj = -1, np.nan
            print(f"  [{i+1}/{n_cases}] {name}: ADJ_WALD ERROR: {e}")
            rows.append({
                "name": name, "type": case_type, "true_k": true_k,
                "k_wald": k_wald, "ari_wald": ari_wald,
                "k_adj": -1, "ari_adj": np.nan,
                "k_fixA": -1, "ari_fixA": np.nan,
                "k_fixB": -1, "ari_fixB": np.nan,
            })
            continue

        # Compute max observed ratio for Fix A
        null_recs = [r for r in records if r.is_null_like and np.isfinite(r.stat) and r.df > 0]
        if null_recs:
            max_ratio = max(r.stat / r.df for r in null_recs)
        else:
            max_ratio = 1.0

        # 3. Fix A: cap at max observed ratio
        try:
            predict_a = lambda m, bl, n, mr=max_ratio: _predict_c_fix_a(m, bl, n, mr)
            patched_a = simulate_fix(records, model, tree, sdf, predict_a)
            k_a, ari_a = decompose_with_patched_sdf(tree, patched_a, data_df, true_labels)
        except Exception:
            k_a, ari_a = -1, np.nan

        # 4. Fix B: drop β₂
        try:
            patched_b = simulate_fix(records, model, tree, sdf, _predict_c_fix_b)
            k_b, ari_b = decompose_with_patched_sdf(tree, patched_b, data_df, true_labels)
        except Exception:
            k_b, ari_b = -1, np.nan

        # Status indicator
        marker = ""
        if k_adj == 1 and k_wald > 1:
            marker = " *** K=1 ***"
        elif k_a != k_adj or k_b != k_adj:
            marker = " (changed)"

        print(
            f"  [{i+1:>2}/{n_cases}] {name:<35} true={true_k:>2} | "
            f"wald=K{k_wald:<3} adj=K{k_adj:<3} fixA=K{k_a:<3} fixB=K{k_b:<3}"
            f"{marker}"
        )

        rows.append({
            "name": name, "type": case_type, "true_k": true_k,
            "k_wald": k_wald, "ari_wald": ari_wald,
            "k_adj": k_adj, "ari_adj": ari_adj,
            "k_fixA": k_a, "ari_fixA": ari_a,
            "k_fixB": k_b, "ari_fixB": ari_b,
        })
        gc.collect()

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    # Save CSV
    out_dir = repo_root / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fix_comparison_A_vs_B.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Total time: {elapsed:.1f}s\n")

    # ── Summary ──────────────────────────────────────────────────────────
    valid = df[df["k_wald"] >= 0].copy()
    methods = {
        "wald": ("k_wald", "ari_wald"),
        "adj_wald (current)": ("k_adj", "ari_adj"),
        "Fix A (cap max)": ("k_fixA", "ari_fixA"),
        "Fix B (drop β₂)": ("k_fixB", "ari_fixB"),
    }

    print("=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(f"{'Method':<22} {'Mean ARI':>9} {'Med ARI':>9} {'Exact K':>9} {'K=1':>5} {'Cases':>6}")
    print("-" * 65)
    for label, (kcol, acol) in methods.items():
        v = valid[valid[kcol] >= 0]
        mean_ari = v[acol].mean()
        med_ari = v[acol].median()
        exact_k = (v[kcol] == v["true_k"]).sum()
        k1 = (v[kcol] == 1).sum()
        print(f"{label:<22} {mean_ari:>9.3f} {med_ari:>9.3f} {exact_k:>6}/{len(v):<3} {k1:>5} {len(v):>6}")

    # ── Head-to-head ─────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("HEAD-TO-HEAD (ARI, tolerance=0.01)")
    print("=" * 95)
    combos = [
        ("Fix A (cap max)", "adj_wald (current)"),
        ("Fix B (drop β₂)", "adj_wald (current)"),
        ("Fix A (cap max)", "wald"),
        ("Fix B (drop β₂)", "wald"),
        ("Fix A (cap max)", "Fix B (drop β₂)"),
    ]
    for l1, l2 in combos:
        _, a1 = methods[l1]
        _, a2 = methods[l2]
        wins = (valid[a1] > valid[a2] + 0.01).sum()
        losses = (valid[a2] > valid[a1] + 0.01).sum()
        ties = len(valid) - wins - losses
        print(f"  {l1:<22} vs {l2:<22}: {wins}W / {losses}L / {ties}T")

    # ── Cases where K=1 collapsed — did fixes help? ──────────────────────
    k1_cases = valid[(valid["k_adj"] == 1) & (valid["k_wald"] > 1)]
    if len(k1_cases) > 0:
        print(f"\n{'='*95}")
        print(f"K=1 COLLAPSE CASES ({len(k1_cases)} cases) — Did fixes help?")
        print(f"{'='*95}")
        print(f"{'Name':<35} {'true':>4} {'wald':>5} {'adj':>5} {'fixA':>5} {'fixB':>5} | {'ARI_w':>6} {'ARI_a':>6} {'ARI_A':>6} {'ARI_B':>6}")
        print("-" * 110)
        for _, row in k1_cases.iterrows():
            print(
                f"{row['name']:<35} {row['true_k']:>4} "
                f"{row['k_wald']:>5.0f} {row['k_adj']:>5.0f} {row['k_fixA']:>5.0f} {row['k_fixB']:>5.0f} | "
                f"{row['ari_wald']:>6.3f} {row['ari_adj']:>6.3f} {row['ari_fixA']:>6.3f} {row['ari_fixB']:>6.3f}"
            )

    # ── Cases where fixes changed K (non-K=1) ───────────────────────────
    changed = valid[
        ((valid["k_fixA"] != valid["k_adj"]) | (valid["k_fixB"] != valid["k_adj"]))
        & ~((valid["k_adj"] == 1) & (valid["k_wald"] > 1))
    ]
    if len(changed) > 0:
        print(f"\n{'='*95}")
        print(f"OTHER CASES WHERE FIXES CHANGED K ({len(changed)} cases)")
        print(f"{'='*95}")
        print(f"{'Name':<35} {'true':>4} {'wald':>5} {'adj':>5} {'fixA':>5} {'fixB':>5} | {'ARI_a':>6} {'ARI_A':>6} {'ARI_B':>6}")
        print("-" * 100)
        for _, row in changed.iterrows():
            print(
                f"{row['name']:<35} {row['true_k']:>4} "
                f"{row['k_wald']:>5.0f} {row['k_adj']:>5.0f} {row['k_fixA']:>5.0f} {row['k_fixB']:>5.0f} | "
                f"{row['ari_adj']:>6.3f} {row['ari_fixA']:>6.3f} {row['ari_fixB']:>6.3f}"
            )


if __name__ == "__main__":
    main()
