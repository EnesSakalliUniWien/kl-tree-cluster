#!/usr/bin/env python3
"""Diagnose WHY the 2 remaining test cases get K=1.

Traces Gate 2 (edge) and Gate 3 (sibling) decisions at every node
to find exactly where the pipeline kills splits.

Cases:
1. unbalanced 96x36 (complete linkage, as in integration test)
2. 'clear' from SMALL_TEST_CASES (benchmark pipeline)
3. feature_matrix.tsv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

print(
    f"Config: EIGENVALUE_WHITENING={config.EIGENVALUE_WHITENING}, "
    f"SPECTRAL_METHOD={config.SPECTRAL_METHOD}, "
    f"SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}, "
    f"FELSENSTEIN_SCALING={config.FELSENSTEIN_SCALING}"
)


def diagnose_case(name, data_df, linkage_method="average"):
    print(f"\n{'='*70}")
    print(f"  CASE: {name}  ({data_df.shape[0]} x {data_df.shape[1]}, linkage={linkage_method})")
    print(f"{'='*70}")

    Z = linkage(pdist(data_df.values, metric="hamming"), method=linkage_method)
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    # Full decomposition — populates tree.stats_df with both gates
    decomp = tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)
    K = decomp["num_clusters"]
    td = tree.stats_df

    # Spectral info from attrs
    spectral_dims = td.attrs.get("_spectral_dims", {})
    pca_projs = td.attrs.get("_pca_projections", {})

    # Gate 2 summary
    sig_col = "Child_Parent_Divergence_Significant"
    internal = [n for n in tree.nodes if tree.out_degree(n) > 0]
    n_sig = td[sig_col].sum() if sig_col in td.columns else 0
    print(f"\n  Gate 2: {n_sig} significant out of {len(internal)} internal nodes")
    print(f"  Final K = {K}")

    # Gate-by-gate trace: BFS from root, show first 30 internal nodes
    root = next(n for n, d in tree.in_degree() if d == 0)

    print(
        f"\n  {'Node':>8s} {'nLeaf':>5s} {'specK':>5s} {'pcaK':>4s} "
        f"{'G2_L':>5s} {'G2_R':>5s} {'G3_raw_p':>10s} {'G3_bh_p':>10s} "
        f"{'diff':>5s} {'skip':>5s} {'Decision':>10s}"
    )
    print(
        f"  {'-'*8} {'-'*5} {'-'*5} {'-'*4} "
        f"{'-'*5} {'-'*5} {'-'*10} {'-'*10} "
        f"{'-'*5} {'-'*5} {'-'*10}"
    )

    queue = [root]
    shown = 0
    while queue and shown < 30:
        node = queue.pop(0)
        children = list(tree.successors(node))
        if len(children) != 2:
            continue

        left, right = children
        n_leaves = tree.nodes[node].get("leaf_count", "?")
        sk = spectral_dims.get(node, "?")
        pk = pca_projs[node].shape[0] if node in pca_projs else "?"

        g2_l = td.loc[left, sig_col] if left in td.index and sig_col in td.columns else "?"
        g2_r = td.loc[right, sig_col] if right in td.index and sig_col in td.columns else "?"

        raw_p = td.loc[node].get("Sibling_Raw_P_value", "?")
        bh_p = td.loc[node].get("Sibling_BH_P_value", "?")
        sib_diff = td.loc[node].get("Sibling_BH_Different", "?")
        sib_skip = td.loc[node].get("Sibling_Divergence_Skipped", "?")

        g2_pass = (g2_l == True) or (g2_r == True)
        if not g2_pass:
            decision = "MERGE(G2)"
        elif sib_skip is True:
            decision = "MERGE(skip)"
        elif sib_diff is True:
            decision = "SPLIT"
        else:
            decision = "MERGE(G3)"

        def fmt_p(v):
            if isinstance(v, (float, np.floating)):
                return f"{v:.4e}"
            return str(v)

        print(
            f"  {node:>8s} {str(n_leaves):>5s} {str(sk):>5s} {str(pk):>4s} "
            f"{str(g2_l):>5s} {str(g2_r):>5s} {fmt_p(raw_p):>10s} {fmt_p(bh_p):>10s} "
            f"{str(sib_diff):>5s} {str(sib_skip):>5s} {decision:>10s}"
        )

        if decision == "SPLIT":
            queue.extend(children)

        shown += 1

    # Calibration audit
    audit = td.attrs.get("sibling_divergence_audit")
    if audit:
        print(
            f"\n  Calibration: method={audit.get('calibration_method', '?')}, "
            f"global_c_hat={audit.get('global_c_hat', '?')}, "
            f"n_calibration={audit.get('calibration_n', '?')}"
        )
        diag = audit.get("diagnostics", {})
        if "regression_beta" in diag:
            print(f"  Regression beta={diag['regression_beta']}")
            print(f"  R2={diag.get('regression_r2', '?')}")
        if "max_observed_ratio" in diag:
            print(f"  max_observed_ratio={diag['max_observed_ratio']}")

    # Show distribution of Gate 3 raw p-values for nodes where Gate 2 passed
    if "Sibling_Raw_P_value" in td.columns:
        g2_pass_parents = set()
        for node in internal:
            children = list(tree.successors(node))
            if len(children) == 2:
                l, r = children
                if (l in td.index and td.loc[l, sig_col] == True) or (
                    r in td.index and td.loc[r, sig_col] == True
                ):
                    g2_pass_parents.add(node)

        raw_ps = td.loc[list(g2_pass_parents), "Sibling_Raw_P_value"].dropna()
        if len(raw_ps) > 0:
            print(f"\n  Gate 3 raw p-values (Gate 2 passed, n={len(raw_ps)}):")
            print(
                f"    min={raw_ps.min():.4e}, median={raw_ps.median():.4e}, "
                f"max={raw_ps.max():.4e}"
            )
            n_under_05 = (raw_ps < 0.05).sum()
            print(f"    {n_under_05}/{len(raw_ps)} have raw p < 0.05")

        bh_ps = td.loc[list(g2_pass_parents), "Sibling_BH_P_value"].dropna()
        if len(bh_ps) > 0:
            print(f"  Gate 3 BH-adjusted p-values (Gate 2 passed, n={len(bh_ps)}):")
            print(
                f"    min={bh_ps.min():.4e}, median={bh_ps.median():.4e}, " f"max={bh_ps.max():.4e}"
            )
            n_under_05_bh = (bh_ps < 0.05).sum()
            print(f"    {n_under_05_bh}/{len(bh_ps)} have BH p < 0.05")


# ============================================================
# Case 1: Unbalanced 96x36 with complete linkage
# ============================================================
from benchmarks.shared.generators import generate_random_feature_matrix

data_dict, _ = generate_random_feature_matrix(
    n_rows=96,
    n_cols=36,
    entropy_param=0.25,
    n_clusters=4,
    random_seed=2024,
    balanced_clusters=False,
)
data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
diagnose_case("unbalanced_96x36", data_df, linkage_method="complete")

# ============================================================
# Case 2: 'clear' from SMALL_TEST_CASES
# ============================================================
try:
    from benchmarks.shared.generators import generate_gaussian_binary
    from tests.test_cases_config import SMALL_TEST_CASES

    case = next((c for c in SMALL_TEST_CASES if c["name"] == "clear"), None)
    if case is not None:
        data_dict2, true_labels2 = generate_gaussian_binary(
            n_samples=case["n_samples"],
            n_features=case["n_features"],
            n_clusters=case["n_clusters"],
            cluster_std=case["cluster_std"],
            seed=case["seed"],
        )
        data_df2 = pd.DataFrame.from_dict(data_dict2, orient="index").astype(int)
        diagnose_case("clear_case", data_df2)
    else:
        print("\n  'clear' case not found")
except Exception as e:
    print(f"\n  Error loading clear case: {e}")
    import traceback

    traceback.print_exc()

# ============================================================
# Case 3: feature_matrix.tsv
# ============================================================
fpath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feature_matrix.tsv"
)
if os.path.exists(fpath):
    data_fm = pd.read_csv(fpath, sep="\t", index_col=0)
    diagnose_case("feature_matrix.tsv", data_fm)
else:
    print("\n  feature_matrix.tsv not found")
    print("\n  feature_matrix.tsv not found")
