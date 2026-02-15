"""A/B benchmark: Wald (old) vs Cousin F-test (new) on the full benchmark suite.

Uses the shared benchmark pipeline infrastructure to materialize cases properly.
Runs each case with both sibling test methods and compares ARI, NMI, cluster count.
"""

import sys
from pathlib import Path

# Ensure project root is on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from benchmarks.shared.util.decomposition import _labels_from_decomposition
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def run_case_with_method(data_df, true_labels, distance_condensed, meta, method):
    """Run decomposition with a specific sibling test method."""
    original = config.SIBLING_TEST_METHOD
    config.SIBLING_TEST_METHOD = method
    try:
        Z = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
        result = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.ALPHA_LOCAL,
            sibling_alpha=config.SIBLING_ALPHA,
        )
        predicted = np.array(_labels_from_decomposition(result, data_df.index.tolist()))
        assigned = predicted != -1
        true_arr = np.array(true_labels)

        if assigned.sum() < 2:
            ari = nmi = 0.0
        else:
            ari = adjusted_rand_score(true_arr[assigned], predicted[assigned])
            nmi = normalized_mutual_info_score(true_arr[assigned], predicted[assigned])

        return {
            "k_pred": result["num_clusters"],
            "ari": ari,
            "nmi": nmi,
        }
    except Exception as e:
        return {
            "k_pred": -1,
            "ari": np.nan,
            "nmi": np.nan,
            "error": str(e),
        }
    finally:
        config.SIBLING_TEST_METHOD = original


def main():
    cases = get_default_test_cases()
    print(f"Running A/B benchmark on {len(cases)} cases...")
    print("=" * 110)

    rows = []
    errors = []
    for i, tc in enumerate(cases):
        name = tc.get("name", f"case_{i}")
        k_true = tc.get("n_clusters", "?")

        # Materialize case data using the benchmark generator
        try:
            data_df, true_labels, X_continuous, meta = generate_case_data(tc)
        except Exception as e:
            print(f"  [{i+1:3d}/{len(cases)}] {name:40s}  SKIP (generator error: {e})")
            errors.append({"case": name, "error": str(e)})
            continue

        # Compute distance matrix
        generator = tc.get("generator", "gaussian")
        if generator == "sbm" and meta.get("distance_condensed") is not None:
            distance_condensed = meta["distance_condensed"]
        else:
            distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)

        r_w = run_case_with_method(data_df, true_labels, distance_condensed, meta, "wald")
        r_c = run_case_with_method(data_df, true_labels, distance_condensed, meta, "cousin_ftest")

        rows.append(
            {
                "case": name,
                "k_true": k_true,
                "wald_k": r_w["k_pred"],
                "cousin_k": r_c["k_pred"],
                "wald_ari": r_w["ari"],
                "cousin_ari": r_c["ari"],
                "wald_nmi": r_w["nmi"],
                "cousin_nmi": r_c["nmi"],
            }
        )

        w_err = r_w.get("error", "")
        c_err = r_c.get("error", "")
        status = "ERR" if w_err or c_err else "OK"
        err_msg = f"  W:{w_err} C:{c_err}" if status == "ERR" else ""

        w_ari = f"{r_w['ari']:.3f}" if np.isfinite(r_w["ari"]) else "  N/A"
        c_ari = f"{r_c['ari']:.3f}" if np.isfinite(r_c["ari"]) else "  N/A"

        print(
            f"  [{i+1:3d}/{len(cases)}] {name:40s}  "
            f"K:{k_true:>2} -> W={r_w['k_pred']:3d} C={r_c['k_pred']:3d}  "
            f"ARI: W={w_ari} C={c_ari}  {status}{err_msg}"
        )

    if not rows:
        print("No cases completed successfully.")
        return

    df = pd.DataFrame(rows)

    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    for metric in ["ari", "nmi"]:
        w = df[f"wald_{metric}"].dropna()
        c = df[f"cousin_{metric}"].dropna()
        print(f"\n  {metric.upper()}:")
        print(f"    Wald:   mean={w.mean():.4f}  median={w.median():.4f}")
        print(f"    Cousin: mean={c.mean():.4f}  median={c.median():.4f}")
        print(
            f"    Cousin wins: {(c > w).sum()}/{len(w)}, "
            f"Wald wins: {(w > c).sum()}/{len(w)}, "
            f"Tied: {(w == c).sum()}/{len(w)}"
        )

    # Cluster count accuracy
    df["wald_k_exact"] = df["wald_k"] == df["k_true"]
    df["cousin_k_exact"] = df["cousin_k"] == df["k_true"]
    print("\n  Exact K match:")
    print(f"    Wald:   {df['wald_k_exact'].sum()}/{len(df)}")
    print(f"    Cousin: {df['cousin_k_exact'].sum()}/{len(df)}")

    # Over/under splitting
    df["wald_over"] = df["wald_k"] > df["k_true"]
    df["cousin_over"] = df["cousin_k"] > df["k_true"]
    df["wald_under"] = df["wald_k"] < df["k_true"]
    df["cousin_under"] = df["cousin_k"] < df["k_true"]
    print("\n  Over-splitting (K_pred > K_true):")
    print(f"    Wald:   {df['wald_over'].sum()}/{len(df)}")
    print(f"    Cousin: {df['cousin_over'].sum()}/{len(df)}")
    print("\n  Under-splitting (K_pred < K_true):")
    print(f"    Wald:   {df['wald_under'].sum()}/{len(df)}")
    print(f"    Cousin: {df['cousin_under'].sum()}/{len(df)}")

    # Cases where cousin is much worse
    worse = df[df["cousin_ari"] < df["wald_ari"] - 0.1]
    if len(worse) > 0:
        print(f"\n  Cases where cousin ARI is >0.1 worse ({len(worse)}):")
        for _, row in worse.iterrows():
            print(
                f"    {row['case']:40s}  K:{row['k_true']:>2} -> "
                f"W={row['wald_k']:.0f} C={row['cousin_k']:.0f}  "
                f"ARI: W={row['wald_ari']:.3f} C={row['cousin_ari']:.3f}"
            )

    # Cases where cousin is much better
    better = df[df["cousin_ari"] > df["wald_ari"] + 0.1]
    if len(better) > 0:
        print(f"\n  Cases where cousin ARI is >0.1 better ({len(better)}):")
        for _, row in better.iterrows():
            print(
                f"    {row['case']:40s}  K:{row['k_true']:>2} -> "
                f"W={row['wald_k']:.0f} C={row['cousin_k']:.0f}  "
                f"ARI: W={row['wald_ari']:.3f} C={row['cousin_ari']:.3f}"
            )

    if errors:
        print(f"\n  Generator errors: {len(errors)}")
        for e in errors:
            print(f"    {e['case']}: {e['error']}")

    return df


if __name__ == "__main__":
    main()
